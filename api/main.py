from __future__ import annotations

import asyncio
import http.client
import ipaddress
import json
import math
import os
import re
import socket
import subprocess
import traceback
import unicodedata
import urllib.parse
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from numbers import Real
from threading import Event, Lock, Thread
from fastapi import Body
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

import logging
import time

# Configure standard logging to include timestamps
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, Depends, Query, Request, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_serializer, field_validator, model_validator
from sqlalchemy.orm import Session
import pandas as pd
import requests

from api.database import UserDB, VersionStatsDB, FeedbackDB, SponsorDB, ProviderDB, init_db, get_db, get_db_ctx
from api.job_store import get_job_store as _new_job_store
from api.services import auth_service, portfolio_import_service, report_service, token_service, watchlist_service, scheduled_service, tracking_board_service, feedback_service, sponsor_service, role_routing_service, custom_prompt_service

def _get_real_ip(request: Request) -> Optional[str]:
    """Extract real client IP, preferring Cloudflare/proxy headers."""
    if request is None:
        return None
    # Cloudflare Tunnel injects the real client IP here
    ip = request.headers.get("CF-Connecting-IP")
    if ip:
        return ip.strip()
    # Standard proxy header fallback
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.graph.data_collector import DataCollector
from tradingagents.agents.utils.prompt_injection import DEFAULT_PLACEMENT

# 全局共享 DataCollector：同一 ticker+date 的数据只拉一次，所有 job 复用缓存
_shared_data_collector = DataCollector()
from tradingagents.dataflows.trade_calendar import (
    TradeCalendarUnavailableError,
    cn_today_str,
    now_cn,
)
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.graph.intent_parser import parse_intent as _parse_intent
from tradingagents.agents.utils.context_utils import USER_CONTEXT_KEYS, normalize_user_context
from tradingagents.agents.utils.agent_states import current_tracker_var


def _cors_allow_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    default_origins = [
        "http://127.0.0.1:5174",
        "http://localhost:5174",
        "http://127.0.0.1:5175",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]
    if not raw:
        return default_origins
    return [item.strip() for item in raw.split(",") if item.strip()]


def _cors_allow_origin_regex() -> str | None:
    raw = os.getenv("CORS_ALLOW_ORIGIN_REGEX", "").strip()
    return raw or None


def _report_version_stats() -> None:
    """Report anonymous version stats to the official site."""
    import threading, uuid

    def _send():
        try:
            requests.post(
                "https://app.510168.xyz/api/version-stats",
                json={"v": APP_VERSION, "nonce": uuid.uuid4().hex},
                timeout=30,
            )
        except Exception as exc:
            logger.debug("version-stats report failed: %s", exc)

    threading.Thread(target=_send, daemon=True).start()


def _resolve_scheduled_trade_date(trade_date: str) -> str:
    """Resolve scheduler defaults using the current CN market phase.

    Scheduled/manual triggers pass today's local date. Treat that value as an
    omitted ordinary-analysis date so pre-open and intraday runs use the last
    completed session; non-today dates retain their explicit date semantics.
    """
    today = cn_today_str()
    explicit = bool(trade_date and trade_date != today)
    return _normalize_analysis_trade_date(
        trade_date if explicit else None,
        explicit=explicit,
    )


def _normalize_analysis_trade_date(
    trade_date: Optional[str],
    *,
    explicit: Optional[bool] = None,
    now: Optional[datetime] = None,
) -> str:
    """Resolve an ordinary-analysis date against the CN calendar and session.

    A missing/empty date is a default: during pre-open, lunch, or an active
    session it resolves to the latest completed trading day, while post-close
    it may resolve to today's trading day.  A non-empty date is explicit by
    default and is only rolled back for a weekend/holiday; this preserves a
    user's request for today's unfinished data so providers can report a gap
    instead of silently substituting yesterday.

    If the calendar cannot be loaded, explicit dates remain usable for the
    caller's requested-date semantics. Omitted defaults fail closed and return
    an explicit unavailable marker rather than selecting today's date.
    """
    from tradingagents.dataflows.trade_calendar import (
        TradeCalendarUnavailableError,
        resolve_cn_analysis_date,
    )

    raw = str(trade_date or "").strip()
    if raw.startswith("【数据获取失败】交易日历") and explicit is not True:
        return raw
    if explicit is None:
        explicit = bool(raw)
    fallback = raw or cn_today_str()
    try:
        return resolve_cn_analysis_date(
            raw or None,
            explicit=bool(explicit),
            now=now,
        )
    except TradeCalendarUnavailableError as exc:
        logger.warning(
            "Analysis trade_date calendar unavailable for %s date; %s",
            "explicit" if explicit else "default",
            exc,
        )
        if not explicit:
            raise
        return fallback
    except (TypeError, ValueError) as exc:
        logger.warning("Analysis trade_date is invalid: %r: %s", raw, exc)
        if explicit:
            raise ValueError(f"分析日期无法解析：{raw!r}") from exc
        raise
    except Exception as exc:  # pragma: no cover - defensive: never break a request
        logger.warning(
            "Analysis trade_date normalization failed for %s date: %s",
            "explicit" if explicit else "default",
            exc,
        )
        if not explicit:
            raise
        return fallback


def _build_scheduled_analyze_request(
    db: Session,
    user_id: str,
    symbol: str,
    horizon: str,
    trade_date: str,
    scheduled_user_context: Optional[Dict[str, Any]] = None,
) -> "AnalyzeRequest":
    scheduled_user_context = scheduled_user_context or _build_imported_user_context(db, user_id, symbol)
    # Read user's saved analyst selection from DB
    user_cfg = auth_service.get_user_llm_config(db, user_id)
    selected = None
    if user_cfg and user_cfg.default_analysts:
        try:
            selected = json.loads(user_cfg.default_analysts)
        except Exception:
            pass
    req = AnalyzeRequest(
        symbol=symbol,
        trade_date=trade_date,
        horizons=[horizon],
        query=f"定时分析 {symbol}",
        user_intent={
            "ticker": symbol,
            "horizons": [horizon],
            "focus_areas": [],
            "specific_questions": [],
            "user_context": scheduled_user_context,
        },
        objective=scheduled_user_context.get("objective"),
        current_position=scheduled_user_context.get("current_position"),
        current_position_pct=scheduled_user_context.get("current_position_pct"),
        average_cost=scheduled_user_context.get("average_cost"),
        user_notes=scheduled_user_context.get("user_notes"),
    )
    if selected:
        req.selected_analysts = selected
    return req


async def _run_manual_trigger(
    task: dict,
    requested_trade_date: str,
    job_id: str,
) -> None:
    """Execute a manual-trigger analysis (no scheduler concurrency control).

    Used by the /v1/scheduled/{id}/trigger and /v1/scheduled/batch/trigger
    endpoints. Calls _run_job directly then records the test result.
    """
    task_id = task["id"]
    user_id = task["user_id"]
    symbol = task["symbol"]
    horizon = task.get("horizon") or "short"

    actual_trade_date: Optional[str] = None

    try:
        actual_trade_date = _resolve_scheduled_trade_date(requested_trade_date)
        _log(f"[Manual Trigger] {symbol} trade_date={actual_trade_date} (requested={requested_trade_date})")
        with get_db_ctx() as db:
            scheduled_user_context = task.get("manual_user_context") or _build_imported_user_context(
                db, user_id, symbol
            )
            req = _build_scheduled_analyze_request(
                db=db,
                user_id=user_id,
                symbol=symbol,
                horizon=horizon,
                trade_date=actual_trade_date,
                scheduled_user_context=scheduled_user_context,
            )

        await _run_job(job_id, req, False, True, user_id, "scheduled_manual")
        job_state = _get_job(job_id)
        if job_state.get("status") == "failed":
            raise RuntimeError(job_state.get("error") or f"manual trigger job {job_id} failed")
        with get_db_ctx() as db:
            scheduled_service.record_manual_test_result(db, task_id, "success", report_id=job_id)
        _log(f"[Manual Trigger] Completed {symbol}")
    except Exception as e:
        error_text = f"{type(e).__name__}: {e}"
        logger.error(f"[Manual Trigger] Failed {symbol}: {e}\n{traceback.format_exc()}")
        _set_job(
            job_id,
            status="failed",
            error=error_text,
            finished_at=_utcnow_iso(),
            overtime=False,
            overtime_at=None,
        )
        _emit_job_event(job_id, "job.failed", {"job_id": job_id, "error": error_text})
        with get_db_ctx() as db:
            scheduled_service.record_manual_test_result(db, task_id, "failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and cleanup on shutdown."""
    # Security startup guard (DAV-66): refuse to boot without a TA_APP_SECRET_KEY
    # unless the operator explicitly opted into the insecure default for local
    # development (TA_ALLOW_DEFAULT_SECRET=1). Must run before init_db() so no
    # data is ever written using the well-known default key.
    auth_service.ensure_secure_secret_configured()
    # 全局 socket 默认超时：akshare 等库内部的 requests 调用不传 timeout，
    # 网络丢包时 TLS 握手/读会永久阻塞，僵尸线程逐渐占满线程池（见 healthz 探针）。
    # uvicorn/asyncio 的服务端 socket 显式 setblocking(False)，不受此影响；
    # httpx/openai SDK 自带超时配置，也不受影响。
    socket.setdefaulttimeout(float(os.getenv("TA_SOCKET_DEFAULT_TIMEOUT", "60")))
    _log(f"Global socket default timeout set to {socket.getdefaulttimeout()}s.")
    # Raise the AnyIO thread limiter ceiling so frequent sync endpoints
    # (tracking-board polling, /v1/jobs/{id} polling, akshare-backed
    # market endpoints) cannot starve each other when the event loop is
    # also running long-lived `_run_job` tasks.
    try:
        from anyio import to_thread as _anyio_to_thread

        limiter = _anyio_to_thread.current_default_thread_limiter()
        desired = int(os.getenv("ANYIO_THREAD_LIMIT", "120"))
        if limiter.total_tokens < desired:
            limiter.total_tokens = desired
            _log(f"AnyIO thread limiter raised to {desired}.")
    except Exception as exc:
        _log(f"Could not raise AnyIO thread limiter: {exc}")

    # Default asyncio executor is used by `asyncio.to_thread`. The CPython
    # default is `min(32, cpu_count + 4)`, which is too small when many
    # `_run_job_inner` coroutines fan out concurrent `to_thread` calls for
    # DB writes, LLM extraction, and akshare data collection.
    global _default_executor
    new_default_executor: Optional[ThreadPoolExecutor] = None
    try:
        loop = asyncio.get_running_loop()
        executor_workers = int(os.getenv("ASYNCIO_DEFAULT_EXECUTOR_WORKERS", "64"))
        new_default_executor = ThreadPoolExecutor(
            max_workers=executor_workers,
            thread_name_prefix="ta-asyncio",
        )
        loop.set_default_executor(new_default_executor)
        _default_executor = new_default_executor
        _log(f"Default asyncio executor set to {executor_workers} workers.")
    except Exception as exc:
        _log(f"Could not configure default asyncio executor: {exc}")

    identity = await asyncio.to_thread(_get_runtime_identity)
    _log(_runtime_identity_log_line(identity))
    init_db()
    _log("Database initialized.")
    store = get_job_store()
    store.clear()
    _background_tasks.clear()

    # Security: reaching this point without a custom key means the operator
    # explicitly opted into the insecure built-in default (TA_ALLOW_DEFAULT_SECRET=1)
    # for local development. Warn loudly so it cannot be confused with a secure boot.
    if not auth_service.is_custom_secret_configured():
        _log("=" * 70)
        _log("WARNING: TA_APP_SECRET_KEY is not set and TA_ALLOW_DEFAULT_SECRET=1.")
        _log("Using hardcoded default key. ALL encryption and JWT signing")
        _log("is INSECURE. Set TA_APP_SECRET_KEY for any non-local deployment.")
        _log("=" * 70)

    _report_version_stats()
    # Pre-load trade calendar (uses mini_racer/V8 which is not thread-safe)
    from tradingagents.dataflows.trade_calendar import _load_cn_trade_dates
    _load_cn_trade_dates()
    _log("Trade calendar pre-loaded.")
    # Pre-load stock + ETF name map
    await asyncio.to_thread(_load_cn_stock_map)
    _log("Stock map pre-loaded on startup.")

    # Recover orphan pending/running reports left by interrupted processes.
    # Without this, DB "running" zombies accumulate and any UI/scheduler that
    # keys off active status keeps fighting real work for LLM capacity.
    try:
        from api.services import report_service as _report_service

        with get_db_ctx() as _db:
            # No in-memory jobs are live yet at startup (store was just cleared).
            stats = _report_service.recover_stale_active_reports(
                _db,
                active_job_ids=[],
                error_message="进程中断，启动恢复流程标记",
            )
        _log(
            f"Recovered stale active reports: failed={stats.get('failed', 0)} "
            f"(error marked as process-interrupt recovery)."
        )
    except Exception as exc:
        _log(f"Stale report recovery failed (non-fatal): {exc}")

    yield
    _log("Shutting down: Cleaning up resources...")
    _executor.shutdown(wait=True)
    if new_default_executor is not None:
        new_default_executor.shutdown(wait=False)
    _log("Executor shutdown complete.")


_is_prod = os.getenv("ENV", "").lower() == "prod"


def _get_version() -> str:
    """Get app version: APP_VERSION env > package metadata > 'dev'."""
    v = os.getenv("APP_VERSION")
    if v:
        return v
    try:
        from importlib.metadata import version as pkg_version
        return pkg_version("tradingagents")
    except Exception:
        return "dev"


APP_VERSION = _get_version()


_RUNTIME_IDENTITY_UNKNOWN = "unknown"
_RUNTIME_IDENTITY_GIT_TIMEOUT_SECONDS = 0.5
_RUNTIME_IDENTITY_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$", re.IGNORECASE)
_RUNTIME_SOURCE_ID = "tradingagents-api"
_RUNTIME_SOURCE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RuntimeIdentity:
    """Immutable code identity captured once on first runtime use."""

    commit_sha: str
    build_identity: str
    version: str
    source_id: str
    source_root: str

    def public_payload(self) -> Dict[str, str]:
        return {
            "commit_sha": self.commit_sha,
            "build_identity": self.build_identity,
            "version": self.version,
            "source_id": self.source_id,
        }

    def log_payload(self) -> Dict[str, str]:
        payload = self.public_payload()
        payload["source_root"] = self.source_root
        return payload


def _normalize_runtime_commit_sha(value: Any) -> str:
    candidate = str(value or "").strip()
    return (
        candidate
        if _RUNTIME_IDENTITY_SHA_RE.fullmatch(candidate)
        else _RUNTIME_IDENTITY_UNKNOWN
    )


def _resolve_runtime_commit_sha(source_root: Path) -> str:
    """Read the exact checkout SHA without guessing when Git metadata is absent."""
    try:
        # Ambient Git variables can redirect discovery to an unrelated checkout;
        # identity must describe this source root or remain explicitly unknown.
        git_env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        }
        result = subprocess.run(
            [
                "git",
                "-C",
                str(source_root),
                "rev-parse",
                "--show-toplevel",
                "--verify",
                "HEAD^{commit}",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=_RUNTIME_IDENTITY_GIT_TIMEOUT_SECONDS,
            env=git_env,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(lines) != 2 or Path(lines[0]).resolve() != source_root.resolve():
            return _RUNTIME_IDENTITY_UNKNOWN
        return _normalize_runtime_commit_sha(lines[1])
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError, RuntimeError):
        return _RUNTIME_IDENTITY_UNKNOWN
    except Exception as exc:  # pragma: no cover - defensive subprocess boundary
        logger.warning("[Runtime Identity] Git metadata unavailable: %s", exc)
        return _RUNTIME_IDENTITY_UNKNOWN


def _build_runtime_identity(
    source_root: Path,
    commit_sha: str,
    app_version: str,
) -> RuntimeIdentity:
    normalized_sha = _normalize_runtime_commit_sha(commit_sha)
    build_identity = (
        f"{_RUNTIME_SOURCE_ID}@{normalized_sha}"
        if normalized_sha != _RUNTIME_IDENTITY_UNKNOWN
        else _RUNTIME_IDENTITY_UNKNOWN
    )
    return RuntimeIdentity(
        commit_sha=normalized_sha,
        build_identity=build_identity,
        version=app_version,
        source_id=_RUNTIME_SOURCE_ID,
        source_root=str(source_root),
    )


_RUNTIME_IDENTITY_CACHE: Optional[RuntimeIdentity] = None
_RUNTIME_IDENTITY_LOCK = Lock()


def _get_runtime_identity() -> RuntimeIdentity:
    """Resolve identity once, keeping Git failures out of every health probe."""
    global _RUNTIME_IDENTITY_CACHE
    if _RUNTIME_IDENTITY_CACHE is not None:
        return _RUNTIME_IDENTITY_CACHE
    with _RUNTIME_IDENTITY_LOCK:
        if _RUNTIME_IDENTITY_CACHE is None:
            _RUNTIME_IDENTITY_CACHE = _build_runtime_identity(
                source_root=_RUNTIME_SOURCE_ROOT,
                commit_sha=_resolve_runtime_commit_sha(_RUNTIME_SOURCE_ROOT),
                app_version=APP_VERSION,
            )
    return _RUNTIME_IDENTITY_CACHE


def _runtime_identity_log_line(identity: RuntimeIdentity) -> str:
    return "[Runtime Identity] " + json.dumps(
        identity.log_payload(),
        ensure_ascii=False,
        sort_keys=True,
    )


app = FastAPI(
    title="TradingAgents-AShare API",
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url=None if _is_prod else "/docs",
    redoc_url=None if _is_prod else "/redoc",
    openapi_url=None if _is_prod else "/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_origin_regex=_cors_allow_origin_regex(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=int(os.getenv("TA_MAX_WORKERS", "2")))
# lifespan 中创建的 asyncio 默认 executor，healthz 探针用它报告饱和度
_default_executor: Optional[ThreadPoolExecutor] = None

# ── Singleton job store (in-memory or Redis depending on REDIS_URL) ─────────
_job_store_instance: Optional[Any] = None

def get_job_store():
    global _job_store_instance
    if _job_store_instance is None:
        _job_store_instance = _new_job_store()
    return _job_store_instance

# Runtime config overrides via PATCH /v1/config
_global_config_overrides: Dict[str, Any] = {}

# Allowlist for config_overrides from client requests.
# Security: prevents injection of api_key, backend_url, or other sensitive keys.
_CONFIG_OVERRIDES_ALLOWLIST = {
    "llm_provider", "deep_think_llm", "quick_think_llm",
    "max_debate_rounds", "max_risk_discuss_rounds",
    "prompt_language",
}
# Hold references to fire-and-forget tasks so they are not garbage collected
_background_tasks: set = set()

# ── A-share stock name → code cache ──────────────────────────────────────────
_cn_stock_map: Optional[Dict[str, str]] = None  # name -> "XXXXXX.SH/SZ"
_cn_stock_reverse_map: Optional[Dict[str, str]] = None  # code -> name
# NFKC-normalized name → code view of the same cache (built lazily).
_cn_stock_map_norm: Optional[Dict[str, str]] = None
_cn_stock_map_norm_src: Optional[Dict[str, str]] = None  # source map this view is built from
_cn_stock_map_lock = Lock()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Soft deadline for long-running analyses.  This keeps the historical
# TA_JOB_TIMEOUT setting API-compatible, but crossing it is now an overtime
# notification rather than a terminal failure.  TA_JOB_HARD_TIMEOUT is the
# resource-safety backstop that releases scheduler capacity if a workflow is
# genuinely wedged.  Set either value to 0 to disable that deadline.
_JOB_TIMEOUT = int(os.getenv("TA_JOB_TIMEOUT", "1800"))
_JOB_HARD_TIMEOUT = int(os.getenv("TA_JOB_HARD_TIMEOUT", "7200"))


def _create_tracked_task(coro, *, label: str = "Background task") -> asyncio.Task:
    """Create an asyncio task and keep a reference to prevent GC.
    Also logs unhandled exceptions via a done callback."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)

    def _on_done(t: asyncio.Task):
        _background_tasks.discard(t)
        if not t.cancelled() and t.exception():
            logger.error("%s failed: %s", label, t.exception())

    task.add_done_callback(_on_done)
    return task


def _log(msg: str):
    """Helper to log with timestamp via standard logging."""
    logger.info(msg)


def _serialize_datetime_utc(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat()


class _TokenDatetimeFieldsMixin:
    """Shared JSON serializer for token ``created_at``/``last_used_at`` (audit P2-6)."""

    @field_serializer("created_at", "last_used_at", when_used="json")
    def serialize_token_datetimes(self, value: Optional[datetime]) -> Optional[str]:
        return _serialize_datetime_utc(value)


_cn_stock_map_loaded_at: float = 0  # timestamp of last successful load
_cn_stock_map_last_failure_at: float = 0  # timestamp of last failed load (0 = no recent failure)
_STOCK_MAP_TTL = 7 * 86400  # 7 days
# After a failed load, back off this long before retrying AkShare. A short
# failure must not hammer a rate-limited endpoint nor freeze the cache for the
# full 7-day TTL. Configurable via TA_STOCK_MAP_RETRY_INTERVAL (seconds).
_STOCK_MAP_FAILURE_RETRY_INTERVAL = int(os.getenv("TA_STOCK_MAP_RETRY_INTERVAL", "1800"))
_cn_stock_map_refresh_inflight = False
_cn_stock_map_refresh_event = Event()
_cn_stock_map_refresh_event.set()


def _stock_map_refresh_needed(now: Optional[float] = None) -> bool:
    """Return whether the name map needs a non-blocking refresh attempt."""
    current_time = time.time() if now is None else now
    cache_is_fresh = (
        _cn_stock_map is not None
        and _cn_stock_reverse_map is not None
        and _cn_stock_map_loaded_at > 0
        and _cn_stock_map_last_failure_at == 0
        and current_time - _cn_stock_map_loaded_at <= _STOCK_MAP_TTL
    )
    if cache_is_fresh:
        return False

    # A recent failure is deliberately left as an empty placeholder by the
    # loader, but it must not cause every report request to retry the provider.
    return not (
        _cn_stock_map_last_failure_at
        and current_time - _cn_stock_map_last_failure_at < _STOCK_MAP_FAILURE_RETRY_INTERVAL
    )


def _refresh_cn_stock_map_worker(refresh_event: Event) -> None:
    """Refresh the stock map outside the request thread and cache lock."""
    _run_cn_stock_map_refresh(refresh_event)


def _schedule_cn_stock_map_refresh() -> None:
    """Start at most one provider refresh without delaying the report response."""
    global _cn_stock_map_refresh_inflight, _cn_stock_map_refresh_event
    if not _stock_map_refresh_needed():
        return

    with _cn_stock_map_lock:
        if not _stock_map_refresh_needed() or _cn_stock_map_refresh_inflight:
            return
        _cn_stock_map_refresh_inflight = True
        refresh_event = Event()
        _cn_stock_map_refresh_event = refresh_event

    try:
        Thread(
            target=_refresh_cn_stock_map_worker,
            args=(refresh_event,),
            name="ta-stock-map-refresh",
            daemon=True,
        ).start()
    except Exception as exc:  # pragma: no cover - thread creation is platform code
        with _cn_stock_map_lock:
            if _cn_stock_map_refresh_event is refresh_event:
                _cn_stock_map_refresh_inflight = False
        refresh_event.set()
        _log(f"[StockMap] Could not start background refresh: {exc}")


def _fetch_cn_stock_map() -> Tuple[Dict[str, str], int, int]:
    """Fetch and validate stock/fund names without touching shared cache state."""
    import akshare as ak

    result: Dict[str, str] = {}
    # A-share stocks (static list, no anti-crawl issue)
    df = ak.stock_info_a_code_name()
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        code = str(row.get("code", "")).strip()
        if name and code:
            result[name] = _normalize_symbol(code)
    stock_count = len(result)

    # ETF / funds are supplemental; a provider failure here does not discard
    # a valid stock source, but a stock source with no usable rows is invalid.
    fund_count = 0
    try:
        fund_df = ak.fund_name_em()
        existing_codes = set(result.values())
        for _, row in fund_df.iterrows():
            code = str(row.get("基金代码", "")).strip()
            name = str(row.get("基金简称", "")).strip()
            if name and code and len(code) == 6 and code.isdigit():
                normalized = _normalize_symbol(code)
                if normalized not in existing_codes:
                    result[name] = normalized
                    existing_codes.add(normalized)
        fund_count = len(result) - stock_count
    except Exception as fe:
        _log(f"[StockMap] ETF/fund load skipped: {fe}")

    if stock_count == 0:
        raise RuntimeError("AkShare returned an empty stock name map")
    return result, stock_count, fund_count


def _finish_cn_stock_map_refresh(
    refresh_event: Event,
    result: Optional[Dict[str, str]] = None,
    stock_count: int = 0,
    fund_count: int = 0,
    error: Optional[Exception] = None,
) -> Dict[str, str]:
    """Publish one refresh outcome while holding the lock only briefly."""
    global _cn_stock_map, _cn_stock_reverse_map, _cn_stock_map_norm, _cn_stock_map_norm_src
    global _cn_stock_map_loaded_at, _cn_stock_map_last_failure_at, _cn_stock_map_refresh_inflight
    if error is None and not result:
        error = RuntimeError("AkShare returned an empty stock name map")
    now = time.time()
    with _cn_stock_map_lock:
        if error is None:
            loaded_map = result or {}
            _cn_stock_map = loaded_map
            _cn_stock_reverse_map = {
                code: name for name, code in loaded_map.items()
            }
            _cn_stock_map_norm = None
            _cn_stock_map_norm_src = None
            _cn_stock_map_loaded_at = now
            _cn_stock_map_last_failure_at = 0
            message = (
                f"[StockMap] Loaded {stock_count} stocks + {fund_count} ETFs/funds = "
                f"{len(loaded_map)} total."
            )
        else:
            _cn_stock_map = {}
            _cn_stock_reverse_map = {}
            _cn_stock_map_norm = None
            _cn_stock_map_norm_src = None
            _cn_stock_map_loaded_at = 0
            _cn_stock_map_last_failure_at = now
            message = (
                f"[StockMap] Failed to load: {error}; will retry in "
                f"{_STOCK_MAP_FAILURE_RETRY_INTERVAL // 60} minutes"
            )
        _cn_stock_map_refresh_inflight = False
    refresh_event.set()
    _log(message)
    return _cn_stock_map


def _run_cn_stock_map_refresh(refresh_event: Event) -> Dict[str, str]:
    """Run provider I/O outside the cache lock and publish its outcome."""
    try:
        result, stock_count, fund_count = _fetch_cn_stock_map()
    except Exception as exc:
        return _finish_cn_stock_map_refresh(refresh_event, error=exc)
    return _finish_cn_stock_map_refresh(
        refresh_event,
        result=result,
        stock_count=stock_count,
        fund_count=fund_count,
    )


def _load_cn_stock_map() -> Dict[str, str]:
    """Lazy-load and cache A-share stock + ETF/fund name→code mapping (7-day TTL).

    Uses akshare stock_info_a_code_name (static list, no anti-crawl) for A-shares,
    plus fund_name_em for ETFs/funds.

    Failure handling (DAV-92): a failed load records ``_cn_stock_map_last_failure_at``
    and is NOT treated as a valid cache, so the 7-day TTL only starts on success.
    Subsequent calls back off for ``_STOCK_MAP_FAILURE_RETRY_INTERVAL`` instead of
    retrying on every request or freezing for the full TTL. An empty stock source
    is treated as a failed load even when the fund source has rows.
    """
    global _cn_stock_map, _cn_stock_reverse_map, _cn_stock_map_norm, _cn_stock_map_norm_src
    global _cn_stock_map_refresh_inflight, _cn_stock_map_refresh_event
    wait_event: Optional[Event] = None
    owner_event: Optional[Event] = None
    now = time.time()

    with _cn_stock_map_lock:
        if _cn_stock_map is not None and (now - _cn_stock_map_loaded_at) > _STOCK_MAP_TTL:
            _cn_stock_map = None  # expire cache
            _cn_stock_reverse_map = None
            _cn_stock_map_norm = None
            _cn_stock_map_norm_src = None
        cache_is_fresh = (
            _cn_stock_map is not None
            and _cn_stock_reverse_map is not None
            and _cn_stock_map_loaded_at > 0
            and _cn_stock_map_last_failure_at == 0
            and now - _cn_stock_map_loaded_at <= _STOCK_MAP_TTL
        )
        if cache_is_fresh:
            return _cn_stock_map
        # A recent failed load must not hammer AkShare; wait out the retry window.
        if _cn_stock_map_last_failure_at and (
            now - _cn_stock_map_last_failure_at
        ) < _STOCK_MAP_FAILURE_RETRY_INTERVAL:
            return _cn_stock_map if _cn_stock_map is not None else {}
        if _cn_stock_map_refresh_inflight:
            wait_event = _cn_stock_map_refresh_event
        else:
            _cn_stock_map_refresh_inflight = True
            owner_event = Event()
            _cn_stock_map_refresh_event = owner_event

    if wait_event is not None:
        wait_event.wait()
        with _cn_stock_map_lock:
            return _cn_stock_map if _cn_stock_map is not None else {}
    return _run_cn_stock_map_refresh(owner_event)


def _get_reverse_stock_map() -> Dict[str, str]:
    """Return code→name mapping."""
    _load_cn_stock_map()
    return dict(_cn_stock_reverse_map or {})


def _get_reverse_stock_map_cached_only() -> Dict[str, str]:
    """Return code→name mapping only from already-warmed cache.

    For list pages we prefer a fast response over blocking on a cold AkShare lookup.
    When the cache is cold we simply return an empty mapping and let the UI fall back
    to stock codes. Search endpoints can still call _load_cn_stock_map() explicitly.
    """
    if _cn_stock_map is None or _cn_stock_reverse_map is None:
        return {}
    return dict(_cn_stock_reverse_map)


def _get_report_reverse_stock_map() -> Dict[str, str]:
    """Return cached report names and refresh a cold cache asynchronously."""
    code_to_name = _get_reverse_stock_map_cached_only()
    _schedule_cn_stock_map_refresh()
    return code_to_name


def _get_normalized_stock_map() -> Dict[str, str]:
    """NFKC-normalized name→code view of the stock map, cached per source map.

    A-share names may carry a full-width share-class letter (京东方Ａ); all
    matching in ``_search_cn_stock_by_name`` runs on this normalized view so a
    half-width query (京东方A) still hits. The view is rebuilt whenever the
    source ``_cn_stock_map`` object is replaced (TTL refresh or test seeding).
    """
    stock_map = _load_cn_stock_map()
    global _cn_stock_map_norm, _cn_stock_map_norm_src
    if _cn_stock_map_norm is None or _cn_stock_map_norm_src is not stock_map:
        _cn_stock_map_norm = {
            unicodedata.normalize("NFKC", name): code
            for name, code in stock_map.items()
        }
        _cn_stock_map_norm_src = stock_map
    return _cn_stock_map_norm


_CN_STOCK_INTENT_PREFIXES = (
    "帮我分析一下", "帮我分析", "请分析一下", "帮忙分析一下", "帮我看看",
    "帮忙看看", "请分析", "分析一下", "帮我查一下", "查一下", "解读一下",
    "复盘一下", "分析", "解读", "复盘", "看看", "看下", "关注", "推荐",
    "查询", "请问", "请", "我想", "我要", "给我", "帮忙",
)
_CN_STOCK_INTENT_SUFFIXES = (
    "怎么样呢", "怎么样啊", "走势怎么样", "如何看待", "怎么样", "怎么看",
    "走势如何", "如何呢", "如何", "呢", "吧", "啊",
)
_CN_STOCK_STRIP_MAX_ROUNDS = 5


def _strip_cn_stock_intent_words(text: str) -> str:
    """Strip leading request verbs / trailing interrogative particles.

    Only used as a last resort for whole-sentence queries (LLM-failure fallback,
    e.g. "分析京东方" → "京东方"). Conservative on purpose: only unambiguous
    request words are stripped, never nouns/adjectives that could begin a company
    name (e.g. 今天国际 must not lose its 今天 prefix).
    """
    s = text.strip()
    for _ in range(_CN_STOCK_STRIP_MAX_ROUNDS):
        before = s
        for prefix in _CN_STOCK_INTENT_PREFIXES:
            if s.startswith(prefix):
                s = s[len(prefix):].lstrip(" ，,、\t")
                break
        for suffix in _CN_STOCK_INTENT_SUFFIXES:
            if s.endswith(suffix):
                s = s[:-len(suffix)].rstrip(" ，,、\t")
                break
        s = s.strip()
        if s == before:
            break
    return s


def _match_cn_stock_name(norm_map: Dict[str, str], norm_query: str) -> Optional[str]:
    """Exact → substring → shortest-name scoring, all on NFKC-normalized text."""
    if norm_query in norm_map:
        return norm_map[norm_query]
    candidates = [(name, code) for name, code in norm_map.items()
                  if norm_query in name or name in norm_query]
    if len(candidates) == 1:
        return candidates[0][1]
    # Multiple partial matches: pick the one with shortest name (closest match).
    if candidates:
        candidates.sort(key=lambda x: len(x[0]))
        return candidates[0][1]
    return None


def _search_cn_stock_by_name(query: str) -> Optional[str]:
    """Look up A-share stock code by company name (exact then partial match).

    Matching runs on NFKC-normalized text so a full-width share-class letter in
    the map (京东方Ａ) matches a half-width query (京东方A) and vice versa. As a
    last resort for whole sentences (LLM-failure fallback passes the raw user
    message), request intent words are stripped and the remainder is matched.
    """
    query = query.strip()
    if not query:
        return None
    norm_map = _get_normalized_stock_map()
    norm_query = unicodedata.normalize("NFKC", query)
    hit = _match_cn_stock_name(norm_map, norm_query)
    if hit:
        return hit
    # 原文兜底：剥离常见意图词后再匹配（"分析京东方" → "京东方"）。
    stripped = _strip_cn_stock_intent_words(norm_query)
    if stripped and stripped != norm_query:
        hit = _match_cn_stock_name(norm_map, stripped)
        if hit:
            return hit
    return None


def _split_watchlist_batch_text(text: str) -> List[str]:
    return [token.strip() for token in re.split(r"[\s,，、；;]+", text.strip()) if token.strip()]


def _resolve_watchlist_identifier(
    raw: str,
    name_to_code: Dict[str, str],
    code_to_name: Dict[str, str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    token = raw.strip()
    if not token:
        return None, None, "输入为空"
    if token in name_to_code:
        symbol = name_to_code[token]
        return symbol, code_to_name.get(symbol, token), None
    symbol = _normalize_symbol(token)
    if symbol in code_to_name:
        return symbol, code_to_name.get(symbol, symbol), None
    return None, None, f"未识别的股票代码或名称: {token}"


_auth_scheme = HTTPBearer(auto_error=False)

FIXED_TEAMS = {
    "Analyst Team": [
        "Market Analyst",
        "Social Analyst",
        "News Analyst",
        "Fundamentals Analyst",
        "Macro Analyst",
        "Smart Money Analyst",
        "Volume Price Analyst",
    ],
    "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
    "Trading Team": ["Trader"],
    "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
    "Portfolio Management": ["Portfolio Manager"],
}
ANALYST_ORDER = ["market", "social", "news", "fundamentals", "macro", "smart_money", "volume_price"]
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Social Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
    "macro": "Macro Analyst",
    "volume_price": "Volume Price Analyst",
    "smart_money": "Smart Money Analyst",
    "bull": "Bull Researcher",
    "bear": "Bear Researcher",
    "Bull_Initial": "Bull Researcher",
    "Bear_Initial": "Bear Researcher",
    "Bull_Rebuttal": "Bull Researcher",
    "Bear_Rebuttal": "Bear Researcher",
    "research_manager": "Research Manager",
    "trader": "Trader",
    "aggressive": "Aggressive Analyst",
    "neutral": "Neutral Analyst",
    "conservative": "Conservative Analyst",
    "portfolio_manager": "Portfolio Manager",
}
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
    "macro": "macro_report",
    "smart_money": "smart_money_report",
    "volume_price": "volume_price_report",
}

# All analysts always run — each uses its own natural time window
# (technical/funds → short, fundamentals/macro → medium)
def _get_horizon_analysts(horizon: str, available: List[str]) -> List[str]:
    """Return all available analysts regardless of horizon."""
    return list(available)


_SUPPORTED_ANALYSIS_HORIZONS = ("short", "medium")
_DUAL_HORIZON_QUERY_RE = re.compile(
    r"(?:短线|短期).{0,16}(?:中线|中期)|(?:中线|中期).{0,16}(?:短线|短期)"
    r"|short(?:[- ]term)?.{0,24}medium(?:[- ]term)?"
    r"|medium(?:[- ]term)?.{0,24}short(?:[- ]term)?",
    re.IGNORECASE,
)
_MEDIUM_HORIZON_QUERY_RE = re.compile(
    r"中线|中期|几个月|季度|长期|趋势投资|medium(?:[- ]term)?",
    re.IGNORECASE,
)


def _normalize_analysis_horizons(
    raw: Any,
    *,
    query: Optional[str] = None,
) -> List[str]:
    """Normalize API horizon intent without changing upstream analyst windows.

    The API accepts only the two supported horizon names and treats an
    explicit natural-language request for both horizons as authoritative.
    Unknown or missing values retain the historical short-horizon default.
    """
    values = raw if isinstance(raw, (list, tuple)) else []
    normalized = {
        str(value).strip().lower()
        for value in values
        if str(value).strip().lower() in _SUPPORTED_ANALYSIS_HORIZONS
    }
    query_text = str(query or "")
    if _DUAL_HORIZON_QUERY_RE.search(query_text):
        normalized.update(_SUPPORTED_ANALYSIS_HORIZONS)
    elif not normalized and _MEDIUM_HORIZON_QUERY_RE.search(query_text):
        normalized.add("medium")
    if not normalized:
        normalized.add("short")
    return [horizon for horizon in _SUPPORTED_ANALYSIS_HORIZONS if horizon in normalized]


def _announcements_file() -> Path:
    return Path(__file__).resolve().parent / "announcements.json"


def _load_latest_announcement() -> Optional[Dict[str, Any]]:
    path = _announcements_file()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _log(f"[Announcements] Failed to read {path.name}: {exc}")
        return None

    announcements = raw.get("announcements") if isinstance(raw, dict) else raw
    if not isinstance(announcements, list):
        return None

    for item in announcements:
        if not isinstance(item, dict):
            continue
        if item.get("active", True) is False:
            continue
        return item
    return None


class UserContextInput(BaseModel):
    objective: Optional[str] = Field(None, description="用户目标动作，如建仓/加仓/减仓/止损/观察")
    risk_profile: Optional[str] = Field(None, description="风险偏好，如保守/平衡/激进")
    investment_horizon: Optional[str] = Field(None, description="持有周期，如短线/波段/中线")
    cash_available: Optional[float] = Field(None, description="可用资金")
    current_position: Optional[float] = Field(None, description="当前持仓数量")
    current_position_pct: Optional[float] = Field(None, description="当前仓位占比")
    average_cost: Optional[float] = Field(None, description="当前持仓成本")
    max_loss_pct: Optional[float] = Field(None, description="最大容忍亏损百分比")
    constraints: List[str] = Field(default_factory=list, description="用户的硬约束列表")
    user_notes: Optional[str] = Field(None, description="用户补充说明")


class AnalyzeRequest(UserContextInput):
    symbol: str = Field(default="", description="股票代码，如 600519.SH（当 query 包含代码时可省略）")
    trade_date: str = Field(default_factory=cn_today_str, description="交易日期 YYYY-MM-DD")
    # ``None`` means infer from Pydantic's fields-set for internal callers;
    # API/chat paths set this explicitly before resolving a default date.
    trade_date_explicit: Optional[bool] = Field(default=None, exclude=True, repr=False)
    selected_analysts: List[str] = Field(
        default_factory=lambda: ["market", "social", "news", "fundamentals", "macro", "smart_money", "volume_price"]
    )
    config_overrides: Dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = False
    # When set, triggers intent-driven analysis via streaming dual-horizon path
    query: Optional[str] = Field(default=None, description="自然语言查询，如：分析贵州茅台短线机会")
    horizons: List[str] = Field(default_factory=lambda: ["short"], description="分析周期列表，如 ['short'] 或 ['short','medium']")
    # Pre-parsed intent from _ai_extract_symbol_and_date (avoids second LLM call in _run_job)
    user_intent: Optional[Dict[str, Any]] = Field(default=None, description="预解析的用户意图，由 chat_completions 传入")


class AnalyzeResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    created_at: str


class BatchScheduledTriggerJob(BaseModel):
    item_id: str
    job_id: str
    symbol: str
    name: str
    status: Literal["pending", "running", "completed", "failed"]
    created_at: str
    current_position: Optional[float] = None
    average_cost: Optional[float] = None
    waiting_ahead_count: Optional[int] = None
    scheduled_running_count: Optional[int] = None
    scheduled_concurrency_limit: Optional[int] = None


class BatchScheduledTriggerResponse(BaseModel):
    summary: Dict[str, int]
    jobs: List[BatchScheduledTriggerJob]


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    symbol: Optional[str] = None
    trade_date: Optional[str] = None
    error: Optional[str] = None
    overtime: bool = False
    overtime_at: Optional[str] = None
    waiting_ahead_count: Optional[int] = None
    scheduled_running_count: Optional[int] = None
    scheduled_concurrency_limit: Optional[int] = None


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(UserContextInput):
    model: Optional[str] = "tradingagents-ashare"
    messages: List[ChatMessage]
    stream: bool = True
    selected_analysts: List[str] = Field(
        default_factory=lambda: ["market", "social", "news", "fundamentals", "macro", "smart_money", "volume_price"]
    )
    config_overrides: Dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = False


class KlineResponse(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    candles: List[Dict[str, Any]]


# Report API Models
# Re-export for backward compat: single implementation lives in report_service (audit P2-7).
_strict_unit_interval = report_service._strict_unit_interval


def _strict_report_probability(value: Any) -> Any:
    return report_service._strict_unit_interval(value, "probability")


def _strict_report_confidence(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("confidence must be a finite integer in [0, 100]")
    try:
        confidence = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("confidence must be a finite integer in [0, 100]") from exc
    if not math.isfinite(confidence) or not confidence.is_integer() or not 0.0 <= confidence <= 100.0:
        raise ValueError("confidence must be a finite integer in [0, 100]")
    return int(confidence)


class ReportCreateRequest(BaseModel):
    symbol: str = Field(..., description="股票代码")
    trade_date: str = Field(..., description="交易日期 YYYY-MM-DD")
    decision: Optional[str] = Field(None, description="交易决策")
    result_data: Optional[Dict[str, Any]] = Field(None, description="完整分析结果")
    probability: Optional[float] = None
    data_gaps: List[str] = Field(default_factory=list)
    falsification_conditions: List[str] = Field(default_factory=list)
    not_applicable: bool = False

    @field_validator("probability", mode="before")
    @classmethod
    def _validate_probability(cls, value: Any) -> Any:
        return _strict_report_probability(value)

    @model_validator(mode="after")
    def _validate_structured_boundary(self):
        structured = self.result_data.get("structured") if isinstance(self.result_data, dict) else None
        if structured is not None:
            if not isinstance(structured, dict):
                raise ValueError("structured report must be an object")
            if "confidence" in structured:
                _strict_report_confidence(structured["confidence"])
            if "probability" in structured:
                _strict_report_probability(structured["probability"])
        self.result_data = report_service.canonicalize_report_result_data(self.result_data)
        return self


class ReportResponse(BaseModel):
    id: str
    user_id: Optional[str]
    symbol: str
    name: Optional[str] = None
    trade_date: str
    status: Literal["pending", "running", "completed", "failed"] = "completed"
    error: Optional[str] = None
    decision: Optional[str]
    direction: Optional[str]
    confidence: Optional[int]
    probability: Optional[float] = None
    target_price: Optional[float]
    stop_loss_price: Optional[float]
    risk_items: Optional[List[Dict[str, Any]]] = None
    key_metrics: Optional[List[Dict[str, Any]]] = None
    data_gaps: List[str] = Field(default_factory=list)
    falsification_conditions: List[str] = Field(default_factory=list)
    not_applicable: bool = False
    analyst_traces: Optional[List[Dict[str, Any]]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    waiting_ahead_count: Optional[int] = None
    scheduled_running_count: Optional[int] = None
    scheduled_concurrency_limit: Optional[int] = None

    model_config = {"from_attributes": True}

    @field_serializer("created_at", "updated_at", when_used="json")
    def serialize_report_datetimes(self, value: Optional[datetime]) -> Optional[str]:
        return _serialize_datetime_utc(value)


class ReportDetailResponse(ReportResponse):
    market_report: Optional[str]
    sentiment_report: Optional[str]
    news_report: Optional[str]
    fundamentals_report: Optional[str]
    macro_report: Optional[str]
    smart_money_report: Optional[str]
    volume_price_report: Optional[str]
    game_theory_report: Optional[str]
    investment_plan: Optional[str]
    trader_investment_plan: Optional[str]
    final_trade_decision: Optional[str]
    result_data: Optional[Dict[str, Any]]


class ReportListResponse(BaseModel):
    total: int
    reports: List[ReportResponse]


class ReportBatchDeleteRequest(BaseModel):
    report_ids: List[str] = Field(default_factory=list)


class ReportBatchDeleteResponse(BaseModel):
    deleted_ids: List[str]
    missing_ids: List[str]


class LatestReportsBySymbolsRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)


class LatestReportsBySymbolsResponse(BaseModel):
    reports: List[ReportResponse]


class PortfolioOverviewResponse(BaseModel):
    watchlist: List[dict]
    scheduled: List[dict]
    latest_reports: List[ReportResponse]
    portfolio_import: Optional[dict] = None


class WatchlistAddRequest(BaseModel):
    text: Optional[str] = None
    symbol: Optional[str] = None


class ScheduledBatchIdsRequest(BaseModel):
    item_ids: List[str] = Field(default_factory=list)


class ScheduledBatchUpdateRequest(BaseModel):
    item_ids: List[str] = Field(default_factory=list)
    is_active: Optional[bool] = None
    horizon: Optional[str] = None
    trigger_time: Optional[str] = None


class AnnouncementItemResponse(BaseModel):
    title: str
    detail: str


class AnnouncementResponse(BaseModel):
    id: str
    tag: Optional[str] = None
    title: str
    summary: Optional[str] = None
    published_at: str
    items: List[AnnouncementItemResponse]
    cta_label: Optional[str] = None
    cta_path: Optional[str] = None


class LatestAnnouncementResponse(BaseModel):
    announcement: Optional[AnnouncementResponse] = None


class UserResponse(BaseModel):
    id: str
    email: str
    created_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None
    email_report_enabled: bool = True

    model_config = {"from_attributes": True}

    @field_serializer("created_at", "last_login_at", when_used="json")
    def serialize_user_datetimes(self, value: Optional[datetime]) -> Optional[str]:
        return _serialize_datetime_utc(value)


class AuthRequestCodeRequest(BaseModel):
    email: str


class AuthVerifyCodeRequest(BaseModel):
    email: str
    code: str


class AuthVerifyCodeResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class UserRuntimeConfigResponse(BaseModel):
    llm_provider: str
    deep_think_llm: str
    quick_think_llm: str
    backend_url: str
    max_debate_rounds: int
    max_risk_discuss_rounds: int
    has_api_key: bool = False
    has_wecom_webhook: bool = False
    wecom_webhook_display: Optional[str] = None
    server_fallback_enabled: bool = True
    email_report_enabled: bool = True
    wecom_report_enabled: bool = True
    default_analysts: List[str] = Field(default_factory=lambda: ["market", "social", "news", "fundamentals", "macro", "smart_money", "volume_price"])


class UserRuntimeConfigUpdateRequest(BaseModel):
    llm_provider: Optional[str] = None
    deep_think_llm: Optional[str] = None
    quick_think_llm: Optional[str] = None
    backend_url: Optional[str] = None
    max_debate_rounds: Optional[int] = None
    max_risk_discuss_rounds: Optional[int] = None
    email_report_enabled: Optional[bool] = None
    wecom_report_enabled: Optional[bool] = None
    api_key: Optional[str] = None
    wecom_webhook_url: Optional[str] = None
    clear_api_key: bool = False
    clear_wecom_webhook: bool = False
    warmup: bool = True
    force_warmup: bool = False
    default_analysts: Optional[List[str]] = None


class UserRuntimeWarmupRequest(UserRuntimeConfigUpdateRequest):
    prompt: str = "你好"


class RuntimeWarmupResult(BaseModel):
    model: str
    targets: List[str] = Field(default_factory=list)
    content: Optional[str] = None
    error: Optional[str] = None


class UserRuntimeWarmupResponse(BaseModel):
    prompt: str
    results: List[RuntimeWarmupResult]


class WecomWebhookWarmupRequest(BaseModel):
    wecom_webhook_url: Optional[str] = None
    content: Optional[str] = None


class WecomWebhookWarmupResponse(BaseModel):
    sent: bool = True
    message: str
    webhook_display: Optional[str] = None


class ProviderCreateRequest(BaseModel):
    provider_type: str
    display_name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    enabled: bool = True


class ProviderUpdateRequest(BaseModel):
    display_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    enabled: Optional[bool] = None
    clear_api_key: bool = False


class ProviderResponse(BaseModel):
    id: str
    user_id: str
    provider_type: str
    display_name: str
    base_url: Optional[str] = None
    has_api_key: bool = False
    api_key_masked: Optional[str] = None
    enabled: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ModelProfilesSyncRequest(BaseModel):
    models: List[str]
    provider_id: Optional[str] = None

class ModelProfileCreateRequest(BaseModel):
    provider_id: Optional[str] = None
    model_name: str
    display_name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    extra_params: Optional[Dict[str, Any]] = None
    tier: Optional[str] = None
    is_default: bool = False


class ModelProfileUpdateRequest(BaseModel):
    provider_id: Optional[str] = None
    model_name: Optional[str] = None
    display_name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    extra_params: Optional[Dict[str, Any]] = None
    tier: Optional[str] = None
    is_default: Optional[bool] = None


class ModelProfileResponse(BaseModel):
    id: str
    user_id: str
    provider_id: str
    provider_display_name: Optional[str] = None
    provider_type: Optional[str] = None
    model_name: str
    display_name: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    extra_params: Optional[Dict[str, Any]] = None
    tier: Optional[str] = None
    is_default: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class RoleBindingItem(BaseModel):
    target_type: str
    target_key: str
    model_profile_id: str


class RoleBindingsUpdateRequest(BaseModel):
    bindings: List[RoleBindingItem]


class RoleBindingResponse(BaseModel):
    id: str
    target_type: str
    target_key: str
    model_profile_id: str
    model_profile_display_name: Optional[str] = None
    model_name: Optional[str] = None
    provider_type: Optional[str] = None


class PresetApplyRequest(BaseModel):
    preset_mode: str
    bull_profile_id: Optional[str] = None
    bear_profile_id: Optional[str] = None
    manager_profile_id: Optional[str] = None
    quick_profile_id: Optional[str] = None
    deep_profile_id: Optional[str] = None


class CustomPromptItem(BaseModel):
    target_type: str  # 'global' | 'role' | 'group'
    target_key: str = ""  # '' for global; role_key or group_key otherwise
    prompt_text: str
    enabled: bool = True


class CustomPromptsUpdateRequest(BaseModel):
    prompts: List[CustomPromptItem]


class CustomPromptResponse(BaseModel):
    id: str
    target_type: str
    target_key: str
    prompt_text: str
    prompt_hash: str
    char_count: int
    enabled: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ResolvedCustomPromptResponse(BaseModel):
    role_key: str
    global_text: str
    override_text: str
    override_source: Optional[str] = None  # 'role' | 'group' | None
    resolved_text: str
    resolved_length: int
    resolved_hash: Optional[str] = None


class CustomPromptMigrateRequest(BaseModel):
    legacy_text: str


class PromptInjectionSwitchResponse(BaseModel):
    enabled: bool


class PromptInjectionSwitchUpdateRequest(BaseModel):
    enabled: bool


class PortfolioPositionItem(BaseModel):
    symbol: str = Field(..., description="股票代码，如 600519.SH 或 600519")
    name: Optional[str] = Field(None, description="股票名称")
    current_position: Optional[float] = Field(None, description="持仓数量")
    available_position: Optional[float] = Field(None, description="可用数量")
    average_cost: Optional[float] = Field(None, description="成本价")
    market_value: Optional[float] = Field(None, description="市值")
    current_position_pct: Optional[float] = Field(None, description="仓位占比 %")


class PortfolioImportSyncRequest(BaseModel):
    positions: List[PortfolioPositionItem] = Field(..., description="持仓列表")
    source: str = Field("manual", description="持仓来源标识")
    auto_apply_scheduled: bool = Field(True, description="是否自动将持仓股票加入定时任务")


class UserTokenResponse(_TokenDatetimeFieldsMixin, BaseModel):
    id: str
    name: str
    token: str
    token_hint: Optional[str] = None
    last_used_at: Optional[datetime] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class UserTokenListItem(_TokenDatetimeFieldsMixin, BaseModel):
    """Token info for list endpoint — never exposes the full token."""
    id: str
    name: str
    token_hint: Optional[str] = None
    last_used_at: Optional[datetime] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class UserTokenCreateRequest(BaseModel):
    name: str


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _user_config_overrides(user_id: Optional[str], db: Optional[Session] = None) -> Dict[str, Any]:
    if not user_id:
        return {}

    def _query(sess: Session) -> Dict[str, Any]:
        user_cfg = auth_service.get_user_llm_config(sess, user_id)
        if not user_cfg:
            return {}
        result: Dict[str, Any] = {}
        for key in (
            "llm_provider",
            "backend_url",
            "quick_think_llm",
            "deep_think_llm",
            "max_debate_rounds",
            "max_risk_discuss_rounds",
        ):
            value = getattr(user_cfg, key, None)
            if value is not None:
                result[key] = value
        api_key = auth_service.decrypt_secret(user_cfg.api_key_encrypted)
        if api_key:
            result["api_key"] = api_key
        return result

    if db is not None:
        return _query(db)
    with get_db_ctx() as own_db:
        return _query(own_db)


def _build_runtime_config(overrides: Dict[str, Any], user_id: Optional[str] = None, db: Optional[Session] = None) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    server_fallback_enabled = os.getenv("ALLOW_SERVER_LLM_FALLBACK", "1").strip().lower() in ("1", "true", "yes", "on")
    config["server_fallback_enabled"] = server_fallback_enabled

    # Security: filter request overrides to allowlist only
    overrides = {k: v for k, v in overrides.items() if k in _CONFIG_OVERRIDES_ALLOWLIST}

    # Apply global config overrides (from PATCH /v1/config)
    if _global_config_overrides:
        config = _deep_merge(config, dict(_global_config_overrides))
    
    # Fetch user specific overrides from DB (pass db to reuse caller's session)
    user_overrides = _user_config_overrides(user_id, db=db)

    # ── Critical: Filter out empty strings before merging ──
    # This prevents an empty DB field from wiping out an Env Var default.
    filtered_user_overrides = {k: v for k, v in user_overrides.items() if v not in (None, "", [])}
    filtered_request_overrides = {k: v for k, v in overrides.items() if v not in (None, "", [])}

    if filtered_user_overrides:
        config = _deep_merge(config, filtered_user_overrides)
    if filtered_request_overrides:
        config = _deep_merge(config, filtered_request_overrides)

    # ── Intelligent fallback between models ──
    # If one is provided but the other is missing (even after env var merge), cross-fill.
    quick = config.get("quick_think_llm")
    deep = config.get("deep_think_llm")

    if not deep and quick:
        config["deep_think_llm"] = quick
    if not quick and deep:
        config["quick_think_llm"] = deep

    # ── Pass user_id through to the graph ──
    # TradingAgentsGraph consumes config["user_id"] to resolve per-role model
    # bindings (resolve_all_roles) and open its own DB session. Without this,
    # role_llms silently falls back to the default quick/deep models.
    if user_id:
        config["user_id"] = user_id

    return config


class RequireUser:
    def __init__(self, allow_api_token: bool = True):
        self.allow_api_token = allow_api_token

    def __call__(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_auth_scheme),
    ) -> UserDB:
        with get_db_ctx() as db:
            if credentials:
                token = credentials.credentials
                # 1. 优先尝试 JWT (网页登录)
                try:
                    payload = auth_service.decode_access_token(token)
                    user_id = str(payload.get("sub") or "")
                    user = auth_service.get_user_by_id(db, user_id)
                    if user and user.is_active:
                        db.expunge(user)
                        return user
                except Exception:
                    pass

                # 2. 尝试 API Token (仅在允许时)
                if self.allow_api_token and token.startswith(token_service.TOKEN_PREFIX):
                    user = token_service.verify_token(db, token)
                    if user and user.is_active:
                        db.expunge(user)
                        return user

            # 本地单用户/未登录回退至默认本地账户
            user = auth_service.get_or_create_default_user(db)
            db.expunge(user)
            return user


# 快捷依赖定义
_require_api_user = RequireUser(allow_api_token=True)    # 允许 API Token
_require_web_user = RequireUser(allow_api_token=False)   # 仅限网页登录


def _set_job(job_key: str, **kwargs) -> None:
    # Callers may pass job_id=<value> as a stored field.  Since
    # store.set_job()'s first positional param is also called job_id,
    # we must strip it from kwargs to avoid a "got multiple values" TypeError.
    # _get_job() always injects job_id back into the returned dict.
    kwargs.pop("job_id", None)
    get_job_store().set_job(job_key, **kwargs)


def _get_job(job_key: str) -> Dict[str, Any]:
    d = get_job_store().get_job(job_key)
    if d:
        d.setdefault("job_id", job_key)
    return d


def _emit_job_event(job_id: str, event: str, data: Dict[str, Any]) -> None:
    get_job_store().emit_event(job_id, event, data)


def _attach_job_runtime_state(target: Any, job_id: Optional[str]) -> Any:
    if not job_id:
        return target
    job = _get_job(job_id)
    if not job:
        return target

    for field in ("waiting_ahead_count", "scheduled_running_count", "scheduled_concurrency_limit"):
        value = job.get(field)
        if value is not None or hasattr(target, field):
            setattr(target, field, value)
    return target


def _extract_request_user_context(request: UserContextInput) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key in USER_CONTEXT_KEYS:
        value = getattr(request, key, None)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if key == "constraints" and not value:
            continue
        payload[key] = value
    return payload


def _merge_user_context_payload(
    explicit_context: Dict[str, Any],
    inferred_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    merged = normalize_user_context(inferred_context or {})
    merged.update(normalize_user_context(explicit_context or {}))
    return merged


def _compose_analysis_user_context(
    db: Session,
    user_id: str,
    symbol: str,
    *,
    explicit_context: Optional[Dict[str, Any]] = None,
    inferred_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    imported_context = _build_manual_imported_user_context(db, user_id, symbol)
    merged_with_imported = _merge_user_context_payload(inferred_context or {}, imported_context)
    return _merge_user_context_payload(explicit_context or {}, merged_with_imported)


def _apply_user_context_to_request(request: "AnalyzeRequest", user_context: Dict[str, Any]) -> "AnalyzeRequest":
    request.objective = user_context.get("objective")
    request.risk_profile = user_context.get("risk_profile")
    request.investment_horizon = user_context.get("investment_horizon")
    request.cash_available = user_context.get("cash_available")
    request.current_position = user_context.get("current_position")
    request.current_position_pct = user_context.get("current_position_pct")
    request.average_cost = user_context.get("average_cost")
    request.max_loss_pct = user_context.get("max_loss_pct")
    request.constraints = user_context.get("constraints", [])
    request.user_notes = user_context.get("user_notes")
    return request


def _is_empty_debate_state(state: Any) -> bool:
    """Return True if state is None, not a dict, or an unpopulated debate state."""
    if not isinstance(state, dict) or not state:
        return True
    count = state.get("count")
    try:
        count_int = int(count) if count is not None else 0
    except (ValueError, TypeError):
        count_int = 0
    if count_int > 0:
        return False
    for text_key in (
        "history",
        "bull_history",
        "bear_history",
        "aggressive_history",
        "conservative_history",
        "neutral_history",
        "judge_decision",
        "latest_risk_verdict",
        "current_speaker",
        "latest_speaker",
    ):
        if str(state.get(text_key) or "").strip():
            return False
    if state.get("claims"):
        return False
    return True


def _build_result_payload(final_state: Dict[str, Any]) -> Dict[str, Any]:
    market_context = final_state.get("market_context") or {}
    market_data_context = final_state.get("market_data_context") or {}
    daily_context = market_data_context.get("daily") or {}
    failure_ledger = market_data_context.get("data_failure_ledger") or []
    data_gaps = [
        str(entry.get("gap"))
        for entry in failure_ledger
        if isinstance(entry, dict) and entry.get("gap")
    ]
    baseline = final_state.get("trade_date")
    # ``market_context.data_as_of`` is a conceptual session date, not proof
    # that a completed daily bar was fetched.  Only normalized daily data can
    # establish the report's actual data cutoff.
    data_as_of = daily_context.get("as_of")
    return {
        "symbol": final_state.get("company_of_interest"),
        "horizon": final_state.get("horizon", "short"),
        "trade_date": baseline,
        "analysis_baseline_date": baseline,
        "data_as_of": data_as_of,
        "data_gaps": data_gaps,
        "fund_flow_consensus_guard": final_state.get("fund_flow_consensus_guard", {"blocked": True, "direction_allowed": False, "status": "not_checked"}),
        "direction": None,
        "instrument_context": final_state.get("instrument_context"),
        "market_context": market_context,
        "market_data_context": market_data_context,
        "user_context": final_state.get("user_context"),
        "workflow_context": final_state.get("workflow_context"),
        "market_report": final_state.get("market_report"),
        "sentiment_report": final_state.get("sentiment_report"),
        "news_report": final_state.get("news_report"),
        "fundamentals_report": final_state.get("fundamentals_report"),
        "macro_report": final_state.get("macro_report"),
        "smart_money_report": final_state.get("smart_money_report"),
        "volume_price_report": final_state.get("volume_price_report"),
        "game_theory_report": final_state.get("game_theory_report"),
        "game_theory_signals": final_state.get("game_theory_signals"),
        "analyst_traces": final_state.get("analyst_traces"),
        "investment_plan": final_state.get("investment_plan"),
        "trader_investment_plan": final_state.get("trader_investment_plan"),
        "investment_debate_state": final_state.get("investment_debate_state"),
        "risk_debate_state": final_state.get("risk_debate_state"),
        "risk_feedback_state": final_state.get("risk_feedback_state"),
        "final_trade_decision": final_state.get("final_trade_decision"),
    }


class AgentProgressTracker:
    # 阶段标题映射
    STAGE_TITLES = {
        "market_analysis": "市场分析完成",
        "sentiment_analysis": "舆情分析完成",
        "news_analysis": "新闻分析完成",
        "fundamentals_analysis": "基本面分析完成",
        "research_decision": "研究团队决策",
        "trader_plan": "交易计划制定",
        "risk_assessment": "风险评估完成",
        "final_decision": "最终决策",
    }
    
    def __init__(self, selected_analysts: List[str], job_id: str, horizon: Optional[str] = None):
        self.job_id = job_id
        self.horizon = horizon
        self.selected_analysts = [a.lower() for a in selected_analysts]
        self.status: Dict[str, str] = {}
        self.start_times: Dict[str, float] = {}  # 记录每个 agent 开始时间
        self.report_sections: Dict[str, Optional[str]] = {
            "market_report": None,
            "sentiment_report": None,
            "news_report": None,
            "fundamentals_report": None,
            "macro_report": None,
            "smart_money_report": None,
            "volume_price_report": None,
            "game_theory_report": None,
            "investment_plan": None,
            "trader_investment_plan": None,
            "final_trade_decision": None,
        }
        # 跟踪已完成的阶段，避免重复发送里程碑
        self._completed_stages: set = set()
        # 跟踪已发送的 writing 状态，避免重复发送
        self._writing_status_sent: set = set()
        
        for team_agents in FIXED_TEAMS.values():
            for agent in team_agents:
                self.status[agent] = "pending"

        # 未选中的分析师标记为 skipped（仍展示，便于固定 12-agent 看板）
        for key in ANALYST_ORDER:
            agent = ANALYST_AGENT_NAMES[key]
            if key not in self.selected_analysts:
                self.status[agent] = "skipped"

    def _emit_milestone(self, stage: str, summary: str = "") -> None:
        """发送用户可见的里程碑事件"""
        if stage in self._completed_stages:
            return
        self._completed_stages.add(stage)
        
        title = self.STAGE_TITLES.get(stage, stage)
        _emit_job_event(
            self.job_id,
            "agent.milestone",
            {
                "stage": stage,
                "title": title,
                "summary": summary,
                "timestamp": _utcnow_iso(),
                "horizon": self.horizon,
            },
        )
        _log(f"[Milestone] {title}: {summary[:100]}...")

    def _emit_report_chunked(self, job_id: str, section: str, content: str) -> None:
        """将报告内容分片发送，直接透传不做人工延迟
        
        按较大块分片（如按段落），让前端自然渲染
        """
        # 按段落分割，保持Markdown结构
        paragraphs = content.split('\n\n')
        
        for i, para in enumerate(paragraphs):
            if not para.strip():
                continue
                
            _emit_job_event(
                job_id,
                "agent.report.chunk",
                {
                    "section": section,
                    "chunk": para + '\n\n',
                    "index": i,
                    "is_complete": False,
                    "horizon": self.horizon,
                },
            )
        
        # 发送完成标记
        _emit_job_event(
            job_id,
            "agent.report.chunk",
            {
                "section": section,
                "chunk": "",
                "index": -1,
                "is_complete": True,
                "horizon": self.horizon,
            },
        )

    def snapshot(self) -> Dict[str, Any]:
        agents = []
        for team, members in FIXED_TEAMS.items():
            for m in members:
                agents.append({"team": team, "agent": m, "status": self.status.get(m, "pending")})
        return {"agents": agents, "horizon": self.horizon}

    def _set_status(self, agent: str, status: str) -> None:
        prev = self.status.get(agent)
        if prev == status:
            return
        self.status[agent] = status
        
        # 记录时间
        if status == "in_progress":
            self.start_times[agent] = time.time()
        elif status == "completed" and agent in self.start_times:
            duration = time.time() - self.start_times[agent]
            _log(f"[Timer] Agent {agent} ({self.horizon or 'main'}) finished in {duration:.2f}s")

        _emit_job_event(
            self.job_id,
            "agent.status",
            {"agent": agent, "status": status, "previous_status": prev, "horizon": self.horizon},
        )

    def _update_research_team_status(self, status: str) -> None:
        for agent in ["Bull Researcher", "Bear Researcher", "Research Manager"]:
            self._set_status(agent, status)

    def _generate_stage_summary(self, stage: str, chunk: Dict[str, Any]) -> str:
        """根据阶段生成简要总结"""
        if stage == "market_analysis":
            report = chunk.get("market_report", "")
            # 提取关键信息
            if "支撑" in report or "压力" in report:
                return "技术面关键位已识别"
            return "技术面分析完成"
        elif stage == "sentiment_analysis":
            return "舆情数据已收集"
        elif stage == "news_analysis":
            return "新闻影响已评估"
        elif stage == "fundamentals_analysis":
            return "基本面指标已计算"
        elif stage == "research_decision":
            return "多空观点已形成"
        elif stage == "trader_plan":
            return "交易策略已制定"
        elif stage == "risk_assessment":
            return "风险水平已评估"
        elif stage == "final_decision":
            decision = chunk.get("final_trade_decision", "")
            return f"最终建议: {decision[:50]}..." if len(decision) > 50 else f"最终建议: {decision}"
        return ""

    def _emit_writing_status(self, agent_name: str, report_type: str) -> None:
        """发送正在编写报告的状态（每个agent只发送一次）"""
        # 检查是否已经发送过
        status_key = f"{agent_name}:{report_type}"
        if status_key in self._writing_status_sent:
            return
        self._writing_status_sent.add(status_key)
        
        report_names = {
            "market_report": "市场分析",
            "sentiment_report": "舆情分析",
            "news_report": "新闻分析",
            "fundamentals_report": "基本面分析",
            "investment_plan": "投资计划",
            "trader_investment_plan": "交易计划",
            "final_trade_decision": "最终交易决策",
        }
        _emit_job_event(
            self.job_id,
            "agent.writing",
            {
                "agent": agent_name,
                "report": report_type,
                "report_name": report_names.get(report_type, report_type),
                "status": "writing",
                "horizon": self.horizon,
            },
        )

    def _emit_token(self, agent_name: str, report_type: str, token: str) -> None:
        """推送 Token 级别的流式内容（跳过空 token，避免思维模型推理阶段刷屏）"""
        if not token:
            return
        _emit_job_event(
            self.job_id,
            "agent.token",
            {
                "agent": agent_name,
                "report": report_type,
                "token": token,
                "horizon": self.horizon,
            },
        )

    def emit_debate_token(
        self, debate: str, agent: str, round_num: int, token: str, model_name: Optional[str] = None,
    ) -> None:
        """推送辩论 token（流式输出，每个 chunk 调用一次）"""
        if not token:
            return
        try:
            _emit_job_event(
                self.job_id,
                "agent.debate.token",
                {
                    "debate": debate,
                    "agent": agent,
                    "round": round_num,
                    "token": token,
                    "horizon": self.horizon,
                    "model_name": model_name,
                },
            )
        except Exception:
            pass

    def emit_debate_message(
        self, debate: str, agent: str, round_num: int,
        content: str, is_verdict: bool = False, model_name: Optional[str] = None,
    ) -> None:
        """推送辩论消息（每个 agent 每轮完成后调用一次）"""
        if not content:
            return
        try:
            _emit_job_event(
                self.job_id,
                "agent.debate",
                {
                    "debate": debate,
                    "agent": agent,
                    "round": round_num,
                    "content": content,
                    "is_verdict": is_verdict,
                    "horizon": self.horizon,
                    "model_name": model_name,
                },
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to emit debate message for %s in %s", agent, debate, exc_info=True,
            )

def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content)


def _generate_tool_description(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """生成工具调用的可读描述"""
    if tool_name == "get_indicators":
        indicator = tool_args.get("indicator")
        if isinstance(indicator, str) and indicator:
            indicator_map = {
                "close_50_sma": "50日均线",
                "close_200_sma": "200日均线",
                "close_10_ema": "10日EMA",
                "close_20_ema": "20日EMA",
                "rsi": "RSI",
                "macd": "MACD",
                "boll": "布林中轨",
                "boll_ub": "布林上轨",
                "boll_lb": "布林下轨",
                "atr": "ATR波动率",
                "vwma": "VWMA量价均线",
                "obv": "OBV能量潮",
            }
            return f"计算 {indicator_map.get(indicator, indicator)}"
        return "获取技术指标"
    elif tool_name == "get_stock_data":
        return "获取股票历史数据"
    elif tool_name == "get_fundamentals":
        metrics = tool_args.get("metrics", [])
        if metrics:
            return f"获取 {', '.join(metrics[:2])}{' 等' if len(metrics) > 2 else ''} 基本面数据"
        return "获取基本面数据"
    elif tool_name == "get_income_statement":
        return "获取利润表"
    elif tool_name == "get_balance_sheet":
        return "获取资产负债表"
    elif tool_name == "get_cash_flow":
        return "获取现金流量表"
    elif tool_name == "get_news":
        return "获取相关新闻"
    elif tool_name == "get_social_sentiment":
        return "获取舆情数据"
    return f"调用 {tool_name}"


async def _run_job(
    job_id: str,
    request: AnalyzeRequest,
    stream_events: bool = False,
    save_report: bool = True,
    user_id: Optional[str] = None,
    request_source: str = "api",
) -> None:
    # Keep the workflow in its own task so crossing the soft deadline does not
    # cancel to_thread work.  The hard deadline cancels the coroutine and
    # releases its caller/scheduler slot.  A sync function already submitted by
    # asyncio.to_thread may finish later, but cancellation prevents the
    # coroutine from resuming into report or terminal-state writes.
    started_monotonic = time.monotonic()
    inner_task = asyncio.create_task(
        _run_job_inner(job_id, request, stream_events, save_report, user_id, request_source)
    )
    try:
        if _JOB_TIMEOUT > 0:
            soft_wait_seconds = _JOB_TIMEOUT
            if _JOB_HARD_TIMEOUT > 0:
                soft_wait_seconds = min(soft_wait_seconds, _JOB_HARD_TIMEOUT)
            done, _ = await asyncio.wait({inner_task}, timeout=soft_wait_seconds)
            if inner_task not in done and not inner_task.done():
                # When the configured hard limit is no greater than the soft
                # limit, skip the misleading overtime notice and fall through
                # to the terminal hard-limit handling below.
                if _JOB_HARD_TIMEOUT <= 0 or _JOB_TIMEOUT < _JOB_HARD_TIMEOUT:
                    overtime_at = _utcnow_iso()
                    elapsed_seconds = time.monotonic() - started_monotonic
                    message = (
                        f"分析耗时较长（已超过 {_JOB_TIMEOUT} 秒），后台仍在继续，"
                        "正在等待最终结果，请勿重复提交。"
                    )
                    _log(f"[Job {job_id}] {message}")
                    _set_job(
                        job_id,
                        status="running",
                        overtime=True,
                        overtime_at=overtime_at,
                        error=None,
                    )
                    _emit_job_event(
                        job_id,
                        "job.overtime",
                        {
                            "job_id": job_id,
                            "elapsed_seconds": elapsed_seconds,
                            "soft_timeout_seconds": _JOB_TIMEOUT,
                            "overtime_at": overtime_at,
                            "message": message,
                        },
                    )

        if _JOB_HARD_TIMEOUT > 0 and not inner_task.done():
            remaining_seconds = max(
                0.0,
                _JOB_HARD_TIMEOUT - (time.monotonic() - started_monotonic),
            )
            done, _ = await asyncio.wait({inner_task}, timeout=remaining_seconds)
            if inner_task not in done and not inner_task.done() and inner_task.cancel():
                try:
                    await inner_task
                except asyncio.CancelledError:
                    pass

                elapsed_seconds = time.monotonic() - started_monotonic
                err_msg = (
                    f"任务达到硬性运行上限（{_JOB_HARD_TIMEOUT} 秒），已终止。"
                    "请检查模型或数据源的请求超时配置后重试。"
                )
                _log(f"[Job {job_id}] {err_msg}")
                _set_job(
                    job_id,
                    status="failed",
                    error=err_msg,
                    overtime=False,
                    overtime_at=None,
                    finished_at=_utcnow_iso(),
                )
                try:
                    with get_db_ctx() as db:
                        report_service.mark_report_failed(db, job_id, err_msg)
                except Exception:
                    pass
                _emit_job_event(
                    job_id,
                    "job.failed",
                    {
                        "job_id": job_id,
                        "error": err_msg,
                        "elapsed_seconds": elapsed_seconds,
                        "hard_timeout_seconds": _JOB_HARD_TIMEOUT,
                    },
                )
                return

        await inner_task
    except asyncio.CancelledError:
        # Preserve normal application shutdown/caller cancellation semantics.
        if not inner_task.done():
            inner_task.cancel()
        raise
    except Exception as exc:
        # _run_job_inner handles expected workflow errors itself.  This guards
        # failures before its try block (initialisation/configuration) as well.
        err_msg = f"{type(exc).__name__}: {exc}"
        _log(f"[Job {job_id}] failed: {err_msg}")
        _set_job(
            job_id,
            status="failed",
            error=err_msg,
            overtime=False,
            overtime_at=None,
            finished_at=_utcnow_iso(),
        )
        try:
            with get_db_ctx() as db:
                report_service.mark_report_failed(db, job_id, err_msg)
        except Exception:
            pass
        _emit_job_event(job_id, "job.failed", {"job_id": job_id, "error": err_msg})


async def _save_report_or_raise(
    job_id: str,
    save_callable: Callable[[], Any],
    *,
    stage: str,
) -> None:
    """Run a report DB finalizer without turning persistence errors into success."""
    try:
        await asyncio.to_thread(save_callable)
    except Exception as exc:
        message = f"Failed to {stage} report for job {job_id}: {exc}"
        _log(message)
        raise RuntimeError(message) from exc


_INJECT_ROLES = ("bull_researcher", "bear_researcher", "research_manager", "trader", "risk_manager")


def _attach_custom_prompt_snapshot(result: Dict[str, Any], prompt_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Attach a deep-copied custom_prompt_snapshot onto result_data before it is saved.

    Both save paths (dual-horizon and single-path) must call this exact function so
    the snapshot shape and isolation guarantee (deepcopy, not shared reference) stay
    identical across paths — no per-path reimplementation.
    """
    result["custom_prompt_snapshot"] = deepcopy(prompt_snapshot)
    return result


def _merge_deduplicated_strings(existing: Any, incoming: Any) -> List[str]:
    """Merge ordered string lists without clearing graph/ledger semantics."""
    merged: List[str] = []
    seen: set[str] = set()
    for value in (existing or []) + (incoming or []):
        normalized = report_service._normalize_gap_text(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            merged.append(normalized)
    return merged


def _apply_structured_report_fields(
    result: Dict[str, Any],
    *,
    structured: Optional[Any],
    graph_decision: Optional[str],
    resolved: Dict[str, Any],
) -> str:
    """Apply one deterministic structured-report contract to every save path."""
    legal_decisions = {"BUY", "SELL", "HOLD"}
    structured_decision = getattr(structured, "decision", None) if structured else None
    if structured_decision not in legal_decisions:
        structured_decision = None
    fallback_decision = graph_decision if graph_decision in legal_decisions else None
    decision = structured_decision or fallback_decision or "UNKNOWN"
    structured_probability = getattr(structured, "probability", None) if structured else None
    structured_data_gaps = getattr(structured, "data_gaps", []) if structured else []
    structured_falsification = (
        getattr(structured, "falsification_conditions", []) if structured else []
    )
    existing_data_gaps = (
        result.get("data_gaps") if isinstance(result.get("data_gaps"), list) else []
    )
    existing_falsification = (
        result.get("falsification_conditions")
        if isinstance(result.get("falsification_conditions"), list)
        else []
    )
    structured_not_applicable = bool(
        getattr(structured, "not_applicable", False) if structured else False
    )
    if structured_not_applicable:
        not_applicable = True
    else:
        not_applicable = bool(result.get("not_applicable"))
    result.update(
        {
            "decision": decision,
            "direction": resolved.get("direction") or (decision if decision in legal_decisions else None),
            "confidence": resolved.get("confidence"),
            "probability": structured_probability,
            "data_gaps": report_service.merge_data_gaps(
                result_data=result,
                llm_data_gaps=[*existing_data_gaps, *(structured_data_gaps or [])],
            ),
            "falsification_conditions": _merge_deduplicated_strings(
                existing_falsification,
                structured_falsification,
            ),
            "not_applicable": not_applicable,
            "target_price": resolved.get("target_price"),
            "stop_loss_price": resolved.get("stop_loss_price"),
        }
    )
    return decision


def _build_custom_prompt_snapshot(
    frozen_bundle: Dict[str, Dict[str, Any]],
    injection_enabled: bool,
) -> Dict[str, Any]:
    """Build the persisted prompt snapshot from a frozen job bundle."""
    return {
        "enabled": injection_enabled,
        "placement": DEFAULT_PLACEMENT,
        "roles": deepcopy(frozen_bundle),
    }


def _resolve_and_freeze_custom_prompts(
    db: Session,
    user_id: Optional[str],
) -> Tuple[Dict[str, Dict[str, Any]], bool]:
    """Resolve custom prompts from DB and freeze into an immutable bundle for a job.

    Must happen after init_report, before any TradingAgentsGraph is constructed.
    Only reads users.prompt_injection_enabled first; only calls resolve_all_roles_prompts
    when the switch is on. Frozen value is immutable for this job's lifetime.

    Returns:
        (frozen_bundle, injection_enabled)
    """
    injection_enabled = custom_prompt_service.get_prompt_injection_enabled(db, user_id) if user_id else False
    if injection_enabled:
        resolved = custom_prompt_service.resolve_all_roles_prompts(db, user_id)
        raw_bundle: Dict[str, Dict[str, Any]] = {}
        for r in resolved:
            rk = r["role_key"]
            if rk not in _INJECT_ROLES:
                continue
            text = r["resolved_text"] or ""
            if text and r["resolved_length"] > custom_prompt_service.RESOLVED_PROMPT_MAX_CHARS:
                raise ValueError(
                    f"[custom_prompt] {rk} resolved text "
                    f"{r['resolved_length']} chars > "
                    f"{custom_prompt_service.RESOLVED_PROMPT_MAX_CHARS} limit, "
                    f"hash={r['resolved_hash']}"
                )
            raw_bundle[rk] = {
                "resolved_text": text,
                "resolved_hash": r["resolved_hash"],
                "resolved_length": r["resolved_length"],
                "injected": bool(text),
            }
        for rk in _INJECT_ROLES:
            if rk not in raw_bundle:
                raw_bundle[rk] = {"resolved_text": "", "resolved_hash": None, "resolved_length": 0, "injected": False}
        return deepcopy(raw_bundle), True
    else:
        frozen_bundle = {
            rk: {"resolved_text": "", "resolved_hash": None, "resolved_length": 0, "injected": False}
            for rk in _INJECT_ROLES
        }
        return frozen_bundle, False


async def _run_job_inner(
    job_id: str,
    request: AnalyzeRequest,
    stream_events: bool = False,
    save_report: bool = True,
    user_id: Optional[str] = None,
    request_source: str = "api",
) -> None:
    job_start_t = time.time()
    # DAV-105: resolve defaults once more at the job boundary so every entry
    # point shares the same CN session rule. The explicitness marker prevents a
    # user-requested current date from being rewritten to the prior session.
    explicit_date = request.trade_date_explicit
    if explicit_date is None:
        explicit_date = "trade_date" in request.model_fields_set and bool(
            str(request.trade_date or "").strip()
        )
    try:
        request.trade_date = _normalize_analysis_trade_date(
            request.trade_date if explicit_date else None,
            explicit=explicit_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    request.trade_date_explicit = explicit_date
    # Normalize for logic but keep original for display
    display_name = request.symbol
    normalized_symbol = _normalize_symbol(request.symbol)

    # ── Step 0: Initialize report in DB (short-lived session) ──
    def _init_and_configure():
        with get_db_ctx() as db:
            try:
                report_service.init_report(
                    db=db,
                    report_id=job_id,
                    symbol=normalized_symbol,
                    trade_date=request.trade_date,
                    user_id=user_id,
                )
                report_service.update_report_partial(db, job_id, status="running")
                db.commit()
            except Exception as e:
                _log(f"CRITICAL: Failed to initialize report in DB: {e}")

            frozen_bundle, injection_enabled = _resolve_and_freeze_custom_prompts(db, user_id)

        cfg = _build_runtime_config(request.config_overrides, user_id=user_id)
        return cfg, frozen_bundle, injection_enabled

    config, _frozen_bundle, _injection_enabled = await asyncio.to_thread(_init_and_configure)

    # Build the snapshot dict written into result_data on both save paths.
    # Keep this explicit at the job boundary; graph/factory defaults use the same constant.
    _PROMPT_PLACEMENT = DEFAULT_PLACEMENT
    _prompt_snapshot = _build_custom_prompt_snapshot(_frozen_bundle, _injection_enabled)
    # Extract just the resolved texts for passing to TradingAgentsGraph
    _custom_prompts_for_graph = {rk: v["resolved_text"] for rk, v in _frozen_bundle.items()}

    _set_job(
        job_id,
        status="running",
        started_at=_utcnow_iso(),
        symbol=normalized_symbol,
        error=None,
    )

    _emit_job_event(
        job_id,
        "job.running",
        {
            "job_id": job_id,
            "symbol": normalized_symbol,
            "display_name": display_name,
            "trade_date": request.trade_date
        },
    )
    # Ensure request object uses the normalized symbol for internal logic
    request.symbol = normalized_symbol
    user_context_payload = _extract_request_user_context(request)
    tracker = AgentProgressTracker(request.selected_analysts, job_id)
    _emit_job_event(job_id, "agent.snapshot", tracker.snapshot())

    try:
        if request.dry_run:
            result = {
                "mode": "dry_run",
                "symbol": request.symbol,
                "trade_date": request.trade_date,
                "selected_analysts": request.selected_analysts,
                "user_context": user_context_payload,
                "llm_provider": config.get("llm_provider"),
                "data_vendors": config.get("data_vendors"),
            }
            _set_job(
                job_id,
                status="completed",
                result=result,
                decision="DRY_RUN",
                error=None,
                overtime=False,
                overtime_at=None,
                finished_at=_utcnow_iso(),
            )
            _emit_job_event(
                job_id,
                "job.completed",
                {"job_id": job_id, "decision": "DRY_RUN", "result": result},
            )
            return

        _shared_data_collector.ref(request.symbol, request.trade_date)
        graph = TradingAgentsGraph(
            selected_analysts=request.selected_analysts,
            debug=False,
            config=config,
            data_collector=_shared_data_collector,
            custom_prompts=_custom_prompts_for_graph,
            custom_prompt_placement=_PROMPT_PLACEMENT,
        )
        final_state: Optional[Dict[str, Any]] = None

        request.horizons = _normalize_analysis_horizons(request.horizons, query=request.query)

        # ── Dual-horizon intent-driven path ──────────────────────────────────
        if request.query or len(request.horizons) > 1:
            # 1. 组装用户意图
            intent_start_t = time.time()
            ticker = request.symbol or display_name

            # 优先使用已由 chat_completions 预解析的 intent（单次 LLM），避免二次调用
            if request.user_intent:
                user_intent = dict(request.user_intent)
                user_intent["ticker"] = ticker
                user_intent["horizons"] = request.horizons
            else:
                if request.query:
                    # 直接 POST /v1/analyze 时的兜底（无预解析 intent）
                    user_intent = await asyncio.to_thread(
                        _parse_intent,
                        request.query,
                        graph.quick_thinking_llm,
                        fallback_ticker=ticker,
                    )
                    if not request.horizons:
                        request.horizons = user_intent["horizons"]
                    request.horizons = _normalize_analysis_horizons(
                        request.horizons,
                        query=request.query,
                    )
                else:
                    # Explicit structured dual-horizon requests have no
                    # natural-language parser input; keep the API contract
                    # self-contained instead of calling the upstream parser.
                    user_intent = {
                        "raw_query": "",
                        "ticker": ticker,
                        "horizons": request.horizons,
                        "focus_areas": [],
                        "specific_questions": [],
                    }
                user_intent["horizons"] = request.horizons
            _log(f"[Timer] Intent Parsing took {time.time() - intent_start_t:.2f}s")

            inferred_user_context = user_intent.get("user_context") or {}
            user_context_payload = _merge_user_context_payload(
                user_context_payload,
                inferred_user_context,
            )
            user_intent["user_context"] = user_context_payload

            # Use normalized ticker from intent parser if available
            ticker = user_intent.get("ticker") or ticker

            # 2. 一次性采集数据，短线/中线共用缓存
            lookback_label = "14天关键行情" if request.horizons == ["short"] else "90天全量行情、财务、新闻、资金"
            _emit_job_event(job_id, "agent.tool_call", {
                "agent": "数据采集", "tool": "data_collector",
                "description": f"预加载 {ticker} 近{lookback_label}数据…",
            })
            _log(f"[DualHorizon] Collecting data for {ticker} {request.trade_date} (horizons={request.horizons})…")
            collect_start_t = time.time()
            collected_pool = await asyncio.to_thread(
                graph.data_collector.collect,
                ticker,
                request.trade_date,
                horizons=request.horizons,
            )
            market_data_context = (
                collected_pool.get("market_data_context")
                if isinstance(collected_pool, dict)
                else None
            )
            _log(f"[Timer] Data Collection step in _run_job took {time.time() - collect_start_t:.2f}s")

            _emit_job_event(job_id, "agent.tool_call", {
                "agent": "数据采集", "tool": "data_collector",
                "description": "数据采集完成，开始多维度分析",
            })

            report_keys = (
                "market_report", "sentiment_report", "news_report", "fundamentals_report",
                "macro_report", "smart_money_report", "volume_price_report",
                "investment_plan", "trader_investment_plan", "final_trade_decision",
            )

            horizon_states: Dict[str, Any] = {}

            async def _process_horizon(horizon: str):
                """Async helper to run analysis for a single horizon."""
                # 根据周期过滤 analyst，共享已采集的数据缓存
                horizon_analysts = _get_horizon_analysts(horizon, request.selected_analysts)
                horizon_graph = TradingAgentsGraph(
                    selected_analysts=horizon_analysts,
                    debug=False,
                    config=config,
                    data_collector=graph.data_collector,
                    custom_prompts=_custom_prompts_for_graph,
                    custom_prompt_placement=_PROMPT_PLACEMENT,
                )

                horizon_label = "短线" if horizon == "short" else "中线"
                _emit_job_event(job_id, "agent.horizon_start", {
                    "horizon": horizon, "label": horizon_label,
                })
                # 每轮重置 tracker，前端进度条重新走一遍
                h_tracker = AgentProgressTracker(horizon_analysts, job_id, horizon=horizon)
                _emit_job_event(job_id, "agent.snapshot", h_tracker.snapshot())
                # 告知前端本轮参与的 analyst 即将开始
                for analyst_key in ANALYST_ORDER:
                    if analyst_key in horizon_analysts:
                        aname = ANALYST_AGENT_NAMES[analyst_key]
                        h_tracker._set_status(aname, "in_progress")
                        h_tracker._emit_writing_status(aname, ANALYST_REPORT_MAP[analyst_key])

                h_args = horizon_graph.propagator.get_graph_args()

                # Use thread_id for LangGraph checkpointer persistence
                if "config" not in h_args:
                    h_args["config"] = {}
                h_args["config"]["configurable"] = {"thread_id": f"{job_id}_{horizon}"}

                init_state = horizon_graph.propagator.create_initial_state(
                    ticker, request.trade_date,
                    user_context=user_context_payload,
                    selected_analysts=horizon_analysts,
                    request_source=request_source,
                    user_intent=user_intent,
                    horizon=horizon,
                    market_data_context=market_data_context,
                )
                last_report: Dict[str, str] = {}
                seen: Dict[str, bool] = {}   # 追踪哪些字段已出现过，避免重复事件
                horizon_final = None

                # Preserve the historical incremental ReportDB updates for
                # single-horizon query/chat jobs. Dual-horizon jobs defer
                # persistence until aggregation so the two graphs cannot
                # overwrite each other's flattened columns.
                def _horizon_partial_update(updates: dict):
                    with get_db_ctx() as _hdb:
                        report_service.update_report_partial(_hdb, job_id, **updates)

                # 通过 ContextVar 将 tracker 传入 async 节点（LangGraph 不传递 schema 外的字段）
                _tracker_token = current_tracker_var.set(h_tracker)
                try:
                    async for chunk in horizon_graph.graph.astream(init_state, **h_args):
                        horizon_final = chunk

                        # ── 并行感知的状态推进 ──────────────────
                        # 1. 每个 analyst 报告首次出现 → completed
                        for analyst_key in ANALYST_ORDER:
                            if analyst_key not in horizon_analysts:
                                continue
                            rkey = ANALYST_REPORT_MAP[analyst_key]
                            aname = ANALYST_AGENT_NAMES[analyst_key]
                            if chunk.get(rkey) and not seen.get(rkey):
                                seen[rkey] = True
                                h_tracker._set_status(aname, "completed")

                        # 2. 分析师全部完成后 → Bull/Bear/ResearchManager 开始
                        all_analysts_done = all(
                            seen.get(ANALYST_REPORT_MAP.get(a, "")) for a in h_tracker.selected_analysts
                        )
                        if all_analysts_done and not seen.get("_research_started"):
                            seen["_research_started"] = True
                            h_tracker._set_status(ANALYST_AGENT_NAMES["bull"], "in_progress")
                            h_tracker._set_status(ANALYST_AGENT_NAMES["bear"], "in_progress")
                            h_tracker._set_status(ANALYST_AGENT_NAMES["research_manager"], "in_progress")

                        # 3. research judge → 研究团队完成, Trader 开始
                        debate = chunk.get("investment_debate_state") or {}
                        if debate.get("judge_decision") and not seen.get("judge_decision"):
                            seen["judge_decision"] = True
                            for r_key in ["bull", "bear", "research_manager"]:
                                h_tracker._set_status(ANALYST_AGENT_NAMES[r_key], "completed")
                            h_tracker._set_status(ANALYST_AGENT_NAMES["trader"], "in_progress")
                            h_tracker._emit_writing_status(ANALYST_AGENT_NAMES["trader"], "trader_investment_plan")

                        # 4. trader plan → Trader completed, 风控开始
                        if chunk.get("trader_investment_plan") and not seen.get("trader_investment_plan"):
                            seen["trader_investment_plan"] = True
                            h_tracker._set_status(ANALYST_AGENT_NAMES["trader"], "completed")
                            h_tracker._set_status(ANALYST_AGENT_NAMES["aggressive"], "in_progress")

                        # 5. risk judge → 风控全部完成
                        risk = chunk.get("risk_debate_state") or {}
                        if risk.get("judge_decision") and not seen.get("risk_judge_decision"):
                            seen["risk_judge_decision"] = True
                            for r_key in ["aggressive", "neutral", "conservative", "portfolio_manager"]:
                                h_tracker._set_status(ANALYST_AGENT_NAMES[r_key], "completed")
                        # ── end 并行感知 ────────────────────────────────────────────

                        # 报告分片推送。双 horizon 的持久化只在最终聚合后写入，
                        # 避免两个 graph 并发把不同 horizon 的字段相互覆盖。
                        db_updates = {}
                        for key in report_keys:
                            value = chunk.get(key)
                            if value and value != last_report.get(key):
                                last_report[key] = value
                                db_updates[key] = str(value)
                                h_tracker._emit_report_chunked(job_id, key, str(value))
                        if db_updates and len(request.horizons) == 1:
                            await asyncio.to_thread(_horizon_partial_update, db_updates)
                except Exception as e:
                    _log(
                        f"Error during horizon streaming ({horizon}): {e!r}\n"
                        f"{traceback.format_exc()}"
                    )
                    raise
                finally:
                    current_tracker_var.reset(_tracker_token)

                horizon_states[horizon] = horizon_final
                for agent, st in h_tracker.status.items():
                    if st not in ("completed", "skipped"):
                        h_tracker._set_status(agent, "completed")
                _emit_job_event(job_id, "agent.horizon_done", {"horizon": horizon})

            # 3. 按解析出的 horizons 并行运行 astream()，事件实时推给前端
            results = await asyncio.gather(
                *[_process_horizon(h) for h in request.horizons],
                return_exceptions=True,
            )
            horizon_errors: Dict[str, Exception] = {}
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    horizon = request.horizons[i]
                    tb = "".join(traceback.format_exception(type(r), r, r.__traceback__))
                    _log(f"Horizon '{horizon}' failed: {r!r}\n{tb}")
                    horizon_errors[horizon] = r
                    _emit_job_event(
                        job_id,
                        "agent.horizon_failed",
                        {
                            "horizon": horizon,
                            "status": "failed",
                            "error": _humanize_analysis_error(str(r)),
                            "impact": f"{horizon} horizon is unavailable; no decision is produced for this horizon.",
                        },
                    )
            if horizon_errors and len(request.horizons) == 1:
                raise RuntimeError(
                    "Horizon analysis failed: "
                    + "; ".join(f"{horizon}: {error}" for horizon, error in horizon_errors.items())
                )
            if len(horizon_errors) == len(request.horizons):
                raise RuntimeError(
                    "All requested horizons failed: "
                    + "; ".join(f"{horizon}: {error}" for horizon, error in horizon_errors.items())
                )

            model_snapshot = {
                role: {
                    "provider_type": cfg.get("provider_type"),
                    "model_name": cfg.get("model_name"),
                    "base_url": cfg.get("base_url"),
                    "resolved_via": cfg.get("resolved_via"),
                    "fallback_used": cfg.get("fallback_used"),
                    "profile_display_name": cfg.get("profile_display_name"),
                    "provider_display_name": cfg.get("provider_display_name"),
                }
                for role, cfg in getattr(graph, "role_resolved_configs", {}).items()
            }

            if len(request.horizons) > 1:
                # Each horizon owns its structured fields.  Keeping the
                # values nested prevents a primary horizon from being
                # accidentally presented as the result of the other graph.
                horizon_results: Dict[str, Dict[str, Any]] = {}
                for horizon in request.horizons:
                    if horizon in horizon_errors:
                        horizon_results[horizon] = {
                            "horizon": horizon,
                            "status": "failed",
                            "error": _humanize_analysis_error(str(horizon_errors[horizon])),
                            "not_applicable": None,
                            "impact": (
                                f"{horizon} horizon is unavailable; downstream consumers must use only the "
                                "completed horizon and treat this result as partial."
                            ),
                        }
                        continue

                    horizon_result = graph._build_horizon_result(
                        horizon,
                        horizon_states.get(horizon) or {},
                    )
                    structured = None
                    try:
                        structured = await asyncio.to_thread(
                            report_service.extract_structured_data,
                            final_trade_decision=horizon_result.get("final_trade_decision", ""),
                            fundamentals_report=horizon_result.get("fundamentals_report", ""),
                            config=config,
                        )
                    except Exception as exc:
                        _log(f"Structured extraction failed for {horizon} (non-fatal): {exc}")

                    resolved = await asyncio.to_thread(
                        report_service.resolve_report_fields,
                        result_data=horizon_result,
                        confidence_override=structured.confidence if structured else None,
                        target_price_override=structured.target_price if structured else None,
                        stop_loss_override=structured.stop_loss_price if structured else None,
                    )
                    graph_decision = graph.process_signal(
                        horizon_result.get("final_trade_decision", "")
                    )
                    decision = _apply_structured_report_fields(
                        horizon_result,
                        structured=structured,
                        graph_decision=graph_decision,
                        resolved=resolved,
                    )
                    horizon_result.update(
                        {
                            "status": "completed",
                            "risk_items": (
                                [item.model_dump() for item in structured.risks]
                                if structured
                                else []
                            ),
                            "key_metrics": (
                                [item.model_dump() for item in structured.key_metrics]
                                if structured
                                else []
                            ),
                        }
                    )
                    horizon_results[horizon] = horizon_result

                def _not_requested(horizon: str) -> Dict[str, Any]:
                    return {"horizon": horizon, "status": "not_requested"}

                short_r = horizon_results.get("short", _not_requested("short"))
                medium_r = horizon_results.get("medium", _not_requested("medium"))
                horizon_status = {
                    horizon: horizon_results[horizon].get("status", "completed")
                    for horizon in request.horizons
                }
                all_data_gaps: List[str] = []
                seen_gaps: set[str] = set()
                for horizon in request.horizons:
                    horizon_payload = horizon_results[horizon]
                    for gap in horizon_payload.get("data_gaps", []):
                        if gap not in seen_gaps:
                            seen_gaps.add(gap)
                            all_data_gaps.append(gap)
                    if horizon_payload.get("status") == "failed":
                        failure_gap = (
                            f"【数据获取失败】{horizon} horizon："
                            f"{horizon_payload.get('error') or '分析失败'}"
                        )
                        if failure_gap not in seen_gaps:
                            seen_gaps.add(failure_gap)
                            all_data_gaps.append(failure_gap)
                horizon_metadata = report_service.aggregate_horizon_metadata(
                    (
                        (horizon, horizon_results[horizon])
                        for horizon in request.horizons
                    ),
                    requested_horizons=request.horizons,
                )
                result = {
                    "symbol": ticker,
                    "trade_date": request.trade_date,
                    "analysis_baseline_date": request.trade_date,
                    "data_as_of": next(
                        (
                            horizon_results[horizon].get("data_as_of")
                            for horizon in request.horizons
                            if horizon_results[horizon].get("data_as_of")
                        ),
                        None,
                    ),
                    "mode": "dual_horizon",
                    "status": "partial" if horizon_errors else "completed",
                    "requested_horizons": list(request.horizons),
                    "horizon_status": horizon_status,
                    "failed_horizons": [horizon for horizon in request.horizons if horizon in horizon_errors],
                    "user_intent": user_intent,
                    "model_config_snapshot": model_snapshot,
                    "market_data_context": {
                        horizon: horizon_results[horizon].get("market_data_context")
                        for horizon in request.horizons
                        if horizon_results[horizon].get("status") == "completed"
                    },
                    "short_term": short_r,
                    "medium_term": medium_r,
                    "horizons": {"short": short_r, "medium": medium_r},
                    "data_gaps": all_data_gaps,
                    "falsification_conditions": horizon_metadata["falsification_conditions"],
                    "falsification_conditions_by_horizon": (
                        horizon_metadata["falsification_conditions_by_horizon"]
                    ),
                    "not_applicable": horizon_metadata["not_applicable"],
                    "not_applicable_by_horizon": horizon_metadata["not_applicable_by_horizon"],
                    "analyst_traces": [
                        trace
                        for horizon in request.horizons
                        for trace in horizon_results[horizon].get("analyst_traces", [])
                    ],
                }
                _attach_custom_prompt_snapshot(result, _prompt_snapshot)

                if save_report:
                    def _save_dual_report_sync():
                        with get_db_ctx() as save_db:
                            report_service.create_report(
                                db=save_db,
                                symbol=request.symbol,
                                trade_date=request.trade_date,
                                decision=None,
                                result_data=result,
                                user_id=user_id,
                                risk_items=None,
                                key_metrics=None,
                                probability=None,
                                data_gaps=result["data_gaps"],
                                falsification_conditions=result["falsification_conditions"],
                                not_applicable=result["not_applicable"],
                                confidence_override=None,
                                target_price_override=None,
                                stop_loss_override=None,
                                report_id=job_id,
                                analyst_traces=result.get("analyst_traces"),
                            )
                            save_db.commit()

                    await _save_report_or_raise(job_id, _save_dual_report_sync, stage="save")

                _set_job(
                    job_id,
                    status="completed",
                    result=result,
                    decision=None,
                    error=None,
                    overtime=False,
                    overtime_at=None,
                    finished_at=_utcnow_iso(),
                )
                _emit_job_event(
                    job_id,
                    "job.completed",
                    {
                        "job_id": job_id,
                        "status": result["status"],
                        "result": result,
                        "mode": "dual_horizon",
                        "data_gaps": result["data_gaps"],
                        "falsification_conditions": result["falsification_conditions"],
                        "not_applicable": result["not_applicable"],
                        "horizon_status": result["horizon_status"],
                        "failed_horizons": result["failed_horizons"],
                    },
                )
                _log(f"Job completed successfully: {job_id}")
                _log(f"[Timer] TOTAL Job execution (dual_horizon) took {time.time() - job_start_t:.2f}s")
                return

            short_r = graph._build_horizon_result("short", horizon_states.get("short") or {})
            medium_r = graph._build_horizon_result("medium", horizon_states.get("medium") or {})
            primary_r = short_r if horizon_states.get("short") else medium_r
            graph_decision = graph.process_signal(primary_r.get("final_trade_decision", ""))
            model_snapshot = {
                role: {
                    "provider_type": cfg.get("provider_type"),
                    "model_name": cfg.get("model_name"),
                    "base_url": cfg.get("base_url"),
                    "resolved_via": cfg.get("resolved_via"),
                    "fallback_used": cfg.get("fallback_used"),
                    "profile_display_name": cfg.get("profile_display_name"),
                    "provider_display_name": cfg.get("provider_display_name"),
                }
                for role, cfg in getattr(graph, "role_resolved_configs", {}).items()
            }
            result = {
                "symbol": ticker,
                "trade_date": request.trade_date,
                "analysis_baseline_date": request.trade_date,
                "data_as_of": primary_r.get("data_as_of"),
                "data_gaps": list(primary_r.get("data_gaps") or []),
                "mode": "dual_horizon",
                "user_intent": user_intent,
                "model_config_snapshot": model_snapshot,
                "market_data_context": primary_r.get("market_data_context"),
                "short_term": short_r,
                "medium_term": medium_r,
                "decision": graph_decision or "UNKNOWN",
                # Hoist primary horizon's report fields to top level so that
                # resolve_report_fields / create_report can find them directly.
                "final_trade_decision": primary_r.get("final_trade_decision", ""),
                "investment_plan": primary_r.get("investment_plan", ""),
                "trader_investment_plan": primary_r.get("trader_investment_plan", ""),
                "investment_debate_state": primary_r.get("investment_debate_state"),
                "risk_debate_state": primary_r.get("risk_debate_state"),
                "market_report": primary_r.get("market_report", ""),
                "sentiment_report": primary_r.get("sentiment_report", ""),
                "news_report": primary_r.get("news_report", ""),
                "fundamentals_report": primary_r.get("fundamentals_report", ""),
                "macro_report": primary_r.get("macro_report", ""),
                "smart_money_report": primary_r.get("smart_money_report", ""),
                "volume_price_report": primary_r.get("volume_price_report", ""),
                "analyst_traces": (
                    short_r.get("analyst_traces", []) + medium_r.get("analyst_traces", [])
                ),
            }
            # LLM 结构化提取（目标价、止损、信心、风险、关键指标）
            # 注意：必须在 _set_job(status="completed") 之前完成，否则 SSE 超时
            # 会因为看到 status="completed" 而提前关闭流，导致 job.completed 事件丢失。
            structured = None
            try:
                structured = await asyncio.to_thread(
                    report_service.extract_structured_data,
                    final_trade_decision=primary_r.get("final_trade_decision", ""),
                    fundamentals_report=primary_r.get("fundamentals_report", ""),
                    config=config,
                )
            except Exception as e:
                _log(f"Structured extraction failed (non-fatal): {e}")

            resolved = await asyncio.to_thread(
                report_service.resolve_report_fields,
                result_data=result,
                confidence_override=structured.confidence if structured else None,
                target_price_override=structured.target_price if structured else None,
                stop_loss_override=structured.stop_loss_price if structured else None,
            )
            decision = _apply_structured_report_fields(
                result,
                structured=structured,
                graph_decision=graph_decision,
                resolved=resolved,
            )
            _attach_custom_prompt_snapshot(result, _prompt_snapshot)

            # 自动保存报告到数据库
            if save_report:
                def _save_report_sync():
                    with get_db_ctx() as save_db:
                        report_service.create_report(
                            db=save_db,
                            symbol=request.symbol,
                            trade_date=request.trade_date,
                            decision=decision,
                            result_data=result,
                            user_id=user_id,
                            risk_items=([r.model_dump() for r in structured.risks] if structured else None),
                            key_metrics=([m.model_dump() for m in structured.key_metrics] if structured else None),
                            probability=result["probability"],
                            data_gaps=result["data_gaps"],
                            falsification_conditions=result["falsification_conditions"],
                            not_applicable=result["not_applicable"],
                            confidence_override=result["confidence"],
                            target_price_override=result["target_price"],
                            stop_loss_override=result["stop_loss_price"],
                            report_id=job_id,
                            analyst_traces=result.get("analyst_traces"),
                        )
                        save_db.commit()

                await _save_report_or_raise(job_id, _save_report_sync, stage="save")

            # 所有后处理完成后再标记 completed，防止 SSE 超时提前关闭流
            _set_job(job_id, status="completed", result=result,
                     decision=decision, error=None, overtime=False,
                     overtime_at=None,
                     finished_at=_utcnow_iso())
            _emit_job_event(job_id, "job.completed", {
                "job_id": job_id, "decision": decision,
                "direction": result["direction"],
                "result": result, "mode": "dual_horizon",
                "risk_items": [r.model_dump() for r in structured.risks] if structured else [],
                "key_metrics": [m.model_dump() for m in structured.key_metrics] if structured else [],
                "probability": result["probability"],
                "data_gaps": result["data_gaps"],
                "falsification_conditions": result["falsification_conditions"],
                "not_applicable": result["not_applicable"],
                "confidence": result["confidence"],
                "target_price": result["target_price"],
                "stop_loss_price": result["stop_loss_price"],
            })
            _log(f"Job completed successfully: {job_id}")
            _log(f"[Timer] TOTAL Job execution (dual_horizon) took {time.time() - job_start_t:.2f}s")
            return
        # ── End dual-horizon path ─────────────────────────────────────────────

        if stream_events:
            collected_pool = await asyncio.to_thread(
                graph.data_collector.collect,
                request.symbol,
                request.trade_date,
                horizons=request.horizons,
            )
            market_data_context = (
                collected_pool.get("market_data_context")
                if isinstance(collected_pool, dict)
                else None
            )
            init_state = graph.propagator.create_initial_state(
                request.symbol,
                request.trade_date,
                user_context=user_context_payload,
                selected_analysts=request.selected_analysts,
                request_source=request_source,
                horizon=request.horizons[0] if request.horizons else "short",
                market_data_context=market_data_context,
            )
            args = graph.propagator.get_graph_args()
            
            # Pass job_id as thread_id for LangGraph checkpointer persistence
            if "config" not in args:
                args["config"] = {}
            args["config"]["configurable"] = {"thread_id": job_id}

            report_keys = (
                "market_report",
                "sentiment_report",
                "news_report",
                "fundamentals_report",
                "macro_report",
                "smart_money_report",
                "volume_price_report",
                "investment_plan",
                "trader_investment_plan",
                "final_trade_decision",
            )
            last_report: Dict[str, str] = {}
            seen: Dict[str, bool] = {}

            accumulated_state: Dict[str, Any] = dict(init_state) if isinstance(init_state, dict) else {}
            final_state = accumulated_state
            _tracker_token = current_tracker_var.set(tracker)
            try:
                async for chunk in graph.graph.astream(init_state, **args):
                    if isinstance(chunk, dict):
                        for k, v in chunk.items():
                            if k in ("investment_debate_state", "risk_debate_state", "risk_feedback_state"):
                                if _is_empty_debate_state(v) and not _is_empty_debate_state(accumulated_state.get(k)):
                                    continue
                                accumulated_state[k] = v
                            else:
                                if v is None and k in accumulated_state and accumulated_state[k] is not None:
                                    continue
                                accumulated_state[k] = v
                        final_state = accumulated_state
                    else:
                        final_state = chunk
                    # ── 并行感知的状态推进 ──────────────────
                    # 1. 每个 analyst 报告首次出现 → completed
                    for analyst_key in ANALYST_ORDER:
                        if analyst_key not in request.selected_analysts:
                            continue
                        rkey = ANALYST_REPORT_MAP[analyst_key]
                        aname = ANALYST_AGENT_NAMES[analyst_key]
                        if chunk.get(rkey) and not seen.get(rkey):
                            seen[rkey] = True
                            tracker._set_status(aname, "completed")

                    # 2. 分析师全部完成 → 研究团队开始
                    all_analysts_done = all(
                        seen.get(ANALYST_REPORT_MAP.get(a, "")) for a in tracker.selected_analysts
                    )
                    if all_analysts_done and not seen.get("_research_started"):
                        seen["_research_started"] = True
                        tracker._set_status(ANALYST_AGENT_NAMES["bull"], "in_progress")
                        tracker._set_status(ANALYST_AGENT_NAMES["bear"], "in_progress")
                        tracker._set_status(ANALYST_AGENT_NAMES["research_manager"], "in_progress")

                    debate = chunk.get("investment_debate_state") or {}
                    if debate.get("judge_decision") and not seen.get("judge_decision"):
                        seen["judge_decision"] = True
                        for r_key in ["bull", "bear", "research_manager"]:
                            tracker._set_status(ANALYST_AGENT_NAMES[r_key], "completed")
                        tracker._set_status(ANALYST_AGENT_NAMES["trader"], "in_progress")

                    if chunk.get("trader_investment_plan") and not seen.get("trader_investment_plan"):
                        seen["trader_investment_plan"] = True
                        tracker._set_status(ANALYST_AGENT_NAMES["trader"], "completed")
                        tracker._set_status(ANALYST_AGENT_NAMES["aggressive"], "in_progress")

                    risk = chunk.get("risk_debate_state") or {}
                    if risk.get("judge_decision") and not seen.get("risk_judge_decision"):
                        seen["risk_judge_decision"] = True
                        for r_key in ["aggressive", "neutral", "conservative", "portfolio_manager"]:
                            tracker._set_status(ANALYST_AGENT_NAMES[r_key], "completed")
                    # ────────────────────────────────────────────

                    # ── Partial DB Persistence & UI Streaming ──
                    db_updates = {}
                    for key in report_keys:
                        value = chunk.get(key)
                        if value and value != last_report.get(key):
                            last_report[key] = value
                            db_updates[key] = str(value)
                            # 立即推送报告分片，前端即可“即产即看”
                            tracker._emit_report_chunked(job_id, key, str(value))
                    
                    if db_updates:
                        def _partial_update(updates=db_updates):
                            with get_db_ctx() as _db:
                                report_service.update_report_partial(_db, job_id, **updates)
                        await asyncio.to_thread(_partial_update)
                    
                    # ── Message & Tool Call Handling ──
                    messages = chunk.get("messages", [])
                    if messages:
                        msg = messages[-1]
                        content = _extract_message_text(getattr(msg, "content", ""))
                        agent_name = getattr(msg, "name", None)
                        msg_type = getattr(msg, "type", "unknown")  # human/system/ai/tool

                        if content:
                            if agent_name:
                                _log(f"[Agent Message] {agent_name}: {content[:200]}...")
                            elif msg_type in ("human", "system"):
                                # Graph 入口的初始 prompt，不是 agent 产出，跳过
                                pass
                            else:
                                _log(f"[Agent Message] {msg_type}: {content[:200]}...")

                        for tool_call in getattr(msg, "tool_calls", []) or []:
                            tool_name = tool_call.get("name", "unknown") if isinstance(tool_call, dict) else getattr(tool_call, "name", "unknown")
                            tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
                            _log(f"[Tool Call] {agent_name or msg_type}: {tool_name}")

                            agent_display = agent_name
                            if not agent_display:
                                tool_to_agent = {
                                    "get_stock_data": "数据获取",
                                    "get_indicators": "技术分析师",
                                    "get_fundamentals": "基本面分析师",
                                    "get_income_statement": "基本面分析师",
                                    "get_balance_sheet": "基本面分析师",
                                    "get_cash_flow": "基本面分析师",
                                    "get_news": "新闻分析师",
                                    "get_social_sentiment": "舆情分析师",
                                }
                                agent_display = tool_to_agent.get(tool_name, "系统")

                            tool_description = _generate_tool_description(tool_name, tool_args)
                            _emit_job_event(
                                job_id,
                                "agent.tool_call",
                                {
                                    "agent": agent_display,
                                    "tool": tool_name,
                                    "description": tool_description,
                                },
                            )
                
            except Exception as e:
                _log(f"Error during default streaming: {e}")
            finally:
                current_tracker_var.reset(_tracker_token)
        else:
            single_horizon = request.horizons[0] if request.horizons else "short"
            if single_horizon == "short":
                final_state, _ = await asyncio.to_thread(
                    graph.propagate,
                    request.symbol,
                    request.trade_date,
                    user_context=user_context_payload,
                    selected_analysts=request.selected_analysts,
                    request_source=request_source,
                    thread_id=job_id,
                )
            else:
                # TradingAgentsGraph.propagate historically defaults the
                # propagator state to short.  Keep the API-only medium path
                # explicit without changing the upstream graph contract.
                collected_pool = await asyncio.to_thread(
                    graph.data_collector.collect,
                    request.symbol,
                    request.trade_date,
                    horizons=request.horizons,
                )
                market_data_context = (
                    collected_pool.get("market_data_context")
                    if isinstance(collected_pool, dict)
                    else None
                )
                init_state = graph.propagator.create_initial_state(
                    request.symbol,
                    request.trade_date,
                    user_context=user_context_payload,
                    selected_analysts=request.selected_analysts,
                    request_source=request_source,
                    horizon=single_horizon,
                    market_data_context=market_data_context,
                )
                args = graph.propagator.get_graph_args()
                if "config" not in args:
                    args["config"] = {}
                args["config"]["configurable"] = {"thread_id": job_id}
                final_state = await asyncio.to_thread(
                    graph.graph.invoke,
                    init_state,
                    **args,
                )

        if not final_state:
            raise RuntimeError("graph returned empty final state")

        graph_decision = graph.process_signal(final_state["final_trade_decision"])
        result = _build_result_payload(final_state)
        result["decision"] = graph_decision or "UNKNOWN"

        # 全量收口为 completed/skipped
        for agent, status in tracker.status.items():
            if status not in ("completed", "skipped"):
                tracker._set_status(agent, "completed")

        # LLM 结构化提取（非阻塞，失败不影响主流程）
        # 注意：_set_job(status="completed") 必须在此之后调用，否则 SSE 超时会提前关闭流
        structured = None
        try:
            structured = await asyncio.to_thread(
                report_service.extract_structured_data,
                final_trade_decision=result.get("final_trade_decision", ""),
                fundamentals_report=result.get("fundamentals_report", ""),
                config=config,
            )
        except Exception as e:
            _log(f"Structured extraction failed (non-fatal): {e}")

        # 一次性解析所有字段（方向、信心、目标价等）
        resolved = await asyncio.to_thread(
            report_service.resolve_report_fields,
            result_data=result,
            confidence_override=structured.confidence if structured else None,
            target_price_override=structured.target_price if structured else None,
            stop_loss_override=structured.stop_loss_price if structured else None,
        )

        # 注入结果字典以便通知和保存使用
        decision = _apply_structured_report_fields(
            result,
            structured=structured,
            graph_decision=graph_decision,
            resolved=resolved,
        )
        _attach_custom_prompt_snapshot(result, _prompt_snapshot)

        # 自动保存/收口报告到数据库
        if save_report:
            def _save_report_final_sync():
                with get_db_ctx() as save_db:
                    report_service.create_report(
                        db=save_db,
                        symbol=request.symbol,
                        trade_date=request.trade_date,
                        decision=decision,
                        result_data=result,
                        user_id=user_id,
                        risk_items=([r.model_dump() for r in structured.risks] if structured else None),
                        key_metrics=([m.model_dump() for m in structured.key_metrics] if structured else None),
                        probability=result["probability"],
                        data_gaps=result["data_gaps"],
                        falsification_conditions=result["falsification_conditions"],
                        not_applicable=result["not_applicable"],
                        confidence_override=result["confidence"],
                        target_price_override=result["target_price"],
                        stop_loss_override=result["stop_loss_price"],
                        report_id=job_id,
                        analyst_traces=result.get("analyst_traces"),
                    )
                    save_db.commit()

            await _save_report_or_raise(job_id, _save_report_final_sync, stage="finalize")
        # 所有后处理完成后再标记 completed，防止 SSE 超时提前关闭流
        _set_job(
            job_id,
            status="completed",
            result=result,
            decision=decision,
            error=None,
            overtime=False,
            overtime_at=None,
            finished_at=_utcnow_iso(),
        )
        _emit_job_event(
            job_id,
            "job.completed",
            {
                "job_id": job_id,
                "decision": decision,
                "direction": result["direction"],
                "result": result,
                "risk_items": [r.model_dump() for r in structured.risks] if structured else [],
                "key_metrics": [m.model_dump() for m in structured.key_metrics] if structured else [],
                "probability": result["probability"],
                "data_gaps": result["data_gaps"],
                "falsification_conditions": result["falsification_conditions"],
                "not_applicable": result["not_applicable"],
                "confidence": result["confidence"],
                "target_price": result["target_price"],
                "stop_loss_price": result["stop_loss_price"],
            },
        )
        _log(f"Job completed successfully: {job_id}")
        _log(f"[Timer] TOTAL Job execution (single_horizon) took {time.time() - job_start_t:.2f}s")
    except Exception as exc:
        err_msg = _humanize_analysis_error(f"{type(exc).__name__}: {exc}")
        _set_job(
            job_id,
            status="failed",
            error=err_msg,
            overtime=False,
            overtime_at=None,
            traceback=traceback.format_exc(),
            finished_at=_utcnow_iso(),
        )
        
        # ── Persistent failure recording (short-lived session) ──
        try:
            def _record_failure():
                with get_db_ctx() as err_db:
                    report_service.mark_report_failed(err_db, job_id, f"{err_msg}\n\n{traceback.format_exc()}")
            await asyncio.to_thread(_record_failure)
        except Exception as db_exc:
            _log(f"Failed to record failure in DB: {db_exc}")

        _emit_job_event(
            job_id,
            "job.failed",
            {"job_id": job_id, "error": err_msg},
        )
    finally:
        _shared_data_collector.evict(request.symbol, request.trade_date)


_ANALYSIS_ERROR_HINTS: List[tuple] = [
    (r"Insufficient Balance|Error code: 402",
     "您配置的大模型 API Key 余额不足。请前往模型服务商充值，或在「设置」中更换其他模型。"),
    (r"DataInspectionFailed|sensitive words detect|data_inspection",
     "模型服务商的内容安全审查拦截了本次分析输出（A股分析内容偶发误伤）。请重试一次；若频繁出现，建议在「设置」中更换其他模型服务商。"),
    (r"Error code: 429|too.?many.?requests|throttling|rate.?limit",
     "模型服务限流（请求过于频繁或额度受限）。请稍后重试，或在「设置」中更换模型。"),
    (r"Error code: 401|Authorization Failed|invalid.*api.?key|authentication",
     "模型 API Key 无效或已过期。请在「设置」中检查 API Key 配置并点击「测试」验证。"),
    (r"Unsupported model|invalid_parameter.*model|model.*not.*(exist|found)",
     "配置的模型名称不被服务商支持（可能已下线或改名）。请在「设置」中更换模型名称。"),
    (r"Error code: 5\d\d|overloaded|InternalError|upload file failed",
     "模型服务端暂时故障。请稍后重试；若持续失败，建议在「设置」中更换模型。"),
    (r"Connection error|peer closed connection|Request timed out|timed?.?out|ConnectTimeout|Connection refused",
     "连接模型服务失败（网络波动或服务不可达）。请稍后重试，并确认「设置」中的 Base URL 可以访问。"),
]


def _humanize_analysis_error(err: str) -> str:
    """把 LLM/网络的原始报错翻译成用户能看懂的提示与建议动作。

    识别不了的错误原样返回；识别出的保留截断后的原始错误便于反馈排查。
    """
    for pat, hint in _ANALYSIS_ERROR_HINTS:
        if re.search(pat, err, re.IGNORECASE):
            return f"{hint}\n\n（原始错误：{err[:200]}）"
    return err


def _normalize_symbol(raw: str) -> str:
    s = raw.strip().upper()
    # Priority: 6-digit CN stock code
    m = re.search(r"(\d{6})(?:\.(SH|SZ|SS))?", s)
    if m:
        code = m.group(1)
        suffix = m.group(2)
        if suffix:
            if suffix == "SS":
                return f"{code}.SH"
            return f"{code}.{suffix}"
        market = "SH" if code.startswith(("5", "6", "9")) else "SZ"
        return f"{code}.{market}"
    # Fallback: 1-6 letter ticker
    m2 = re.search(r"([A-Z]{1,6}(?:\.[A-Z]{1,3})?)", s)
    if m2:
        return m2.group(1)
        
    # Final Fallback: Check Chinese Name Map (e.g. "三花智控" -> "002050.SZ")
    stock_map = _load_cn_stock_map()
    if s in stock_map:
        return stock_map[s]
        
    return s


def _extract_chat_text(messages: List[ChatMessage]) -> str:
    if not messages:
        return ""
    last = messages[-1]
    return _extract_message_text(last.content)


_ANALYSIS_REQUIREMENTS_BOUNDARY_RE = re.compile(
    r"\r?\n\r?\n[ \t]*\[分析要求\][ \t]+"
)


def _original_question_for_extraction(text: str) -> str:
    """Keep frontend-appended requirements out of symbol and date extraction."""
    match = _ANALYSIS_REQUIREMENTS_BOUNDARY_RE.search(text)
    return text[:match.start()].strip() if match else text


def _extract_explicit_analysis_date(text: str) -> Optional[str]:
    """Extract only dates stated by the user; never invent today's date."""
    patterns = (
        r"(?<!\d)(20\d{2})[-/](\d{1,2})[-/](\d{1,2})(?!\d)",
        r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)",
        r"(?<!\d)(20\d{2})年(\d{1,2})月(\d{1,2})日?",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            return datetime(
                int(match.group(1)), int(match.group(2)), int(match.group(3))
            ).strftime("%Y-%m-%d")
        except ValueError:
            continue

    if re.search(r"(?:今天|今日)", text):
        return cn_today_str()
    if re.search(r"(?:昨天|昨日)", text):
        return (now_cn().date() - timedelta(days=1)).strftime("%Y-%m-%d")
    return None


def _extract_symbol_and_date(text: str) -> tuple[Optional[str], Optional[str]]:
    text = _original_question_for_extraction(text)
    date = _extract_explicit_analysis_date(text)

    # Priority 1: A-Share 6-digit code (even if stuck to Chinese characters)
    sym_match = re.search(r"(\d{6}(?:\.(?:SH|SZ|SS))?)", text, re.IGNORECASE)
    if sym_match:
        return _normalize_symbol(sym_match.group(1)), date

    # Priority 2: US Stocks or other Tickers (use boundaries for letters to avoid partial words)
    us_match = re.search(r"\b([A-Z]{1,6}(?:\.[A-Z]{1,3})?)\b", text.upper())
    if us_match:
        return us_match.group(1), date

    return None, date


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _parse_stock_csv(raw: str) -> List[Dict[str, Any]]:
    if not raw:
        return []
    lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.startswith("#")]
    if not lines:
        return []

    try:
        df = pd.read_csv(StringIO("\n".join(lines)))
    except Exception:
        return []

    if "Date" not in df.columns:
        return []

    rename_map = {k: k.strip() for k in df.columns}
    df = df.rename(columns=rename_map)
    required = ["Date", "Open", "High", "Low", "Close"]
    for col in required:
        if col not in df.columns:
            return []

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"]).sort_values("Date")
    if df.empty:
        return []

    candles: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        candles.append(
            {
                "date": row["Date"].strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]) if "Volume" in df.columns and pd.notna(row.get("Volume")) else None,
            }
        )
    return candles


CN_INDEX_SYMBOL_MAP = {
    "000001.SH": "sh000001",
    "399001.SZ": "sz399001",
    "399006.SZ": "sz399006",
    "000300.SH": "sh000300",
    "000688.SH": "sh000688",
    "000905.SH": "sh000905",
    "000852.SH": "sh000852",
    "899050.BJ": "bj899050",
}


def _is_cn_index_symbol(symbol: str) -> bool:
    return symbol.upper() in CN_INDEX_SYMBOL_MAP


def _normalize_kline_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    col_map = {
        "日期": "Date",
        "date": "Date",
        "Date": "Date",
        "开盘": "Open",
        "open": "Open",
        "Open": "Open",
        "最高": "High",
        "high": "High",
        "High": "High",
        "最低": "Low",
        "low": "Low",
        "Low": "Low",
        "收盘": "Close",
        "close": "Close",
        "Close": "Close",
        "成交量": "Volume",
        "volume": "Volume",
        "Volume": "Volume",
        "成交额": "Amount",
        "amount": "Amount",
        "Amount": "Amount",
        "涨跌幅": "ChangePercent",
        "涨跌额": "Change",
        "换手率": "TurnoverRate",
    }
    out = df.rename(columns=col_map).copy()
    required = ["Date", "Open", "High", "Low", "Close"]
    if any(col not in out.columns for col in required):
        return pd.DataFrame()

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date")
    for col in ["Open", "High", "Low", "Close", "Volume", "Amount", "ChangePercent", "Change", "TurnoverRate"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out.reset_index(drop=True)


def _fetch_index_kline(symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    import akshare as ak  # type: ignore

    symbol_key = symbol.upper()
    vendor_symbol = CN_INDEX_SYMBOL_MAP.get(symbol_key)
    if not vendor_symbol:
        return []

    yyyymmdd_start = start_date.replace("-", "")
    yyyymmdd_end = end_date.replace("-", "")
    last_exc: Exception | None = None

    for fetcher in (
        lambda: ak.stock_zh_index_daily_em(
            symbol=vendor_symbol,
            start_date=yyyymmdd_start,
            end_date=yyyymmdd_end,
        ),
        lambda: ak.stock_zh_index_daily(symbol=vendor_symbol),
        lambda: ak.index_zh_a_hist(
            symbol=symbol_key.split(".")[0],
            period="daily",
            start_date=yyyymmdd_start,
            end_date=yyyymmdd_end,
        ),
    ):
        try:
            raw_df = fetcher()
            df = _normalize_kline_df(raw_df)
            if df.empty:
                continue
            df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
            if df.empty:
                continue
            candles: List[Dict[str, Any]] = []
            prev_close: float | None = None
            for _, row in df.iterrows():
                close = float(row["Close"])
                change = float(row["Change"]) if "Change" in df.columns and pd.notna(row.get("Change")) else (close - prev_close if prev_close is not None else None)
                change_pct = (
                    float(row["ChangePercent"])
                    if "ChangePercent" in df.columns and pd.notna(row.get("ChangePercent"))
                    else ((change / prev_close) * 100 if prev_close not in (None, 0) and change is not None else None)
                )
                candles.append(
                    {
                        "date": row["Date"].strftime("%Y-%m-%d"),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": close,
                        "volume": float(row["Volume"]) if "Volume" in df.columns and pd.notna(row.get("Volume")) else None,
                        "amount": float(row["Amount"]) if "Amount" in df.columns and pd.notna(row.get("Amount")) else None,
                        "change": change,
                        "change_percent": change_pct,
                        "turnover_rate": float(row["TurnoverRate"]) if "TurnoverRate" in df.columns and pd.notna(row.get("TurnoverRate")) else None,
                    }
                )
                prev_close = close
            return candles
        except Exception as exc:
            last_exc = exc
            continue

    if last_exc:
        _log(f"[kline] index fetch failed for {symbol}: {type(last_exc).__name__}: {last_exc}")
    return []


async def _stream_job_events(job_id: str):
    store = get_job_store()
    yield _sse_pack("job.ready", {"job_id": job_id})
    async for event in store.subscribe(job_id):
        evt_name = event["event"]
        yield _sse_pack(evt_name, event["data"])
        if evt_name in ("job.completed", "job.failed"):
            yield "event: done\ndata: [DONE]\n\n"
            return


@app.get("/healthz")
async def healthz():
    """健康检查，同时探测 asyncio 默认线程池是否被僵尸线程占满。

    向默认 executor 提交一个 no-op，5 秒内排不上队即判定饱和并返回 503
    （生产事故：无超时的网络调用把 64 个 worker 全部占死，所有依赖
    to_thread 的接口静默挂起，前端表现为 Cloudflare 524）。
    """
    identity = _get_runtime_identity()
    payload: Dict[str, Any] = {
        "status": "ok",
        **identity.public_payload(),
    }
    if _default_executor is not None:
        payload["executor_queued"] = _default_executor._work_queue.qsize()
        payload["executor_threads"] = len(_default_executor._threads)
    try:
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(loop.run_in_executor(None, int), timeout=5)
    except asyncio.TimeoutError:
        payload["status"] = "thread_pool_starved"
        return JSONResponse(status_code=503, content=payload)
    return payload


# Simple in-memory rate limiter for version stats: {ip: last_timestamp}
_vs_rate_limit: Dict[str, float] = {}
_VS_RATE_INTERVAL = 3600  # at most once per hour per IP


@app.post("/api/version-stats")
def version_stats(payload: Dict[str, Any] = Body(...), request: Request = None, db: Session = Depends(get_db)):
    """Collect anonymous version statistics from deployed instances."""
    remote_ip = _get_real_ip(request)

    # Rate limit by IP
    now = time.time()
    if remote_ip:
        last = _vs_rate_limit.get(remote_ip, 0)
        if now - last < _VS_RATE_INTERVAL:
            return {"status": "ok"}
        _vs_rate_limit[remote_ip] = now

    record = VersionStatsDB(
        version=str(payload.get("v", ""))[:50],
        nonce=str(payload.get("nonce", ""))[:64],
        remote_ip=remote_ip,
    )
    db.add(record)
    db.commit()
    return {"status": "ok"}


_RESOLVABLE_SYMBOL_RE = re.compile(
    r"^("
    r"\d{6}\.(SH|SZ|BJ)"          # A 股 / 北交所
    r"|\d{4,5}\.HK"                # 港股
    r"|[A-Z][A-Z0-9.\-]{0,10}"     # 美股 / 通用 ticker
    r")$"
)


@app.get("/v1/market/kline", response_model=KlineResponse)
def get_kline(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> KlineResponse:
    end = end_date or cn_today_str()
    if start_date:
        start = start_date
    else:
        start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")

    if _is_cn_index_symbol(symbol):
        candles = _fetch_index_kline(symbol, start, end)
    else:
        # Normalize symbol (convert "阳光电源" -> "300274.SZ")
        original = symbol
        symbol = _normalize_symbol(symbol)
        if not _RESOLVABLE_SYMBOL_RE.match(symbol):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unrecognized symbol {original!r} (normalized to {symbol!r}); "
                    f"expected formats: '300394.SZ' / 'AAPL' / '00700.HK'"
                ),
            )
        config = _build_runtime_config({})
        set_config(config)
        raw = route_to_vendor("get_stock_data", symbol, start, end)
        candles = _parse_stock_csv(raw)
    if not candles:
        raise HTTPException(status_code=404, detail="no kline data")
    return KlineResponse(
        symbol=symbol,
        start_date=start,
        end_date=end,
        candles=candles,
    )


def _normalize_ths_code(code: str) -> str:
    """Convert THS/XQ code like SH601xxx → 601xxx.SH"""
    code = str(code).strip()
    if code.upper().startswith("SH"):
        return f"{code[2:]}.SH"
    if code.upper().startswith("SZ"):
        return f"{code[2:]}.SZ"
    if code.upper().startswith("BJ") or code.upper().startswith("NQ"):
        return f"{code[2:]}.BJ"
    # Bare 6-digit code — guess exchange
    if code.startswith(("6", "5")):
        return f"{code}.SH"
    if code.startswith(("0", "3", "2")):
        return f"{code}.SZ"
    return code


@app.get("/v1/market/hot-stocks")
def get_hot_stocks(source: str = "em", limit: int = 30) -> Dict:
    """Return hot A-share stocks from different sources.
    
    Args:
        source: Data source selection
            - 'em': 东方财富热榜 (EastMoney hot stocks)
            - 'xq': 雪球热门 (Xueqiu most-followed stocks)
            - 'ths': 连涨榜 (Consecutive rising stocks, not general hot list)
        limit: Maximum number of stocks to return
    
    Returns:
        Dict with stocks list, total count, source info, and fallback status
    """
    import akshare as ak

    # 定义数据源尝试顺序（如果主数据源失败，自动尝试备用源）
    source_configs = {
        "em": ("stock_hot_rank_em", None, "东方财富热榜"),
        "xq": ("stock_hot_follow_xq", "最热门", "雪球热门"),
        "ths": ("stock_rank_lxsz_ths", None, "连涨榜"),
    }

    if source not in source_configs:
        raise HTTPException(status_code=400, detail=f"Unknown source: {source}")

    # 尝试主数据源，失败则尝试其他源
    sources_to_try = [source] + [s for s in ["xq", "em", "ths"] if s != source]
    last_error = None

    for src in sources_to_try:
        try:
            func_name, param, desc = source_configs[src]
            func = getattr(ak, func_name)

            # 调用 akshare 函数
            if param:
                df = func(symbol=param).head(limit)
            else:
                df = func().head(limit)

            stocks = []

            if src == "em":
                for i, (_, row) in enumerate(df.iterrows()):
                    stocks.append({
                        "rank": i + 1,
                        "symbol": _normalize_ths_code(str(row.get("代码", ""))),
                        "name": str(row.get("股票名称", "")),
                        "price": float(row.get("最新价", 0) or 0),
                        "change": float(row.get("涨跌额", 0) or 0),
                        "change_pct": float(row.get("涨跌幅", 0) or 0),
                        "extra": "",
                    })

            elif src == "xq":
                for i, (_, row) in enumerate(df.iterrows()):
                    stocks.append({
                        "rank": i + 1,
                        "symbol": _normalize_ths_code(str(row.get("股票代码", ""))),
                        "name": str(row.get("股票简称", "")),
                        "price": float(row.get("最新价", 0) or 0),
                        "change": 0.0,
                        "change_pct": 0.0,
                        "extra": f"关注 {int(row.get('关注', 0)):,}",
                    })

            elif src == "ths":
                for i, (_, row) in enumerate(df.iterrows()):
                    days = int(row.get("连涨天数", 0) or 0)
                    change_pct = float(row.get("连续涨跌幅", 0) or 0)
                    stocks.append({
                        "rank": i + 1,
                        "symbol": _normalize_ths_code(str(row.get("股票代码", ""))),
                        "name": str(row.get("股票简称", "")),
                        "price": float(row.get("收盘价", 0) or 0),
                        "change": 0.0,
                        "change_pct": change_pct,
                        "extra": f"连涨{days}天",
                    })

            # 成功获取数据
            fallback_msg = f" (fallback from {source_configs[source][2]})" if src != source else ""
            _log(f"Hot stocks: successfully fetched from {desc}{fallback_msg}")
            return {
                "stocks": stocks,
                "total": len(stocks),
                "source": src,
                "requested_source": source,
                "fallback": src != source,
            }

        except Exception as e:
            last_error = e
            _log(f"Hot stocks: {desc} failed - {type(e).__name__}: {str(e)[:100]}")
            continue

    # 所有数据源都失败
    raise HTTPException(
        status_code=503,
        detail=f"All data sources failed. Last error: {type(last_error).__name__}: {str(last_error)[:200]}"
    )


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    current_user: UserDB = Depends(_require_api_user),
) -> AnalyzeResponse:
    # Ordinary-analysis entry: resolve omitted dates with the current CN
    # session, while preserving an explicitly requested trading day.
    explicit_date = "trade_date" in request.model_fields_set and bool(
        str(request.trade_date or "").strip()
    )
    try:
        request.trade_date = _normalize_analysis_trade_date(
            request.trade_date if explicit_date else None,
            explicit=explicit_date,
        )
    except TradeCalendarUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"交易日历不可用：{exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    request.trade_date_explicit = explicit_date
    explicit_context = _extract_request_user_context(request)

    def _load_user_context() -> Dict[str, Any]:
        with get_db_ctx() as db:
            return _compose_analysis_user_context(
                db,
                current_user.id,
                request.symbol,
                explicit_context=explicit_context,
            )

    # Don't block the event loop on a sync SQLite read while the scheduler
    # process may be holding write locks.
    merged_user_context = await asyncio.to_thread(_load_user_context)
    _apply_user_context_to_request(request, merged_user_context)

    job_id = uuid4().hex
    now = _utcnow_iso()
    _set_job(
        job_id,
        job_id=job_id,
        user_id=current_user.id,
        status="pending",
        created_at=now,
        started_at=None,
        finished_at=None,
        symbol=request.symbol,
        trade_date=request.trade_date,
        error=None,
        overtime=False,
        overtime_at=None,
        result=None,
        decision=None,
    )
    _emit_job_event(
        job_id,
        "job.created",
        {"job_id": job_id, "symbol": request.symbol, "trade_date": request.trade_date},
    )
    if request.dry_run:
        await _run_job(job_id, request, True, True, current_user.id, "api")
        final_status = _get_job(job_id).get("status", "completed")
        return AnalyzeResponse(job_id=job_id, status=final_status, created_at=now)
    _create_tracked_task(_run_job(job_id, request, True, True, current_user.id, "api"))
    return AnalyzeResponse(job_id=job_id, status="pending", created_at=now)


def _require_job_owner(job_id: str, current_user: UserDB) -> Dict[str, Any]:
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    owner_id = job.get("user_id")
    if owner_id and owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, current_user: UserDB = Depends(_require_api_user)) -> JobStatusResponse:
    job = _require_job_owner(job_id, current_user)
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        finished_at=job.get("finished_at"),
        symbol=job["symbol"],
        trade_date=job["trade_date"],
        error=job.get("error"),
        overtime=bool(job.get("overtime", False)),
        overtime_at=job.get("overtime_at"),
        waiting_ahead_count=job.get("waiting_ahead_count"),
        scheduled_running_count=job.get("scheduled_running_count"),
        scheduled_concurrency_limit=job.get("scheduled_concurrency_limit"),
    )


@app.get("/v1/jobs/{job_id}/result")
def get_job_result(job_id: str, current_user: UserDB = Depends(_require_api_user)) -> Dict[str, Any]:
    job = _require_job_owner(job_id, current_user)
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"job status is {job['status']}")
    return {
        "job_id": job_id,
        "status": job["status"],
        "decision": job.get("decision"),
        "result": job.get("result"),
        "finished_at": job.get("finished_at"),
    }


@app.get("/v1/jobs/{job_id}/events")
def stream_job_events(job_id: str, current_user: UserDB = Depends(_require_api_user)):
    _require_job_owner(job_id, current_user)
    return StreamingResponse(
        _stream_job_events(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


async def _ai_extract_symbol_and_date_streaming(
    text: str, config: Dict[str, Any], job_id: str
) -> tuple[Optional[str], Optional[str], List[str], List[str], List[str], Dict[str, Any]]:
    """
    Async streaming version of _ai_extract_symbol_and_date.
    Emits agent.token events so the frontend can show streaming output during extraction.
    """
    from tradingagents.llm_clients.factory import create_llm_client
    import json as _json

    today = datetime.now().strftime("%Y-%m-%d")
    # 兜底：先用 regex 直接从原文抽 symbol/date，LLM 失败 / 限流 / 返回 null 时
    # 至少不会把用户已经明确输入的代码也判为"无法识别"。
    extraction_text = _original_question_for_extraction(text)
    fast_symbol, fast_date = _extract_symbol_and_date(extraction_text)
    llm_name: Optional[str] = None
    llm_date: Optional[str] = None
    llm_horizons: List[str] = ["short"]
    llm_focus_areas: List[str] = []
    llm_specific_questions: List[str] = []
    llm_user_context: Dict[str, Any] = {}

    try:
        client = create_llm_client(
            provider=config.get("llm_provider", "openai"),
            model=config.get("quick_think_llm"),
            base_url=config.get("backend_url"),
            api_key=config.get("api_key"),
        )
        prompt = f"""你是金融数据助手。从用户消息中提取以下字段并以 JSON 输出。

字段说明：
- stock_name：用户提到的公司名称或股票代码原文（如"华盛天成"、"贵州茅台"、"600519"、"AAPL"）；美股直接填 ticker。
- date：YYYY-MM-DD 格式。今天是 {today}；只有用户明确提到日期时才填写，否则填 null，不要推断今天。
- horizons：分析周期。默认只选短线；若用户明确同时提到"短线与中线/短期和中期"，或同时提到 short and medium，必须返回 ["short", "medium"]；
  仅提到"中线/中期/几个月/季度/长期/趋势投资"→ ["medium"]；其他情况（含未提及）→ ["short"]。
- focus_areas：用户关注的分析维度关键词列表，如 ["技术面", "资金面", "业绩"]，未提及则 []。
- specific_questions：用户提出的具体问题列表，如 ["近期有无催化剂？", "主力是否出货？"]，未提及则 []。
- user_context：从自然语言中提取的账户与约束对象。若未提及返回 {{}}。可包含：
  * objective：建仓 / 加仓 / 减仓 / 止损 / 观察 / 持有处理
  * risk_profile：保守 / 平衡 / 激进
  * investment_horizon：短线 / 波段 / 中线 / 长期
  * cash_available / current_position / current_position_pct / average_cost / max_loss_pct：数字
  * constraints：字符串数组
  * user_notes：仅保留重要但未能结构化归类的信息

仅输出 JSON，不要任何其他文字：
{{"stock_name": "...", "date": null, "horizons": ["short"], "focus_areas": [], "specific_questions": [], "user_context": {{}}}}

如果无法识别股票标的：{{"stock_name": null, "date": null, "horizons": ["short"], "focus_areas": [], "specific_questions": [], "user_context": {{}}}}

用户消息："{extraction_text}"
"""
        llm = client.get_llm()
        _log(f"[LLM Debug] Streaming StockExtract with model: {getattr(llm, 'model_name', 'unknown')}")

        full_content = ""
        async for chunk in llm.astream(prompt):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_content += token
            if token:
                _emit_job_event(job_id, "agent.token", {
                    "agent": "意图解析",
                    "report": "stock_extract",
                    "token": token,
                })

        _log(f"[LLM Debug] StockExtract response: {full_content[:200]}")
        m = re.search(r"\{.*\}", full_content, re.DOTALL)
        if m:
            data = _json.loads(m.group(0))
            llm_name = (data.get("stock_name") or "").strip() or None
            # The regex extractor is the source of truth for explicit dates;
            # the LLM must not turn an omitted date into an implicit today.
            llm_date = fast_date
            llm_horizons = data.get("horizons") or ["short"]
            llm_focus_areas = data.get("focus_areas") or []
            llm_specific_questions = data.get("specific_questions") or []
            llm_user_context = normalize_user_context(data.get("user_context") or {})
    except Exception as e:
        _log(f"[StockExtract streaming] LLM failed: {e}")

    if not llm_name:
        if fast_symbol:
            _log(f"[StockExtract] LLM 未返回 stock_name，使用 regex 兜底: {fast_symbol}")
            return fast_symbol, fast_date, llm_horizons, llm_focus_areas, llm_specific_questions, llm_user_context
        # LLM 挂掉(限流/模型下线/网络)且原文没有代码时，拿原文在本地股票名单里
        # 搜一次：_search_cn_stock_by_name 支持"名称是输入子串"的匹配，
        # "分析一下 飞沃科技" 可以不经 LLM 直接命中 301232.SZ
        local_code = await asyncio.to_thread(_search_cn_stock_by_name, extraction_text)
        if local_code:
            _log(f"[StockExtract] LLM 失败，本地名单从原文兜底命中: {local_code}")
            return local_code, fast_date, llm_horizons, llm_focus_areas, llm_specific_questions, llm_user_context
        return None, None, llm_horizons, llm_focus_areas, llm_specific_questions, llm_user_context

    _log(f"[StockExtract] extracted name='{llm_name}', date={llm_date}, horizons={llm_horizons}")
    if re.match(r"^\d{6}(?:\.(?:SH|SZ|SS))?$", llm_name, re.IGNORECASE) or re.match(r"^[A-Za-z]{1,6}(\.[A-Za-z]+)?$", llm_name):
        symbol = _normalize_symbol(llm_name)
        if symbol:
            return symbol, llm_date, llm_horizons, llm_focus_areas, llm_specific_questions, llm_user_context

    local_code = await asyncio.to_thread(_search_cn_stock_by_name, llm_name)
    if local_code:
        return local_code, llm_date, llm_horizons, llm_focus_areas, llm_specific_questions, llm_user_context

    fallback = _normalize_symbol(llm_name)
    if fallback and re.search(r"\d{6}|[A-Za-z]{2,}", fallback):
        return fallback, llm_date, llm_horizons, llm_focus_areas, llm_specific_questions, llm_user_context

    # 最后兜底：LLM 返回了名字但所有 resolver 都解析不出，且 regex 找到了清晰代码
    if fast_symbol:
        _log(f"[StockExtract] LLM 名 '{llm_name}' 无法解析为代码，使用 regex 兜底: {fast_symbol}")
        return fast_symbol, llm_date or fast_date, llm_horizons, llm_focus_areas, llm_specific_questions, llm_user_context

    return None, llm_date, llm_horizons, llm_focus_areas, llm_specific_questions, llm_user_context


def _ai_extract_symbol_and_date(
    text: str, config: Dict[str, Any]
) -> tuple[Optional[str], Optional[str], List[str], List[str], List[str], Dict[str, Any]]:
    """
    Single-LLM extraction: stock name, date, horizons, focus_areas, specific_questions.
    Then resolves the stock name to an authoritative code via akshare.
    Returns (symbol, date, horizons, focus_areas, specific_questions, inferred_user_context).
    """
    from tradingagents.llm_clients.factory import create_llm_client
    import json as _json

    today = datetime.now().strftime("%Y-%m-%d")
    # 兜底：先用 regex 直接从原文抽 symbol/date，LLM 失败 / 限流 / 返回 null 时
    # 至少不会把用户已经明确输入的代码也判为"无法识别"。
    extraction_text = _original_question_for_extraction(text)
    fast_symbol, fast_date = _extract_symbol_and_date(extraction_text)

    llm_name: Optional[str] = None
    llm_date: Optional[str] = None
    llm_horizons: List[str] = ["short"]
    llm_focus_areas: List[str] = []
    llm_specific_questions: List[str] = []
    llm_user_context: Dict[str, Any] = {}
    try:
        client = create_llm_client(
            provider=config.get("llm_provider", "openai"),
            model=config.get("quick_think_llm"),
            base_url=config.get("backend_url"),
            api_key=config.get("api_key"),
        )
        prompt = f"""你是金融数据助手。从用户消息中提取以下字段并以 JSON 输出。

字段说明：
- stock_name：用户提到的公司名称或股票代码原文（如"华盛天成"、"贵州茅台"、"600519"、"AAPL"）；美股直接填 ticker。
- date：YYYY-MM-DD 格式。今天是 {today}；只有用户明确提到日期时才填写，否则填 null，不要推断今天。
- horizons：分析周期。默认只选短线；若用户明确同时提到"短线与中线/短期和中期"，或同时提到 short and medium，必须返回 ["short", "medium"]；
  仅提到"中线/中期/几个月/季度/长期/趋势投资"→ ["medium"]；其他情况（含未提及）→ ["short"]。
- focus_areas：用户关注的分析维度关键词列表，如 ["技术面", "资金面", "业绩"]，未提及则 []。
- specific_questions：用户提出的具体问题列表，如 ["近期有无催化剂？", "主力是否出货？"]，未提及则 []。
- user_context：从自然语言中提取的账户与约束对象。若未提及返回 {{}}。可包含：
  * objective：建仓 / 加仓 / 减仓 / 止损 / 观察 / 持有处理
  * risk_profile：保守 / 平衡 / 激进
  * investment_horizon：短线 / 波段 / 中线 / 长期
  * cash_available / current_position / current_position_pct / average_cost / max_loss_pct：数字
  * constraints：字符串数组
  * user_notes：仅保留重要但未能结构化归类的信息

仅输出 JSON，不要任何其他文字：
{{"stock_name": "...", "date": null, "horizons": ["short"], "focus_areas": [], "specific_questions": [], "user_context": {{}}}}

如果无法识别股票标的：{{"stock_name": null, "date": null, "horizons": ["short"], "focus_areas": [], "specific_questions": [], "user_context": {{}}}}

用户消息："{extraction_text}"
"""
        llm = client.get_llm()
        
        # 调试日志：打印请求参数
        target_url = getattr(llm, 'openai_api_base', 'default')
        _log(f"[LLM Debug] Requesting StockExtract with model: {getattr(llm, 'model_name', 'unknown')} at {target_url}")
        _log(f"[LLM Debug] Prompt: {prompt[:500]}...")

        response = llm.invoke(prompt)
        raw = response if isinstance(response, str) else getattr(response, "content", str(response))
        
        # 调试日志：打印原始响应
        _log(f"[LLM Debug] Raw Response: {raw}")

        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = _json.loads(m.group(0))
            llm_name = (data.get("stock_name") or "").strip() or None
            # The regex extractor is the source of truth for explicit dates;
            # the LLM must not turn an omitted date into an implicit today.
            llm_date = fast_date
            llm_horizons = data.get("horizons") or ["short"]
            llm_focus_areas = data.get("focus_areas") or []
            llm_specific_questions = data.get("specific_questions") or []
            llm_user_context = normalize_user_context(data.get("user_context") or {})
    except Exception as e:
        _log(f"[StockExtract] LLM failed: {e}")

    if not llm_name:
        if fast_symbol:
            _log(f"[StockExtract] LLM 未返回 stock_name，使用 regex 兜底: {fast_symbol}")
            return fast_symbol, fast_date, llm_horizons, llm_focus_areas, llm_specific_questions, llm_user_context
        # LLM 挂掉且原文没有代码时，拿原文在本地股票名单里搜一次（同流式版本）
        local_code = _search_cn_stock_by_name(extraction_text)
        if local_code:
            _log(f"[StockExtract] LLM 失败，本地名单从原文兜底命中: {local_code}")
            return local_code, fast_date, llm_horizons, llm_focus_areas, llm_specific_questions, llm_user_context
        _log(f"[StockExtract] LLM returned no stock name for: '{text[:40]}'")
        return None, None, llm_horizons, llm_focus_areas, llm_specific_questions, llm_user_context

    _log(f"[StockExtract] LLM extracted name='{llm_name}', date={llm_date}, horizons={llm_horizons}")

    # ── Step 2: If looks like a direct code (digits / letters), normalize it ──
    if re.match(r"^\d{6}(?:\.(?:SH|SZ|SS))?$", llm_name, re.IGNORECASE) or re.match(r"^[A-Za-z]{1,6}(\.[A-Za-z]+)?$", llm_name):
        symbol = _normalize_symbol(llm_name)
        if symbol:
            _log(f"[StockExtract] Direct code: {symbol}")
            return symbol, llm_date, llm_horizons, llm_focus_areas, llm_specific_questions, llm_user_context

    # ── Step 3: Search akshare A-share name database ──────────────────────────
    local_code = _search_cn_stock_by_name(llm_name)
    if local_code:
        _log(f"[StockExtract] akshare match: '{llm_name}' → {local_code}")
        return local_code, llm_date, llm_horizons, llm_focus_areas, llm_specific_questions, llm_user_context

    # ── Step 4: Last resort — treat LLM name as a raw code ────────────────────
    # _normalize_symbol 在找不到代码时会原样返回，需要校验结果包含数字/英文，
    # 避免把"天孚通讯"这种纯中文 LLM 输出当成 symbol 透传给 provider。
    fallback = _normalize_symbol(llm_name)
    if fallback and re.search(r"\d{6}|[A-Za-z]{2,}", fallback):
        _log(f"[StockExtract] Fallback normalize: '{llm_name}' → {fallback}")
        return fallback, llm_date, llm_horizons, llm_focus_areas, llm_specific_questions, llm_user_context

    # 最后兜底：LLM 给了名字但所有 resolver 都解析不出，且 regex 找到了清晰代码
    if fast_symbol:
        _log(f"[StockExtract] LLM 名 '{llm_name}' 无法解析为代码，使用 regex 兜底: {fast_symbol}")
        return fast_symbol, llm_date or fast_date, llm_horizons, llm_focus_areas, llm_specific_questions, llm_user_context

    _log(f"[StockExtract] Could not resolve '{llm_name}' to a stock code")
    return None, llm_date, llm_horizons, llm_focus_areas, llm_specific_questions, llm_user_context

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    current_user: UserDB = Depends(_require_api_user),
):
    text = _extract_chat_text(request.messages)
    config = await asyncio.to_thread(_build_runtime_config, request.config_overrides, user_id=current_user.id)

    # ── 流式模式：立刻返回 SSE 流，在后台异步提取意图再启动任务 ──────────────────
    # 这样用户提交查询后立刻收到 job.ready，不用等待 thinking 模型的 StockExtract。
    if request.stream:
        job_id = uuid4().hex

        async def _extract_and_run():
            now = _utcnow_iso()
            _set_job(
                job_id,
                job_id=job_id,
                user_id=current_user.id,
                status="pending",
                created_at=now,
                started_at=None,
                finished_at=None,
                symbol=None,
                trade_date=None,
                error=None,
                result=None,
                decision=None,
                request_source="chat",
            )
            try:
                symbol, trade_date, horizons, focus_areas, specific_questions, inferred_user_context = \
                    await _ai_extract_symbol_and_date_streaming(text, config, job_id)
                horizons = _normalize_analysis_horizons(horizons, query=text)
                date_explicit = bool(str(trade_date or "").strip())
                try:
                    trade_date = _normalize_analysis_trade_date(
                        trade_date if date_explicit else None,
                        explicit=date_explicit,
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc

                if not symbol:
                    _emit_job_event(job_id, "job.failed", {
                        "error": "抱歉，我没能从您的消息中识别出股票标的。请输入代码（如 600519.SH）或可识别的公司名称。"
                    })
                    return

                pre_intent = {
                    "raw_query": text,
                    "ticker": symbol,
                    "horizons": horizons,
                    "focus_areas": focus_areas,
                    "specific_questions": specific_questions,
                }
                explicit_context = _extract_request_user_context(request)

                def _load_user_context() -> Dict[str, Any]:
                    with get_db_ctx() as db:
                        return _compose_analysis_user_context(
                            db,
                            current_user.id,
                            symbol,
                            explicit_context=explicit_context,
                            inferred_context=inferred_user_context,
                        )

                merged_user_context = await asyncio.to_thread(_load_user_context)
                pre_intent["user_context"] = merged_user_context
                analyze_req = AnalyzeRequest(
                    symbol=symbol,
                    trade_date=trade_date,
                    trade_date_explicit=date_explicit,
                    selected_analysts=request.selected_analysts,
                    config_overrides=request.config_overrides,
                    dry_run=request.dry_run,
                    query=text,
                    horizons=horizons,
                    user_intent=pre_intent,
                    objective=merged_user_context.get("objective"),
                    risk_profile=merged_user_context.get("risk_profile"),
                    investment_horizon=merged_user_context.get("investment_horizon"),
                    cash_available=merged_user_context.get("cash_available"),
                    current_position=merged_user_context.get("current_position"),
                    current_position_pct=merged_user_context.get("current_position_pct"),
                    average_cost=merged_user_context.get("average_cost"),
                    max_loss_pct=merged_user_context.get("max_loss_pct"),
                    constraints=merged_user_context.get("constraints", []),
                    user_notes=merged_user_context.get("user_notes"),
                )
                now = _utcnow_iso()
                _set_job(
                    job_id,
                    job_id=job_id,
                    user_id=current_user.id,
                    status="pending",
                    created_at=now,
                    started_at=None,
                    finished_at=None,
                    symbol=analyze_req.symbol,
                    trade_date=analyze_req.trade_date,
                    error=None,
                    result=None,
                    decision=None,
                    request_source="chat",
                )
                _emit_job_event(
                    job_id,
                    "job.created",
                    {"job_id": job_id, "symbol": analyze_req.symbol, "trade_date": analyze_req.trade_date},
                )
                await _run_job(job_id, analyze_req, True, True, current_user.id, "chat")
            except Exception as exc:
                err_msg = _humanize_analysis_error(str(exc))
                _log(f"[chat] _extract_and_run failed: {exc}")
                _set_job(
                    job_id,
                    status="failed",
                    error=err_msg,
                    finished_at=_utcnow_iso(),
                    overtime=False,
                    overtime_at=None,
                )
                try:
                    with get_db_ctx() as db:
                        report_service.mark_report_failed(db, job_id, err_msg)
                except Exception:
                    pass
                _emit_job_event(job_id, "job.failed", {"job_id": job_id, "error": err_msg})

        _create_tracked_task(_extract_and_run())
        return StreamingResponse(
            _stream_job_events(job_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # ── 非流式模式：保持原有阻塞行为 ─────────────────────────────────────────────
    symbol, trade_date, horizons, focus_areas, specific_questions, inferred_user_context = \
        await asyncio.to_thread(_ai_extract_symbol_and_date, text, config)
    horizons = _normalize_analysis_horizons(horizons, query=text)
    date_explicit = bool(str(trade_date or "").strip())
    try:
        trade_date = _normalize_analysis_trade_date(
            trade_date if date_explicit else None,
            explicit=date_explicit,
        )
    except TradeCalendarUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"交易日历不可用：{exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not symbol:
        raise HTTPException(status_code=400, detail="抱歉，我没能从您的消息中识别出股票标的。请输入代码（如 600519.SH）或可识别的公司名称。")

    pre_intent = {
        "raw_query": text,
        "ticker": symbol,
        "horizons": horizons,
        "focus_areas": focus_areas,
        "specific_questions": specific_questions,
    }
    explicit_context = _extract_request_user_context(request)

    def _load_user_context_nonstream() -> Dict[str, Any]:
        with get_db_ctx() as db:
            return _compose_analysis_user_context(
                db,
                current_user.id,
                symbol,
                explicit_context=explicit_context,
                inferred_context=inferred_user_context,
            )

    merged_user_context = await asyncio.to_thread(_load_user_context_nonstream)
    pre_intent["user_context"] = merged_user_context
    analyze_req = AnalyzeRequest(
        symbol=symbol,
        trade_date=trade_date,
        trade_date_explicit=date_explicit,
        selected_analysts=request.selected_analysts,
        config_overrides=request.config_overrides,
        dry_run=request.dry_run,
        query=text,
        horizons=horizons,
        user_intent=pre_intent,
        objective=merged_user_context.get("objective"),
        risk_profile=merged_user_context.get("risk_profile"),
        investment_horizon=merged_user_context.get("investment_horizon"),
        cash_available=merged_user_context.get("cash_available"),
        current_position=merged_user_context.get("current_position"),
        current_position_pct=merged_user_context.get("current_position_pct"),
        average_cost=merged_user_context.get("average_cost"),
        max_loss_pct=merged_user_context.get("max_loss_pct"),
        constraints=merged_user_context.get("constraints", []),
        user_notes=merged_user_context.get("user_notes"),
    )
    job_id = uuid4().hex
    now = _utcnow_iso()
    _set_job(
        job_id,
        job_id=job_id,
        user_id=current_user.id,
        status="pending",
        created_at=now,
        started_at=None,
        finished_at=None,
        symbol=analyze_req.symbol,
        trade_date=analyze_req.trade_date,
        error=None,
        result=None,
        decision=None,
    )
    _emit_job_event(
        job_id,
        "job.created",
        {"job_id": job_id, "symbol": analyze_req.symbol, "trade_date": analyze_req.trade_date},
    )
    if request.dry_run:
        await _run_job(job_id, analyze_req, True, True, current_user.id, "chat")
        status_text = _get_job(job_id).get("status", "completed")
        decision_text = _get_job(job_id).get("decision", "DRY_RUN")
        return {
            "id": f"chatcmpl-{job_id}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": (
                            f"已完成分析任务：{job_id}\n"
                            f"symbol={analyze_req.symbol}, trade_date={analyze_req.trade_date}\n"
                            f"status={status_text}, decision={decision_text}"
                        ),
                    },
                }
            ],
        }
    _create_tracked_task(_run_job(job_id, analyze_req, True, True, current_user.id, "chat"))
    return {
        "id": f"chatcmpl-{job_id}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": (
                        f"已启动分析任务：{job_id}\n"
                        f"symbol={analyze_req.symbol}, trade_date={analyze_req.trade_date}\n"
                        f"可通过 /v1/jobs/{job_id} 与 /v1/jobs/{job_id}/result 查询结果。"
                    ),
                },
            }
        ],
    }


# Report API Endpoints
@app.post("/v1/reports", response_model=ReportResponse)
def create_report_endpoint(
    request: ReportCreateRequest,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_api_user),
):
    """手动创建报告（通常由系统自动调用）."""
    report = report_service.create_report(
        db=db,
        symbol=request.symbol,
        trade_date=request.trade_date,
        decision=request.decision,
        result_data=request.result_data,
        user_id=current_user.id,
        probability=request.probability,
        data_gaps=request.data_gaps,
        falsification_conditions=request.falsification_conditions,
        not_applicable=request.not_applicable,
    )
    return report


@app.get("/v1/announcements/latest", response_model=LatestAnnouncementResponse)
def get_latest_announcement():
    return {"announcement": _load_latest_announcement()}


@app.get("/v1/reports", response_model=ReportListResponse)
def list_reports(
    symbol: Optional[str] = Query(None, description="按股票代码筛选"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_api_user),
):
    """获取报告列表."""
    total = report_service.count_reports(db=db, user_id=current_user.id, symbol=symbol)
    reports = report_service.get_reports_by_user(
        db=db,
        user_id=current_user.id,
        symbol=symbol,
        skip=skip,
        limit=limit,
    )
    code_to_name = _get_report_reverse_stock_map()
    for r in reports:
        r.name = code_to_name.get(r.symbol, r.symbol)
        _attach_job_runtime_state(r, str(getattr(r, "id", "")))
    return {"total": total, "reports": reports}


@app.post("/v1/reports/latest-by-symbols", response_model=LatestReportsBySymbolsResponse)
def list_latest_reports_by_symbols(
    body: LatestReportsBySymbolsRequest,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_api_user),
):
    reports = report_service.get_latest_reports_by_symbols(
        db=db,
        user_id=current_user.id,
        symbols=body.symbols,
    )
    return {"reports": reports}


@app.get("/v1/reports/{report_id}", response_model=ReportDetailResponse)
def get_report_endpoint(
    report_id: str,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_api_user),
):
    """获取报告详情."""
    report = report_service.get_report(db, report_id, user_id=current_user.id)
    if not report:
        raise HTTPException(status_code=404, detail="报告不存在")
    if str(report.status or "") in report_service.ACTIVE_REPORT_STATUSES and not _get_job(report_id):
        report = report_service.finalize_orphan_report(db, report)
    code_to_name = _get_report_reverse_stock_map()
    report.name = code_to_name.get(report.symbol, report.symbol)
    _attach_job_runtime_state(report, report_id)
    return report


@app.delete("/v1/reports/{report_id}")
def delete_report_endpoint(
    report_id: str,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_api_user),
):
    """删除报告."""
    success = report_service.delete_report(db, report_id, user_id=current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="报告不存在")
    return {"message": "报告已删除"}


@app.post("/v1/reports/batch/delete", response_model=ReportBatchDeleteResponse)
def batch_delete_reports_endpoint(
    body: ReportBatchDeleteRequest,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_api_user),
):
    try:
        return report_service.batch_delete_reports(db, body.report_ids, user_id=current_user.id)
    except ValueError as e:
        raise HTTPException(400, str(e))


# ─── API Token Endpoints ────────────────────────────────────────────────────

@app.get("/v1/tokens", response_model=List[UserTokenListItem])
def list_tokens(
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    """获取当前用户的所有 API Token（不返回完整 token）。"""
    return token_service.list_user_tokens(db, current_user.id)


@app.post("/v1/tokens", response_model=UserTokenResponse)
def create_token(
    request: UserTokenCreateRequest,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    """创建一个新的 API Token。完整 token 仅在此接口返回一次。"""
    try:
        return token_service.create_token(db, current_user.id, request.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/v1/tokens/{token_id}")
def delete_token(
    token_id: str,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    """吊销并删除一个 API Token。"""
    success = token_service.delete_token(db, current_user.id, token_id)
    if not success:
        raise HTTPException(status_code=404, detail="Token 不存在")
    return {"message": "Token 已吊销"}


# ─── Backtest Endpoints ───────────────────────────────────────────────────────

from api.services import backtest_service as _bt


class BacktestRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    selected_analysts: List[str] = ["market", "news", "fundamentals", "sentiment"]
    hold_days: int = 5
    # Keep the raw value until service validation so FastAPI/Pydantic cannot
    # coerce bool or numeric strings into an int before the strict check.
    sample_interval: Any = 7
    config_overrides: Optional[Dict[str, Any]] = None


@app.post("/v1/backtest")
def submit_backtest(
    request: BacktestRequest,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_api_user),
) -> Dict:
    """提交历史回测任务，返回 job_id."""
    config = _build_runtime_config(request.config_overrides or {}, user_id=current_user.id, db=db)
    try:
        job_id = _bt.submit(
            user_id=current_user.id,
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            selected_analysts=request.selected_analysts,
            hold_days=request.hold_days,
            sample_interval=request.sample_interval,
            config=config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except _bt.BacktestQueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    return {"job_id": job_id, "status": "pending"}


@app.get("/v1/backtest")
def list_backtests(current_user: UserDB = Depends(_require_api_user)) -> Dict:
    """列出所有回测任务."""
    jobs = _bt.list_jobs(user_id=current_user.id)
    return {"jobs": jobs, "total": len(jobs)}


def _require_backtest_owner(job_id: str, current_user: UserDB) -> Dict[str, Any]:
    job = _bt.get_job(job_id, user_id=current_user.id)
    if not job:
        raise HTTPException(status_code=404, detail="回测任务不存在")
    return job


@app.get("/v1/backtest/{job_id}")
def get_backtest(job_id: str, current_user: UserDB = Depends(_require_api_user)) -> Dict:
    """获取回测任务状态和结果."""
    return _require_backtest_owner(job_id, current_user)


@app.delete("/v1/backtest/{job_id}")
def delete_backtest(job_id: str, current_user: UserDB = Depends(_require_api_user)) -> Dict:
    """删除回测任务."""
    if not _bt.delete_job(job_id, user_id=current_user.id):
        raise HTTPException(status_code=404, detail="回测任务不存在")
    return {"message": "已删除"}


# ─── Calibration Endpoints ────────────────────────────────────────────────────

from api.services import calibration_service as _calibration  # noqa: E402

# /v1/calibration is a synchronous, I/O-heavy endpoint (it resolves a price
# window per evaluated report).  Beyond the service-level concurrency guard and
# per-filter-key cache, we rate-limit it per source IP so a burst cannot occupy
# worker threads or exhaust the shared data-provider quota.
_CALIBRATION_RATE_WINDOW_SECONDS = 60
_CALIBRATION_RATE_MAX = int(os.getenv("CALIBRATION_RATE_MAX", "5"))
_calibration_rate_hits: Dict[str, List[float]] = {}
_calibration_rate_lock = Lock()


def _enforce_calibration_rate_limit(request: Request) -> None:
    remote_ip = _get_real_ip(request)
    now = time.time()
    with _calibration_rate_lock:
        hits = _calibration_rate_hits.setdefault(remote_ip, [])
        hits[:] = [t for t in hits if now - t < _CALIBRATION_RATE_WINDOW_SECONDS]
        if len(hits) >= _CALIBRATION_RATE_MAX:
            raise HTTPException(status_code=429, detail="校准度接口请求过于频繁，请稍后重试")
        hits.append(now)
        # Periodic cleanup: once the table grows past a threshold, drop buckets
        # whose entries have all expired so memory cannot grow unbounded with
        # the number of distinct source IPs.
        if len(_calibration_rate_hits) > 1024:
            empty_buckets = [
                ip for ip, timestamps in _calibration_rate_hits.items() if not timestamps
            ]
            for ip in empty_buckets:
                _calibration_rate_hits.pop(ip, None)


@app.get("/v1/calibration")
def get_calibration(
    request: Request,
    start_date: Optional[str] = Query(None, description="起始分析日期 (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="结束分析日期 (YYYY-MM-DD)"),
    symbol: Optional[str] = Query(None, description="股票代码"),
    prompt_version: Optional[str] = Query(None, description="提示词版本（resolved_hash 子串）"),
    model: Optional[str] = Query(None, description="模型名称（子串匹配 model_config_snapshot）"),
    hold_days: int = Query(_calibration.DEFAULT_HOLD_DAYS, ge=1, le=60),
    limit: Optional[int] = Query(None, ge=1, le=_calibration.MAX_CALIBRATION_LIMIT),
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_api_user),
) -> Dict:
    """返回历史报告的校准度统计：可靠性曲线 + Brier score.

    按报告的 probability（0–1）分桶 0-50 / 50-60 / 60-70 / 70-80 / 80+，
    统计各桶实际上涨率；支持按日期段 / 提示词版本 / 模型过滤。
    结果可归因到每份报告冻结的 custom_prompt_snapshot / model_config_snapshot。
    结果按过滤器键缓存；并发计算受限，命中上限返回 429。
    """
    _enforce_calibration_rate_limit(request)
    try:
        return _calibration.compute_calibration(
            db,
            user_id=current_user.id,
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            prompt_version=prompt_version,
            model=model,
            hold_days=hold_days,
            limit=limit,
        )
    except _calibration.CalibrationBusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


# ─── Runtime Config Endpoints ────────────────────────────────────────────────

_CONFIG_ALLOWED_KEYS = {
    "llm_provider", "deep_think_llm", "quick_think_llm",
    "backend_url", "max_debate_rounds", "max_risk_discuss_rounds",
}
_CONFIG_PREFERENCE_KEYS = {"email_report_enabled", "wecom_report_enabled"}
_CONFIG_MODEL_KEYS = ("llm_provider", "backend_url", "quick_think_llm", "deep_think_llm")
_CONFIG_MODEL_LABELS = {
    "quick_think_llm": "常规模型",
    "deep_think_llm": "推理模型",
}
_CONFIG_PROBE_TIMEOUT_SECONDS = 12.0
_CONFIG_PROBE_PROMPT = "Reply with the single word OK."
_CONFIG_WARMUP_TIMEOUT_SECONDS = 20.0
_CONFIG_WARMUP_PROMPT = "Reply with the single word OK."


def _mask_secret_value(value: Optional[str], *, head: int = 4, tail: int = 4) -> Optional[str]:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    if len(normalized) <= head + tail:
        return "*" * max(6, len(normalized))
    return f"{normalized[:head]}{'*' * max(6, len(normalized) - head - tail)}{normalized[-tail:]}"


def _mask_wecom_webhook(webhook_url: Optional[str]) -> Optional[str]:
    normalized = str(webhook_url or "").strip()
    if not normalized:
        return None
    prefix = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key="
    if normalized.startswith(prefix):
        masked_key = _mask_secret_value(normalized[len(prefix):])
        return f"{prefix}{masked_key}"
    if normalized.startswith("http"):
        if "key=" in normalized:
            base, key = normalized.rsplit("key=", 1)
            return f"{base}key={_mask_secret_value(key)}"
        return _mask_secret_value(normalized, head=18, tail=8)
    return _mask_secret_value(normalized)


def _warmup_model_names(config: Dict[str, Any]) -> List[str]:
    seen: set[str] = set()
    models: List[str] = []
    for key in ("quick_think_llm", "deep_think_llm"):
        value = str(config.get(key) or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        models.append(value)
    return models


def _should_trigger_config_warmup(
    before_cfg: UserRuntimeConfigResponse,
    after_cfg: UserRuntimeConfigResponse,
    updates: UserRuntimeConfigUpdateRequest,
) -> bool:
    if not updates.warmup:
        return False
    if updates.force_warmup:
        return True
    if updates.api_key:
        return True
    before = before_cfg.model_dump()
    after = after_cfg.model_dump()
    return any(before.get(key) != after.get(key) for key in _CONFIG_MODEL_KEYS)


def _build_pending_runtime_config(
    updates: UserRuntimeConfigUpdateRequest,
    user_id: str,
    db: Session,
) -> Dict[str, Any]:
    config = _build_runtime_config({}, user_id=user_id, db=db)
    for key in _CONFIG_ALLOWED_KEYS:
        value = getattr(updates, key, None)
        if value is not None:
            config[key] = value

    if updates.clear_api_key:
        config["api_key"] = ""
    elif updates.api_key:
        config["api_key"] = updates.api_key

    quick = config.get("quick_think_llm")
    deep = config.get("deep_think_llm")
    if not deep and quick:
        config["deep_think_llm"] = quick
    if not quick and deep:
        config["quick_think_llm"] = deep
    return config


def _should_probe_runtime_config(
    before_cfg: UserRuntimeConfigResponse,
    pending_cfg: Dict[str, Any],
    updates: UserRuntimeConfigUpdateRequest,
) -> bool:
    del before_cfg, pending_cfg
    if updates.clear_api_key:
        return False
    return bool(updates.api_key)


def _probe_runtime_config(config: Dict[str, Any]) -> Dict[str, str]:
    from tradingagents.llm_clients.factory import create_llm_client

    provider = str(config.get("llm_provider") or "openai")
    base_url = config.get("backend_url")
    api_key = str(config.get("api_key") or "").strip()
    model = str(config.get("quick_think_llm") or config.get("deep_think_llm") or "").strip()

    if not model or not api_key:
        return {"status": "skipped", "reason": "missing_model_or_key"}

    try:
        client = create_llm_client(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=_CONFIG_PROBE_TIMEOUT_SECONDS,
            max_retries=0,
        )
        llm = client.get_llm()
        response = llm.invoke(_CONFIG_PROBE_PROMPT)
        raw = response if isinstance(response, str) else getattr(response, "content", str(response))
        preview = str(raw).strip().replace("\n", " ")[:80] or "<empty>"
        return {"status": "ok", "model": model, "preview": preview}
    except Exception as exc:
        detail = str(exc).strip()
        lowered = detail.lower()
        if "401" in lowered or "invalid authentication" in lowered or "authenticationerror" in lowered:
            raise HTTPException(
                status_code=400,
                detail="模型 Key 验证失败：上游返回 401 Invalid Authentication，请检查 API Key 是否正确。",
            ) from exc
        raise HTTPException(
            status_code=400,
            detail=f"模型连接验证失败：{detail[:200] or 'unknown error'}",
        ) from exc


def _invoke_runtime_warmup(
    config: Dict[str, Any],
    prompt: str,
    user_id: str,
    timeout: float = _CONFIG_WARMUP_TIMEOUT_SECONDS,
) -> List[Dict[str, Any]]:
    from tradingagents.llm_clients.factory import create_llm_client

    provider = str(config.get("llm_provider") or "openai")
    base_url = config.get("backend_url")
    api_key = config.get("api_key")

    targets_dict: Dict[Tuple[str, str, Optional[str], Optional[str]], List[str]] = {}
    for key in ("quick_think_llm", "deep_think_llm"):
        m = str(config.get(key) or "").strip()
        if m:
            lbl = _CONFIG_MODEL_LABELS.get(key, key)
            targets_dict.setdefault((provider, m, base_url, api_key), []).append(lbl)

    try:
        with get_db_ctx() as db:
            resolved_roles = role_routing_service.resolve_all_roles(db, user_id, config)
            for r_key, r_cfg in resolved_roles.items():
                m = str(r_cfg.get("model_name") or "").strip()
                prov = str(r_cfg.get("provider_type") or provider)
                b_url = r_cfg.get("base_url") or base_url
                a_key = r_cfg.get("api_key") or api_key
                if m:
                    lbl = f"角色: {r_key}"
                    lbls = targets_dict.setdefault((prov, m, b_url, a_key), [])
                    if lbl not in lbls:
                        lbls.append(lbl)
    except Exception as err:
        logger.warning(f"[LLM Warmup] Failed resolving roles for warmup: {err}")

    targets = [(prov, model, b_url, a_key, lbls) for (prov, model, b_url, a_key), lbls in targets_dict.items()]

    if not targets:
        raise HTTPException(status_code=400, detail="请先配置至少一个可用模型。")

    _log(
        f"[LLM Warmup] user={user_id} invoking provider={provider} "
        f"models={[t[1] for t in targets]} base_url={base_url or 'default'}"
    )

    results: List[Dict[str, Any]] = []
    errors: List[str] = []
    for prov, model, b_url, a_key, labels in targets:
        try:
            client = create_llm_client(
                provider=prov,
                model=model,
                base_url=b_url,
                api_key=a_key,
                timeout=timeout,
                max_retries=0,
            )
            llm = client.get_llm()
            response = llm.invoke(prompt)
            raw = response if isinstance(response, str) else getattr(response, "content", str(response))
            content = str(raw).strip() or "<empty>"
            preview = content.replace("\n", " ")[:80]
            _log(f"[LLM Warmup] user={user_id} model={model} success response={preview}")
            results.append({
                "model": model,
                "targets": labels,
                "content": content,
                "error": None,
            })
        except Exception as exc:
            detail = str(exc).strip() or "unknown error"
            errors.append(f"{model}: {detail}")
            logger.warning(
                "[LLM Warmup] user=%s model=%s failed: %s",
                user_id,
                model,
                exc,
            )
            results.append({
                "model": model,
                "targets": labels,
                "content": None,
                "error": detail[:200],
            })

    if not any(item.get("content") for item in results):
        raise HTTPException(
            status_code=400,
            detail=f"模型 warmup 失败：{'; '.join(errors)[:300]}",
        )

    return results


def _run_config_warmup(config: Dict[str, Any], user_id: str) -> None:
    models = _warmup_model_names(config)
    if not models:
        _log(f"[LLM Warmup] user={user_id} skipped: no models configured")
        return
    try:
        _invoke_runtime_warmup(config, _CONFIG_WARMUP_PROMPT, user_id, timeout=_CONFIG_WARMUP_TIMEOUT_SECONDS)
    except HTTPException as exc:
        logger.warning("[LLM Warmup] user=%s failed: %s", user_id, exc.detail)


def _config_response_for_user(user: Optional[UserDB], db: Session) -> UserRuntimeConfigResponse:
    cfg = _build_runtime_config({}, user_id=user.id if user else None, db=db)
    user_cfg = auth_service.get_user_llm_config(db, user.id) if user else None
    webhook_url = auth_service.decrypt_secret(getattr(user_cfg, "wecom_webhook_encrypted", None))
    return UserRuntimeConfigResponse(
        llm_provider=cfg["llm_provider"],
        deep_think_llm=cfg["deep_think_llm"],
        quick_think_llm=cfg["quick_think_llm"],
        backend_url=cfg["backend_url"],
        max_debate_rounds=cfg["max_debate_rounds"],
        max_risk_discuss_rounds=cfg["max_risk_discuss_rounds"],
        has_api_key=bool(user_cfg and user_cfg.api_key_encrypted),
        has_wecom_webhook=bool(webhook_url),
        wecom_webhook_display=_mask_wecom_webhook(webhook_url),
        server_fallback_enabled=bool(cfg.get("server_fallback_enabled", True)),
        email_report_enabled=user.email_report_enabled if user and hasattr(user, 'email_report_enabled') else True,
        wecom_report_enabled=user.wecom_report_enabled if user and hasattr(user, "wecom_report_enabled") else True,
        default_analysts=json.loads(user_cfg.default_analysts) if user_cfg and user_cfg.default_analysts else ["market", "social", "news", "fundamentals", "macro", "smart_money", "volume_price"],
    )


@app.post("/v1/auth/request-code")
def request_login_code(request: AuthRequestCodeRequest):
    email = auth_service.normalize_email(request.email)
    if not re.match(r"^[^@\s]+@[^@\s.]+\.[^@\s.]+$", email):
        raise HTTPException(status_code=400, detail="邮箱格式不正确")
    with get_db_ctx() as db:
        code = auth_service.upsert_login_code(db, email)
    # DB session 已释放，SMTP 不会阻塞连接池
    dev_code = auth_service.send_login_code(email, code)
    response = {"message": "验证码已发送"}
    if dev_code:
        response["dev_code"] = dev_code
    return response


@app.post("/v1/auth/verify-code", response_model=AuthVerifyCodeResponse)
def verify_login_code(body: AuthVerifyCodeRequest, request: Request, db: Session = Depends(get_db)):
    user = auth_service.verify_login_code(db, body.email, body.code, client_ip=_get_real_ip(request))
    if not user:
        raise HTTPException(status_code=400, detail="验证码错误或已过期")
    access_token = auth_service.create_access_token(user)
    return AuthVerifyCodeResponse(access_token=access_token, user=user)


@app.get("/v1/auth/me", response_model=UserResponse)
def get_me(current_user: UserDB = Depends(_require_web_user)):
    return current_user




# --- Multi-Provider & Role-Based Model Routing Endpoints ---

@app.get("/v1/providers", response_model=List[ProviderResponse])
def get_user_providers(
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    return role_routing_service.list_providers(db, current_user.id)


@app.post("/v1/providers", response_model=ProviderResponse)
def create_user_provider(
    body: ProviderCreateRequest,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    return role_routing_service.create_provider(
        db,
        current_user.id,
        provider_type=body.provider_type,
        display_name=body.display_name or body.model_name,
        base_url=body.base_url,
        api_key=body.api_key,
        enabled=body.enabled,
    )


@app.patch("/v1/providers/{provider_id}", response_model=ProviderResponse)
def update_user_provider(
    provider_id: str,
    body: ProviderUpdateRequest,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    res = role_routing_service.update_provider(
        db,
        current_user.id,
        provider_id,
        display_name=body.display_name or body.model_name,
        base_url=body.base_url,
        api_key=body.api_key,
        enabled=body.enabled,
        clear_api_key=body.clear_api_key,
    )
    if not res:
        raise HTTPException(status_code=404, detail="Provider not found")
    return res


@app.delete("/v1/providers/{provider_id}")
def delete_user_provider(
    provider_id: str,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    success = role_routing_service.delete_provider(db, current_user.id, provider_id)
    if not success:
        raise HTTPException(status_code=404, detail="Provider not found")
    return {"status": "ok", "deleted_provider_id": provider_id}


@app.post("/v1/model-profiles/sync", response_model=List[ModelProfileResponse])
def sync_user_model_profiles(
    body: ModelProfilesSyncRequest,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    return role_routing_service.sync_model_profiles_from_names(
        db, current_user.id, body.models, body.provider_id
    )

@app.get("/v1/model-profiles", response_model=List[ModelProfileResponse])
def get_user_model_profiles(
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    return role_routing_service.list_model_profiles(db, current_user.id)


@app.post("/v1/model-profiles", response_model=ModelProfileResponse)
def create_user_model_profile(
    body: ModelProfileCreateRequest,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    return role_routing_service.create_model_profile(
        db,
        current_user.id,
        provider_id=body.provider_id,
        model_name=body.model_name,
        display_name=body.display_name or body.model_name,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        extra_params=body.extra_params,
        tier=body.tier,
        is_default=body.is_default,
    )


@app.patch("/v1/model-profiles/{profile_id}", response_model=ModelProfileResponse)
def update_user_model_profile(
    profile_id: str,
    body: ModelProfileUpdateRequest,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    res = role_routing_service.update_model_profile(
        db,
        current_user.id,
        profile_id,
        provider_id=body.provider_id,
        model_name=body.model_name,
        display_name=body.display_name or body.model_name,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        extra_params=body.extra_params,
        tier=body.tier,
        is_default=body.is_default,
    )
    if not res:
        raise HTTPException(status_code=404, detail="Model profile not found")
    return res


@app.delete("/v1/model-profiles/{profile_id}")
def delete_user_model_profile(
    profile_id: str,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    success = role_routing_service.delete_model_profile(db, current_user.id, profile_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model profile not found")
    return {"status": "ok", "deleted_profile_id": profile_id}


@app.get("/v1/role-bindings", response_model=List[RoleBindingResponse])
def get_user_role_bindings(
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    return role_routing_service.get_role_bindings(db, current_user.id)


@app.patch("/v1/role-bindings", response_model=List[RoleBindingResponse])
def update_user_role_bindings(
    body: RoleBindingsUpdateRequest,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    raw_items = [item.model_dump() for item in body.bindings]
    return role_routing_service.update_role_bindings(db, current_user.id, raw_items)


@app.post("/v1/role-bindings/presets")
def apply_user_role_preset(
    body: PresetApplyRequest,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    try:
        return role_routing_service.apply_role_preset(
            db,
            current_user.id,
            preset_mode=body.preset_mode,
            bull_profile_id=body.bull_profile_id,
            bear_profile_id=body.bear_profile_id,
            manager_profile_id=body.manager_profile_id,
            quick_profile_id=body.quick_profile_id,
            deep_profile_id=body.deep_profile_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/role-bindings/resolved")
def get_resolved_user_role_bindings(
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    runtime_cfg = _config_response_for_user(current_user, db).model_dump()
    return role_routing_service.resolve_all_roles(db, current_user.id, runtime_cfg)


# --- Custom Analysis Prompts Endpoints (Phase B: persistence only, no injection yet) ---

@app.get("/v1/custom-prompts", response_model=List[CustomPromptResponse])
def get_user_custom_prompts(
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    return custom_prompt_service.list_custom_prompts(db, current_user.id)


@app.patch("/v1/custom-prompts", response_model=List[CustomPromptResponse])
def update_user_custom_prompts(
    body: CustomPromptsUpdateRequest,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    raw_items = [item.model_dump() for item in body.prompts]
    try:
        return custom_prompt_service.replace_custom_prompts(db, current_user.id, raw_items)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/v1/custom-prompts/resolved", response_model=List[ResolvedCustomPromptResponse])
def get_resolved_user_custom_prompts(
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    return custom_prompt_service.resolve_all_roles_prompts(db, current_user.id)


@app.post("/v1/custom-prompts/migrate", response_model=List[CustomPromptResponse])
def migrate_user_custom_prompt(
    body: CustomPromptMigrateRequest,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    try:
        return custom_prompt_service.migrate_legacy_prompt(db, current_user.id, body.legacy_text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/v1/custom-prompts/switch", response_model=PromptInjectionSwitchResponse)
def get_user_prompt_injection_switch(
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    return {"enabled": custom_prompt_service.get_prompt_injection_enabled(db, current_user.id)}


@app.patch("/v1/custom-prompts/switch", response_model=PromptInjectionSwitchResponse)
def update_user_prompt_injection_switch(
    body: PromptInjectionSwitchUpdateRequest,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    enabled = custom_prompt_service.set_prompt_injection_enabled(db, current_user.id, body.enabled)
    return {"enabled": enabled}


@app.get("/v1/config", response_model=UserRuntimeConfigResponse)
def get_runtime_config(
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    """获取当前用户运行时配置。"""
    return _config_response_for_user(current_user, db)


@app.patch("/v1/config")
def update_runtime_config(
    updates: UserRuntimeConfigUpdateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    """更新当前用户运行时配置，下次分析时生效。"""
    normalized_wecom_webhook = None
    if updates.wecom_webhook_url:
        from api.services.wecom_notification_service import normalize_webhook_url

        try:
            normalized_wecom_webhook = normalize_webhook_url(updates.wecom_webhook_url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    persistent_user = db.query(UserDB).filter(UserDB.id == current_user.id).first() or current_user
    before_cfg = _config_response_for_user(persistent_user, db)
    pending_cfg = _build_pending_runtime_config(updates, persistent_user.id, db)
    if _should_probe_runtime_config(before_cfg, pending_cfg, updates):
        probe = _probe_runtime_config(pending_cfg)
        _log(
            f"[LLM Probe] user={persistent_user.id} provider={pending_cfg.get('llm_provider')} "
            f"model={probe.get('model', '')} status={probe.get('status')}"
        )
    row = auth_service.upsert_user_llm_config(
        db,
        persistent_user.id,
        llm_provider=updates.llm_provider,
        deep_think_llm=updates.deep_think_llm,
        quick_think_llm=updates.quick_think_llm,
        backend_url=updates.backend_url,
        max_debate_rounds=updates.max_debate_rounds,
        max_risk_discuss_rounds=updates.max_risk_discuss_rounds,
        api_key=updates.api_key,
        wecom_webhook_url=normalized_wecom_webhook,
        clear_api_key=updates.clear_api_key,
        clear_wecom_webhook=updates.clear_wecom_webhook,
        default_analysts=updates.default_analysts,
    )
    user_pref_updated = False
    if updates.email_report_enabled is not None:
        persistent_user.email_report_enabled = updates.email_report_enabled
        user_pref_updated = True
    if updates.wecom_report_enabled is not None:
        persistent_user.wecom_report_enabled = updates.wecom_report_enabled
        user_pref_updated = True
    if user_pref_updated:
        db.commit()
    current_cfg = _config_response_for_user(persistent_user, db)
    warmup_models = _warmup_model_names(current_cfg.model_dump())
    should_warmup = _should_trigger_config_warmup(before_cfg, current_cfg, updates)
    warmup_payload: Dict[str, Any]
    if should_warmup and warmup_models:
        warmup_payload = {
            "requested": True,
            "triggered": True,
            "status": "scheduled",
            "models": warmup_models,
            "message": f"模型配置已保存，后台正在预热 {len(warmup_models)} 个模型。",
        }
        background_tasks.add_task(
            _run_config_warmup,
            _build_runtime_config({}, user_id=persistent_user.id, db=db),
            persistent_user.id,
        )
    elif updates.warmup:
        warmup_payload = {
            "requested": True,
            "triggered": False,
            "status": "skipped",
            "models": warmup_models,
            "message": "模型配置已保存，本次未触发 warmup。",
        }
    else:
        warmup_payload = {
            "requested": False,
            "triggered": False,
            "status": "disabled",
            "models": [],
            "message": "模型配置已保存。",
        }
    filtered = {
        k: v
        for k, v in updates.model_dump().items()
        if v is not None
        and k not in {"api_key", "wecom_webhook_url", "warmup", "force_warmup"}
        and (
            k in _CONFIG_ALLOWED_KEYS
            or k in _CONFIG_PREFERENCE_KEYS
            or (k in {"clear_api_key", "clear_wecom_webhook"} and bool(v))
        )
    }
    return {
        "message": "用户配置已更新",
        "applied": filtered,
        "has_api_key": bool(row.api_key_encrypted),
        "current": current_cfg,
        "warmup": warmup_payload,
    }



_MODELS_FETCH_ALLOWLIST_ENV = "TA_MODELS_FETCH_ALLOWLIST"
_MODELS_FETCH_TIMEOUT_SECONDS = 8.0
_MODELS_FETCH_DEFAULT_URL = "http://host.docker.internal:8317/v1"
_MODELS_FETCH_GENERIC_ERROR = "无法获取模型列表"
_MODELS_FETCH_METADATA_IPS = frozenset({"169.254.169.254", "fd00:ec2::254"})
# 受信本地默认主机：Docker host-gateway + 裸机回环。仅当这些主机显式写入
# TA_MODELS_FETCH_ALLOWLIST（且必须钉死端口）时才放行解析出的私网/回环 IP，
# 其余主机一律保持 fail-closed。
_MODELS_FETCH_TRUSTED_LOCAL_HOSTS = frozenset({
    "host.docker.internal",
    "localhost",
    "127.0.0.1",
    "::1",
})
# auth_service.get_or_create_default_user 创建的默认本地账号 id。
_DEFAULT_LOCAL_USER_ID = "local-default-user"


class _ModelsFetchError(Exception):
    """Internal marker; API responses must expose only the generic message."""


def _split_models_allowlist_entry(raw: str) -> tuple[str, Optional[int]]:
    entry = raw.strip()
    if not entry or any(ch in entry for ch in "/?#@"):
        raise _ModelsFetchError("invalid allowlist entry")
    if entry.startswith("["):
        end = entry.find("]")
        if end == -1:
            raise _ModelsFetchError("invalid IPv6 allowlist entry")
        host = entry[1:end].strip().casefold().rstrip(".")
        suffix = entry[end + 1:].strip()
        if not suffix:
            return host, None
        if not suffix.startswith(":"):
            raise _ModelsFetchError("invalid allowlist port")
        port_text = suffix[1:].strip()
    else:
        colon_count = entry.count(":")
        if colon_count == 0:
            host = entry.casefold().rstrip(".")
            port_text = None
        elif colon_count == 1:
            host_text, port_text = entry.rsplit(":", 1)
            host = host_text.strip().casefold().rstrip(".")
        else:
            host = entry.casefold().rstrip(".")
            port_text = None
    if not host:
        raise _ModelsFetchError("empty allowlist host")
    if port_text is not None:
        if not port_text.isdigit():
            raise _ModelsFetchError("invalid allowlist port")
        port = int(port_text)
        if not 1 <= port <= 65535:
            raise _ModelsFetchError("allowlist port out of range")
        return host, port
    return host, None


def _parse_models_fetch_allowlist(raw: Optional[str] = None) -> Optional[Dict[str, set[Optional[int]]]]:
    raw_value = (raw if raw is not None else os.getenv(_MODELS_FETCH_ALLOWLIST_ENV, "")).strip()
    if not raw_value:
        return None
    allow_by_host: Dict[str, set[Optional[int]]] = {}
    for raw_entry in re.split(r"[,;]", raw_value):
        entry = raw_entry.strip()
        if not entry:
            continue
        host, port = _split_models_allowlist_entry(entry)
        allow_by_host.setdefault(host, set()).add(port)
    return allow_by_host or None


def _parse_models_fetch_url(base_url: str) -> urllib.parse.SplitResult:
    url = (base_url or "").strip()
    if not url:
        raise _ModelsFetchError("empty base_url")
    if any(ch in url for ch in ("\x00", "\r", "\n")):
        raise _ModelsFetchError("invalid control characters")
    try:
        parsed = urllib.parse.urlsplit(url)
        explicit_port = parsed.port
    except ValueError as exc:
        raise _ModelsFetchError("invalid base_url") from exc
    if parsed.scheme not in ("http", "https"):
        raise _ModelsFetchError("unsupported scheme")
    host = parsed.hostname
    if not host or any(ch.isspace() for ch in host):
        raise _ModelsFetchError("missing or invalid host")
    if parsed.username is not None or parsed.password is not None:
        raise _ModelsFetchError("credentials not allowed")
    if parsed.query or parsed.fragment:
        raise _ModelsFetchError("query and fragment not allowed")
    if explicit_port is not None and not 1 <= explicit_port <= 65535:
        raise _ModelsFetchError("port out of range")
    return parsed


def _models_fetch_port(parsed: urllib.parse.SplitResult) -> int:
    return parsed.port or (443 if parsed.scheme == "https" else 80)


def _models_fetch_host_allowed(
    host: str,
    port: int,
    allowlist: Dict[str, set[Optional[int]]],
    *,
    require_explicit_port: bool = False,
) -> bool:
    allowed_ports = allowlist.get(host.casefold().rstrip("."))
    if allowed_ports is None:
        return False
    if require_explicit_port:
        # 受信本地主机必须钉死端口；裸 host 形式（任意端口）不满足。
        return port in allowed_ports
    return None in allowed_ports or port in allowed_ports


def _is_models_fetch_ip_blocked(ip_text: str) -> bool:
    if ip_text in _MODELS_FETCH_METADATA_IPS:
        return True
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return True
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return _is_models_fetch_ip_blocked(str(ip.ipv4_mapped))
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_unspecified
        or ip.is_multicast
        or not ip.is_global
    )


def _is_models_fetch_local_ip(ip_text: str) -> bool:
    """回环或私网地址——受信本地主机解析目标的放行范围。"""
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return _is_models_fetch_local_ip(str(ip.ipv4_mapped))
    return ip.is_loopback or ip.is_private


def _resolve_models_fetch_target(host: str, port: int, *, trust_local: bool = False) -> str:
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise _ModelsFetchError("DNS resolution failed") from exc
    safe_ip: Optional[str] = None
    for address in addresses:
        ip_text = address[4][0]
        # 云元数据地址无条件拦截，受信本地放行也不覆盖。
        if ip_text in _MODELS_FETCH_METADATA_IPS:
            raise _ModelsFetchError("resolved address is blocked")
        if _is_models_fetch_ip_blocked(ip_text):
            # 仅受信本地主机（host.docker.internal / 回环）在已通过白名单时，
            # 放行其解析出的回环/私网地址；其余主机一律 fail-closed。
            if not (trust_local and _is_models_fetch_local_ip(ip_text)):
                raise _ModelsFetchError("resolved address is blocked")
        if safe_ip is None:
            safe_ip = ip_text
    if safe_ip is None:
        raise _ModelsFetchError("no addresses resolved")
    return safe_ip


class _SafeHTTPConnection(http.client.HTTPConnection):
    def __init__(
        self,
        host: str,
        port: Optional[int] = None,
        timeout: Optional[float] = None,
        source_address=None,
        blocksize: int = 8192,
        safe_ip: Optional[str] = None,
    ) -> None:
        super().__init__(host, port, timeout=timeout, source_address=source_address, blocksize=blocksize)
        self._safe_ip = safe_ip or self.host
        self._create_connection = self._safe_create_connection

    def _safe_create_connection(self, address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
        return socket.create_connection((self._safe_ip, self.port), timeout, source_address)


class _SafeHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        host: str,
        port: Optional[int] = None,
        *,
        timeout: Optional[float] = None,
        source_address=None,
        context=None,
        blocksize: int = 8192,
        safe_ip: Optional[str] = None,
    ) -> None:
        super().__init__(
            host,
            port,
            timeout=timeout,
            source_address=source_address,
            context=context,
            blocksize=blocksize,
        )
        self._safe_ip = safe_ip or self.host
        self._create_connection = self._safe_create_connection

    def _safe_create_connection(self, address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
        return socket.create_connection((self._safe_ip, self.port), timeout, source_address)


def _build_models_fetch_connection(
    scheme: str,
    host: str,
    port: int,
    safe_ip: str,
    timeout: float,
) -> http.client.HTTPConnection:
    if scheme == "https":
        return _SafeHTTPSConnection(host, port, timeout=timeout, safe_ip=safe_ip)
    return _SafeHTTPConnection(host, port, timeout=timeout, safe_ip=safe_ip)


def _models_fetch_path(parsed: urllib.parse.SplitResult) -> str:
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/models"):
        return path
    if path.endswith("/v1"):
        return f"{path}/models"
    return f"{path}/v1/models" if path else "/v1/models"


def _models_fetch_url(parsed: urllib.parse.SplitResult, path: str) -> str:
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _extract_models_from_payload(body: str) -> list[str]:
    data = json.loads(body)
    models: list[str] = []
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        items = data["data"]
    elif isinstance(data, list):
        items = data
    else:
        return models
    for item in items:
        if isinstance(item, dict) and "id" in item:
            models.append(str(item["id"]))
        elif isinstance(item, str):
            models.append(item)
    return models


def _fetch_available_models(base_url: str, api_key: str) -> tuple[list[str], str]:
    allowlist = _parse_models_fetch_allowlist()
    if allowlist is None:
        raise _ModelsFetchError("models fetch allowlist is not configured")
    parsed = _parse_models_fetch_url(base_url)
    host = (parsed.hostname or "").casefold().rstrip(".")
    port = _models_fetch_port(parsed)
    trust_local = host in _MODELS_FETCH_TRUSTED_LOCAL_HOSTS
    # 受信本地主机必须钉死端口（require_explicit_port），防止裸 host 变任意端口可探。
    if not _models_fetch_host_allowed(host, port, allowlist, require_explicit_port=trust_local):
        raise _ModelsFetchError("host is not allowlisted")
    safe_ip = _resolve_models_fetch_target(host, port, trust_local=trust_local)
    path = _models_fetch_path(parsed)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    conn: Optional[http.client.HTTPConnection] = None
    try:
        conn = _build_models_fetch_connection(
            parsed.scheme,
            host,
            port,
            safe_ip,
            _MODELS_FETCH_TIMEOUT_SECONDS,
        )
        conn.request("GET", path, headers=headers)
        response = conn.getresponse()
        body = response.read().decode("utf-8")
        if response.status < 200 or response.status >= 300:
            raise _ModelsFetchError(f"HTTP status {response.status}")
        models = _extract_models_from_payload(body)
        return sorted(set(models)), _models_fetch_url(parsed, path)
    except (_ModelsFetchError, OSError, http.client.HTTPException, UnicodeDecodeError, ValueError) as exc:
        raise _ModelsFetchError("model fetch failed") from exc
    finally:
        if conn is not None:
            conn.close()


def _is_loopback_client(client_host: Optional[str]) -> bool:
    """直接 TCP 对端是否为回环来源（用于匿名回退的收窄）。

    必须使用 request.client 的真实对端，而不是 X-Forwarded-For /
    CF-Connecting-IP——那些头可被客户端伪造。
    """
    if not client_host:
        return False
    try:
        ip = ipaddress.ip_address(client_host)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return _is_loopback_client(str(ip.ipv4_mapped))
    return ip.is_loopback


class FetchModelsRequest(BaseModel):
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    provider_id: Optional[str] = None


@app.post("/v1/models/fetch")
def fetch_available_models(
    payload: FetchModelsRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_api_user),
):
    """从白名单允许的 Base URL 抓取可用模型列表。"""
    # 匿名回退收窄：默认本地账号(local@tradingagents.local)仅在回环来源下可用。
    # 非回环来源（局域网/暴露端口）必须提供登录或 API Token，否则拒绝匿名调用。
    # 判定用 request.client 真实对端，不信任可伪造的代理头。
    if current_user.id == _DEFAULT_LOCAL_USER_ID and not _is_loopback_client(
        request.client.host if request.client else None
    ):
        raise HTTPException(status_code=401, detail="authentication required")
    base_url = (payload.base_url or "").strip()
    api_key = (payload.api_key or "").strip()

    if payload.provider_id:
        provider = db.query(ProviderDB).filter(
            ProviderDB.id == payload.provider_id,
            ProviderDB.user_id == current_user.id
        ).first()
        if provider:
            base_url = base_url or provider.base_url
            if not api_key and provider.api_key_ciphertext:
                api_key = auth_service.decrypt_secret(provider.api_key_ciphertext) or ""

    if not base_url:
        user_cfg = auth_service.get_user_llm_config(db, current_user.id)
        if user_cfg:
            base_url = user_cfg.custom_base_url or ""
            if not api_key and user_cfg.llm_api_key_ciphertext:
                api_key = auth_service.decrypt_secret(user_cfg.llm_api_key_ciphertext) or ""

    if not base_url:
        base_url = _MODELS_FETCH_DEFAULT_URL
    try:
        models_sorted, target_url = _fetch_available_models(base_url, api_key)
    except Exception as exc:
        logger.warning("[fetch_available_models] rejected: %s", exc)
        return {
            "ok": False,
            "error": _MODELS_FETCH_GENERIC_ERROR,
            "models": [],
            "count": 0,
        }
    if models_sorted and current_user:
        try:
            role_routing_service.sync_model_profiles_from_names(db, current_user.id, models_sorted)
        except Exception as sync_err:
            logger.warning("[fetch_available_models] model profile sync failed: %s", sync_err)
    return {
        "ok": True,
        "models": models_sorted,
        "count": len(models_sorted),
        "url": target_url,
    }


@app.post("/v1/config/warmup", response_model=UserRuntimeWarmupResponse)
def warmup_runtime_config(
    request: UserRuntimeWarmupRequest,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    pending_cfg = _build_pending_runtime_config(request, current_user.id, db)
    prompt = (request.prompt or "").strip() or "你好"
    results = _invoke_runtime_warmup(pending_cfg, prompt, current_user.id)
    return {
        "prompt": prompt,
        "results": results,
    }


@app.post("/v1/config/wecom/warmup", response_model=WecomWebhookWarmupResponse)
async def warmup_wecom_webhook(
    request: WecomWebhookWarmupRequest,
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(_require_web_user),
):
    from api.services.wecom_notification_service import build_test_message, normalize_webhook_url, send_message

    webhook_url = (request.wecom_webhook_url or "").strip()
    if not webhook_url:
        user_cfg = auth_service.get_user_llm_config(db, current_user.id)
        webhook_url = auth_service.decrypt_secret(getattr(user_cfg, "wecom_webhook_encrypted", None)) or ""
    if not webhook_url:
        raise HTTPException(status_code=400, detail="请先填写或保存企业微信 Webhook")
    try:
        webhook_url = normalize_webhook_url(webhook_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        sent = await asyncio.to_thread(send_message, build_test_message(request.content), webhook_url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Webhook 测试发送失败：{exc}") from exc
    if not sent:
        raise HTTPException(status_code=400, detail="Webhook 测试发送失败，请检查地址或机器人状态")

    return {
        "sent": True,
        "message": "Webhook 测试发送成功",
        "webhook_display": _mask_wecom_webhook(webhook_url),
    }


# ── Stock Search ──────────────────────────────────────────────────────────────

@app.get("/v1/market/stock-search")
def search_stocks(
    q: str = Query("", min_length=1, max_length=20),
    current_user: UserDB = Depends(_require_api_user),
):
    """Search stocks by code prefix or name substring."""
    q = q.strip()
    if not q:
        return {"results": []}

    name_to_code = _load_cn_stock_map()
    code_to_name = _get_reverse_stock_map()
    results = []
    q_upper = q.upper()

    for code, name in code_to_name.items():
        if code.upper().startswith(q_upper) or code.split(".")[0].startswith(q):
            results.append({"symbol": code, "name": name})
            if len(results) >= 20:
                break

    if len(results) < 20:
        for name, code in name_to_code.items():
            if q in name and not any(r["symbol"] == code for r in results):
                results.append({"symbol": code, "name": name})
                if len(results) >= 20:
                    break

    return {"results": results}


def _annotate_scheduled_with_imported_context(items: List[dict], db: Session, user_id: str) -> List[dict]:
    imported_map: Dict[str, Dict[str, Any]] = {}
    for item in portfolio_import_service.list_imported_positions(db, user_id):
        imported_map[item["symbol"]] = item
    for item in items:
        imported = imported_map.get(item["symbol"])
        item["has_imported_context"] = imported is not None
        item["imported_current_position"] = imported.get("current_position") if imported else None
        item["imported_average_cost"] = imported.get("average_cost") if imported else None
        item["imported_trade_points_count"] = imported.get("trade_points_count") if imported else 0
    return items


def _merge_imported_user_context(*contexts: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    note_parts: List[str] = []
    for ctx in contexts:
        if not ctx:
            continue
        for key, value in ctx.items():
            if key == "user_notes":
                if value:
                    note_parts.append(str(value).strip())
                continue
            if value is not None:
                merged[key] = value
    if note_parts:
        merged["user_notes"] = "\n\n".join(part for part in note_parts if part)
    return normalize_user_context(merged)


def _build_imported_user_context(db: Session, user_id: str, symbol: str) -> Dict[str, Any]:
    context = portfolio_import_service.build_scheduled_user_context(db, user_id, symbol)
    return _merge_imported_user_context(context)


def _build_manual_imported_user_context(db: Session, user_id: str, symbol: str) -> Dict[str, Any]:
    """Build imported position context for manual/ad-hoc analysis runs."""
    return _build_imported_user_context(db, user_id, symbol)


def _attach_stock_names(items: List[dict], code_to_name: Dict[str, str]) -> List[dict]:
    for item in items:
        symbol = str(item.get("symbol") or "").upper()
        item["name"] = code_to_name.get(symbol, symbol or item.get("name") or "")
    return items


@app.get("/v1/portfolio/imports")
def get_portfolio_import_state(
    current_user: UserDB = Depends(_require_api_user),
    db: Session = Depends(get_db),
):
    return portfolio_import_service.get_import_state(db, current_user.id)


@app.post("/v1/portfolio/imports")
def sync_portfolio_import(
    body: PortfolioImportSyncRequest,
    current_user: UserDB = Depends(_require_api_user),
    db: Session = Depends(get_db),
):
    try:
        return portfolio_import_service.sync_positions(
            db=db,
            user_id=current_user.id,
            positions=[p.model_dump() for p in body.positions],
            source=body.source,
            auto_apply_scheduled=body.auto_apply_scheduled,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/v1/portfolio/imports", status_code=204)
def clear_portfolio_import_state(
    current_user: UserDB = Depends(_require_api_user),
    db: Session = Depends(get_db),
):
    portfolio_import_service.clear_imported_portfolio(db, current_user.id)


@app.post("/v1/portfolio/parse-image")
async def parse_position_image_endpoint(
    file: UploadFile = File(...),
    current_user: UserDB = Depends(_require_api_user),
):
    """Parse a broker position screenshot using server-side VLM."""
    from api.services.vlm_position_parser import parse_position_image

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "只支持图片文件")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(400, "图片不能超过 10MB")

    try:
        positions = await asyncio.to_thread(parse_position_image, image_bytes, file.content_type)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.warning("[parse-image] VLM parsing failed: %s", exc)
        raise HTTPException(500, "图片解析失败，请稍后重试") from exc

    return {"positions": positions}


@app.get("/v1/dashboard/tracking-board")
def get_dashboard_tracking_board(
    current_user: UserDB = Depends(_require_api_user),
    db: Session = Depends(get_db),
):
    return tracking_board_service.get_tracking_board(db, current_user.id)


# ── Watchlist ─────────────────────────────────────────────────────────────────

@app.get("/v1/watchlist")
def list_watchlist(
    current_user: UserDB = Depends(_require_api_user),
    db: Session = Depends(get_db),
):
    items = watchlist_service.list_watchlist(db, current_user.id)
    _attach_stock_names(items, _get_reverse_stock_map())
    return {"items": items}


@app.post("/v1/watchlist")
def add_to_watchlist(
    body: WatchlistAddRequest,
    current_user: UserDB = Depends(_require_api_user),
    db: Session = Depends(get_db),
):
    text = str(body.text or body.symbol or "").strip()
    if not text:
        raise HTTPException(400, "text or symbol is required")

    tokens = _split_watchlist_batch_text(text)
    if not tokens:
        raise HTTPException(400, "至少提供一个股票代码或名称")

    name_to_code = _load_cn_stock_map()
    code_to_name = _get_reverse_stock_map()

    resolved_entries: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    for idx, token in enumerate(tokens):
        symbol, name, error = _resolve_watchlist_identifier(token, name_to_code, code_to_name)
        if error:
            results.append({
                "_order": idx,
                "input": token,
                "status": "invalid",
                "message": error,
            })
            continue
        resolved_entries.append({
            "_order": idx,
            "input": token,
            "symbol": symbol,
            "name": name,
        })

    add_results = watchlist_service.add_watchlist_items(
        db,
        current_user.id,
        [entry["symbol"] for entry in resolved_entries],
    )
    for entry, result in zip(resolved_entries, add_results):
        item = result.get("item")
        if item:
            item["name"] = entry["name"]
            item["has_scheduled"] = False
        results.append({
            "_order": entry["_order"],
            "input": entry["input"],
            "symbol": entry["symbol"],
            "name": entry["name"],
            "status": result["status"],
            "message": result["message"],
            "item": item,
        })

    results.sort(key=lambda row: row["_order"])
    for row in results:
        row.pop("_order", None)
    summary = {
        "total": len(tokens),
        "added": sum(1 for row in results if row["status"] == "added"),
        "duplicate": sum(1 for row in results if row["status"] == "duplicate"),
        "failed": sum(1 for row in results if row["status"] in {"invalid", "failed"}),
    }
    message_parts = [f"共处理 {summary['total']} 项"]
    if summary["added"]:
        message_parts.append(f"新增 {summary['added']} 项")
    if summary["duplicate"]:
        message_parts.append(f"重复 {summary['duplicate']} 项")
    if summary["failed"]:
        message_parts.append(f"失败 {summary['failed']} 项")
    return {
        "message": "，".join(message_parts),
        "summary": summary,
        "results": results,
    }


@app.delete("/v1/watchlist/{item_id}", status_code=204)
def delete_from_watchlist(
    item_id: str,
    current_user: UserDB = Depends(_require_api_user),
    db: Session = Depends(get_db),
):
    if not watchlist_service.delete_watchlist_item(db, current_user.id, item_id):
        raise HTTPException(404, "未找到该自选股")


# ── Scheduled Analysis ────────────────────────────────────────────────────────

@app.get("/v1/scheduled")
def list_scheduled_analyses(
    current_user: UserDB = Depends(_require_api_user),
    db: Session = Depends(get_db),
):
    items = scheduled_service.list_scheduled(db, current_user.id)
    _attach_stock_names(items, _get_reverse_stock_map_cached_only())
    return {"items": _annotate_scheduled_with_imported_context(items, db, current_user.id)}


@app.get("/v1/portfolio/overview", response_model=PortfolioOverviewResponse)
def get_portfolio_overview(
    current_user: UserDB = Depends(_require_api_user),
    db: Session = Depends(get_db),
):
    code_to_name = _get_reverse_stock_map()

    watchlist_items = watchlist_service.list_watchlist(db, current_user.id)
    _attach_stock_names(watchlist_items, code_to_name)

    scheduled_items = scheduled_service.list_scheduled(db, current_user.id)
    _attach_stock_names(scheduled_items, code_to_name)
    scheduled_items = _annotate_scheduled_with_imported_context(scheduled_items, db, current_user.id)

    latest_reports = report_service.get_latest_reports_by_symbols(
        db=db,
        user_id=current_user.id,
        symbols=[item["symbol"] for item in watchlist_items],
    )
    for report in latest_reports:
        report.name = code_to_name.get(report.symbol, report.symbol)

    portfolio_import = portfolio_import_service.get_import_state(db, current_user.id)

    return {
        "watchlist": watchlist_items,
        "scheduled": scheduled_items,
        "latest_reports": latest_reports,
        "portfolio_import": portfolio_import,
    }


@app.post("/v1/scheduled", status_code=201)
def create_scheduled_analysis(
    body: dict,
    current_user: UserDB = Depends(_require_api_user),
    db: Session = Depends(get_db),
):
    symbol = body.get("symbol", "").strip().upper()
    horizon = body.get("horizon", "short")
    trigger_time = body.get("trigger_time", "20:00")
    if not symbol:
        raise HTTPException(400, "symbol is required")
    code_to_name = _get_reverse_stock_map()
    if symbol not in code_to_name:
        raise HTTPException(400, f"未知的股票代码: {symbol}")
    try:
        item = scheduled_service.create_scheduled(db, current_user.id, symbol, horizon, trigger_time)
        item["name"] = code_to_name.get(symbol, symbol)
        _annotate_scheduled_with_imported_context([item], db, current_user.id)
        return item
    except ValueError as e:
        raise HTTPException(400, str(e))


def _extract_scheduled_update_kwargs(body: dict) -> dict:
    kwargs = {}
    if "is_active" in body:
        kwargs["is_active"] = bool(body["is_active"])
    if "horizon" in body:
        kwargs["horizon"] = body["horizon"]
    if "trigger_time" in body:
        kwargs["trigger_time"] = body["trigger_time"]
    return kwargs


@app.patch("/v1/scheduled/batch")
def batch_update_scheduled_analyses(
    body: ScheduledBatchUpdateRequest,
    current_user: UserDB = Depends(_require_api_user),
    db: Session = Depends(get_db),
):
    kwargs = _extract_scheduled_update_kwargs(body.model_dump(exclude_unset=True))
    if not kwargs:
        raise HTTPException(400, "至少提供一个更新字段")
    try:
        items = scheduled_service.batch_update_scheduled(
            db,
            current_user.id,
            body.item_ids,
            **kwargs,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    code_to_name = _get_reverse_stock_map()
    for item in items:
        item["name"] = code_to_name.get(item["symbol"], item["symbol"])
    return {"items": _annotate_scheduled_with_imported_context(items, db, current_user.id)}


@app.post("/v1/scheduled/batch/delete")
def batch_delete_scheduled_analyses(
    body: ScheduledBatchIdsRequest,
    current_user: UserDB = Depends(_require_api_user),
    db: Session = Depends(get_db),
):
    try:
        return scheduled_service.batch_delete_scheduled(db, current_user.id, body.item_ids)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/v1/scheduled/batch/trigger", response_model=BatchScheduledTriggerResponse)
async def trigger_scheduled_analyses_batch(
    body: ScheduledBatchIdsRequest,
    current_user: UserDB = Depends(_require_api_user),
    db: Session = Depends(get_db),
):
    if not body.item_ids:
        raise HTTPException(400, "请至少选择 1 个定时任务")

    requested_trade_date = cn_today_str()
    try:
        actual_trade_date = _resolve_scheduled_trade_date(requested_trade_date)
    except TradeCalendarUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"交易日历不可用：{exc}") from exc
    code_to_name = _get_reverse_stock_map()
    jobs: List[Dict[str, Any]] = []
    with_position_context = 0
    available_tasks = {
        task["id"]: task
        for task in scheduled_service.list_scheduled(db, current_user.id)
    }
    valid_item_ids = []
    missing_item_ids = []
    for raw_item_id in body.item_ids:
        item_id = str(raw_item_id or "").strip()
        if not item_id:
            continue
        if item_id in available_tasks:
            valid_item_ids.append(item_id)
        else:
            missing_item_ids.append(item_id)

    if not valid_item_ids:
        raise HTTPException(400, "选中的定时任务已失效，请刷新页面后重试")

    if missing_item_ids:
        _log(
            f"[Scheduled Batch Trigger] user={current_user.id} skipped missing item_ids={missing_item_ids}"
        )

    for item_id in valid_item_ids:
        task = available_tasks[item_id]

        task_snapshot = dict(task)
        task_snapshot["user_id"] = current_user.id
        task_snapshot["manual_user_context"] = _build_manual_imported_user_context(db, current_user.id, task["symbol"])

        scheduled_user_context = task_snapshot["manual_user_context"]
        if scheduled_user_context.get("current_position") is not None:
            with_position_context += 1

        now = _utcnow_iso()
        job_id = uuid4().hex
        _set_job(
            job_id,
            job_id=job_id,
            status="pending",
            created_at=now,
            symbol=task["symbol"],
            trade_date=actual_trade_date,
            user_id=current_user.id,
            request_source="scheduled_manual_batch",
        )
        _emit_job_event(
            job_id,
            "job.queued",
            {"job_id": job_id, "symbol": task["symbol"], "trade_date": actual_trade_date},
        )
        _create_tracked_task(
            _run_manual_trigger(
                task_snapshot,
                requested_trade_date,
                job_id,
            )
        )

        jobs.append({
            "item_id": task["id"],
            "job_id": job_id,
            "symbol": task["symbol"],
            "name": code_to_name.get(task["symbol"], task["symbol"]),
            "status": "pending",
            "created_at": now,
            "current_position": scheduled_user_context.get("current_position"),
            "average_cost": scheduled_user_context.get("average_cost"),
        })

    return {
        "summary": {
            "total": len(jobs),
            "with_position_context": with_position_context,
        },
        "jobs": jobs,
    }


@app.post("/v1/scheduled/{item_id}/trigger", response_model=AnalyzeResponse)
async def trigger_scheduled_analysis_once(
    item_id: str,
    current_user: UserDB = Depends(_require_api_user),
    db: Session = Depends(get_db),
):
    task = scheduled_service.get_scheduled(db, current_user.id, item_id)
    if task is None:
        raise HTTPException(404, "未找到该定时任务")

    requested_trade_date = cn_today_str()
    try:
        actual_trade_date = _resolve_scheduled_trade_date(requested_trade_date)
    except TradeCalendarUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"交易日历不可用：{exc}") from exc
    now = _utcnow_iso()
    job_id = uuid4().hex

    task_snapshot = dict(task)
    task_snapshot["user_id"] = current_user.id
    task_snapshot["manual_user_context"] = _build_manual_imported_user_context(db, current_user.id, task["symbol"])

    _set_job(
        job_id,
        job_id=job_id,
        status="pending",
        created_at=now,
        symbol=task["symbol"],
        trade_date=actual_trade_date,
        user_id=current_user.id,
        request_source="scheduled_manual",
    )
    _emit_job_event(
        job_id,
        "job.queued",
        {"job_id": job_id, "symbol": task["symbol"], "trade_date": actual_trade_date},
    )
    _create_tracked_task(
        _run_manual_trigger(
            task_snapshot,
            requested_trade_date,
            job_id,
        )
    )
    return AnalyzeResponse(job_id=job_id, status="pending", created_at=now)


@app.patch("/v1/scheduled/{item_id}")
def update_scheduled_analysis(
    item_id: str,
    body: dict,
    current_user: UserDB = Depends(_require_api_user),
    db: Session = Depends(get_db),
):
    kwargs = _extract_scheduled_update_kwargs(body)
    try:
        result = scheduled_service.update_scheduled(db, current_user.id, item_id, **kwargs)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if result is None:
        raise HTTPException(404, "未找到该定时任务")
    code_to_name = _get_reverse_stock_map()
    result["name"] = code_to_name.get(result["symbol"], result["symbol"])
    _annotate_scheduled_with_imported_context([result], db, current_user.id)
    return result


@app.delete("/v1/scheduled/{item_id}", status_code=204)
def delete_scheduled_analysis(
    item_id: str,
    current_user: UserDB = Depends(_require_api_user),
    db: Session = Depends(get_db),
):
    if not scheduled_service.delete_scheduled(db, current_user.id, item_id):
        raise HTTPException(404, "未找到该定时任务")


# ─── Sponsor endpoints (public, no auth) ────────────────────────────────────


class SponsorItem(BaseModel):
    id: str
    sponsor_type: str
    name: str
    github: Optional[str] = None
    avatar: Optional[str] = None
    email: Optional[str] = None
    provider: Optional[str] = None
    date: str
    # NOTE: amount is intentionally excluded from the public API


class SponsorsResponse(BaseModel):
    money: List[SponsorItem]
    token: List[SponsorItem]


def _sponsor_to_item(s: SponsorDB) -> SponsorItem:
    return SponsorItem(
        id=s.id,
        sponsor_type=s.sponsor_type,
        name=s.name,
        github=s.github,
        avatar=s.avatar,
        email=s.email,
        provider=s.provider,
        date=s.date,
    )


@app.get("/v1/sponsors", response_model=SponsorsResponse)
def list_sponsors(db: Session = Depends(get_db)):
    """Public endpoint: list all visible sponsors grouped by type."""
    all_sponsors = sponsor_service.list_sponsors(db)
    money = [_sponsor_to_item(s) for s in all_sponsors if s.sponsor_type == "money"]
    token = [_sponsor_to_item(s) for s in all_sponsors if s.sponsor_type == "token"]
    return SponsorsResponse(money=money, token=token)


# ─── Feedback endpoints ─────────────────────────────────────────────────────


class FeedbackCreateRequest(BaseModel):
    subject: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=5000)


class FeedbackItem(BaseModel):
    id: str
    user_email: str
    subject: str
    content: str
    admin_reply: Optional[str] = None
    replied_at: Optional[datetime] = None
    is_read: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_serializer("replied_at", "created_at", "updated_at")
    def serialize_dt(self, v: Optional[datetime], _info: Any) -> Optional[str]:
        return v.isoformat() if v else None


class FeedbackListResponse(BaseModel):
    total: int
    feedbacks: List[FeedbackItem]


class FeedbackUnreadResponse(BaseModel):
    unread_count: int


def _fb_to_item(fb: FeedbackDB) -> FeedbackItem:
    return FeedbackItem(
        id=fb.id,
        user_email=fb.user_email,
        subject=fb.subject,
        content=fb.content,
        admin_reply=fb.admin_reply,
        replied_at=fb.replied_at,
        is_read=fb.is_read,
        created_at=fb.created_at,
        updated_at=fb.updated_at,
    )


@app.post("/v1/feedbacks", response_model=FeedbackItem, status_code=201)
def create_feedback(
    req: FeedbackCreateRequest,
    current_user: UserDB = Depends(_require_web_user),
    db: Session = Depends(get_db),
):
    fb = feedback_service.create_feedback(db, current_user, req.subject, req.content)
    return _fb_to_item(fb)


@app.get("/v1/feedbacks", response_model=FeedbackListResponse)
def list_feedbacks(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: UserDB = Depends(_require_web_user),
    db: Session = Depends(get_db),
):
    items, total = feedback_service.list_feedbacks(db, current_user.id, page, page_size)
    return FeedbackListResponse(total=total, feedbacks=[_fb_to_item(fb) for fb in items])


@app.get("/v1/feedbacks/unread-count", response_model=FeedbackUnreadResponse)
def feedback_unread_count(
    current_user: UserDB = Depends(_require_web_user),
    db: Session = Depends(get_db),
):
    count = feedback_service.unread_count(db, current_user.id)
    return FeedbackUnreadResponse(unread_count=count)


@app.get("/v1/feedbacks/{feedback_id}", response_model=FeedbackItem)
def get_feedback(
    feedback_id: str,
    current_user: UserDB = Depends(_require_web_user),
    db: Session = Depends(get_db),
):
    fb = feedback_service.get_feedback(db, feedback_id)
    if not fb or fb.user_id != current_user.id:
        raise HTTPException(404, "未找到该反馈")
    # auto mark read
    if not fb.is_read and fb.admin_reply:
        feedback_service.mark_read(db, feedback_id, current_user.id)
        fb.is_read = True
    return _fb_to_item(fb)


@app.post("/v1/feedbacks/{feedback_id}/read")
def mark_feedback_read(
    feedback_id: str,
    current_user: UserDB = Depends(_require_web_user),
    db: Session = Depends(get_db),
):
    fb = feedback_service.mark_read(db, feedback_id, current_user.id)
    if not fb:
        raise HTTPException(404, "未找到该反馈")
    return {"ok": True}


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ─── Static Files & SPA Routing ──────────────────────────────────────────────

# Serve uploaded files (avatars etc.) from shared uploads directory
_uploads_dir = Path(os.getenv("UPLOAD_DIR", str(Path(__file__).parent.parent / "uploads")))
if _uploads_dir.is_dir():
    app.mount("/uploads", StaticFiles(directory=str(_uploads_dir)), name="uploads")

@app.get("/v1/dataproviders/health")
def check_data_providers_health():
    """Return health status of integrated data providers."""
    from tradingagents.dataflows.interface import TOOLS_CATEGORIES, _registry
    providers_status = []
    for name in _registry.list_names():
        prov = _registry.get(name)
        providers_status.append({
            "name": name,
            "status": "healthy",
            "is_placeholder": getattr(prov, "is_placeholder", False),
        })
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "providers": providers_status,
        "categories": list(TOOLS_CATEGORIES.keys()),
    }


# Mount frontend if dist exists
dist_path = os.path.join(os.getcwd(), "frontend/dist")
if os.path.exists(dist_path):
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_path, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # 1. Define and resolve the absolute safe root
        base_path = os.path.realpath(dist_path)
        
        # 2. Resolve the requested path (handling .. and symlinks)
        # We lstrip("/") to prevent os.path.join from treating it as an absolute path
        fullpath = os.path.realpath(os.path.join(base_path, full_path.lstrip("/")))
        
        # 3. Security Check: The normalized path must start with the base_path
        if not fullpath.startswith(base_path):
            return FileResponse(os.path.join(base_path, "index.html"))
            
        # 4. Final check: if it's a valid file, serve it
        if os.path.isfile(fullpath):
            return FileResponse(fullpath)
            
        # Otherwise fallback to index.html for SPA routing
        return FileResponse(os.path.join(base_path, "index.html"))


def run() -> None:
    import uvicorn
    from pathlib import Path

    log_config = str(Path(__file__).parent / "logging_config.yaml")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False, log_config=log_config)
