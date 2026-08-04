"""Standalone scheduler process.

Runs independently of the FastAPI API server. Checks every minute for
scheduled analysis tasks to trigger and executes them with concurrency
control via a simple ``asyncio.Semaphore``.

Start with::

    python -m scheduler.main
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _log(msg: str):
    logger.info(msg)


# ── Concurrency ──────────────────────────────────────────────────────────────
SCHEDULER_CONCURRENCY = int(os.getenv("SCHEDULER_CONCURRENCY", "3"))

# How long to wait for in-flight jobs after SIGTERM/SIGINT before cancelling.
# A job cancelled here stays persisted as "running" (its last_job_id was
# written at claim time), so startup recovery re-queues it — the analysis is
# not lost even when the drain times out.
SCHEDULER_GRACEFUL_SHUTDOWN_SECONDS = int(os.getenv("SCHEDULER_GRACEFUL_SHUTDOWN_SECONDS", "900"))

# Heartbeat interval for persisting running-job liveness to the DB.
SCHEDULER_HEARTBEAT_SECONDS = int(os.getenv("SCHEDULER_HEARTBEAT_SECONDS", "30"))

# A persisted heartbeat older than this many seconds marks a still-"running"
# task as abandoned at startup (rather than genuinely in-flight when the
# previous process died). The heartbeat refreshes every
# SCHEDULER_HEARTBEAT_SECONDS, so this is a small multiple of that interval;
# widen it for long analyses or long maintenance windows.
SCHEDULER_HEARTBEAT_STALE_AFTER_SECONDS = int(
    os.getenv("SCHEDULER_HEARTBEAT_STALE_AFTER_SECONDS", "900")
)

_semaphore: Optional[asyncio.Semaphore] = None
_executor: Optional[ThreadPoolExecutor] = None

# Hold references to fire-and-forget tasks so they are not garbage collected
_background_tasks: set = set()


def _create_tracked_task(coro, *, label: str = "Background task") -> asyncio.Task:
    """Create an asyncio task and keep a reference to prevent GC."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)

    def _on_done(t: asyncio.Task):
        _background_tasks.discard(t)
        if not t.cancelled() and t.exception():
            logger.error("%s failed: %s", label, t.exception())

    task.add_done_callback(_on_done)
    return task


# ── Imports from api & tradingagents ─────────────────────────────────────────
from api.database import (
    ScheduledAnalysisDB,
    ReportDB,
    UserDB,
    init_db,
    get_db_ctx,
)
from api.job_store import get_job_store as _new_job_store
from api.services import (
    auth_service,
    report_service,
    scheduled_service,
)

# Thin wrappers & job runner from the API module
from api.main import (
    _build_imported_user_context,
    _build_scheduled_analyze_request,
    _resolve_scheduled_trade_date,
    _run_job,
    _set_job,
    _get_job,
    _emit_job_event,
    get_job_store,
)

from tradingagents.dataflows.providers.cn_akshare_provider import set_scheduled_task_context


# ── Semaphore-based concurrency slot ─────────────────────────────────────────

@asynccontextmanager
async def _concurrency_slot(job_id: str, symbol: str):
    """Acquire/release a concurrency slot for a scheduled job."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(SCHEDULER_CONCURRENCY)

    if SCHEDULER_CONCURRENCY <= 0:
        # 0 = unlimited concurrency
        yield
        return

    _log(
        f"[Scheduler] Waiting for slot job={job_id} symbol={symbol}"
    )
    await _semaphore.acquire()
    try:
        _log(
            f"[Scheduler] Acquired slot job={job_id} symbol={symbol}"
        )
        yield
    finally:
        _semaphore.release()
        _log(
            f"[Scheduler] Released slot job={job_id} symbol={symbol}"
        )


# ── Notification ─────────────────────────────────────────────────────────────

async def _send_scheduled_report_notifications(
    user_id: str, report_id: str, symbol: str
) -> None:
    """Send configured scheduled report notifications (email & WeCom)."""
    try:
        from api.services.email_report_service import send_report_email_with_retry
        from api.services.wecom_notification_service import send_report_message_with_retry

        def _load_notification_targets():
            email_user = None
            report_to_send = None
            webhook_url = None
            wecom_report_enabled = True
            with get_db_ctx() as db:
                user = db.query(UserDB).filter(UserDB.id == user_id).first()
                report = db.query(ReportDB).filter(ReportDB.id == report_id).first()
                user_cfg = auth_service.get_user_llm_config(db, user_id)
                webhook_url = auth_service.decrypt_secret(
                    getattr(user_cfg, "wecom_webhook_encrypted", None)
                )
                if report:
                    db.expunge(report)
                    report_to_send = report
                if user:
                    wecom_report_enabled = getattr(user, "wecom_report_enabled", True)
                    if getattr(user, "email_report_enabled", True):
                        db.expunge(user)
                        email_user = user
            return email_user, report_to_send, webhook_url, wecom_report_enabled

        email_user, report_to_send, webhook_url, wecom_report_enabled = (
            await asyncio.to_thread(_load_notification_targets)
        )
        if email_user and report_to_send:
            _log(f"[Scheduler] Sending email report for {symbol} to {email_user.email}")
            _create_tracked_task(
                send_report_email_with_retry(email_user, report_to_send),
                label=f"Email notification task ({symbol})",
            )
        if report_to_send and webhook_url and wecom_report_enabled:
            _log(f"[Scheduler] Sending WeCom report for {symbol}")
            _create_tracked_task(
                send_report_message_with_retry(report_to_send, webhook_url),
                label=f"WeCom notification task ({symbol})",
            )
    except Exception as e:
        logger.warning(f"[Scheduler] Notification send failed for {symbol}: {e}")


# ── Single scheduled analysis execution ──────────────────────────────────────

async def _run_scheduled_analysis_once(
    task: dict,
    requested_trade_date: str,
    job_id: str,
    *,
    mark_schedule_run: bool,
) -> None:
    """Execute one scheduled analysis, optionally recording it as the daily run."""
    task_id = task["id"]
    user_id = task["user_id"]
    symbol = task["symbol"]
    horizon = task.get("horizon") or "short"

    actual_trade_date = _resolve_scheduled_trade_date(requested_trade_date)
    _log(f"[Scheduler] {symbol} trade_date={actual_trade_date} (requested={requested_trade_date})")

    set_scheduled_task_context(True)

    def _build_request_sync():
        with get_db_ctx() as db:
            scheduled_user_context = task.get("manual_user_context") or _build_imported_user_context(
                db, user_id, symbol
            )
            return _build_scheduled_analyze_request(
                db=db,
                user_id=user_id,
                symbol=symbol,
                horizon=horizon,
                trade_date=actual_trade_date,
                scheduled_user_context=scheduled_user_context,
            )

    def _record_success_sync():
        with get_db_ctx() as db:
            if mark_schedule_run:
                scheduled_service.mark_run_success(db, task_id, requested_trade_date, job_id)
            else:
                scheduled_service.record_manual_test_result(db, task_id, "success", report_id=job_id)

    def _record_failure_sync():
        with get_db_ctx() as db:
            if mark_schedule_run:
                scheduled_service.mark_run_failed(db, task_id, requested_trade_date)
            else:
                scheduled_service.record_manual_test_result(db, task_id, "failed")

    try:
        async with _concurrency_slot(job_id, symbol):
            req = await asyncio.to_thread(_build_request_sync)

            await _run_job(
                job_id,
                req,
                False,
                True,
                user_id,
                "scheduled" if mark_schedule_run else "scheduled_manual",
            )
        job_state = _get_job(job_id)
        if job_state.get("status") == "failed":
            raise RuntimeError(job_state.get("error") or f"scheduled analysis job {job_id} failed")
        await asyncio.to_thread(_record_success_sync)
        _log(f"[Scheduler] Completed {symbol}")

        await _send_scheduled_report_notifications(user_id, job_id, symbol)
    except Exception as e:
        logger.error(f"[Scheduler] Failed {symbol}: {e}\n{traceback.format_exc()}")
        try:
            await asyncio.to_thread(_record_failure_sync)
        except Exception as db_exc:
            logger.error(f"[Scheduler] Could not record failure: {db_exc}")


async def _run_job_heartbeat(task_id: str, job_id: str) -> None:
    """Persist a heartbeat for a running scheduled job.

    Refreshes ``last_job_heartbeat_at`` every ``SCHEDULER_HEARTBEAT_SECONDS``
    while the analysis runs, so startup recovery can tell a genuinely in-flight
    task from an abandoned one.  ``_run_scheduled_job`` cancels this task when
    the analysis reaches a terminal state.
    """
    while True:
        await asyncio.sleep(SCHEDULER_HEARTBEAT_SECONDS)

        def _touch() -> None:
            with get_db_ctx() as db:
                item = (
                    db.query(ScheduledAnalysisDB)
                    .filter(ScheduledAnalysisDB.id == task_id)
                    .first()
                )
                if item is not None and item.last_job_id == job_id:
                    item.last_job_heartbeat_at = datetime.now(timezone.utc)
                    db.commit()

        try:
            await asyncio.to_thread(_touch)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"[Scheduler] Heartbeat write failed for job {job_id}: {exc}")


async def _run_scheduled_job(task: dict, trade_date: str):
    """Execute a single scheduled analysis job.

    Args:
        task: dict with keys id, user_id, symbol, horizon, job_id (plain values,
              not an ORM instance, to avoid DetachedInstanceError).
        trade_date: YYYY-MM-DD string.

    ``job_id`` is generated and persisted on the scheduled row at claim time
    (see ``_scheduler_loop``), so a crash mid-analysis leaves a recoverable
    trace for ``_recover_stale_tasks``.
    """
    user_id = task["user_id"]
    symbol = task["symbol"]
    job_id = task["job_id"]
    task_id = task["id"]

    _log(f"[Scheduler] Running {symbol} for user={user_id} job={job_id}")
    heartbeat = asyncio.create_task(_run_job_heartbeat(task_id, job_id))
    try:
        await _run_scheduled_analysis_once(
            task,
            trade_date,
            job_id,
            mark_schedule_run=True,
        )
    finally:
        heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass
        get_job_store().delete_job(job_id)


# ── Scheduler loop ───────────────────────────────────────────────────────────

def _claim_pending_tasks(today: str, current_hhmm: str):
    """Claim all pending scheduled tasks, persisting job ids, and return snapshots.

    Each claimed task gets a fresh ``job_id`` written to ``last_job_id`` /
    ``last_job_started_at`` on its scheduled row before the analysis launches,
    so a crash mid-run leaves a recoverable trace (see ``_recover_stale_tasks``).

    Returns a list of plain dicts (keys: id, user_id, symbol, horizon, job_id).
    """
    with get_db_ctx() as db:
        tasks = scheduled_service.get_pending_tasks(db, today, current_hhmm)
        if not tasks:
            return []
        claimed_at = datetime.now(timezone.utc)
        snapshots = []
        for task in tasks:
            job_id = uuid4().hex
            task.last_run_date = today
            task.last_run_status = "running"
            task.last_job_id = job_id
            task.last_job_started_at = claimed_at
            task.last_job_heartbeat_at = claimed_at
            snapshots.append(
                {
                    "id": task.id,
                    "user_id": task.user_id,
                    "symbol": task.symbol,
                    "horizon": task.horizon,
                    "job_id": job_id,
                }
            )
        db.commit()
        return snapshots


async def _scheduler_loop(stop_event: asyncio.Event):
    """Background loop: check every minute for scheduled tasks to trigger.

    Each task has its own trigger_time (HH:MM). The scheduler runs on trading
    days only, outside of trading hours (before 9:15 or after 15:00). Tasks
    are triggered when current time >= task.trigger_time and the task hasn't
    run today yet.

    *stop_event* is set by the SIGTERM/SIGINT handler; when set, the loop
    stops claiming new tasks and returns so ``_startup`` can drain in-flight
    jobs.
    """
    from tradingagents.dataflows.trade_calendar import is_cn_trading_day
    from zoneinfo import ZoneInfo

    _log("[Scheduler] Loop started.")
    while not stop_event.is_set():
        # Wake every 60s, or immediately once shutdown is requested.
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=60)
        except asyncio.TimeoutError:
            pass
        if stop_event.is_set():
            break
        try:
            now = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
            today = now.strftime("%Y-%m-%d")
            current_hhmm = now.strftime("%H:%M")

            if not is_cn_trading_day(today):
                continue
            time_val = now.hour * 60 + now.minute
            if 8 * 60 < time_val < 20 * 60:
                continue

            task_snapshots = await asyncio.to_thread(_claim_pending_tasks, today, current_hhmm)
            if not task_snapshots:
                continue

            _log(f"[Scheduler] Launching {len(task_snapshots)} tasks (staggered)")
            for i, snap in enumerate(task_snapshots):
                if i > 0:
                    await asyncio.sleep(1)
                _create_tracked_task(_run_scheduled_job(snap, today))

        except Exception as e:
            logger.error(f"[Scheduler] Error: {e}")


async def _drain_inflight_jobs(timeout_seconds: int) -> None:
    """Wait for in-flight background jobs to finish on shutdown.

    Used after the scheduler loop exits (SIGTERM/SIGINT).  If *timeout_seconds*
    elapses, the remaining jobs are cancelled.  A job cancelled here stays
    ``running`` in the DB with its ``last_job_id`` persisted, so the next
    startup's recovery re-queues it — the analysis is not lost.
    """
    deadline = time.monotonic() + max(0, timeout_seconds)
    while _background_tasks:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        pending = list(_background_tasks)
        _log(f"[Scheduler] Waiting for {len(pending)} in-flight job(s) ...")
        try:
            await asyncio.wait(pending, timeout=min(5.0, remaining))
        except Exception as exc:
            logger.warning(f"[Scheduler] Drain wait interrupted: {exc}")
            break
    remaining = list(_background_tasks)
    if remaining:
        _log(f"[Scheduler] Shutdown drain timed out; cancelling {len(remaining)} job(s).")
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)


# ── Stale task recovery ──────────────────────────────────────────────────────

def _run_date_start_utc(run_date: str):
    """Start of a Shanghai-local run date as a naive UTC datetime.

    Reports persist ``created_at`` in UTC (``report_service`` uses
    ``datetime.now(timezone.utc)``), while ``last_run_date`` is the Shanghai
    local date the scheduler claimed the run for.  Comparing a UTC timestamp
    against a bare date string misclassifies runs that fall in the first 8 UTC
    hours of a Shanghai day (see AGENTS.md 3.5 — no implicit date shapes).
    """
    from zoneinfo import ZoneInfo

    naive_local = datetime.fromisoformat(f"{run_date}T00:00:00")
    aware_local = naive_local.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    return aware_local.astimezone(timezone.utc).replace(tzinfo=None)


def _heartbeat_is_abandoned(heartbeat_at: Optional[datetime]) -> bool:
    """True when a persisted heartbeat is older than the abandoned threshold.

    ``last_job_heartbeat_at`` is refreshed every ``SCHEDULER_HEARTBEAT_SECONDS``
    while a job runs, so a row whose heartbeat has been silent for longer than
    ``SCHEDULER_HEARTBEAT_STALE_AFTER_SECONDS`` was not actively running when
    this process started — treat it as abandoned rather than in-flight. A
    missing heartbeat is treated as not abandoned so rows written before the
    heartbeat column existed are still re-queued conservatively.
    """
    if heartbeat_at is None:
        return False
    if heartbeat_at.tzinfo is None:
        # SQLite persists DATETIME columns as naive UTC (the writes here are
        # aware UTC); normalize so the age comparison is tz-safe.
        heartbeat_at = heartbeat_at.replace(tzinfo=timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - heartbeat_at).total_seconds()
    return age_seconds > SCHEDULER_HEARTBEAT_STALE_AFTER_SECONDS


def _recover_stale_tasks():
    """Recover tasks stuck in 'running' state (from previous crash/restart).

    A row still ``running`` at startup means its analysis did not reach a
    terminal state before the previous process exited:

    - If a completed report exists for the claimed run, treat it as success.
    - If the run persisted a ``last_job_id``, the task was in-flight when the
      previous process exited. A fresh heartbeat (``last_job_heartbeat_at``
      within ``SCHEDULER_HEARTBEAT_STALE_AFTER_SECONDS``) confirms it was
      genuinely running — re-queue it for today so the analysis is not silently
      lost. A stale heartbeat means the task was abandoned, so mark it ``stale``
      and clear the run date.
    - Otherwise mark it stale and clear the run date (legacy behavior).
    """
    # In the split deployment (REDIS_URL set) the shared job store survives a
    # scheduler restart, so its running set cross-checks the DB "running" rows.
    # Best-effort only: a down Redis must not block startup recovery.
    try:
        store_running_jobs = get_job_store().running_jobs()
    except Exception as exc:
        logger.warning(f"[Scheduler] Could not inspect job store at startup: {exc}")
        store_running_jobs = {}

    with get_db_ctx() as db:
        stale = (
            db.query(ScheduledAnalysisDB)
            .filter(ScheduledAnalysisDB.last_run_status == "running")
            .all()
        )
        if stale:
            recovered_count = 0
            reset_count = 0
            requeued_count = 0
            for item in stale:
                run_start_utc = (
                    _run_date_start_utc(item.last_run_date)
                    if item.last_run_date
                    else None
                )
                has_report = (
                    item.last_report_id
                    and run_start_utc is not None
                    and db.query(ReportDB)
                    .filter(
                        ReportDB.id == item.last_report_id,
                        ReportDB.status == "completed",
                        ReportDB.created_at >= run_start_utc,
                    )
                    .first()
                )
                if has_report:
                    item.last_run_status = "success"
                    recovered_count += 1
                elif item.last_job_id:
                    if _heartbeat_is_abandoned(item.last_job_heartbeat_at):
                        # The heartbeat went silent longer than the stale
                        # threshold ago, so the task was not actively running
                        # when this process started — it is abandoned. Mark it
                        # stale like legacy rows instead of re-queueing it.
                        _log(
                            f"[Scheduler] Marking task {item.id} stale: last "
                            f"heartbeat older than "
                            f"{SCHEDULER_HEARTBEAT_STALE_AFTER_SECONDS}s."
                        )
                        item.last_run_status = "stale"
                        item.last_run_date = None
                        item.last_job_id = None
                        item.last_job_started_at = None
                        item.last_job_heartbeat_at = None
                        reset_count += 1
                    else:
                        # Fresh heartbeat (or none): genuinely in-flight when
                        # the previous process died. Reset the run so it is
                        # picked up again today instead of being dropped; clear
                        # the persisted job trace with it.
                        store_confirmed = item.last_job_id in store_running_jobs
                        if store_confirmed:
                            _log(
                                f"[Scheduler] Re-queuing task {item.id}: job "
                                f"{item.last_job_id} still tracked as running in the "
                                "shared job store."
                            )
                        item.last_run_status = None
                        item.last_run_date = None
                        item.last_job_id = None
                        item.last_job_started_at = None
                        item.last_job_heartbeat_at = None
                        requeued_count += 1
                else:
                    item.last_run_status = "stale"
                    item.last_run_date = None
                    reset_count += 1
            db.commit()
            _log(
                f"[Scheduler] Reset {len(stale)} stale 'running' tasks on startup "
                f"(recovered={recovered_count}, requeued={requeued_count}, "
                f"reset_to_stale={reset_count})."
            )
        report_reset = report_service.recover_stale_active_reports(db)
        if report_reset["total"]:
            _log(
                "[Reports] Recovered %s stale active reports on startup (marked failed)."
                % report_reset["total"]
            )


# ── Startup / main ───────────────────────────────────────────────────────────

async def _startup():
    """Initialize DB, pre-load caches, recover stale tasks, then run the loop.

    Installs SIGTERM/SIGINT handlers so container orchestration (docker stop /
    docker-entrypoint forward_stop) triggers a graceful drain of in-flight jobs
    instead of killing them mid-analysis.
    """
    global _semaphore, _executor

    # Security startup guard (DAV-66): refuse to run the scheduler without a
    # TA_APP_SECRET_KEY (unless explicitly opted into the insecure default for
    # local dev). The scheduler decrypts webhook URLs and reads user secrets, so
    # it must fail fast alongside the API server instead of using the default key.
    auth_service.ensure_secure_secret_configured()

    # Each scheduled `_run_job` fans out many `asyncio.to_thread` calls (DB
    # writes, akshare data collection, LLM extraction). The CPython default
    # of `min(32, cpu_count + 4)` is too small to absorb concurrent jobs +
    # the per-tick DB transaction the scheduler loop now runs in to_thread.
    try:
        loop = asyncio.get_running_loop()
        executor_workers = int(
            os.getenv("ASYNCIO_DEFAULT_EXECUTOR_WORKERS", str(max(64, SCHEDULER_CONCURRENCY * 16)))
        )
        loop.set_default_executor(
            ThreadPoolExecutor(
                max_workers=executor_workers,
                thread_name_prefix="ta-sched-asyncio",
            )
        )
        _log(f"[Scheduler] Default asyncio executor set to {executor_workers} workers.")
    except Exception as exc:
        _log(f"[Scheduler] Could not configure default asyncio executor: {exc}")

    init_db()
    _log("Database initialized.")

    _semaphore = asyncio.Semaphore(SCHEDULER_CONCURRENCY)
    _log(f"[Scheduler] Concurrency limit set to {SCHEDULER_CONCURRENCY}")

    _executor = ThreadPoolExecutor(max_workers=SCHEDULER_CONCURRENCY + 2)

    # Recover stale tasks from previous run
    _recover_stale_tasks()

    # Pre-load trade calendar (uses mini_racer/V8 which is not thread-safe)
    from tradingagents.dataflows.trade_calendar import _load_cn_trade_dates

    _load_cn_trade_dates()
    _log("Trade calendar pre-loaded.")

    # Pre-load stock + ETF name map
    from api.main import _load_cn_stock_map

    await asyncio.to_thread(_load_cn_stock_map)
    _log("Stock map pre-loaded on startup.")

    # ── Graceful shutdown: SIGTERM/SIGINT stop the loop, then drain ──
    stop_event = asyncio.Event()

    def _request_shutdown():
        if not stop_event.is_set():
            _log("[Scheduler] Shutdown requested (SIGTERM/SIGINT); draining in-flight jobs ...")
        stop_event.set()

    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(_sig, _request_shutdown)
        except (NotImplementedError, RuntimeError):
            # Not on the main thread or unsupported platform; fall through to
            # the default handler rather than failing startup.
            pass

    try:
        await _scheduler_loop(stop_event)
    finally:
        _log("[Scheduler] Scheduler loop exited; draining in-flight jobs ...")
        await _drain_inflight_jobs(timeout_seconds=SCHEDULER_GRACEFUL_SHUTDOWN_SECONDS)
        _log("[Scheduler] Shutdown complete.")


def main():
    """Entry point for ``python -m scheduler.main``."""
    _log("[Scheduler] Starting standalone scheduler process ...")
    try:
        asyncio.run(_startup())
    except KeyboardInterrupt:
        _log("[Scheduler] Stopped by user.")


# Alias for pyproject.toml script entry (must be sync)
sync_main = main


if __name__ == "__main__":
    main()
