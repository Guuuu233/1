"""
Backtest service — runs historical analysis for a symbol across a date range
and compares each decision against subsequent price performance.

Design: completely non-invasive. Reuses existing TradingAgentsGraph.propagate()
without touching any existing code.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# Backtest jobs are intentionally in-memory only: a process restart drops all
# queued, running, and completed jobs. No DB schema or recovery path is added;
# terminal history is bounded below so the in-memory store cannot grow without
# limit.
# ──────────────────────────────────────────────────────────────────────────────
# In-memory store (no additional DB table — results stored as JSON in the job)
# ──────────────────────────────────────────────────────────────────────────────
_backtest_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()

MAX_BACKTEST_WORKERS = max(1, int(os.getenv("BACKTEST_MAX_WORKERS", "2")))
MAX_BACKTEST_QUEUE = max(1, int(os.getenv("BACKTEST_MAX_QUEUE", "20")))
MAX_RETAINED_BACKTEST_JOBS = max(1, int(os.getenv("BACKTEST_MAX_RETAINED_JOBS", "100")))
MIN_SAMPLE_INTERVAL = 1
MAX_SAMPLE_INTERVAL = 365

# Price basis semantics (DAV-606, DAV-705)
PRICE_BASIS_VENDOR_QFQ: str = "vendor_qfq"
PRICE_BASIS_UNSPECIFIED: str = "unspecified"
PRICE_BASIS_RAW: str = "raw"
PRICE_BASIS_PIT_RAW: str = "pit_raw"
PRICE_BASIS_PIT_ADJUSTED: str = "pit_adjusted"


class BacktestQueueFullError(RuntimeError):
    """Raised when the bounded backtest submission queue is full."""


@dataclass(frozen=True)
class _BacktestTask:
    job_id: str
    symbol: str
    start_date: str
    end_date: str
    selected_analysts: List[str]
    hold_days: int
    sample_interval: int
    config: Dict[str, Any]


_job_queue: "queue.Queue[_BacktestTask]" = queue.Queue(maxsize=MAX_BACKTEST_QUEUE)
_workers: List[threading.Thread] = []
_workers_started = False


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set(job_id: str, **kwargs: Any) -> None:
    with _lock:
        job = _backtest_jobs.get(job_id)
        if job is None:
            # A deleted job must not be resurrected by a queued worker.
            return
        job.update(kwargs)


def _create_job(job_id: str, **kwargs: Any) -> None:
    payload = dict(kwargs)
    payload["job_id"] = job_id
    with _lock:
        _backtest_jobs[job_id] = payload


def get_job(job_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with _lock:
        job = _backtest_jobs.get(job_id)
        if job is None or (user_id is not None and job.get("user_id") != user_id):
            return None
        return dict(job)


def list_jobs(user_id: str) -> List[Dict[str, Any]]:
    with _lock:
        jobs = [
            dict(job)
            for job in _backtest_jobs.values()
            if job.get("user_id") == user_id
        ]
        return sorted(jobs, key=lambda j: j.get("created_at", ""), reverse=True)


def delete_job(job_id: str, user_id: str) -> bool:
    with _lock:
        job = _backtest_jobs.get(job_id)
        if job is None or job.get("user_id") != user_id:
            return False
        del _backtest_jobs[job_id]
        return True


def _prune_old_jobs(keep_job_id: Optional[str] = None) -> None:
    """Drop oldest terminal jobs once the in-memory store exceeds its cap."""
    with _lock:
        terminal = sorted(
            (
                job
                for job in _backtest_jobs.values()
                if job.get("status") in ("completed", "failed")
                and job.get("job_id") != keep_job_id
            ),
            key=lambda job: job.get("created_at") or "",
        )
        # Never treat a negative excess as a Python negative slice; doing so
        # would delete the oldest terminal jobs even when the store is under cap.
        excess = max(0, len(_backtest_jobs) - MAX_RETAINED_BACKTEST_JOBS)
        for job in terminal[:excess]:
            _backtest_jobs.pop(job["job_id"], None)


def validate_sample_interval(sample_interval: int) -> int:
    """Validate an integer sampling interval and return it unchanged."""
    if isinstance(sample_interval, bool) or not isinstance(sample_interval, int):
        raise ValueError("sample_interval must be an integer")
    if sample_interval < MIN_SAMPLE_INTERVAL or sample_interval > MAX_SAMPLE_INTERVAL:
        raise ValueError(
            f"sample_interval must be between {MIN_SAMPLE_INTERVAL} and {MAX_SAMPLE_INTERVAL}"
        )
    return sample_interval


# ──────────────────────────────────────────────────────────────────────────────
# Trading-day utilities (lightweight — no exchange dependency)
# ──────────────────────────────────────────────────────────────────────────────

def _get_trading_dates(start: str, end: str, interval_days: int) -> List[str]:
    """Return a list of weekday dates between start and end, sampled every interval_days."""
    if interval_days < MIN_SAMPLE_INTERVAL:
        raise ValueError("interval_days must be >= 1")
    fmt = "%Y-%m-%d"
    cur = datetime.strptime(start, fmt)
    end_dt = datetime.strptime(end, fmt)
    dates = []
    while cur <= end_dt:
        if cur.weekday() < 5:  # Mon–Fri only
            dates.append(cur.strftime(fmt))
        cur += timedelta(days=interval_days)
    return dates


def _get_price_after(
    symbol: str,
    base_date: str,
    hold_days: int,
    price_basis: Optional[str] = PRICE_BASIS_VENDOR_QFQ,
) -> Optional[float]:
    """Fetch closing price hold_days trading days after base_date using akshare or raw provider.

    Refuses to shorten hold_days when the fetched series is shorter than hold_days;
    returns None to ensure strict T+N evaluation (D-009 / P1-3).
    When price_basis is PRICE_BASIS_VENDOR_QFQ (default): routes to existing vendor get_stock_data.
    When price_basis is PRICE_BASIS_RAW: routes to cn_akshare provider with price_basis="raw".
    When price_basis is neither (unspecified, pit_raw, pit_adjusted, unknown): fails closed (returns None).
    """
    try:
        import pandas as pd

        fmt = "%Y-%m-%d"
        start_dt = datetime.strptime(base_date, fmt)
        # Fetch data starting from base_date + 1 day, extend window for hold_days
        fetch_start = (start_dt + timedelta(days=1)).strftime(fmt)
        fetch_end = (start_dt + timedelta(days=hold_days + 30)).strftime(fmt)

        effective_basis = PRICE_BASIS_VENDOR_QFQ if price_basis is None else price_basis

        if effective_basis == PRICE_BASIS_VENDOR_QFQ:
            from tradingagents.dataflows.interface import route_to_vendor

            csv_data = route_to_vendor("get_stock_data", symbol, fetch_start, fetch_end)
            if not csv_data:
                return None
        elif effective_basis == PRICE_BASIS_RAW:
            from tradingagents.dataflows.interface import _registry

            prov = _registry.get("cn_akshare")
            if prov is None or not hasattr(prov, "get_stock_data"):
                try:
                    from tradingagents.dataflows.providers.cn_akshare_provider import (
                        CnAkshareProvider,
                    )

                    prov = CnAkshareProvider()
                except Exception:
                    prov = None

            if prov is None:
                logger.warning("cn_akshare provider unavailable for raw stock data fetch")
                return None

            try:
                csv_data = prov.get_stock_data(
                    symbol=symbol,
                    start_date=fetch_start,
                    end_date=fetch_end,
                    price_basis=PRICE_BASIS_RAW,
                )
            except Exception as exc:
                logger.warning(
                    "_get_price_after (raw) failed for %s @ %s (hold_days=%s): %s",
                    symbol,
                    base_date,
                    hold_days,
                    exc,
                )
                return None

            if not csv_data or str(csv_data).startswith("【数据获取失败】") or str(csv_data).startswith("No data found"):
                return None
        else:
            logger.warning(
                "_get_price_after: unsupported or unknown price_basis %r for %s; fail-closed",
                price_basis,
                symbol,
            )
            return None

        df = pd.read_csv(pd.io.common.StringIO(csv_data), comment="#")
        # Find column for close price
        close_cols = [c for c in df.columns if "close" in c.lower() or "收盘" in c]
        date_cols = [c for c in df.columns if "date" in c.lower() or "日期" in c or "time" in c.lower()]
        if not close_cols or not date_cols:
            return None

        df = df.sort_values(date_cols[0]).reset_index(drop=True)
        if len(df) < max(1, hold_days):
            return None
        return float(df[close_cols[0]].iloc[hold_days - 1])
    except Exception as exc:
        logger.warning(
            "_get_price_after failed for %s @ %s (hold_days=%s, price_basis=%s): %s",
            symbol,
            base_date,
            hold_days,
            price_basis,
            exc,
        )
        return None


def _get_price_on(
    symbol: str,
    date: str,
    price_basis: Optional[str] = PRICE_BASIS_VENDOR_QFQ,
) -> Optional[float]:
    """Fetch closing price on or just before date.

    When price_basis is PRICE_BASIS_VENDOR_QFQ (default): routes to existing vendor get_stock_data.
    When price_basis is PRICE_BASIS_RAW: routes to cn_akshare provider with price_basis="raw".
    When price_basis is neither (unspecified, pit_raw, pit_adjusted, unknown): fails closed (returns None).
    """
    try:
        import pandas as pd

        fmt = "%Y-%m-%d"
        start = (datetime.strptime(date, fmt) - timedelta(days=5)).strftime(fmt)

        effective_basis = PRICE_BASIS_VENDOR_QFQ if price_basis is None else price_basis

        if effective_basis == PRICE_BASIS_VENDOR_QFQ:
            from tradingagents.dataflows.interface import route_to_vendor

            csv_data = route_to_vendor("get_stock_data", symbol, start, date)
            if not csv_data:
                return None
        elif effective_basis == PRICE_BASIS_RAW:
            from tradingagents.dataflows.interface import _registry

            prov = _registry.get("cn_akshare")
            if prov is None or not hasattr(prov, "get_stock_data"):
                try:
                    from tradingagents.dataflows.providers.cn_akshare_provider import (
                        CnAkshareProvider,
                    )

                    prov = CnAkshareProvider()
                except Exception:
                    prov = None

            if prov is None:
                logger.warning("cn_akshare provider unavailable for raw stock data fetch")
                return None

            try:
                csv_data = prov.get_stock_data(
                    symbol=symbol,
                    start_date=start,
                    end_date=date,
                    price_basis=PRICE_BASIS_RAW,
                )
            except Exception as exc:
                logger.warning(
                    "_get_price_on (raw) failed for %s @ %s: %s",
                    symbol,
                    date,
                    exc,
                )
                return None

            if not csv_data or str(csv_data).startswith("【数据获取失败】") or str(csv_data).startswith("No data found"):
                return None
        else:
            logger.warning(
                "_get_price_on: unsupported or unknown price_basis %r for %s; fail-closed",
                price_basis,
                symbol,
            )
            return None

        df = pd.read_csv(pd.io.common.StringIO(csv_data), comment="#")
        close_cols = [c for c in df.columns if "close" in c.lower() or "收盘" in c]
        date_cols = [c for c in df.columns if "date" in c.lower() or "日期" in c or "time" in c.lower()]
        if not close_cols or not date_cols:
            return None
        df = df.sort_values(date_cols[0]).reset_index(drop=True)
        if df.empty:
            return None
        return float(df[close_cols[0]].iloc[-1])
    except Exception as exc:
        logger.warning(
            "_get_price_on failed for %s @ %s (price_basis=%s): %s",
            symbol,
            date,
            price_basis,
            exc,
        )
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Core backtest runner
# ──────────────────────────────────────────────────────────────────────────────

def _run_single_analysis(
    symbol: str,
    trade_date: str,
    selected_analysts: List[str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one full analysis without SSE. Returns final state dict."""
    from tradingagents.agents.utils.decision_status import decision_status_from_state
    from tradingagents.dataflows.config import set_config
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    set_config(config)
    graph = TradingAgentsGraph(
        selected_analysts=selected_analysts,
        debug=False,
        config=config,
    )
    final_state, _ = graph.propagate(symbol, trade_date)
    decision_raw = final_state.get("final_trade_decision", "")
    decision = graph.process_signal(decision_raw)

    dec_status = decision_status_from_state(final_state)
    analysis_status = (
        final_state.get("analysis_status")
        or (dec_status.analysis_status if dec_status else None)
    )
    trade_action = (
        final_state.get("trade_action")
        or (dec_status.trade_action if dec_status else None)
    )
    price_basis = final_state.get("price_basis") or PRICE_BASIS_VENDOR_QFQ

    return {
        "final_trade_decision": decision_raw,
        "decision": decision,
        "analysis_status": analysis_status,
        "trade_action": trade_action,
        "decision_status": dec_status.to_dict() if dec_status else final_state.get("decision_status"),
        "price_basis": price_basis,
        "final_state": final_state,
    }


def _classify_decision(decision: Any) -> str:
    """Classify decision or state into trade action (BUY / SELL / HOLD / WAIT / NO_TRADE).

    D-009 / P1-3: Does not collapse non-directional / invalid states into HOLD.
    """
    if isinstance(decision, dict):
        status = decision.get("analysis_status")
        action = decision.get("trade_action")
        if status in ("INVALID_RUN", "DATA_ERROR"):
            return "NO_TRADE"
        if status in ("ABSTAIN", "PARTIAL"):
            return "NO_TRADE" if action not in ("WAIT", "NO_TRADE") else action
        if action in ("BUY", "SELL", "HOLD", "WAIT", "NO_TRADE"):
            return action
        decision = decision.get("decision") or decision.get("final_trade_decision") or ""

    d = str(decision or "").strip().upper()
    if not d:
        return "NO_TRADE"
    if any(k in d for k in ["INVALID_RUN", "DATA_ERROR", "INVALID"]):
        return "NO_TRADE"
    if any(k in d for k in ["ABSTAIN", "PARTIAL"]):
        return "NO_TRADE"
    if any(k in d for k in ["WAIT", "观望", "待确认"]):
        return "WAIT"
    if any(k in d for k in ["NO_TRADE", "禁止交易", "不交易"]):
        return "NO_TRADE"
    if any(k in d for k in ["BUY", "增持", "买入", "BULLISH", "BULL", "看多", "偏多"]):
        return "BUY"
    if any(k in d for k in ["SELL", "减持", "卖出", "BEARISH", "BEAR", "看空", "偏空"]):
        return "SELL"
    if any(k in d for k in ["HOLD", "中性", "NEUTRAL"]):
        return "HOLD"
    return "NO_TRADE"


def _compute_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute win rate, returns, and exclusion statistics from backtest records."""
    excluded_invalid = 0
    excluded_abstain = 0
    excluded_no_trade = 0
    excluded_incomplete = 0
    excluded_hold = 0

    valid_trades: List[Dict[str, Any]] = []

    for r in records:
        analysis_status = str(r.get("analysis_status") or "").upper()
        trade_action = str(r.get("trade_action") or r.get("action") or "").upper()
        outcome_status = str(r.get("outcome_status") or "").lower()
        ret = r.get("return_pct")
        err = r.get("error")

        if err is not None or analysis_status in ("INVALID_RUN", "DATA_ERROR"):
            excluded_invalid += 1
        elif analysis_status in ("ABSTAIN", "PARTIAL"):
            excluded_abstain += 1
        elif trade_action in ("WAIT", "NO_TRADE"):
            excluded_no_trade += 1
        elif trade_action == "HOLD":
            excluded_hold += 1
        elif trade_action in ("BUY", "SELL"):
            if analysis_status != "VALID":
                excluded_invalid += 1
            elif outcome_status == "incomplete" or ret is None:
                excluded_incomplete += 1
            else:
                valid_trades.append(r)
        else:
            excluded_invalid += 1

    excluded_total = (
        excluded_invalid
        + excluded_abstain
        + excluded_no_trade
        + excluded_incomplete
        + excluded_hold
    )

    base_stats = {
        "excluded_invalid": excluded_invalid,
        "excluded_abstain": excluded_abstain,
        "excluded_wait_or_no_trade": excluded_no_trade,
        "excluded_no_trade": excluded_no_trade,
        "excluded_incomplete": excluded_incomplete,
        "excluded_hold": excluded_hold,
        "excluded_total": excluded_total,
        "excluded_counts": {
            "invalid": excluded_invalid,
            "abstain": excluded_abstain,
            "wait_or_no_trade": excluded_no_trade,
            "no_trade": excluded_no_trade,
            "incomplete": excluded_incomplete,
            "hold": excluded_hold,
            "total": excluded_total,
        },
    }

    if not valid_trades:
        return {
            "total_signals": 0,
            "win_rate": None,
            "avg_return_pct": None,
            "best_return_pct": None,
            "worst_return_pct": None,
            **base_stats,
        }

    wins = 0
    returns = []
    for t in valid_trades:
        ret = t["return_pct"]
        returns.append(ret)
        # return_pct is already signed so positive return is a winning trade
        if ret > 0:
            wins += 1

    return {
        "total_signals": len(valid_trades),
        "win_rate": round(wins / len(valid_trades) * 100, 1),
        "avg_return_pct": round(sum(returns) / len(returns), 2),
        "best_return_pct": round(max(returns), 2),
        "worst_return_pct": round(min(returns), 2),
        **base_stats,
    }


def _run_backtest(job_id: str, symbol: str, start_date: str, end_date: str,
                  selected_analysts: List[str], hold_days: int, sample_interval: int,
                  config: Dict[str, Any]) -> None:
    """Background thread: run backtest and store results."""
    _set(job_id, status="running", started_at=_utcnow_iso())
    try:
        dates = _get_trading_dates(start_date, end_date, sample_interval)
        total = len(dates)
        _set(job_id, total_dates=total, completed_dates=0, records=[], error=None)

        records: List[Dict[str, Any]] = []

        for i, trade_date in enumerate(dates):
            record: Dict[str, Any] = {
                "date": trade_date,
                "action": "NO_TRADE",
                "analysis_status": "INVALID_RUN",
                "trade_action": "NO_TRADE",
                "price_basis": "unknown",
                "entry_price": None,
                "entry_price_as_of": None,
                "exit_price": None,
                "exit_price_as_of": None,
                "return_pct": None,
                "outcome_status": "excluded_invalid",
                "decision_summary": "",
                "error": None,
            }
            try:
                analysis = _run_single_analysis(symbol, trade_date, selected_analysts, config)

                raw_decision = analysis.get("decision") or ""
                final_decision_text = analysis.get("final_trade_decision") or ""

                analysis_status = analysis.get("analysis_status")
                if not analysis_status and isinstance(analysis.get("decision_status"), dict):
                    analysis_status = analysis["decision_status"].get("analysis_status")
                if not analysis_status and isinstance(analysis.get("final_state"), dict):
                    analysis_status = analysis["final_state"].get("analysis_status")

                trade_action = analysis.get("trade_action")
                if not trade_action and isinstance(analysis.get("decision_status"), dict):
                    trade_action = analysis["decision_status"].get("trade_action")
                if not trade_action and isinstance(analysis.get("final_state"), dict):
                    trade_action = analysis["final_state"].get("trade_action")

                if not analysis_status:
                    text_to_check = f"{raw_decision} {final_decision_text}".upper()
                    if any(k in text_to_check for k in ["INVALID_RUN", "DATA_ERROR"]):
                        analysis_status = "INVALID_RUN"
                    elif any(k in text_to_check for k in ["ABSTAIN", "PARTIAL"]):
                        analysis_status = "ABSTAIN"
                    elif any(k in text_to_check for k in ["BUY", "SELL", "HOLD", "WAIT", "NO_TRADE"]):
                        analysis_status = "VALID"
                    else:
                        analysis_status = "VALID"

                if not trade_action:
                    trade_action = _classify_decision(raw_decision or final_decision_text)

                if analysis_status in ("INVALID_RUN", "DATA_ERROR"):
                    trade_action = "NO_TRADE"
                elif analysis_status in ("ABSTAIN", "PARTIAL"):
                    if trade_action not in ("WAIT", "NO_TRADE"):
                        trade_action = "NO_TRADE"

                price_basis = analysis.get("price_basis") or PRICE_BASIS_VENDOR_QFQ

                record["analysis_status"] = analysis_status
                record["trade_action"] = trade_action
                record["action"] = trade_action
                record["price_basis"] = price_basis
                record["decision_summary"] = (
                    final_decision_text[:200] if final_decision_text else ""
                )

                if analysis_status in ("INVALID_RUN", "DATA_ERROR"):
                    record["outcome_status"] = "excluded_invalid"
                elif analysis_status in ("ABSTAIN", "PARTIAL"):
                    record["outcome_status"] = "excluded_abstain"
                elif trade_action in ("WAIT", "NO_TRADE"):
                    record["outcome_status"] = "excluded_no_trade"
                elif trade_action == "HOLD":
                    record["outcome_status"] = "excluded_hold"
                elif trade_action in ("BUY", "SELL") and analysis_status == "VALID":
                    entry_price = _get_price_on(symbol, trade_date, price_basis=price_basis)
                    exit_price = _get_price_after(symbol, trade_date, hold_days, price_basis=price_basis)

                    if entry_price is not None and entry_price > 0:
                        record["entry_price"] = round(entry_price, 2)
                        record["entry_price_as_of"] = trade_date

                    if exit_price is not None and exit_price > 0:
                        record["exit_price"] = round(exit_price, 2)
                        record["exit_price_as_of"] = f"{trade_date}+T{hold_days}"

                    if (
                        entry_price is not None
                        and exit_price is not None
                        and entry_price > 0
                        and exit_price > 0
                    ):
                        raw_return = (exit_price - entry_price) / entry_price * 100
                        record["return_pct"] = round(
                            raw_return if trade_action == "BUY" else -raw_return, 2
                        )
                        record["outcome_status"] = "ok"
                    else:
                        record["return_pct"] = None
                        record["outcome_status"] = "incomplete"
                else:
                    record["outcome_status"] = "excluded_invalid"

            except Exception as exc:
                logger.exception("Single analysis failed for %s @ %s", symbol, trade_date)
                record["error"] = str(exc)[:200]
                record["analysis_status"] = "INVALID_RUN"
                record["trade_action"] = "NO_TRADE"
                record["action"] = "NO_TRADE"
                record["outcome_status"] = "excluded_invalid"
                record["return_pct"] = None

            records.append(record)
            _set(job_id, completed_dates=i + 1, records=list(records))

        stats = _compute_stats(records)
        _set(job_id,
             status="completed",
             finished_at=_utcnow_iso(),
             records=records,
             stats=stats)
    except Exception as exc:
        logger.exception("Backtest job %s failed", job_id)
        _set(job_id,
             status="failed",
             finished_at=_utcnow_iso(),
             records=[],
             stats=None,
             error=str(exc)[:500])
    finally:
        _prune_old_jobs(job_id)


def _worker_loop() -> None:
    """Consume queued backtest tasks from a fixed worker pool."""
    while True:
        task = _job_queue.get()
        try:
            _run_backtest(
                task.job_id,
                task.symbol,
                task.start_date,
                task.end_date,
                task.selected_analysts,
                task.hold_days,
                task.sample_interval,
                task.config,
            )
        except Exception as exc:
            logger.exception("Backtest worker task %s failed", task.job_id)
            _set(task.job_id,
                 status="failed",
                 finished_at=_utcnow_iso(),
                 records=[],
                 stats=None,
                 error=str(exc)[:500])
            _prune_old_jobs(task.job_id)
        finally:
            _job_queue.task_done()


def _ensure_workers() -> None:
    global _workers_started
    with _lock:
        if _workers_started:
            return
        for index in range(MAX_BACKTEST_WORKERS):
            worker = threading.Thread(
                target=_worker_loop,
                name=f"backtest-worker-{index + 1}",
                daemon=True,
            )
            worker.start()
            _workers.append(worker)
        _workers_started = True


def submit(
    user_id: str,
    symbol: str,
    start_date: str,
    end_date: str,
    selected_analysts: List[str],
    hold_days: int,
    sample_interval: int,
    config: Dict[str, Any],
) -> str:
    """Submit a backtest job. Returns job_id."""
    validate_sample_interval(sample_interval)
    job_id = uuid4().hex
    _create_job(
        job_id=job_id,
        user_id=user_id,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        selected_analysts=selected_analysts,
        hold_days=hold_days,
        sample_interval=sample_interval,
        status="pending",
        created_at=_utcnow_iso(),
        total_dates=0,
        completed_dates=0,
        records=[],
        stats=None,
        error=None,
    )
    _prune_old_jobs()
    task = _BacktestTask(
        job_id=job_id,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        selected_analysts=selected_analysts,
        hold_days=hold_days,
        sample_interval=sample_interval,
        config=config,
    )
    try:
        _job_queue.put_nowait(task)
    except queue.Full as exc:
        delete_job(job_id, user_id)
        raise BacktestQueueFullError("backtest queue is full; retry later") from exc
    _ensure_workers()
    return job_id
