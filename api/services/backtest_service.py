"""
Backtest service — runs historical analysis for a symbol across a date range
and compares each decision against subsequent price performance.

Design: completely non-invasive. Reuses existing TradingAgentsGraph.propagate()
without touching any existing code.
"""
from __future__ import annotations

import json
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
        excess = len(_backtest_jobs) - MAX_RETAINED_BACKTEST_JOBS
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


def _get_price_after(symbol: str, base_date: str, hold_days: int) -> Optional[float]:
    """Fetch closing price hold_days trading days after base_date using akshare."""
    try:
        import akshare as ak
        from tradingagents.dataflows.interface import route_to_vendor
        import pandas as pd

        fmt = "%Y-%m-%d"
        start_dt = datetime.strptime(base_date, fmt)
        # Fetch data starting from base_date + 1 day, extend window for hold_days
        fetch_start = (start_dt + timedelta(days=1)).strftime(fmt)
        fetch_end = (start_dt + timedelta(days=hold_days + 30)).strftime(fmt)

        csv_data = route_to_vendor("get_stock_data", symbol, fetch_start, fetch_end)
        if not csv_data:
            return None

        df = pd.read_csv(pd.io.common.StringIO(csv_data))
        # Find column for close price
        close_cols = [c for c in df.columns if "close" in c.lower() or "收盘" in c]
        date_cols = [c for c in df.columns if "date" in c.lower() or "日期" in c or "time" in c.lower()]
        if not close_cols or not date_cols:
            return None

        df = df.sort_values(date_cols[0]).reset_index(drop=True)
        if len(df) < hold_days:
            hold_days = len(df) - 1
        if hold_days < 1:
            return None
        return float(df[close_cols[0]].iloc[hold_days - 1])
    except Exception:
        return None


def _get_price_on(symbol: str, date: str) -> Optional[float]:
    """Fetch closing price on or just before date."""
    try:
        from tradingagents.dataflows.interface import route_to_vendor
        import pandas as pd

        fmt = "%Y-%m-%d"
        start = (datetime.strptime(date, fmt) - timedelta(days=5)).strftime(fmt)
        csv_data = route_to_vendor("get_stock_data", symbol, start, date)
        if not csv_data:
            return None
        df = pd.read_csv(pd.io.common.StringIO(csv_data))
        close_cols = [c for c in df.columns if "close" in c.lower() or "收盘" in c]
        date_cols = [c for c in df.columns if "date" in c.lower() or "日期" in c or "time" in c.lower()]
        if not close_cols or not date_cols:
            return None
        df = df.sort_values(date_cols[0]).reset_index(drop=True)
        return float(df[close_cols[0]].iloc[-1])
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Core backtest runner
# ──────────────────────────────────────────────────────────────────────────────

def _run_single_analysis(symbol: str, trade_date: str, selected_analysts: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Run one full analysis without SSE. Returns final state dict."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.dataflows.config import set_config

    set_config(config)
    graph = TradingAgentsGraph(
        selected_analysts=selected_analysts,
        debug=False,
        config=config,
    )
    final_state, _ = graph.propagate(symbol, trade_date)
    decision_raw = final_state.get("final_trade_decision", "")
    decision = graph.process_signal(decision_raw)
    return {
        "final_trade_decision": decision_raw,
        "decision": decision,
    }


def _classify_decision(decision: str) -> str:
    """Classify decision as BUY / SELL / HOLD."""
    d = decision.upper()
    if any(k in d for k in ["BUY", "增持", "买入", "BULLISH"]):
        return "BUY"
    if any(k in d for k in ["SELL", "减持", "卖出", "BEARISH"]):
        return "SELL"
    return "HOLD"


def _compute_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute win rate and average return from backtest records."""
    trades = [r for r in records if r.get("action") in ("BUY", "SELL") and r.get("return_pct") is not None]
    if not trades:
        return {"total_signals": 0, "win_rate": None, "avg_return_pct": None, "best_return_pct": None, "worst_return_pct": None}

    wins = 0
    returns = []
    for t in trades:
        ret = t["return_pct"]
        returns.append(ret)
        # Win = positive return for BUY, negative return for SELL
        if t["action"] == "BUY" and ret > 0:
            wins += 1
        elif t["action"] == "SELL" and ret < 0:
            wins += 1

    return {
        "total_signals": len(trades),
        "win_rate": round(wins / len(trades) * 100, 1),
        "avg_return_pct": round(sum(returns) / len(returns), 2),
        "best_return_pct": round(max(returns), 2),
        "worst_return_pct": round(min(returns), 2),
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
                "action": "HOLD",
                "return_pct": None,
                "error": None,
            }
            try:
                analysis = _run_single_analysis(symbol, trade_date, selected_analysts, config)
                action = _classify_decision(analysis["decision"])
                record["action"] = action
                record["decision_summary"] = analysis["final_trade_decision"][:200] if analysis.get("final_trade_decision") else ""

                if action in ("BUY", "SELL"):
                    entry_price = _get_price_on(symbol, trade_date)
                    exit_price = _get_price_after(symbol, trade_date, hold_days)
                    if entry_price and exit_price and entry_price > 0:
                        raw_return = (exit_price - entry_price) / entry_price * 100
                        record["entry_price"] = round(entry_price, 2)
                        record["exit_price"] = round(exit_price, 2)
                        record["return_pct"] = round(raw_return if action == "BUY" else -raw_return, 2)
            except Exception as exc:
                record["error"] = str(exc)[:200]

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
