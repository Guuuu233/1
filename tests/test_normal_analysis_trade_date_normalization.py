"""DAV-98: ordinary-analysis trade_date normalization to the latest trading day.

Covers:
- ``_normalize_analysis_trade_date``: weekend roll-back, trading-day identity,
  None/empty default, no forward rounding, calendar-unavailable fallback
- ``/v1/analyze``: a weekend as-of date is normalized before the job runs
- ``/v1/chat/completions`` (non-streaming): an LLM-extracted weekend date is
  normalized
- ``_run_job_inner``: defensive normalization at the job boundary
"""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from datetime import date, datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from api import main
from api.job_store import InMemoryJobStore
from api.services import report_service
from tradingagents.dataflows import trade_calendar as tc


@pytest.fixture(autouse=True)
def _clear_calendar_cache():
    tc.clear_cn_trade_date_cache()
    yield
    tc.clear_cn_trade_date_cache()


def _seed_calendar(days: list[str]) -> None:
    dates = [date.fromisoformat(d) for d in days]
    tc._TRADE_DATES_CACHE["dates"] = dates
    tc._TRADE_DATES_CACHE["dates_set"] = set(dates)
    tc._TRADE_DATES_CACHE["loaded_at"] = 1e18  # never expire in tests


def _boom(*_args, **_kwargs):
    raise RuntimeError("network down")


# ── helper: _normalize_analysis_trade_date ────────────────────────────


def test_normalize_weekend_rolls_back_to_friday():
    _seed_calendar(
        [
            "2026-08-07",  # Friday
            "2026-08-10",  # Monday
            "2026-08-11",
            "2026-08-12",
            "2026-08-13",
        ]
    )
    # Sunday 2026-08-09 -> Friday 2026-08-07
    assert main._normalize_analysis_trade_date("2026-08-09") == "2026-08-07"
    # Saturday 2026-08-08 -> Friday 2026-08-07
    assert main._normalize_analysis_trade_date("2026-08-08") == "2026-08-07"


def test_normalize_trading_day_identity():
    _seed_calendar(
        [
            "2026-08-07",
            "2026-08-10",
            "2026-08-11",
        ]
    )
    assert main._normalize_analysis_trade_date("2026-08-07") == "2026-08-07"
    assert main._normalize_analysis_trade_date("2026-08-10") == "2026-08-10"


def test_normalize_none_defaults_to_latest_trading_day():
    _seed_calendar(
        [
            "2026-08-07",  # Friday
            "2026-08-10",  # Monday
            "2026-08-11",
        ]
    )
    with patch.object(tc, "now_cn", return_value=datetime(2026, 8, 9, 10, 0, tzinfo=tc.CN_TZ)):
        assert main._normalize_analysis_trade_date(None) == "2026-08-07"
        assert main._normalize_analysis_trade_date("") == "2026-08-07"


@pytest.mark.parametrize(
    ("frozen", "expected"),
    [
        (datetime(2026, 8, 12, 3, 0, tzinfo=tc.CN_TZ), "2026-08-11"),
        (datetime(2026, 8, 12, 10, 0, tzinfo=tc.CN_TZ), "2026-08-11"),
        (datetime(2026, 8, 12, 12, 0, tzinfo=tc.CN_TZ), "2026-08-11"),
        (datetime(2026, 8, 12, 16, 0, tzinfo=tc.CN_TZ), "2026-08-12"),
    ],
)
def test_default_date_uses_latest_completed_session_by_market_phase(frozen, expected):
    _seed_calendar(
        [
            "2026-08-07",
            "2026-08-10",
            "2026-08-11",
            "2026-08-12",
            "2026-08-13",
        ]
    )
    with patch.object(tc, "now_cn", return_value=frozen):
        assert main._normalize_analysis_trade_date(None) == expected


def test_explicit_current_trading_date_is_not_rewritten_before_close():
    _seed_calendar(["2026-08-11", "2026-08-12", "2026-08-13"])
    frozen = datetime(2026, 8, 12, 3, 0, tzinfo=tc.CN_TZ)
    with patch.object(tc, "now_cn", return_value=frozen):
        assert (
            main._normalize_analysis_trade_date(
                "2026-08-12", explicit=True
            )
            == "2026-08-12"
        )


def test_scheduled_today_uses_same_default_resolution():
    _seed_calendar(["2026-08-11", "2026-08-12", "2026-08-13"])
    frozen = datetime(2026, 8, 12, 3, 0, tzinfo=tc.CN_TZ)
    with patch.object(tc, "now_cn", return_value=frozen), patch.object(
        main, "cn_today_str", return_value="2026-08-12"
    ):
        assert main._resolve_scheduled_trade_date("2026-08-12") == "2026-08-11"


def test_normalize_never_rounds_forward():
    _seed_calendar(
        [
            "2026-08-13",  # Thursday
            "2026-08-14",  # Friday
            "2026-08-17",  # Monday
        ]
    )
    # Future Saturday 2026-08-15 -> prior Friday, never the following Monday.
    out = main._normalize_analysis_trade_date("2026-08-15")
    assert out == "2026-08-14"
    assert out <= "2026-08-15"


def test_normalize_calendar_unavailable_preserves_explicit_date():
    tc.clear_cn_trade_date_cache()
    with patch.object(tc, "_fetch_cn_trade_dates_from_akshare", side_effect=_boom), \
         patch.object(tc, "_fetch_cn_trade_dates_from_fuyao", side_effect=_boom):
        assert main._normalize_analysis_trade_date(
            "2026-08-09", explicit=True
        ) == "2026-08-09"


def test_normalize_calendar_unavailable_with_none_fails_closed():
    tc.clear_cn_trade_date_cache()
    with patch.object(tc, "_fetch_cn_trade_dates_from_akshare", side_effect=_boom), \
         patch.object(tc, "_fetch_cn_trade_dates_from_fuyao", side_effect=_boom), \
         patch.object(tc, "now_cn", return_value=datetime(2026, 8, 9, 10, 0, tzinfo=tc.CN_TZ)):
        with pytest.raises(tc.TradeCalendarUnavailableError):
            main._normalize_analysis_trade_date(None)


# ── _run_job_inner defensive normalization ────────────────────────────


def test_run_job_inner_defensively_normalizes_weekend_date():
    store = InMemoryJobStore()
    db = MagicMock()
    request = main.AnalyzeRequest(
        symbol="600519.SH",
        trade_date="2026-08-09",  # Sunday
        dry_run=True,
        horizons=["short"],
        selected_analysts=[],
    )

    async def run():
        await main._run_job_inner("job-normalize", request, stream_events=False, save_report=False)

    with (
        patch.object(main, "_job_store_instance", store),
        patch.object(main, "_build_runtime_config", return_value={}),
        patch.object(main, "_resolve_and_freeze_custom_prompts", return_value=({}, False)),
        patch.object(main, "get_db_ctx", return_value=nullcontext(db)),
        patch.object(report_service, "init_report"),
        patch.object(report_service, "update_report_partial"),
    ):
        asyncio.run(run())

    job = store.get_job("job-normalize")
    assert job["status"] == "completed"
    assert job["result"]["trade_date"] == "2026-08-07"


# ── API integration: endpoints normalize the as-of date ───────────────


def _get_client():
    from fastapi.testclient import TestClient

    return TestClient(main.app, raise_server_exceptions=False)


def _auth_unique(client) -> str:
    from api.database import UserDB, get_db_ctx, init_db
    from api.services import auth_service

    init_db()
    email = auth_service.normalize_email(f"dav98-{uuid4().hex[:8]}@test.com")
    now = datetime.now(tc.CN_TZ)
    with get_db_ctx() as db:
        user = auth_service.get_user_by_email(db, email)
        if not user:
            user = UserDB(
                id=str(uuid4()),
                email=email,
                is_active=True,
                created_at=now,
                updated_at=now,
                last_login_at=now,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
    return auth_service.create_access_token(user)


def _wait_job(client, token: str, job_id: str, timeout: float = 5.0) -> dict:
    import time

    headers = {"Authorization": f"Bearer {token}"}
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/v1/jobs/{job_id}", headers=headers)
        status = r.json().get("status")
        if status in ("completed", "failed"):
            break
        time.sleep(0.2)
    r2 = client.get(f"/v1/jobs/{job_id}/result", headers=headers)
    return r2.json()


def test_analyze_endpoint_normalizes_weekend_trade_date():
    client = _get_client()
    token = _auth_unique(client)
    headers = {"Authorization": f"Bearer {token}"}
    r = client.post(
        "/v1/analyze",
        headers=headers,
        json={"symbol": "600519.SH", "trade_date": "2026-08-09", "dry_run": True},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    result = _wait_job(client, token, job_id)
    assert result["status"] == "completed"
    assert result["result"]["trade_date"] == "2026-08-07"


def test_analyze_endpoint_default_uses_completed_session_before_close():
    _seed_calendar(["2026-08-11", "2026-08-12", "2026-08-13"])
    client = _get_client()
    token = _auth_unique(client)
    headers = {"Authorization": f"Bearer {token}"}
    frozen = datetime(2026, 8, 12, 3, 0, tzinfo=tc.CN_TZ)

    with patch.object(tc, "now_cn", return_value=frozen):
        r = client.post(
            "/v1/analyze",
            headers=headers,
            json={"symbol": "600519.SH", "dry_run": True},
        )
    assert r.status_code == 200
    result = _wait_job(client, token, r.json()["job_id"])
    assert result["result"]["trade_date"] == "2026-08-11"


def test_analyze_endpoint_keeps_trading_day_unchanged():
    client = _get_client()
    token = _auth_unique(client)
    headers = {"Authorization": f"Bearer {token}"}
    r = client.post(
        "/v1/analyze",
        headers=headers,
        json={"symbol": "600519.SH", "trade_date": "2026-08-07", "dry_run": True},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    result = _wait_job(client, token, job_id)
    assert result["status"] == "completed"
    assert result["result"]["trade_date"] == "2026-08-07"


def test_chat_completions_normalizes_extracted_weekend_trade_date():
    client = _get_client()
    token = _auth_unique(client)
    headers = {"Authorization": f"Bearer {token}"}
    captured: dict = {}

    async def fake_run_job(job_id, request, *args, **kwargs):
        captured["request"] = request

    with patch(
        "api.main._ai_extract_symbol_and_date",
        return_value=("600519.SH", "2026-08-09", ["short"], [], [], {}),
    ), patch("api.main._run_job", side_effect=fake_run_job):
        r = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "分析600519短线机会"}],
                "stream": False,
                "dry_run": True,
            },
        )
    assert r.status_code == 200
    assert captured["request"].trade_date == "2026-08-07"
