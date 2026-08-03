"""Offline tests for the calibration service (reliability curve + Brier score).

These tests seed the reports table exactly as the live system persists reports
(completed status, structured probability, and frozen snapshot JSON), then
patch the price fetchers so outcome resolution stays offline and deterministic.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api.database import ReportDB, get_db_ctx, init_db
from api.services import calibration_service as cal
from api.services import report_service


def _result_data(
    *,
    prompt_versions: tuple[str, ...] = ("v1",),
    model_names: tuple[str, ...] = ("gpt-4o-mini",),
) -> dict:
    """Build a result_data dict shaped like a persisted dual-horizon report."""
    roles = {
        f"role-{i}": {
            "resolved_text": f"prompt text {i}",
            "resolved_hash": prompt_versions[i % len(prompt_versions)],
            "resolved_length": 10,
            "injected": True,
        }
        for i in range(3)
    }
    model_snapshot = {
        f"role-{i}": {
            "provider_type": "openai",
            "model_name": model_names[i % len(model_names)],
            "base_url": "https://api.example.com",
            "resolved_via": "binding",
            "fallback_used": False,
            "profile_display_name": None,
            "provider_display_name": None,
        }
        for i in range(3)
    }
    return {
        "mode": "dual_horizon",
        "status": "completed",
        "short_term": {"horizon": "short", "status": "completed"},
        "medium_term": {"horizon": "medium", "status": "completed"},
        "custom_prompt_snapshot": {
            "enabled": True,
            "placement": "prefix",
            "roles": roles,
        },
        "model_config_snapshot": model_snapshot,
    }


def _seed_report(
    *,
    symbol: str,
    trade_date: str,
    probability: float,
    user_id: str,
    prompt_versions: tuple[str, ...] = ("v1",),
    model_names: tuple[str, ...] = ("gpt-4o-mini",),
) -> ReportDB:
    init_db()
    with get_db_ctx() as db:
        report = report_service.create_report(
            db=db,
            symbol=symbol,
            trade_date=trade_date,
            decision="BUY",
            probability=probability,
            result_data=_result_data(
                prompt_versions=prompt_versions,
                model_names=model_names,
            ),
            user_id=user_id,
            report_id=str(uuid4()),
        )
        db.commit()
        return report


def _fake_price_after(entry_price: float) -> float:
    """Return a price_after that yields the given relative outcome."""

    def _resolve(symbol: str, base_date: str, hold_days: int) -> float:
        return entry_price

    return _resolve


def _fake_price_on(entry_price: float) -> float:
    def _resolve(symbol: str, date: str) -> float:
        return entry_price

    return _resolve


def _user_token() -> tuple[str, str]:
    from api.services import auth_service

    init_db()
    now = datetime.now(timezone.utc)
    user_id = str(uuid4())
    email = f"calib-{uuid4().hex[:12]}@test.com"
    with get_db_ctx() as db:
        user = _UserDB(user_id, email, now)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user_id, auth_service.create_access_token(user)


def _UserDB(user_id: str, email: str, now: datetime):
    from api.database import UserDB

    return UserDB(
        id=user_id,
        email=email,
        is_active=True,
        created_at=now,
        updated_at=now,
        last_login_at=now,
    )


@pytest.fixture
def client():
    from api.main import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _reset_calibration_state(monkeypatch):
    """Isolate the module-level cache, concurrency counter and per-IP rate limiter."""
    from api import main as main_mod

    cal._calibration_cache.clear()
    cal._active_calibrations = 0
    main_mod._calibration_rate_hits.clear()
    monkeypatch.setattr(main_mod, "_CALIBRATION_RATE_MAX", 10000)
    yield


class TestReliabilityCurveBucketing:
    def test_buckets_and_rise_rate_are_computed_correctly(self):
        user_id, _ = _user_token()
        # 60-70% bucket: 3 reports, 2 actually rise -> 66.7%
        for symbol, prob, trade_date in [
            ("600519.SH", 0.62, "2024-03-01"),
            ("600519.SH", 0.65, "2024-03-02"),
            ("600519.SH", 0.68, "2024-03-03"),
        ]:
            _seed_report(symbol=symbol, trade_date=trade_date, probability=prob, user_id=user_id)

        def outcome_resolver(report: ReportDB):
            if report.trade_date in ("2024-03-01", "2024-03-02"):
                return True
            return False

        with get_db_ctx() as db:
            result = cal.compute_calibration(
                db,
                user_id=user_id,
                hold_days=5,
                outcome_resolver=outcome_resolver,
            )

        bucket = next(b for b in result["buckets"] if b["bucket"] == "60-70%")
        assert bucket["count"] == 3
        assert bucket["rise_count"] == 2
        assert bucket["rise_rate"] == 66.7
        assert bucket["avg_probability"] == pytest.approx(round((0.62 + 0.65 + 0.68) / 3, 3))
        assert result["sample_size"] == 3
        assert result["skipped_no_outcome"] == 0
        # Empty buckets are still present with null stats.
        assert all(b["rise_rate"] is None for b in result["buckets"] if b["count"] == 0)

    def test_probability_edge_is_bucketed(self):
        user_id, _ = _user_token()
        _seed_report(symbol="600519.SH", trade_date="2024-01-02", probability=0.5, user_id=user_id)
        _seed_report(symbol="600519.SH", trade_date="2024-01-03", probability=1.0, user_id=user_id)

        with get_db_ctx() as db:
            result = cal.compute_calibration(
                db,
                user_id=user_id,
                outcome_resolver=lambda report: True,
            )

        b50 = next(b for b in result["buckets"] if b["bucket"] == "50-60%")
        b80 = next(b for b in result["buckets"] if b["bucket"] == "80+%")
        assert b50["count"] == 1  # 0.5 lands in [0.5, 0.6)
        assert b80["count"] == 1  # 1.0 lands in [0.8, 1.0]
        assert result["sample_size"] == 2

    def test_unresolvable_outcome_is_skipped(self):
        user_id, _ = _user_token()
        _seed_report(symbol="600519.SH", trade_date="2024-01-02", probability=0.6, user_id=user_id)
        _seed_report(symbol="600519.SH", trade_date="2024-01-03", probability=0.9, user_id=user_id)

        with get_db_ctx() as db:
            result = cal.compute_calibration(
                db,
                user_id=user_id,
                outcome_resolver=lambda report: None if report.trade_date == "2024-01-02" else True,
            )

        assert result["sample_size"] == 1
        assert result["skipped_no_outcome"] == 1
        b60 = next(b for b in result["buckets"] if b["bucket"] == "60-70%")
        assert b60["count"] == 0
        b80 = next(b for b in result["buckets"] if b["bucket"] == "80+%")
        assert b80["count"] == 1


class TestBrierScore:
    def test_perfect_calibration_has_near_zero_brier(self):
        user_id, _ = _user_token()
        # probability exactly equals outcome (1.0 -> rise, 0.0 -> fall)
        _seed_report(symbol="600519.SH", trade_date="2024-01-02", probability=1.0, user_id=user_id)
        _seed_report(symbol="600519.SH", trade_date="2024-01-03", probability=0.0, user_id=user_id)

        def resolver(report: ReportDB):
            return report.probability >= 0.5

        with get_db_ctx() as db:
            result = cal.compute_calibration(db, user_id=user_id, outcome_resolver=resolver)

        assert result["brier_score"] == pytest.approx(0.0, abs=1e-6)

    def test_worst_case_brier_is_one(self):
        user_id, _ = _user_token()
        # perfectly wrong predictions
        _seed_report(symbol="600519.SH", trade_date="2024-01-02", probability=1.0, user_id=user_id)
        _seed_report(symbol="600519.SH", trade_date="2024-01-03", probability=0.0, user_id=user_id)

        def resolver(report: ReportDB):
            return report.probability < 0.5  # always the opposite of prediction

        with get_db_ctx() as db:
            result = cal.compute_calibration(db, user_id=user_id, outcome_resolver=resolver)

        assert result["brier_score"] == pytest.approx(1.0)

    def test_empty_sample_has_null_brier(self):
        user_id, _ = _user_token()
        with get_db_ctx() as db:
            result = cal.compute_calibration(db, user_id=user_id, outcome_resolver=lambda r: True)
        assert result["brier_score"] is None
        assert result["sample_size"] == 0


class TestFilters:
    def test_date_range_filters_on_trade_date(self):
        user_id, _ = _user_token()
        _seed_report(symbol="600519.SH", trade_date="2024-01-02", probability=0.6, user_id=user_id)
        _seed_report(symbol="600519.SH", trade_date="2024-06-01", probability=0.6, user_id=user_id)

        with get_db_ctx() as db:
            result = cal.compute_calibration(
                db,
                user_id=user_id,
                start_date="2024-01-01",
                end_date="2024-01-31",
                outcome_resolver=lambda r: True,
            )
        assert result["sample_size"] == 1
        assert result["filters"]["start_date"] == "2024-01-01"
        assert result["filters"]["end_date"] == "2024-01-31"

    def test_symbol_filter(self):
        user_id, _ = _user_token()
        _seed_report(symbol="600519.SH", trade_date="2024-01-02", probability=0.6, user_id=user_id)
        _seed_report(symbol="300750.SZ", trade_date="2024-01-02", probability=0.6, user_id=user_id)

        with get_db_ctx() as db:
            result = cal.compute_calibration(
                db,
                user_id=user_id,
                symbol="600519.SH",
                outcome_resolver=lambda r: True,
            )
        assert result["sample_size"] == 1

    def test_symbol_filter_is_normalized_case_insensitively(self):
        user_id, _ = _user_token()
        _seed_report(symbol="600519.SH", trade_date="2024-01-02", probability=0.6, user_id=user_id)

        with get_db_ctx() as db:
            result = cal.compute_calibration(
                db,
                user_id=user_id,
                symbol="600519.sh",  # lowercase suffix must still match 600519.SH
                outcome_resolver=lambda r: True,
            )
        assert result["sample_size"] == 1
        assert result["filters"]["symbol"] == "600519.sh"

    def test_prompt_version_filter_matches_snapshot_hash(self):
        user_id, _ = _user_token()
        _seed_report(
            symbol="600519.SH",
            trade_date="2024-01-02",
            probability=0.6,
            user_id=user_id,
            prompt_versions=("hash-aaaa",),
        )
        _seed_report(
            symbol="600519.SH",
            trade_date="2024-01-03",
            probability=0.6,
            user_id=user_id,
            prompt_versions=("hash-bbbb",),
        )

        with get_db_ctx() as db:
            result = cal.compute_calibration(
                db,
                user_id=user_id,
                prompt_version="hash-aaaa",
                outcome_resolver=lambda r: True,
            )
        assert result["sample_size"] == 1

    def test_model_filter_matches_snapshot_model_name(self):
        user_id, _ = _user_token()
        _seed_report(
            symbol="600519.SH",
            trade_date="2024-01-02",
            probability=0.6,
            user_id=user_id,
            model_names=("gpt-4o-mini",),
        )
        _seed_report(
            symbol="600519.SH",
            trade_date="2024-01-03",
            probability=0.6,
            user_id=user_id,
            model_names=("deepseek-v3",),
        )

        with get_db_ctx() as db:
            result = cal.compute_calibration(
                db,
                user_id=user_id,
                model="gpt-4o-mini",
                outcome_resolver=lambda r: True,
            )
        assert result["sample_size"] == 1

    def test_user_scoping_excludes_other_users_reports(self):
        user_a, _ = _user_token()
        user_b, _ = _user_token()
        _seed_report(symbol="600519.SH", trade_date="2024-01-02", probability=0.6, user_id=user_a)
        _seed_report(symbol="600519.SH", trade_date="2024-01-03", probability=0.6, user_id=user_b)

        with get_db_ctx() as db:
            result = cal.compute_calibration(db, user_id=user_a, outcome_resolver=lambda r: True)
        assert result["sample_size"] == 1


class TestFilterBeforeLimit:
    def test_snapshot_filter_applies_before_limit_when_scan_covers_matches(self, monkeypatch):
        user_id, _ = _user_token()
        monkeypatch.setattr(cal, "MAX_CALIBRATION_FILTER_SCAN", 20)
        # 10 recent non-matching reports + 3 older matching reports
        for index in range(10):
            _seed_report(
                symbol="600519.SH",
                trade_date=f"2024-04-{index + 1:02d}",
                probability=0.6,
                user_id=user_id,
                prompt_versions=("hash-aaaa",),
            )
        for index in range(3):
            _seed_report(
                symbol="600519.SH",
                trade_date=f"2024-03-{index + 1:02d}",
                probability=0.6,
                user_id=user_id,
                prompt_versions=("hash-bbbb",),
            )

        with get_db_ctx() as db:
            result = cal.compute_calibration(
                db,
                user_id=user_id,
                prompt_version="hash-bbbb",
                limit=3,
                outcome_resolver=lambda r: True,
            )
        assert result["sample_size"] == 3
        assert result["truncated_before_filter"] is False

    def test_scan_cap_reports_truncated_before_filter(self, monkeypatch):
        user_id, _ = _user_token()
        monkeypatch.setattr(cal, "MAX_CALIBRATION_FILTER_SCAN", 5)
        # Older matching reports would exist but fall beyond the scan cap (seeded
        # first, so they are older in created_at).
        for index in range(3):
            _seed_report(
                symbol="600519.SH",
                trade_date=f"2024-03-{index + 1:02d}",
                probability=0.6,
                user_id=user_id,
                prompt_versions=("hash-bbbb",),
            )
        # 5 recent non-matching reports occupy the whole scan window (seeded last).
        for index in range(5):
            _seed_report(
                symbol="600519.SH",
                trade_date=f"2024-04-{index + 1:02d}",
                probability=0.6,
                user_id=user_id,
                prompt_versions=("hash-aaaa",),
            )

        with get_db_ctx() as db:
            result = cal.compute_calibration(
                db,
                user_id=user_id,
                prompt_version="hash-bbbb",
                limit=3,
                outcome_resolver=lambda r: True,
            )
        assert result["sample_size"] == 0
        assert result["truncated_before_filter"] is True

    def test_no_snapshot_filter_never_truncates_before_filter(self):
        user_id, _ = _user_token()
        _seed_report(symbol="600519.SH", trade_date="2024-01-02", probability=0.6, user_id=user_id)
        with get_db_ctx() as db:
            result = cal.compute_calibration(
                db,
                user_id=user_id,
                limit=1,
                outcome_resolver=lambda r: True,
            )
        assert result["truncated_before_filter"] is False


class TestHoldWindowSelection:
    def test_recent_reports_all_skipped_by_hold_window(self, monkeypatch):
        user_id, _ = _user_token()
        # Today fixed to 2026-08-01; hold_days=5 requires 8 calendar days,
        # so any report dated after 2026-07-24 is not yet evaluable.
        today = datetime(2026, 8, 1).date()
        monkeypatch.setattr(cal, "_today", lambda: today)
        _seed_report(symbol="600519.SH", trade_date="2026-07-28", probability=0.6, user_id=user_id)
        _seed_report(symbol="600519.SH", trade_date="2026-07-30", probability=0.6, user_id=user_id)

        with get_db_ctx() as db:
            result = cal.compute_calibration(
                db,
                user_id=user_id,
                hold_days=5,
                outcome_resolver=lambda r: True,
            )

        assert result["sample_size"] == 0
        assert result["skipped_no_outcome"] == 2

    def test_recent_reports_excluded_but_older_evaluable_selected(self, monkeypatch):
        user_id, _ = _user_token()
        today = datetime(2026, 8, 1).date()
        monkeypatch.setattr(cal, "_today", lambda: today)
        _seed_report(symbol="600519.SH", trade_date="2026-07-28", probability=0.6, user_id=user_id)
        _seed_report(symbol="600519.SH", trade_date="2026-06-01", probability=0.6, user_id=user_id)

        with get_db_ctx() as db:
            result = cal.compute_calibration(
                db,
                user_id=user_id,
                hold_days=5,
                outcome_resolver=lambda r: True,
            )

        # The older evaluable report is selected even though a newer one exists;
        # the too-recent one is reported as skipped, not silently dropped.
        assert result["sample_size"] == 1
        assert result["skipped_no_outcome"] == 1


class TestDefaultPriceOutcome:
    def test_default_resolver_uses_strict_price_after(self):
        user_id, _ = _user_token()
        _seed_report(symbol="600519.SH", trade_date="2024-01-02", probability=0.8, user_id=user_id)

        with (
            patch.object(cal, "_get_price_on", side_effect=_fake_price_on(100.0)),
            patch.object(cal, "_get_price_after_strict", side_effect=_fake_price_after(110.0)),
        ):
            with get_db_ctx() as db:
                result = cal.compute_calibration(db, user_id=user_id)
        assert result["sample_size"] == 1
        assert result["buckets"][-1]["rise_count"] == 1
        assert result["buckets"][-1]["rise_rate"] == 100.0


class TestHoldWindowIntegrity:
    def _report(self, trade_date: str):
        from types import SimpleNamespace

        # A detached/expired ORM row can't be read outside a session, so the
        # direct `_resolve_outcome` unit tests use a lightweight stand-in.
        return SimpleNamespace(id="r-1", symbol="600519.SH", trade_date=trade_date)

    def test_hold_window_incomplete_is_skipped_before_price_fetch(self):
        report = self._report("2026-07-30")
        with (
            patch.object(cal, "_today", return_value=datetime(2026, 8, 1).date()),
            patch.object(cal, "_get_price_on", side_effect=_fake_price_on(100.0)),
            patch.object(cal, "_get_price_after_strict", side_effect=_fake_price_after(110.0)),
        ):
            # Even though the patched prices would say "rise", the hold window is
            # not yet complete, so no premature conclusion is drawn.
            assert cal._resolve_outcome(report, 5) is None

    def test_hold_window_complete_after_elapsed(self):
        report = self._report("2026-06-01")
        with (
            patch.object(cal, "_today", return_value=datetime(2026, 8, 1).date()),
            patch.object(cal, "_get_price_on", side_effect=_fake_price_on(100.0)),
            patch.object(cal, "_get_price_after_strict", side_effect=_fake_price_after(110.0)),
        ):
            assert cal._resolve_outcome(report, 5) is True

    def test_exit_non_positive_price_is_skipped(self):
        report = self._report("2024-01-02")
        with (
            patch.object(cal, "_get_price_on", side_effect=_fake_price_on(100.0)),
            patch.object(cal, "_get_price_after_strict", side_effect=_fake_price_after(0.0)),
        ):
            assert cal._resolve_outcome(report, 5) is None

    def test_strict_price_after_refuses_short_series(self):
        short_csv = "date,close\n2024-01-02,100\n2024-01-03,101\n2024-01-04,102\n"
        long_csv = "date,close\n" + "\n".join(
            f"2024-01-{day:02d},{100 + day}" for day in range(2, 12)
        ) + "\n"
        with patch("tradingagents.dataflows.interface.route_to_vendor", return_value=short_csv):
            assert cal._get_price_after_strict("600519.SH", "2024-01-01", 5) is None
        with patch("tradingagents.dataflows.interface.route_to_vendor", return_value=long_csv):
            assert cal._get_price_after_strict("600519.SH", "2024-01-01", 5) == 106.0


class TestApiWiring:
    def test_calibration_route_requires_api_user_dependency(self):
        from api.main import _require_api_user, app

        route = next(r for r in app.routes if getattr(r, "path", None) == "/v1/calibration")
        calls = [dependency.call for dependency in route.dependant.dependencies]
        assert _require_api_user in calls

    def test_calibration_endpoint_returns_curve_and_brier(self, client):
        user_id, token = _user_token()
        _seed_report(symbol="600519.SH", trade_date="2024-01-02", probability=0.8, user_id=user_id)

        with (
            patch.object(cal, "_get_price_on", side_effect=_fake_price_on(100.0)),
            patch.object(cal, "_get_price_after_strict", side_effect=_fake_price_after(110.0)),
        ):
            response = client.get(
                "/v1/calibration",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["sample_size"] == 1
        assert payload["brier_score"] is not None
        assert len(payload["buckets"]) == 5
        assert payload["buckets"][-1]["rise_rate"] == 100.0
        assert payload["filters"]["hold_days"] == 5

    def test_calibration_endpoint_applies_date_and_symbol_params(self, client):
        user_id, token = _user_token()
        _seed_report(symbol="600519.SH", trade_date="2024-01-02", probability=0.6, user_id=user_id)
        _seed_report(symbol="300750.SZ", trade_date="2024-06-01", probability=0.6, user_id=user_id)

        with (
            patch.object(cal, "_get_price_on", side_effect=_fake_price_on(100.0)),
            patch.object(cal, "_get_price_after_strict", side_effect=_fake_price_after(110.0)),
        ):
            response = client.get(
                "/v1/calibration",
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "symbol": "600519.SH",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                    "hold_days": 3,
                },
            )
        assert response.status_code == 200
        payload = response.json()
        assert payload["sample_size"] == 1
        assert payload["filters"]["hold_days"] == 3
        assert payload["filters"]["symbol"] == "600519.SH"


class TestApiParamsAndResourceGuard:
    def test_calibration_endpoint_prompt_version_and_model_params(self, client):
        user_id, token = _user_token()
        _seed_report(
            symbol="600519.SH", trade_date="2024-01-02", probability=0.6,
            user_id=user_id, prompt_versions=("hash-aaaa",), model_names=("gpt-4o-mini",),
        )
        _seed_report(
            symbol="600519.SH", trade_date="2024-01-03", probability=0.6,
            user_id=user_id, prompt_versions=("hash-bbbb",), model_names=("deepseek-v3",),
        )

        with (
            patch.object(cal, "_get_price_on", side_effect=_fake_price_on(100.0)),
            patch.object(cal, "_get_price_after_strict", side_effect=_fake_price_after(110.0)),
        ):
            response = client.get(
                "/v1/calibration",
                headers={"Authorization": f"Bearer {token}"},
                params={"prompt_version": "hash-aaaa", "model": "gpt-4o-mini", "limit": 10},
            )
        assert response.status_code == 200
        payload = response.json()
        assert payload["sample_size"] == 1
        assert payload["filters"]["prompt_version"] == "hash-aaaa"
        assert payload["filters"]["model"] == "gpt-4o-mini"

    def test_calibration_endpoint_limit_param(self, client):
        user_id, token = _user_token()
        for index in range(5):
            _seed_report(symbol="600519.SH", trade_date=f"2024-01-{index + 1:02d}", probability=0.6, user_id=user_id)

        with (
            patch.object(cal, "_get_price_on", side_effect=_fake_price_on(100.0)),
            patch.object(cal, "_get_price_after_strict", side_effect=_fake_price_after(110.0)),
        ):
            response = client.get(
                "/v1/calibration",
                headers={"Authorization": f"Bearer {token}"},
                params={"limit": 2},
            )
        assert response.status_code == 200
        assert response.json()["sample_size"] == 2

    @pytest.mark.parametrize("params", [
        {"hold_days": 61},
        {"hold_days": 0},
        {"limit": 201},
        {"limit": 0},
    ])
    def test_calibration_rejects_invalid_params(self, client, params):
        _, token = _user_token()
        response = client.get(
            "/v1/calibration",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
        )
        assert response.status_code == 422

    def test_calibration_busy_returns_429(self, client, monkeypatch):
        _, token = _user_token()

        def _block(*args, **kwargs):
            raise cal.CalibrationBusyError("校准度计算繁忙，请稍后重试")

        monkeypatch.setattr(cal, "compute_calibration", _block)
        response = client.get(
            "/v1/calibration",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 429
        assert "繁忙" in response.json()["detail"]

    def test_calibration_uses_cache_for_repeated_identical_requests(self, client):
        user_id, token = _user_token()
        _seed_report(symbol="600519.SH", trade_date="2024-01-02", probability=0.6, user_id=user_id)
        headers = {"Authorization": f"Bearer {token}"}

        with (
            patch.object(cal, "_get_price_on", side_effect=_fake_price_on(100.0)),
            patch.object(cal, "_get_price_after_strict", side_effect=_fake_price_after(110.0)),
        ):
            first = client.get("/v1/calibration", headers=headers)
            second = client.get("/v1/calibration", headers=headers)
        assert first.status_code == 200 and second.status_code == 200
        assert first.json() == second.json()
        # The cache key is stored — the second identical request did not recompute.
        assert len(cal._calibration_cache) == 1
