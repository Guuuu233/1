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


class TestDefaultPriceOutcome:
    def test_default_resolver_uses_price_after(self):
        user_id, _ = _user_token()
        _seed_report(symbol="600519.SH", trade_date="2024-01-02", probability=0.8, user_id=user_id)

        with (
            patch.object(cal, "_get_price_on", side_effect=_fake_price_on(100.0)),
            patch.object(cal, "_get_price_after", side_effect=_fake_price_after(110.0)),
        ):
            with get_db_ctx() as db:
                result = cal.compute_calibration(db, user_id=user_id)
        assert result["sample_size"] == 1
        assert result["buckets"][-1]["rise_count"] == 1
        assert result["buckets"][-1]["rise_rate"] == 100.0


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
            patch.object(cal, "_get_price_after", side_effect=_fake_price_after(110.0)),
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
            patch.object(cal, "_get_price_after", side_effect=_fake_price_after(110.0)),
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
