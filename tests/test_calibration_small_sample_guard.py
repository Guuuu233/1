"""Red tests for calibration small sample guard (DAV-758).

Verifies:
1. Five tiers: n=0, n=1, n=threshold-1, n=threshold, n=threshold+1.
2. Under threshold: brier_score is None, all bucket rise_rates are None,
   sample_sufficient is False, min_sample_size is exposed, and insufficient_reason is given.
3. Distinction between "no sample" (n=0) and "insufficient sample" (0 < n < threshold).
4. At/above threshold: sample_sufficient is True, brier_score is float, rise_rates
   are computed, insufficient_reason is None.
5. API endpoint /v1/calibration exposes sample_sufficient, min_sample_size,
   insufficient_reason, and adheres to the small sample guard.
"""
from __future__ import annotations

from unittest.mock import patch
from uuid import uuid4

import pytest

from api.database import ReportDB, get_db_ctx, init_db
from api.services import calibration_service as cal
from api.services import report_service


def _seed_sample_report(
    *,
    symbol: str = "600519.SH",
    trade_date: str = "2024-01-02",
    probability: float = 0.65,
    user_id: str,
    analysis_status: str = "VALID",
    trade_action: str = "BUY",
) -> ReportDB:
    init_db()
    with get_db_ctx() as db:
        res_data = {
            "mode": "dual_horizon",
            "status": "completed",
            "analysis_status": analysis_status,
            "trade_action": trade_action,
        }
        report = ReportDB(
            id=str(uuid4()),
            user_id=user_id,
            symbol=symbol,
            trade_date=trade_date,
            decision=trade_action,
            probability=probability,
            result_data=res_data,
            status="completed",
            analysis_status=analysis_status,
            trade_action=trade_action,
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        return report


def _seed_n_reports(n: int, user_id: str) -> None:
    """Seed n distinct evaluable reports with valid probability in a single batch."""
    init_db()
    with get_db_ctx() as db:
        reports = []
        for i in range(n):
            month = (i // 25) + 1
            day = (i % 25) + 1
            res_data = {
                "mode": "dual_horizon",
                "status": "completed",
                "analysis_status": "VALID",
                "trade_action": "BUY",
            }
            report = ReportDB(
                id=str(uuid4()),
                user_id=user_id,
                symbol=f"600{i % 999:03d}.SH",
                trade_date=f"2024-{month:02d}-{day:02d}",
                decision="BUY",
                probability=0.55 + (0.01 * (i % 30)),
                result_data=res_data,
                status="completed",
                analysis_status="VALID",
                trade_action="BUY",
            )
            reports.append(report)
        db.add_all(reports)
        db.commit()


class TestSmallSampleGuardTiers:
    """Validate the 5 tiers: n=0, n=1, n=threshold-1, n=threshold, n=threshold+1."""

    def test_tier_n_zero(self):
        """Tier 1: n=0 (no samples) -> sample_sufficient=False, brier_score=None, reason indicates no sample."""
        user_id = str(uuid4())
        with get_db_ctx() as db:
            result = cal.compute_calibration(
                db,
                user_id=user_id,
                outcome_resolver=lambda r: True,
            )

        assert result["sample_size"] == 0
        assert result["sample_sufficient"] is False
        assert result["min_sample_size"] == cal.DEFAULT_MIN_CALIBRATION_SAMPLE_SIZE
        assert result["brier_score"] is None
        assert all(b["rise_rate"] is None for b in result["buckets"])
        assert result["insufficient_reason"] is not None
        assert "无" in result["insufficient_reason"]

    def test_tier_n_one(self):
        """Tier 2: n=1 -> sample_sufficient=False, brier_score=None, reason indicates sample insufficient."""
        user_id = str(uuid4())
        _seed_sample_report(user_id=user_id, probability=0.75, trade_date="2024-01-02")

        with get_db_ctx() as db:
            result = cal.compute_calibration(
                db,
                user_id=user_id,
                outcome_resolver=lambda r: True,
            )

        assert result["sample_size"] == 1
        assert result["sample_sufficient"] is False
        assert result["min_sample_size"] == cal.DEFAULT_MIN_CALIBRATION_SAMPLE_SIZE
        assert result["brier_score"] is None
        assert all(b["rise_rate"] is None for b in result["buckets"])
        assert result["insufficient_reason"] is not None
        # Distinguishable from "无样本"
        assert "不足" in result["insufficient_reason"]
        assert "1" in result["insufficient_reason"]

    def test_tier_n_threshold_minus_one(self):
        """Tier 3: n=threshold-1 -> sample_sufficient=False, brier_score=None."""
        user_id = str(uuid4())
        threshold = cal.DEFAULT_MIN_CALIBRATION_SAMPLE_SIZE
        _seed_n_reports(threshold - 1, user_id=user_id)

        with get_db_ctx() as db:
            result = cal.compute_calibration(
                db,
                user_id=user_id,
                outcome_resolver=lambda r: True,
            )

        assert result["sample_size"] == threshold - 1
        assert result["sample_sufficient"] is False
        assert result["min_sample_size"] == threshold
        assert result["brier_score"] is None
        assert all(b["rise_rate"] is None for b in result["buckets"])
        assert result["insufficient_reason"] is not None
        assert "不足" in result["insufficient_reason"]

    def test_tier_n_threshold(self):
        """Tier 4: n=threshold -> sample_sufficient=True, brier_score is float, rise_rate is computed."""
        user_id = str(uuid4())
        threshold = cal.DEFAULT_MIN_CALIBRATION_SAMPLE_SIZE
        _seed_n_reports(threshold, user_id=user_id)

        with get_db_ctx() as db:
            result = cal.compute_calibration(
                db,
                user_id=user_id,
                outcome_resolver=lambda r: True,
            )

        assert result["sample_size"] == threshold
        assert result["sample_sufficient"] is True
        assert result["min_sample_size"] == threshold
        assert result["brier_score"] is not None
        assert isinstance(result["brier_score"], float)
        assert result["insufficient_reason"] is None
        # Non-empty buckets should have computed rise_rate
        non_empty = [b for b in result["buckets"] if b["count"] > 0]
        assert len(non_empty) > 0
        assert all(b["rise_rate"] is not None for b in non_empty)

    def test_tier_n_threshold_plus_one(self):
        """Tier 5: n=threshold+1 -> sample_sufficient=True, brier_score is float, rise_rate is computed."""
        user_id = str(uuid4())
        threshold = cal.DEFAULT_MIN_CALIBRATION_SAMPLE_SIZE
        _seed_n_reports(threshold + 1, user_id=user_id)

        with get_db_ctx() as db:
            result = cal.compute_calibration(
                db,
                user_id=user_id,
                outcome_resolver=lambda r: True,
            )

        assert result["sample_size"] == threshold + 1
        assert result["sample_sufficient"] is True
        assert result["min_sample_size"] == threshold
        assert result["brier_score"] is not None
        assert isinstance(result["brier_score"], float)
        assert result["insufficient_reason"] is None


class TestSmallSampleDistinctionAndContract:
    """Contract requirements: distinction between no-sample and small-sample, named constant, etc."""

    def test_distinguish_no_sample_vs_insufficient_sample(self):
        """insufficient_reason must clearly distinguish n=0 (no sample) vs 0<n<threshold (small sample)."""
        user_0 = str(uuid4())
        user_1 = str(uuid4())
        _seed_sample_report(user_id=user_1, probability=0.7)

        with get_db_ctx() as db:
            res_0 = cal.compute_calibration(db, user_id=user_0, outcome_resolver=lambda r: True)
            res_1 = cal.compute_calibration(db, user_id=user_1, outcome_resolver=lambda r: True)

        assert res_0["sample_sufficient"] is False
        assert res_1["sample_sufficient"] is False
        assert res_0["insufficient_reason"] != res_1["insufficient_reason"]
        # n=0 reason conveys no sample
        assert ("无" in res_0["insufficient_reason"] or "暂无" in res_0["insufficient_reason"])
        # n=1 reason conveys insufficient sample with counts
        assert "不足" in res_1["insufficient_reason"]

    def test_named_constant_exists_and_meets_agent_rules(self):
        """DEFAULT_MIN_CALIBRATION_SAMPLE_SIZE must be a named constant (AGENTS.md §5)."""
        assert hasattr(cal, "DEFAULT_MIN_CALIBRATION_SAMPLE_SIZE")
        val = getattr(cal, "DEFAULT_MIN_CALIBRATION_SAMPLE_SIZE")
        assert isinstance(val, int)
        assert val == 30

    def test_custom_min_sample_size_override(self):
        """Calling compute_calibration with explicit min_sample_size overrides default."""
        user_id = str(uuid4())
        _seed_sample_report(user_id=user_id, probability=0.7)

        with get_db_ctx() as db:
            # Threshold 1: n=1 is now sufficient
            res_suff = cal.compute_calibration(
                db,
                user_id=user_id,
                min_sample_size=1,
                outcome_resolver=lambda r: True,
            )
            # Threshold 5: n=1 is insufficient
            res_insuff = cal.compute_calibration(
                db,
                user_id=user_id,
                min_sample_size=5,
                outcome_resolver=lambda r: True,
            )

        assert res_suff["sample_sufficient"] is True
        assert res_suff["min_sample_size"] == 1
        assert res_suff["brier_score"] is not None
        assert res_suff["insufficient_reason"] is None

        assert res_insuff["sample_sufficient"] is False
        assert res_insuff["min_sample_size"] == 5
        assert res_insuff["brier_score"] is None
        assert res_insuff["insufficient_reason"] is not None


class TestCalibrationEndpointGuard:
    """Verify HTTP endpoint /v1/calibration applies the small sample guard."""

    def test_endpoint_returns_guard_fields_when_small_sample(self):
        from datetime import datetime, timezone
        from fastapi.testclient import TestClient
        from api.database import UserDB
        from api.main import app
        from api.services import auth_service

        client = TestClient(app)
        user_id = str(uuid4())
        now = datetime.now(timezone.utc)
        with get_db_ctx() as db:
            user = UserDB(id=user_id, email=f"dav758-{uuid4().hex[:6]}@example.com", is_active=True, created_at=now, updated_at=now, last_login_at=now)
            db.add(user)
            db.commit()
            db.refresh(user)

        token = auth_service.create_access_token(user)
        _seed_sample_report(user_id=user_id, probability=0.8)

        with (
            patch.object(cal, "_get_price_on", return_value=100.0),
            patch.object(cal, "_get_price_after_strict", return_value=110.0),
        ):
            resp = client.get("/v1/calibration", headers={"Authorization": f"Bearer {token}"})

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["sample_size"] == 1
        assert payload["sample_sufficient"] is False
        assert payload["min_sample_size"] == 30
        assert payload["insufficient_reason"] is not None
        assert payload["brier_score"] is None
        assert all(b["rise_rate"] is None for b in payload["buckets"])
