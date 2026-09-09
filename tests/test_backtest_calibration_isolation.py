"""Tests for backtest and calibration isolation (D-009 / P1-3).

Verifies:
1. backtest_service does not shorten hold_days on short series (no pseudo T+N).
2. INVALID_RUN / DATA_ERROR / ABSTAIN / PARTIAL do not enter backtest win_rate.
3. WAIT / NO_TRADE do not collapse into HOLD win_rate samples.
4. VALID + BUY/SELL with complete window computes returns accurately.
5. Every record preserves analysis_status, trade_action, price_basis,
   entry_price_as_of, exit_price_as_of, and outcome_status.
6. calibration_service exclusion_stats and incomplete outcome accounting.
7. Insufficient sample does not fabricate metrics.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from api.services import backtest_service as bt
from api.services import calibration_service as cal


def _fake_price_after(price: float | None):
    def _resolve(symbol: str, base_date: str, hold_days: int, *args, **kwargs) -> float | None:
        return price
    return _resolve


def _fake_price_on(price: float | None):
    def _resolve(symbol: str, date: str, *args, **kwargs) -> float | None:
        return price
    return _resolve


class TestBacktestHoldDaysStrictness:
    def test_get_price_after_refuses_short_series_without_truncating_hold_days(self):
        """Short price series (< hold_days) must return None, NOT shorten hold_days."""
        short_csv = "date,close\n2024-01-02,100\n2024-01-03,101\n"
        with patch("tradingagents.dataflows.interface.route_to_vendor", return_value=short_csv):
            # hold_days=5 but only 2 rows available
            result = bt._get_price_after("600519.SH", "2024-01-01", 5)
            assert result is None, "Must return None instead of shortening hold_days to len(df)-1"

    def test_get_price_after_returns_price_when_series_sufficient(self):
        """When price series has >= hold_days rows, return exact price at index hold_days - 1."""
        csv_data = "date,close\n" + "\n".join(f"2024-01-{i:02d},{100 + i}" for i in range(2, 10))
        with patch("tradingagents.dataflows.interface.route_to_vendor", return_value=csv_data):
            result = bt._get_price_after("600519.SH", "2024-01-01", 5)
            assert result == 106.0

    def test_backtest_incomplete_series_marks_outcome_incomplete(self, monkeypatch):
        """When hold window price is unavailable (short series), outcome_status is incomplete."""
        job_id = "test-job-incomplete"
        bt._create_job(job_id=job_id, user_id="u1", status="pending")

        analysis_mock = {
            "decision": "BUY",
            "final_trade_decision": "BUY",
            "analysis_status": "VALID",
            "trade_action": "BUY",
        }

        with (
            patch.object(bt, "_get_trading_dates", return_value=["2024-01-02"]),
            patch.object(bt, "_run_single_analysis", return_value=analysis_mock),
            patch.object(bt, "_get_price_on", side_effect=_fake_price_on(100.0)),
            patch.object(bt, "_get_price_after", side_effect=_fake_price_after(None)),
        ):
            bt._run_backtest(
                job_id=job_id,
                symbol="600519.SH",
                start_date="2024-01-02",
                end_date="2024-01-02",
                selected_analysts=["market"],
                hold_days=5,
                sample_interval=1,
                config={},
            )

        job = bt.get_job(job_id, "u1")
        assert job is not None
        assert job["status"] == "completed"
        assert len(job["records"]) == 1
        record = job["records"][0]
        assert record["outcome_status"] == "incomplete"
        assert record["return_pct"] is None
        assert record["analysis_status"] == "VALID"
        assert record["trade_action"] == "BUY"
        assert record["price_basis"] == "vendor_qfq"
        assert record["price_basis"] != "raw"
        assert record["entry_price"] == 100.0
        assert record["entry_price_as_of"] == "2024-01-02"

        stats = job["stats"]
        assert stats["total_signals"] == 0
        assert stats["win_rate"] is None
        assert stats["excluded_incomplete"] == 1


class TestBacktestSemanticExclusions:
    def test_invalid_run_with_buy_text_is_excluded_from_directional_stats(self):
        """INVALID_RUN must not become a BUY/HOLD signal even if raw text contains BUY."""
        job_id = "test-job-invalid"
        bt._create_job(job_id=job_id, user_id="u1", status="pending")

        analysis_mock = {
            "decision": "BUY",
            "final_trade_decision": "BUY 强烈推荐买入（但运行实际失败）",
            "analysis_status": "INVALID_RUN",
            "trade_action": "NO_TRADE",
            "price_basis": "vendor_qfq",
        }

        with (
            patch.object(bt, "_get_trading_dates", return_value=["2024-01-02"]),
            patch.object(bt, "_run_single_analysis", return_value=analysis_mock),
            patch.object(bt, "_get_price_on", side_effect=_fake_price_on(100.0)),
            patch.object(bt, "_get_price_after", side_effect=_fake_price_after(110.0)),
        ):
            bt._run_backtest(
                job_id=job_id,
                symbol="600519.SH",
                start_date="2024-01-02",
                end_date="2024-01-02",
                selected_analysts=["market"],
                hold_days=5,
                sample_interval=1,
                config={},
            )

        job = bt.get_job(job_id, "u1")
        assert job is not None
        record = job["records"][0]
        assert record["analysis_status"] == "INVALID_RUN"
        assert record["trade_action"] == "NO_TRADE"
        assert record["action"] == "NO_TRADE"
        assert record["return_pct"] is None
        assert "excluded" in record["outcome_status"]

        stats = job["stats"]
        assert stats["total_signals"] == 0
        assert stats["win_rate"] is None
        assert stats["excluded_invalid"] >= 1

    def test_wait_trade_action_is_excluded_and_not_counted_as_hold_trade(self):
        """WAIT action must not collapse into a HOLD trade sample."""
        job_id = "test-job-wait"
        bt._create_job(job_id=job_id, user_id="u1", status="pending")

        analysis_mock = {
            "decision": "WAIT",
            "final_trade_decision": "WAIT 观望等待确认",
            "analysis_status": "VALID",
            "trade_action": "WAIT",
            "price_basis": "vendor_qfq",
        }

        with (
            patch.object(bt, "_get_trading_dates", return_value=["2024-01-02"]),
            patch.object(bt, "_run_single_analysis", return_value=analysis_mock),
            patch.object(bt, "_get_price_on", side_effect=_fake_price_on(100.0)),
            patch.object(bt, "_get_price_after", side_effect=_fake_price_after(110.0)),
        ):
            bt._run_backtest(
                job_id=job_id,
                symbol="600519.SH",
                start_date="2024-01-02",
                end_date="2024-01-02",
                selected_analysts=["market"],
                hold_days=5,
                sample_interval=1,
                config={},
            )

        job = bt.get_job(job_id, "u1")
        assert job is not None
        record = job["records"][0]
        assert record["trade_action"] == "WAIT"
        assert record["action"] == "WAIT"
        assert record["return_pct"] is None
        assert "excluded" in record["outcome_status"]

        stats = job["stats"]
        assert stats["total_signals"] == 0
        assert stats["win_rate"] is None
        assert stats["excluded_wait_or_no_trade"] >= 1 or stats["excluded_no_trade"] >= 1

    def test_single_analysis_exception_logs_and_marks_invalid_without_aborting_job(self):
        """When an exception occurs during single analysis, record is marked INVALID_RUN."""
        job_id = "test-job-exc"
        bt._create_job(job_id=job_id, user_id="u1", status="pending")

        with (
            patch.object(bt, "_get_trading_dates", return_value=["2024-01-02"]),
            patch.object(bt, "_run_single_analysis", side_effect=RuntimeError("Provider 502 error")),
        ):
            bt._run_backtest(
                job_id=job_id,
                symbol="600519.SH",
                start_date="2024-01-02",
                end_date="2024-01-02",
                selected_analysts=["market"],
                hold_days=5,
                sample_interval=1,
                config={},
            )

        job = bt.get_job(job_id, "u1")
        assert job is not None
        assert job["status"] == "completed"
        record = job["records"][0]
        assert record["analysis_status"] == "INVALID_RUN"
        assert record["trade_action"] == "NO_TRADE"
        assert record["action"] == "NO_TRADE"
        assert record["outcome_status"] == "excluded_invalid"
        assert "502" in str(record["error"])

        stats = job["stats"]
        assert stats["excluded_invalid"] == 1
        assert stats["total_signals"] == 0

    def test_valid_buy_and_sell_with_complete_window_computes_returns(self):
        """VALID BUY/SELL with complete price window computes returns correctly."""
        records = [
            {
                "date": "2024-01-02",
                "action": "BUY",
                "trade_action": "BUY",
                "analysis_status": "VALID",
                "price_basis": "vendor_qfq",
                "entry_price": 100.0,
                "entry_price_as_of": "2024-01-02",
                "exit_price": 110.0,
                "exit_price_as_of": "2024-01-09",
                "return_pct": 10.0,
                "outcome_status": "ok",
            },
            {
                "date": "2024-01-10",
                "action": "SELL",
                "trade_action": "SELL",
                "analysis_status": "VALID",
                "price_basis": "vendor_qfq",
                "entry_price": 100.0,
                "entry_price_as_of": "2024-01-10",
                "exit_price": 90.0,
                "exit_price_as_of": "2024-01-17",
                "return_pct": 10.0,  # Short profit: (100 - 90)/100 = +10%
                "outcome_status": "ok",
            },
        ]
        stats = bt._compute_stats(records)
        assert stats["total_signals"] == 2
        assert stats["win_rate"] == 100.0
        assert stats["avg_return_pct"] == 10.0
        assert stats["best_return_pct"] == 10.0
        assert stats["worst_return_pct"] == 10.0
        assert stats["excluded_total"] == 0

    def test_compute_stats_breakdown_with_mixed_records(self):
        records = [
            # 1. Valid winning BUY
            {"action": "BUY", "trade_action": "BUY", "analysis_status": "VALID", "return_pct": 5.0, "outcome_status": "ok"},
            # 2. Valid losing BUY
            {"action": "BUY", "trade_action": "BUY", "analysis_status": "VALID", "return_pct": -3.0, "outcome_status": "ok"},
            # 3. Incomplete outcome BUY
            {"action": "BUY", "trade_action": "BUY", "analysis_status": "VALID", "return_pct": None, "outcome_status": "incomplete"},
            # 4. INVALID_RUN
            {"action": "NO_TRADE", "trade_action": "NO_TRADE", "analysis_status": "INVALID_RUN", "return_pct": None, "outcome_status": "excluded_invalid"},
            # 5. ABSTAIN
            {"action": "NO_TRADE", "trade_action": "NO_TRADE", "analysis_status": "ABSTAIN", "return_pct": None, "outcome_status": "excluded_abstain"},
            # 6. WAIT
            {"action": "WAIT", "trade_action": "WAIT", "analysis_status": "VALID", "return_pct": None, "outcome_status": "excluded_no_trade"},
            # 7. Explicit HOLD
            {"action": "HOLD", "trade_action": "HOLD", "analysis_status": "VALID", "return_pct": None, "outcome_status": "excluded_hold"},
        ]
        stats = bt._compute_stats(records)
        assert stats["total_signals"] == 2
        assert stats["win_rate"] == 50.0
        assert stats["excluded_invalid"] == 1
        assert stats["excluded_abstain"] == 1
        assert stats["excluded_no_trade"] == 1
        assert stats["excluded_incomplete"] == 1
        assert stats["excluded_hold"] == 1
        assert stats["excluded_total"] == 5

    def test_classify_decision_semantics(self):
        """Test _classify_decision mapping across strings and structured dicts."""
        assert bt._classify_decision("BUY") == "BUY"
        assert bt._classify_decision("买入") == "BUY"
        assert bt._classify_decision("SELL") == "SELL"
        assert bt._classify_decision("减持") == "SELL"
        assert bt._classify_decision("WAIT") == "WAIT"
        assert bt._classify_decision("观望") == "WAIT"
        assert bt._classify_decision("NO_TRADE") == "NO_TRADE"
        assert bt._classify_decision("INVALID_RUN") == "NO_TRADE"
        assert bt._classify_decision("ABSTAIN") == "NO_TRADE"
        assert bt._classify_decision("HOLD") == "HOLD"
        assert bt._classify_decision("中性") == "HOLD"
        # Unknown should NOT collapse into HOLD
        assert bt._classify_decision("UNKNOWN_RANDOM_TEXT") == "NO_TRADE"

        # Dict input
        assert bt._classify_decision({"analysis_status": "INVALID_RUN", "trade_action": "BUY"}) == "NO_TRADE"
        assert bt._classify_decision({"analysis_status": "ABSTAIN", "trade_action": "BUY"}) == "NO_TRADE"
        assert bt._classify_decision({"analysis_status": "VALID", "trade_action": "BUY"}) == "BUY"
        assert bt._classify_decision({"analysis_status": "VALID", "trade_action": "WAIT"}) == "WAIT"


class TestCalibrationIsolationIntegrity:
    def _seed_report(self, **kwargs):
        from api.database import get_db_ctx, init_db
        from api.services import report_service
        init_db()
        with get_db_ctx() as db:
            rd = {
                "status": "completed",
                "analysis_status": kwargs.get("analysis_status"),
                "trade_action": kwargs.get("trade_action"),
            }
            report = report_service.create_report(
                db=db,
                symbol=kwargs["symbol"],
                trade_date=kwargs["trade_date"],
                decision=kwargs.get("trade_action") or "BUY",
                probability=kwargs.get("probability", 0.7),
                result_data=rd,
                user_id=kwargs["user_id"],
                report_id=str(uuid4()),
            )
            report.probability = kwargs.get("probability", 0.7)
            report.analysis_status = kwargs.get("analysis_status")
            report.trade_action = kwargs.get("trade_action")
            db.commit()
            db.refresh(report)
            return report

    def teardown_method(self):
        from api.database import get_db_ctx, ReportDB, UserDB
        with get_db_ctx() as db:
            db.query(ReportDB).delete()
            db.query(UserDB).delete()
            db.commit()

    def test_calibration_excludes_invalid_abstain_wait_and_incomplete(self):
        from api.database import get_db_ctx, init_db, UserDB
        init_db()
        now = datetime.now(timezone.utc)
        user_id = str(uuid4())
        with get_db_ctx() as db:
            user = UserDB(id=user_id, email=f"cal-{uuid4().hex[:8]}@t.com", is_active=True, created_at=now, updated_at=now, last_login_at=now)
            db.add(user)
            db.commit()

        # 1. Eligible VALID directional report
        self._seed_report(symbol="600519.SH", trade_date="2024-01-02", probability=0.8, user_id=user_id, analysis_status="VALID", trade_action="BUY")
        # 2. INVALID_RUN report
        self._seed_report(symbol="600519.SH", trade_date="2024-01-03", probability=0.8, user_id=user_id, analysis_status="INVALID_RUN", trade_action="NO_TRADE")
        # 3. ABSTAIN report
        self._seed_report(symbol="600519.SH", trade_date="2024-01-04", probability=0.8, user_id=user_id, analysis_status="ABSTAIN", trade_action="NO_TRADE")
        # 4. WAIT report
        self._seed_report(symbol="600519.SH", trade_date="2024-01-05", probability=0.8, user_id=user_id, analysis_status="VALID", trade_action="WAIT")

        with get_db_ctx() as db:
            res = cal.compute_calibration(
                db,
                user_id=user_id,
                hold_days=5,
                outcome_resolver=lambda r: True,
            )

        assert res["sample_size"] == 1
        assert res["excluded_invalid"] >= 1
        assert res["excluded_abstain"] >= 1
        assert res["excluded_no_trade"] >= 1
        assert "excluded_incomplete_outcome" in res or "skipped_no_outcome" in res
        assert res["price_basis"] == "vendor_qfq"
        assert res["price_basis"] != "raw"

    def test_calibration_insufficient_sample_returns_none_metrics(self):
        from api.database import get_db_ctx, init_db, UserDB
        init_db()
        now = datetime.now(timezone.utc)
        user_id = str(uuid4())
        with get_db_ctx() as db:
            user = UserDB(id=user_id, email=f"cal-{uuid4().hex[:8]}@t.com", is_active=True, created_at=now, updated_at=now, last_login_at=now)
            db.add(user)
            db.commit()

        # Only an INVALID report is present (0 eligible samples)
        self._seed_report(symbol="600519.SH", trade_date="2024-01-03", probability=0.8, user_id=user_id, analysis_status="INVALID_RUN", trade_action="NO_TRADE")

        with get_db_ctx() as db:
            res = cal.compute_calibration(
                db,
                user_id=user_id,
                hold_days=5,
                outcome_resolver=lambda r: True,
            )

        assert res["sample_size"] == 0
        assert res["brier_score"] is None
        assert all(b["rise_rate"] is None for b in res["buckets"])
        assert res["excluded_invalid"] >= 1


class TestPriceBasisSemantics:
    """DAV-606: verify price_basis正名: vendor_qfq replaces raw default."""

    def test_constants_defined(self):
        """Named constants must be defined and match allowed values."""
        assert getattr(bt, "PRICE_BASIS_VENDOR_QFQ", None) == "vendor_qfq"
        assert getattr(bt, "PRICE_BASIS_UNSPECIFIED", None) == "unspecified"
        assert getattr(bt, "PRICE_BASIS_RAW", None) == "raw"
        assert getattr(bt, "PRICE_BASIS_PIT_RAW", None) == "pit_raw"
        assert getattr(bt, "PRICE_BASIS_PIT_ADJUSTED", None) == "pit_adjusted"
        assert getattr(cal, "PRICE_BASIS_VENDOR_QFQ", None) == "vendor_qfq"

    def test_single_analysis_defaults_to_vendor_qfq_and_never_raw(self):
        """_run_single_analysis must default price_basis to vendor_qfq, never raw."""
        with patch("tradingagents.graph.trading_graph.TradingAgentsGraph") as mock_graph_cls:
            mock_graph = MagicMock()
            mock_graph.propagate.return_value = ({"final_trade_decision": "BUY"}, {})
            mock_graph.process_signal.return_value = "BUY"
            mock_graph_cls.return_value = mock_graph

            res = bt._run_single_analysis("600519.SH", "2024-01-02", ["market"], {})
            assert res["price_basis"] == "vendor_qfq"
            assert res["price_basis"] != "raw"

    def test_single_analysis_preserves_explicit_price_basis(self):
        """_run_single_analysis preserves explicit caller-provided price_basis."""
        with patch("tradingagents.graph.trading_graph.TradingAgentsGraph") as mock_graph_cls:
            mock_graph = MagicMock()
            mock_graph.propagate.return_value = (
                {"final_trade_decision": "BUY", "price_basis": "unspecified"},
                {},
            )
            mock_graph.process_signal.return_value = "BUY"
            mock_graph_cls.return_value = mock_graph

            res = bt._run_single_analysis("600519.SH", "2024-01-02", ["market"], {})
            assert res["price_basis"] == "unspecified"

    def test_backtest_run_defaults_to_vendor_qfq_when_analysis_omits_price_basis(self):
        """_run_backtest record must default price_basis to vendor_qfq and never raw."""
        job_id = "test-job-price-basis-default"
        bt._create_job(job_id=job_id, user_id="u1", status="pending")

        analysis_mock = {
            "decision": "BUY",
            "final_trade_decision": "BUY",
            "analysis_status": "VALID",
            "trade_action": "BUY",
            # price_basis omitted
        }

        with (
            patch.object(bt, "_get_trading_dates", return_value=["2024-01-02"]),
            patch.object(bt, "_run_single_analysis", return_value=analysis_mock),
            patch.object(bt, "_get_price_on", side_effect=_fake_price_on(100.0)),
            patch.object(bt, "_get_price_after", side_effect=_fake_price_after(110.0)),
        ):
            bt._run_backtest(
                job_id=job_id,
                symbol="600519.SH",
                start_date="2024-01-02",
                end_date="2024-01-02",
                selected_analysts=["market"],
                hold_days=5,
                sample_interval=1,
                config={},
            )

        job = bt.get_job(job_id, "u1")
        assert job is not None
        record = job["records"][0]
        assert record["price_basis"] == "vendor_qfq"
        assert record["price_basis"] != "raw"

    def test_backtest_run_preserves_explicit_price_basis(self):
        """_run_backtest preserves explicit price_basis when declared by analysis."""
        job_id = "test-job-price-basis-explicit"
        bt._create_job(job_id=job_id, user_id="u1", status="pending")

        analysis_mock = {
            "decision": "BUY",
            "final_trade_decision": "BUY",
            "analysis_status": "VALID",
            "trade_action": "BUY",
            "price_basis": "unspecified",
        }

        with (
            patch.object(bt, "_get_trading_dates", return_value=["2024-01-02"]),
            patch.object(bt, "_run_single_analysis", return_value=analysis_mock),
            patch.object(bt, "_get_price_on", side_effect=_fake_price_on(100.0)),
            patch.object(bt, "_get_price_after", side_effect=_fake_price_after(110.0)),
        ):
            bt._run_backtest(
                job_id=job_id,
                symbol="600519.SH",
                start_date="2024-01-02",
                end_date="2024-01-02",
                selected_analysts=["market"],
                hold_days=5,
                sample_interval=1,
                config={},
            )

        job = bt.get_job(job_id, "u1")
        assert job is not None
        record = job["records"][0]
        assert record["price_basis"] == "unspecified"

    def test_calibration_summary_defaults_to_vendor_qfq_and_never_raw(self):
        """Calibration output price_basis must be vendor_qfq and never raw."""
        from api.database import get_db_ctx, init_db, UserDB
        init_db()
        now = datetime.now(timezone.utc)
        user_id = str(uuid4())
        with get_db_ctx() as db:
            user = UserDB(id=user_id, email=f"cal-{uuid4().hex[:8]}@t.com", is_active=True, created_at=now, updated_at=now, last_login_at=now)
            db.add(user)
            db.commit()

            res = cal.compute_calibration(
                db,
                user_id=user_id,
                hold_days=5,
                outcome_resolver=lambda r: True,
            )

        assert res["price_basis"] == "vendor_qfq"
        assert res["price_basis"] != "raw"


class TestV02MultiHorizonReadonlyEvaluationAndIsolation:
    """V-02 tests: multi-horizon readonly evaluation, profile routing, and isolation."""

    def _seed_custom_report(self, user_id: str, **kwargs) -> ReportDB:
        from api.database import get_db_ctx, init_db, ReportDB
        init_db()
        with get_db_ctx() as db:
            rd = kwargs.get("result_data", {})
            report_id = str(uuid4())
            rep = ReportDB(
                id=report_id,
                user_id=user_id,
                symbol=kwargs.get("symbol", "600519.SH"),
                trade_date=kwargs.get("trade_date", "2024-01-02"),
                status="completed",
                decision=kwargs.get("trade_action", "BUY"),
                direction=kwargs.get("direction", "BULL"),
                probability=kwargs.get("probability", 0.8),
                confidence=kwargs.get("confidence", 85),
                analysis_status=kwargs.get("analysis_status", "VALID"),
                trade_action=kwargs.get("trade_action", "BUY"),
                result_data=rd,
            )
            db.add(rep)
            db.commit()
            db.refresh(rep)
            return rep

    def teardown_method(self):
        from api.database import get_db_ctx, ReportDB, UserDB
        with get_db_ctx() as db:
            db.query(ReportDB).delete()
            db.query(UserDB).delete()
            db.commit()

    def test_explicit_short_profile_routes_to_t10_and_isolates_pool(self):
        """Specifying horizon='short' routes hold_days to 10 and excludes medium and legacy samples."""
        from api.database import get_db_ctx, init_db, UserDB
        init_db()
        user_id = str(uuid4())
        now = datetime.now(timezone.utc)
        with get_db_ctx() as db:
            user = UserDB(id=user_id, email=f"v02-{uuid4().hex[:8]}@t.com", is_active=True, created_at=now, updated_at=now, last_login_at=now)
            db.add(user)
            db.commit()

        # 1. Short report (T+10)
        self._seed_custom_report(
            user_id,
            trade_date="2024-01-02",
            probability=0.8,
            analysis_status="VALID",
            trade_action="BUY",
            result_data={
                "horizon": "short",
                "horizon_run_metadata": {
                    "resolved": ["short"],
                    "profile_id": "horizon_profile_v1",
                    "resolution_source": "explicit",
                },
            },
        )
        # 2. Medium report (T+40)
        self._seed_custom_report(
            user_id,
            trade_date="2024-01-03",
            probability=0.75,
            analysis_status="VALID",
            trade_action="BUY",
            result_data={
                "horizon": "medium",
                "horizon_run_metadata": {
                    "resolved": ["medium"],
                    "profile_id": "horizon_profile_v1",
                    "resolution_source": "explicit",
                },
            },
        )
        # 3. Legacy report (no horizon metadata)
        self._seed_custom_report(
            user_id,
            trade_date="2024-01-04",
            probability=0.7,
            analysis_status="VALID",
            trade_action="BUY",
            result_data={},
        )

        with get_db_ctx() as db:
            res = cal.compute_calibration(
                db,
                user_id=user_id,
                horizon="short",
                outcome_resolver=lambda r: True,
            )

        assert res["sample_size"] == 1
        assert res["filters"]["hold_days"] == 10
        assert res["horizon"] == "short"
        assert res["profile_id"] == "horizon_profile_v1"
        assert res["excluded_counts"]["mismatched_horizon"] >= 2

    def test_explicit_medium_profile_routes_to_t40_and_isolates_pool(self):
        """Specifying horizon='medium' routes hold_days to 40 and excludes short and legacy samples."""
        from api.database import get_db_ctx, init_db, UserDB
        init_db()
        user_id = str(uuid4())
        now = datetime.now(timezone.utc)
        with get_db_ctx() as db:
            user = UserDB(id=user_id, email=f"v02-{uuid4().hex[:8]}@t.com", is_active=True, created_at=now, updated_at=now, last_login_at=now)
            db.add(user)
            db.commit()

        # 1. Short report (T+10)
        self._seed_custom_report(
            user_id,
            trade_date="2024-01-02",
            probability=0.8,
            analysis_status="VALID",
            trade_action="BUY",
            result_data={
                "horizon": "short",
                "horizon_run_metadata": {
                    "resolved": ["short"],
                    "profile_id": "horizon_profile_v1",
                    "resolution_source": "explicit",
                },
            },
        )
        # 2. Medium report (T+40)
        self._seed_custom_report(
            user_id,
            trade_date="2024-01-03",
            probability=0.75,
            analysis_status="VALID",
            trade_action="BUY",
            result_data={
                "horizon": "medium",
                "horizon_run_metadata": {
                    "resolved": ["medium"],
                    "profile_id": "horizon_profile_v1",
                    "resolution_source": "explicit",
                },
            },
        )
        # 3. Legacy report (no horizon metadata)
        self._seed_custom_report(
            user_id,
            trade_date="2024-01-04",
            probability=0.7,
            analysis_status="VALID",
            trade_action="BUY",
            result_data={},
        )

        with get_db_ctx() as db:
            res = cal.compute_calibration(
                db,
                user_id=user_id,
                horizon="medium",
                outcome_resolver=lambda r: True,
            )

        assert res["sample_size"] == 1
        assert res["filters"]["hold_days"] == 40
        assert res["horizon"] == "medium"
        assert res["profile_id"] == "horizon_profile_v1"
        assert res["excluded_counts"]["mismatched_horizon"] >= 2

    def test_default_legacy_preserves_t5_and_rejects_explicit_t10_t40_mix(self):
        """Default call without horizon stays at T+5 and excludes explicit single-horizon medium/short."""
        from api.database import get_db_ctx, init_db, UserDB
        init_db()
        user_id = str(uuid4())
        now = datetime.now(timezone.utc)
        with get_db_ctx() as db:
            user = UserDB(id=user_id, email=f"v02-{uuid4().hex[:8]}@t.com", is_active=True, created_at=now, updated_at=now, last_login_at=now)
            db.add(user)
            db.commit()

        # 1. Explicit Medium report (T+40)
        self._seed_custom_report(
            user_id,
            trade_date="2024-01-02",
            probability=0.75,
            analysis_status="VALID",
            trade_action="BUY",
            result_data={
                "horizon": "medium",
                "horizon_run_metadata": {
                    "resolved": ["medium"],
                    "profile_id": "horizon_profile_v1",
                    "resolution_source": "explicit",
                },
            },
        )
        # 2. Legacy report (compatible with default T+5)
        self._seed_custom_report(
            user_id,
            trade_date="2024-01-03",
            probability=0.7,
            analysis_status="VALID",
            trade_action="BUY",
            result_data={},
        )

        with get_db_ctx() as db:
            res = cal.compute_calibration(
                db,
                user_id=user_id,
                outcome_resolver=lambda r: True,
            )

        assert res["sample_size"] == 1
        assert res["filters"]["hold_days"] == 5
        assert res["horizon"] is None
        assert res["excluded_counts"]["mismatched_horizon"] >= 1

    def test_dual_horizon_reports_admitted_to_respective_slices(self):
        """Dual-horizon reports are admitted under both short and medium evaluations with correct slice probability."""
        from api.database import get_db_ctx, init_db, UserDB
        init_db()
        user_id = str(uuid4())
        now = datetime.now(timezone.utc)
        with get_db_ctx() as db:
            user = UserDB(id=user_id, email=f"v02-{uuid4().hex[:8]}@t.com", is_active=True, created_at=now, updated_at=now, last_login_at=now)
            db.add(user)
            db.commit()

        self._seed_custom_report(
            user_id,
            trade_date="2024-01-02",
            probability=0.5,
            analysis_status="VALID",
            trade_action="BUY",
            result_data={
                "mode": "dual_horizon",
                "short_term": {
                    "horizon": "short",
                    "status": "completed",
                    "probability": 0.85,
                    "analysis_status": "VALID",
                    "trade_action": "BUY",
                },
                "medium_term": {
                    "horizon": "medium",
                    "status": "completed",
                    "probability": 0.65,
                    "analysis_status": "VALID",
                    "trade_action": "BUY",
                },
                "horizon_run_metadata": {
                    "resolved": ["short", "medium"],
                    "profile_id": "horizon_profile_v1",
                },
            },
        )

        with get_db_ctx() as db:
            res_short = cal.compute_calibration(
                db,
                user_id=user_id,
                horizon="short",
                outcome_resolver=lambda r: True,
            )
            res_medium = cal.compute_calibration(
                db,
                user_id=user_id,
                horizon="medium",
                outcome_resolver=lambda r: True,
            )

        assert res_short["sample_size"] == 1
        assert res_short["filters"]["hold_days"] == 10
        # 85% lands in 80+% bucket
        b80_short = next(b for b in res_short["buckets"] if b["bucket"] == "80+%")
        assert b80_short["count"] == 1

        assert res_medium["sample_size"] == 1
        assert res_medium["filters"]["hold_days"] == 40
        # 65% lands in 60-70% bucket
        b60_med = next(b for b in res_medium["buckets"] if b["bucket"] == "60-70%")
        assert b60_med["count"] == 1

    def test_invalid_profile_and_corrupted_horizon_excluded_with_counts(self):
        """Corrupted horizon or profile are excluded and accounted for under invalid_profile."""
        from api.database import get_db_ctx, init_db, UserDB
        init_db()
        user_id = str(uuid4())
        now = datetime.now(timezone.utc)
        with get_db_ctx() as db:
            user = UserDB(id=user_id, email=f"v02-{uuid4().hex[:8]}@t.com", is_active=True, created_at=now, updated_at=now, last_login_at=now)
            db.add(user)
            db.commit()

        # Corrupted horizon
        self._seed_custom_report(
            user_id,
            trade_date="2024-01-02",
            probability=0.8,
            analysis_status="VALID",
            trade_action="BUY",
            result_data={
                "horizon": "unsupported_super_long",
                "profile_id": "horizon_profile_v1",
            },
        )
        # Corrupted profile
        self._seed_custom_report(
            user_id,
            trade_date="2024-01-03",
            probability=0.8,
            analysis_status="VALID",
            trade_action="BUY",
            result_data={
                "horizon": "short",
                "profile_id": "unsupported_parallel_profile_v99",
            },
        )

        with get_db_ctx() as db:
            res = cal.compute_calibration(
                db,
                user_id=user_id,
                outcome_resolver=lambda r: True,
            )

        assert res["sample_size"] == 0
        assert res["excluded_counts"]["invalid_profile"] >= 2
        assert res["excluded_invalid_profile"] >= 2

        # Direct caller validation check
        with get_db_ctx() as db:
            with pytest.raises(ValueError, match="Unsupported horizon 'invalid_param'"):
                cal.compute_calibration(db, horizon="invalid_param")

            with pytest.raises(ValueError, match="Unsupported profile_id 'invalid_profile'"):
                cal.compute_calibration(db, horizon="short", profile_id="invalid_profile")

    def test_wait_direction_diagnostic_separated_from_trading_performance(self):
        """WAIT directional diagnostics are computed and presented separately from trading calibration."""
        from api.database import get_db_ctx, init_db, UserDB
        init_db()
        user_id = str(uuid4())
        now = datetime.now(timezone.utc)
        with get_db_ctx() as db:
            user = UserDB(id=user_id, email=f"v02-{uuid4().hex[:8]}@t.com", is_active=True, created_at=now, updated_at=now, last_login_at=now)
            db.add(user)
            db.commit()

        # 1. Traded BUY (price rose -> win)
        self._seed_custom_report(
            user_id, trade_date="2024-01-02", probability=0.8,
            analysis_status="VALID", trade_action="BUY", symbol="600519.SH",
        )
        # 2. Traded SELL (price fell -> win)
        self._seed_custom_report(
            user_id, trade_date="2024-01-03", probability=0.2,
            analysis_status="VALID", trade_action="SELL", symbol="600519.SH",
        )
        # 3. Non-traded WAIT with bullish lean (prob=0.75, price rose -> diagnostic hit)
        self._seed_custom_report(
            user_id, trade_date="2024-01-04", probability=0.75,
            analysis_status="VALID", trade_action="WAIT", symbol="600519.SH",
            result_data={"direction": "BULL"},
        )
        # 4. Non-traded WAIT with bearish lean (winner='bear', price fell -> diagnostic hit)
        self._seed_custom_report(
            user_id, trade_date="2024-01-05", probability=None,
            analysis_status="VALID", trade_action="WAIT", symbol="600519.SH",
            result_data={"investment_debate_state": {"winner": "bear"}},
        )
        # 5. Non-traded WAIT with bearish lean (winner='bear', price rose -> diagnostic miss)
        self._seed_custom_report(
            user_id, trade_date="2024-01-08", probability=None,
            analysis_status="VALID", trade_action="WAIT", symbol="600519.SH",
            result_data={"investment_debate_state": {"winner": "bear"}},
        )

        def mock_outcome(r: ReportDB) -> bool | None:
            return r.trade_date in ("2024-01-02", "2024-01-04", "2024-01-08")

        with get_db_ctx() as db:
            res = cal.compute_calibration(
                db,
                user_id=user_id,
                outcome_resolver=mock_outcome,
            )

        # Traded calibration sample_size must ONLY count the 2 traded reports
        assert res["sample_size"] == 2
        assert res["probability_sample_size"] == 2

        # WAIT diagnostic is reported separately
        wait_diag = res["wait_diagnostic"]
        assert wait_diag["total_wait_count"] == 3
        assert wait_diag["evaluable_wait_count"] == 3
        assert wait_diag["rise_count"] == 2  # Jan 4 and Jan 8 rose
        assert wait_diag["fall_count"] == 1  # Jan 5 fell
        assert wait_diag["rise_rate"] == 66.7
        assert wait_diag["directional_lean_count"] == 3
        assert wait_diag["directional_hits"] == 2  # Jan 4 (bull/rose) & Jan 5 (bear/fell)
        assert wait_diag["directional_hit_rate"] == 66.7

        # Exclusion accounting
        assert res["excluded_wait"] == 3
        assert res["excluded_counts"]["wait"] == 3

    def test_zero_or_insufficient_samples_does_not_fabricate_metrics(self):
        """Zero samples or N < min_sample_size never fabricates Brier score or bucket rise rates."""
        from api.database import get_db_ctx, init_db, UserDB
        init_db()
        user_id = str(uuid4())
        now = datetime.now(timezone.utc)
        with get_db_ctx() as db:
            user = UserDB(id=user_id, email=f"v02-{uuid4().hex[:8]}@t.com", is_active=True, created_at=now, updated_at=now, last_login_at=now)
            db.add(user)
            db.commit()

        # 0 samples
        with get_db_ctx() as db:
            res_zero = cal.compute_calibration(
                db,
                user_id=user_id,
                outcome_resolver=lambda r: True,
            )
        assert res_zero["sample_size"] == 0
        assert res_zero["brier_score"] is None
        assert res_zero["sample_sufficient"] is False
        assert all(b["rise_rate"] is None for b in res_zero["buckets"])

        # 2 samples (< 30)
        self._seed_custom_report(user_id, trade_date="2024-01-02", probability=0.8, analysis_status="VALID", trade_action="BUY")
        self._seed_custom_report(user_id, trade_date="2024-01-03", probability=0.6, analysis_status="VALID", trade_action="BUY")

        with get_db_ctx() as db:
            res_two = cal.compute_calibration(
                db,
                user_id=user_id,
                outcome_resolver=lambda r: True,
            )
        assert res_two["sample_size"] == 2
        assert res_two["brier_score"] is None
        assert res_two["sample_sufficient"] is False
        assert all(b["rise_rate"] is None for b in res_two["buckets"])
        assert "低于最小阈值" in str(res_two["insufficient_reason"])

    def test_backtest_service_routes_profile_and_preserves_hold_days(self):
        """Backtest service routes short to T+10 and medium to T+40, preserving default T+5."""
        job_short = bt.submit(
            user_id="u-v02",
            symbol="600519.SH",
            start_date="2024-01-02",
            end_date="2024-01-10",
            selected_analysts=["market"],
            horizon="short",
        )
        task_short = bt.get_job(job_short, "u-v02")
        assert task_short is not None
        assert task_short["hold_days"] == 10
        assert task_short["horizon"] == "short"
        assert task_short["profile_id"] == "horizon_profile_v1"

        job_med = bt.submit(
            user_id="u-v02",
            symbol="600519.SH",
            start_date="2024-01-02",
            end_date="2024-01-10",
            selected_analysts=["market"],
            horizon="medium",
        )
        task_med = bt.get_job(job_med, "u-v02")
        assert task_med is not None
        assert task_med["hold_days"] == 40
        assert task_med["horizon"] == "medium"

        job_def = bt.submit(
            user_id="u-v02",
            symbol="600519.SH",
            start_date="2024-01-02",
            end_date="2024-01-10",
            selected_analysts=["market"],
        )
        task_def = bt.get_job(job_def, "u-v02")
        assert task_def is not None
        assert task_def["hold_days"] == 5
        assert task_def["horizon"] is None

        with pytest.raises(ValueError, match="Unsupported horizon 'invalid_h'"):
            bt.submit(
                user_id="u-v02",
                symbol="600519.SH",
                start_date="2024-01-02",
                end_date="2024-01-10",
                selected_analysts=["market"],
                horizon="invalid_h",
            )

    def test_backtest_service_wait_direction_diagnostic_separated_from_win_rate(self):
        """Backtest compute_stats calculates wait_diagnostic without mixing into total_signals or win_rate."""
        records = [
            # 1. Winning BUY trade
            {"action": "BUY", "trade_action": "BUY", "analysis_status": "VALID", "return_pct": 5.0, "outcome_status": "ok"},
            # 2. Losing BUY trade
            {"action": "BUY", "trade_action": "BUY", "analysis_status": "VALID", "return_pct": -2.0, "outcome_status": "ok"},
            # 3. Non-traded WAIT with subsequent rise
            {"action": "WAIT", "trade_action": "WAIT", "analysis_status": "VALID", "return_pct": None, "wait_subsequent_return_pct": 4.5, "outcome_status": "excluded_wait"},
            # 4. Non-traded WAIT with subsequent fall
            {"action": "WAIT", "trade_action": "WAIT", "analysis_status": "VALID", "return_pct": None, "wait_subsequent_return_pct": -1.5, "outcome_status": "excluded_wait"},
        ]
        stats = bt._compute_stats(records)
        # Real trades count is strictly 2
        assert stats["total_signals"] == 2
        assert stats["win_rate"] == 50.0
        assert stats["avg_return_pct"] == 1.5

        # WAIT diagnostic is listed separately
        wait_diag = stats["wait_diagnostic"]
        assert wait_diag["total_wait_signals"] == 2
        assert wait_diag["evaluable_wait_signals"] == 2
        assert wait_diag["up_count"] == 1
        assert wait_diag["down_count"] == 1
        assert wait_diag["up_rate"] == 50.0
        assert wait_diag["avg_subsequent_return_pct"] == 1.5

        assert stats["excluded_wait"] == 2
        assert stats["excluded_counts"]["wait"] == 2

    def test_negative_constraints_invariance(self):
        """V-02 negative constraints verification:
        1. No portfolio Sharpe / drawdown metrics generated without portfolio rules.
        2. Confidence is never converted into probability.
        3. cal.DEFAULT_HOLD_DAYS remains 5.
        4. shadow_credit T+5 semantics and constants remain unchanged.
        """
        # 1. No portfolio Sharpe / drawdown in calibration or backtest
        stats = bt._compute_stats([])
        for forbidden_key in ("sharpe", "sharpe_ratio", "max_drawdown", "drawdown", "portfolio_sharpe"):
            assert forbidden_key not in stats, f"Forbidden key {forbidden_key!r} detected in backtest stats"

        # 2. Confidence never converted to probability
        from api.database import ReportDB
        rep = ReportDB(confidence=90, probability=None, result_data={})
        extracted_p = cal._extract_report_probability(rep)
        assert extracted_p is None, "confidence must never be converted to probability"

        # 3. DEFAULT_HOLD_DAYS invariance
        assert cal.DEFAULT_HOLD_DAYS == 5

        # 4. shadow_credit.py T+5 constants and invariants
        from tradingagents.agents.utils import shadow_credit as sc
        assert sc.T_PLUS_5_STATUS_DUE_AND_EVALUATED == "due_and_evaluated"
        assert sc.T_PLUS_5_STATUS_PENDING_DUE == "pending_due"
        assert hasattr(sc, "calculate_t_plus_5_date")
        assert hasattr(sc, "H1B_THRESHOLDS")
