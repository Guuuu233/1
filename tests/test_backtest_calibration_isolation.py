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
    def _resolve(symbol: str, base_date: str, hold_days: int) -> float | None:
        return price
    return _resolve


def _fake_price_on(price: float | None):
    def _resolve(symbol: str, date: str) -> float | None:
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
