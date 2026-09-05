"""Tests for horizon cache and derivative results isolation (H-02c, DAV-675).

Contracts:
1. Sequential runs of short and medium for the same ticker+date: in-memory log_states_dict
   and on-disk logs can both be retrieved separately; later run must not overwrite earlier run.
2. Checkpointer thread_id differs across horizons; raw data cache is not split.
3. collect(..., horizons=...) shares make_cache_key; raw pool is not pruned by horizon.
4. Legacy log without horizon suffix: marked unknown/legacy on read, never interpreted as T+40 run.
5. report_service.get_latest_reports_by_symbols read-side isolation by metadata.
"""

import copy
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base, ReportDB
from api.services.report_service import get_latest_reports_by_symbols
from tradingagents.graph.data_collector import DataCollector, make_cache_key
from tradingagents.graph.horizon_profile import resolve_analysis_horizons
from tradingagents.graph.trading_graph import TradingAgentsGraph


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


def _make_mock_trading_agents_graph():
    with patch("tradingagents.graph.trading_graph.create_llm_client"), \
         patch("tradingagents.graph.trading_graph.FinancialSituationMemory"), \
         patch("tradingagents.graph.trading_graph.GraphSetup"), \
         patch("tradingagents.graph.trading_graph.ConditionalLogic"), \
         patch("tradingagents.graph.trading_graph.Reflector"), \
         patch("tradingagents.graph.trading_graph.SignalProcessor"), \
         patch("tradingagents.graph.trading_graph.set_config"):
        ta = TradingAgentsGraph.__new__(TradingAgentsGraph)
        ta.debug = False
        ta.config = {}
        ta.callbacks = []
        ta.ticker = "600519"
        ta.log_states_dict = {}
        ta.quick_thinking_llm = MagicMock()
        ta.data_collector = DataCollector()
        from tradingagents.graph.propagation import Propagator
        ta.propagator = Propagator()
        ta.graph = MagicMock()
        ta.signal_processor = MagicMock()
        ta.signal_processor.process_signal.return_value = "BUY"
        return ta


class TestHorizonLogIsolationContract1:
    """Contract 1: In-memory log_states_dict and on-disk logs separate short and medium."""

    def test_log_states_dict_separates_short_and_medium(self):
        ta = _make_mock_trading_agents_graph()
        trade_date = "2026-08-26"

        final_state_short = {
            "company_of_interest": "600519",
            "trade_date": trade_date,
            "horizon": "short",
            "market_report": "short market",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
            "investment_debate_state": {"bull_history": "", "bear_history": "", "history": "", "current_response": "", "judge_decision": ""},
            "risk_debate_state": {"aggressive_history": "", "conservative_history": "", "neutral_history": "", "history": "", "judge_decision": ""},
            "investment_plan": "short plan",
            "trader_investment_plan": "short plan",
            "final_trade_decision": "BUY_SHORT",
        }
        final_state_medium = {
            "company_of_interest": "600519",
            "trade_date": trade_date,
            "horizon": "medium",
            "market_report": "medium market",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
            "investment_debate_state": {"bull_history": "", "bear_history": "", "history": "", "current_response": "", "judge_decision": ""},
            "risk_debate_state": {"aggressive_history": "", "conservative_history": "", "neutral_history": "", "history": "", "judge_decision": ""},
            "investment_plan": "medium plan",
            "trader_investment_plan": "medium plan",
            "final_trade_decision": "HOLD_MEDIUM",
        }

        # Log short first, then medium
        ta._log_state(trade_date, final_state_short)
        ta._log_state(trade_date, final_state_medium)

        # Both entries must be distinct and retrievable; medium must not overwrite short
        short_entry = ta.log_states_dict.get(f"{trade_date}_short")
        medium_entry = ta.log_states_dict.get(f"{trade_date}_medium")

        assert short_entry is not None, "Short entry missing from log_states_dict"
        assert medium_entry is not None, "Medium entry missing from log_states_dict"
        assert short_entry["final_trade_decision"] == "BUY_SHORT"
        assert medium_entry["final_trade_decision"] == "HOLD_MEDIUM"
        assert short_entry["market_report"] == "short market"
        assert medium_entry["market_report"] == "medium market"
        assert short_entry.get("horizon") == "short"
        assert medium_entry.get("horizon") == "medium"

    def test_disk_log_files_separate_short_and_medium(self, tmp_path):
        ta = _make_mock_trading_agents_graph()
        trade_date = "2026-08-26"

        final_state_short = {
            "company_of_interest": "600519",
            "trade_date": trade_date,
            "horizon": "short",
            "market_report": "short market",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
            "investment_debate_state": {"bull_history": "", "bear_history": "", "history": "", "current_response": "", "judge_decision": ""},
            "risk_debate_state": {"aggressive_history": "", "conservative_history": "", "neutral_history": "", "history": "", "judge_decision": ""},
            "investment_plan": "",
            "trader_investment_plan": "",
            "final_trade_decision": "BUY_SHORT",
        }
        final_state_medium = {
            "company_of_interest": "600519",
            "trade_date": trade_date,
            "horizon": "medium",
            "market_report": "medium market",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
            "investment_debate_state": {"bull_history": "", "bear_history": "", "history": "", "current_response": "", "judge_decision": ""},
            "risk_debate_state": {"aggressive_history": "", "conservative_history": "", "neutral_history": "", "history": "", "judge_decision": ""},
            "investment_plan": "",
            "trader_investment_plan": "",
            "final_trade_decision": "HOLD_MEDIUM",
        }

        with patch.dict(os.environ, {"TA_RESULTS_DIR": str(tmp_path), "TA_STATE_LOGS": "1"}):
            ta._log_state(trade_date, final_state_short)
            ta._log_state(trade_date, final_state_medium)

            log_dir = tmp_path / "600519" / "TradingAgentsStrategy_logs"
            short_file = log_dir / f"full_states_log_{trade_date}_short.json"
            medium_file = log_dir / f"full_states_log_{trade_date}_medium.json"

            assert short_file.exists(), f"Expected short file {short_file} to exist"
            assert medium_file.exists(), f"Expected medium file {medium_file} to exist"

            # Both files must contain their respective decisions
            with open(short_file) as f:
                short_data = json.load(f)
            with open(medium_file) as f:
                medium_data = json.load(f)

            short_val = short_data.get(f"{trade_date}_short") or short_data.get(trade_date) or short_data
            medium_val = medium_data.get(f"{trade_date}_medium") or medium_data.get(trade_date) or medium_data
            assert short_val["final_trade_decision"] == "BUY_SHORT"
            assert medium_val["final_trade_decision"] == "HOLD_MEDIUM"


class TestCheckpointerThreadIdContract2:
    """Contract 2: Checkpointer thread_id differs on horizon switch; API thread_id not broken."""

    def test_propagate_default_thread_id_differs_by_horizon(self):
        ta = _make_mock_trading_agents_graph()

        with patch.object(ta.data_collector, "collect", return_value={}), \
             patch.object(ta.graph, "invoke") as mock_invoke:
            mock_invoke.return_value = {
                "company_of_interest": "600519",
                "trade_date": "2026-08-26",
                "horizon": "short",
                "final_trade_decision": "BUY",
                "trader_investment_plan": "",
                "investment_plan": "",
                "market_report": "",
                "sentiment_report": "",
                "news_report": "",
                "fundamentals_report": "",
                "investment_debate_state": {"bull_history": "", "bear_history": "", "history": "", "current_response": "", "judge_decision": ""},
                "risk_debate_state": {"aggressive_history": "", "conservative_history": "", "neutral_history": "", "history": "", "judge_decision": ""},
            }

            # Run short
            ta.propagate("600519", "2026-08-26", horizon="short")
            assert mock_invoke.called
            call_kwargs_short = mock_invoke.call_args[1]
            thread_id_short = call_kwargs_short["config"]["configurable"]["thread_id"]

            # Run medium
            mock_invoke.reset_mock()
            mock_invoke.return_value["horizon"] = "medium"
            ta.propagate("600519", "2026-08-26", horizon="medium")
            assert mock_invoke.called
            call_kwargs_medium = mock_invoke.call_args[1]
            thread_id_medium = call_kwargs_medium["config"]["configurable"]["thread_id"]

            assert thread_id_short != thread_id_medium
            assert "short" in thread_id_short
            assert "medium" in thread_id_medium

    def test_propagate_preserves_explicit_thread_id(self):
        ta = _make_mock_trading_agents_graph()

        with patch.object(ta.data_collector, "collect", return_value={}), \
             patch.object(ta.graph, "invoke") as mock_invoke:
            mock_invoke.return_value = {
                "company_of_interest": "600519",
                "trade_date": "2026-08-26",
                "horizon": "medium",
                "final_trade_decision": "BUY",
                "trader_investment_plan": "",
                "investment_plan": "",
                "market_report": "",
                "sentiment_report": "",
                "news_report": "",
                "fundamentals_report": "",
                "investment_debate_state": {"bull_history": "", "bear_history": "", "history": "", "current_response": "", "judge_decision": ""},
                "risk_debate_state": {"aggressive_history": "", "conservative_history": "", "neutral_history": "", "history": "", "judge_decision": ""},
            }

            ta.propagate(
                "600519",
                "2026-08-26",
                thread_id="job_42_medium",
                horizon="medium",
            )
            assert mock_invoke.called
            call_kwargs = mock_invoke.call_args[1]
            assert call_kwargs["config"]["configurable"]["thread_id"] == "job_42_medium"


class TestDataCollectorContract3:
    """Contract 3: collect(..., horizons=...) shares make_cache_key; raw pool is not pruned."""

    def test_make_cache_key_unchanged(self):
        assert make_cache_key("600519", "2026-08-26") == "600519_2026-08-26"

    def test_collect_shares_cache_key_across_horizons(self):
        dc = DataCollector()
        fake_pool = {
            "daily": {"as_of": "2026-08-26", "bars": 100},
            "financials": {"revenue": 1000},
            "news": ["headline 1"],
            "market_data_context": {"daily": {"as_of": "2026-08-26"}},
        }

        with patch("tradingagents.graph.data_collector._fetch_all", return_value=fake_pool) as mock_fetch, \
             patch.object(dc, "_fetch_social_context", return_value=None):

            pool_short = dc.collect("600519", "2026-08-26", horizons=["short"])
            pool_medium = dc.collect("600519", "2026-08-26", horizons=["medium"])

            # _fetch_all should only be called once because the same make_cache_key was used
            assert mock_fetch.call_count == 1
            key = make_cache_key("600519", "2026-08-26")
            assert key in dc._cache

            # Neither pool is pruned by horizon
            assert pool_short["daily"]["bars"] == 100
            assert pool_medium["daily"]["bars"] == 100
            assert "financials" in pool_short
            assert "financials" in pool_medium


class TestLegacyLogContract4:
    """Contract 4: Legacy log without horizon suffix is marked unknown/legacy, never T+40."""

    def test_read_legacy_log_without_horizon_suffix(self, tmp_path):
        ta = _make_mock_trading_agents_graph()
        trade_date = "2026-08-26"
        log_dir = tmp_path / "600519" / "TradingAgentsStrategy_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        legacy_file = log_dir / f"full_states_log_{trade_date}.json"
        legacy_data = {
            trade_date: {
                "company_of_interest": "600519",
                "trade_date": trade_date,
                "final_trade_decision": "BUY_LEGACY",
            }
        }
        with open(legacy_file, "w") as f:
            json.dump(legacy_data, f)

        # Read legacy log via read_state_log helper
        with patch.dict(os.environ, {"TA_RESULTS_DIR": str(tmp_path)}):
            loaded = ta.read_state_log("600519", trade_date)
            assert loaded is not None
            assert loaded["final_trade_decision"] == "BUY_LEGACY"
            # Must mark as legacy/unknown, and NOT as medium / T+40
            assert loaded.get("horizon") in ("legacy", "unknown")
            meta = loaded.get("horizon_run_metadata", {})
            assert meta.get("resolution_source") in ("legacy", "unknown", "default")
            assert 40 not in meta.get("primary_eval_offsets", {}).values()
            assert "medium" not in meta.get("primary_eval_offsets", {})

            # When explicitly querying for medium, the legacy file must NOT be returned as medium
            loaded_med = ta.read_state_log("600519", trade_date, horizon="medium")
            assert loaded_med is None


class TestReportServiceGetLatestIsolationContract5:
    """Read-side isolation in get_latest_reports_by_symbols by metadata."""

    def test_get_latest_reports_filters_by_horizon_metadata(self, db_session):
        symbol = "600519.SH"
        user_id = "user_test_1"

        # Report 1 (older): short horizon
        r_short = ReportDB(
            id="rep_short_1",
            user_id=user_id,
            symbol=symbol,
            trade_date="2026-08-26",
            status="completed",
            decision="BUY",
            result_data={
                "horizon": "short",
                "horizon_run_metadata": {
                    "requested": ["short"],
                    "resolved": ["short"],
                    "resolution_source": "explicit",
                    "profile_id": "horizon_profile_v1",
                    "primary_eval_offsets": {"short": 10},
                },
            },
        )
        # Report 2 (newer): medium horizon
        r_medium = ReportDB(
            id="rep_medium_2",
            user_id=user_id,
            symbol=symbol,
            trade_date="2026-08-26",
            status="completed",
            decision="HOLD",
            result_data={
                "horizon": "medium",
                "horizon_run_metadata": {
                    "requested": ["medium"],
                    "resolved": ["medium"],
                    "resolution_source": "explicit",
                    "profile_id": "horizon_profile_v1",
                    "primary_eval_offsets": {"medium": 40},
                },
            },
        )

        db_session.add(r_short)
        db_session.flush()
        db_session.add(r_medium)
        db_session.commit()

        # When asking for short, must return r_short even though r_medium is newer
        short_res = get_latest_reports_by_symbols(db_session, [symbol], user_id=user_id, horizon="short")
        assert len(short_res) == 1
        assert short_res[0].id == "rep_short_1"
        assert short_res[0].decision == "BUY"

        # When asking for medium, must return r_medium
        medium_res = get_latest_reports_by_symbols(db_session, [symbol], user_id=user_id, horizon="medium")
        assert len(medium_res) == 1
        assert medium_res[0].id == "rep_medium_2"
        assert medium_res[0].decision == "HOLD"

        # When horizon is omitted, returns latest (r_medium), preserving legacy behavior
        latest_res = get_latest_reports_by_symbols(db_session, [symbol], user_id=user_id)
        assert len(latest_res) == 1
        assert latest_res[0].id == "rep_medium_2"
