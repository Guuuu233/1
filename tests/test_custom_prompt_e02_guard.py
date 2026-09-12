"""Red Team Scenarios (RT-1 to RT-16) for DAV-808 / D-015 E-02 prompt guard.

Covers:
  RT-1  - global-level violation text entering research_manager fails closed before model chain
  RT-2  - group=arbiter level violation fails closed (group override)
  RT-3  - role=research_manager level violation fails closed (role override)
  RT-4  - global/group/role combination resolved text validation & atomic rollback
  RT-5  - non-manager role violation halts entire job before graph execution; 5 roles no injection
  RT-6  - before_data and after_data placement intercepted, valid preserved
  RT-7  - switch off with violating text in DB: zero resolver reads, zero linter calls, byte-identical
  RT-8  - explicit negation ("不要/禁止按权重计票") not false positive
  RT-9  - domain-safe weights (仓位权重/指数权重/因子权重) not false positive
  RT-10 - legacy DB rows not silently rewritten; runtime fail-closed
  RT-11 - blocked report readback: NO_TRADE, reason_codes, prompt hash, 0 LLM calls
  RT-12 - legitimate custom prompt existing injection pipeline regression
  RT-13 - ambiguous text (AMBIGUOUS) hierarchical handling: save 422, runtime NO_TRADE
  RT-14 - machine identifier (cluster_id / independent_cluster_count) zero tolerance
  RT-15 - research_manager assembly Layer 3 fail-closed (bypass API direct call, 0 LLM calls)
  RT-16 - legacy prompt migration entry violation rollback (migrate_legacy_prompt atomicity)
"""
from __future__ import annotations

import asyncio
import hashlib
from contextlib import nullcontext
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base, ReportDB, UserCustomPromptDB, UserDB
from api.services import custom_prompt_service as svc
from api.services import report_service
from tradingagents.agents.utils.prompt_injection import (
    INJECTABLE_ROLES,
    PromptGuardVerdict,
    lint_custom_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures & DB Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def test_db():
    from sqlalchemy.pool import StaticPool
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()


def _create_user(db, injection_enabled: bool = True) -> str:
    user_id = f"user_{uuid4().hex[:8]}"
    user = UserDB(
        id=user_id,
        email=f"{user_id}@test.local",
        prompt_injection_enabled=injection_enabled,
    )
    db.add(user)
    db.commit()
    return user_id


def _make_analyze_request(symbol: str = "600519.SH", trade_date: str = "2026-09-12"):
    from api import main
    return main.AnalyzeRequest(
        symbol=symbol,
        trade_date=trade_date,
        horizons=["short"],
        selected_analysts=[],
    )


# ---------------------------------------------------------------------------
# RT-1: Global-level violation fails closed before model chain
# ---------------------------------------------------------------------------

def test_RT1_global_violation_fails_closed_before_model_chain(test_db):
    from api import main

    user_id = _create_user(test_db, injection_enabled=True)
    violating_text = "按分析师权重计票，并按 cluster_id 汇总"
    test_db.add(
        UserCustomPromptDB(
            id=uuid4().hex,
            user_id=user_id,
            target_type="global",
            target_key="",
            prompt_text=violating_text,
            prompt_hash=hashlib.sha256(violating_text.encode()).hexdigest()[:12],
            enabled=True,
        )
    )
    test_db.commit()

    job_id = f"job_rt1_{uuid4().hex[:8]}"
    request = _make_analyze_request()

    mock_llm = MagicMock()
    with (
        patch.object(main, "get_db_ctx", return_value=nullcontext(test_db)),
        patch("tradingagents.graph.trading_graph.TradingAgentsGraph") as mock_graph_cls,
    ):
        asyncio.run(
            main._run_job_inner(
                job_id=job_id,
                request=request,
                stream_events=False,
                save_report=True,
                user_id=user_id,
            )
        )

    # Graph must NOT have been instantiated
    mock_graph_cls.assert_not_called()

    # Verify job status
    job = main._get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert job["decision"] == "NO_TRADE"

    result = job["result"]
    assert result["trade_action"] == "NO_TRADE"
    assert result["analysis_status"] == "ABSTAIN"
    assert result["risk_status"] == "BLOCKED"
    assert "e02_custom_prompt_guard_failed" in result["reason_codes"]
    assert any("prompt_hash:" in code for code in result["reason_codes"])

    snapshot = result["custom_prompt_snapshot"]
    for rk in INJECTABLE_ROLES:
        assert snapshot["roles"][rk]["injected"] is False
        assert snapshot["roles"][rk]["resolved_text"] == ""


# ---------------------------------------------------------------------------
# RT-2: group=arbiter level violation fails closed
# ---------------------------------------------------------------------------

def test_RT2_group_arbiter_violation_fails_closed(test_db):
    from api import main

    user_id = _create_user(test_db, injection_enabled=True)
    violating_text = "根据命题加权计票"
    test_db.add(
        UserCustomPromptDB(
            id=uuid4().hex,
            user_id=user_id,
            target_type="group",
            target_key="arbiter",
            prompt_text=violating_text,
            prompt_hash=hashlib.sha256(violating_text.encode()).hexdigest()[:12],
            enabled=True,
        )
    )
    test_db.commit()

    job_id = f"job_rt2_{uuid4().hex[:8]}"
    request = _make_analyze_request()

    with (
        patch.object(main, "get_db_ctx", return_value=nullcontext(test_db)),
        patch("tradingagents.graph.trading_graph.TradingAgentsGraph") as mock_graph_cls,
    ):
        asyncio.run(
            main._run_job_inner(
                job_id=job_id,
                request=request,
                stream_events=False,
                save_report=True,
                user_id=user_id,
            )
        )

    mock_graph_cls.assert_not_called()
    job = main._get_job(job_id)
    assert job["decision"] == "NO_TRADE"
    assert job["result"]["trade_action"] == "NO_TRADE"
    assert "e02_custom_prompt_guard_failed" in job["result"]["reason_codes"]
    assert all(not r["injected"] for r in job["result"]["custom_prompt_snapshot"]["roles"].values())


# ---------------------------------------------------------------------------
# RT-3: role=research_manager level violation fails closed
# ---------------------------------------------------------------------------

def test_RT3_role_research_manager_violation_fails_closed(test_db):
    from api import main

    user_id = _create_user(test_db, injection_enabled=True)
    violating_text = "按分析师权重计票"
    test_db.add(
        UserCustomPromptDB(
            id=uuid4().hex,
            user_id=user_id,
            target_type="role",
            target_key="research_manager",
            prompt_text=violating_text,
            prompt_hash=hashlib.sha256(violating_text.encode()).hexdigest()[:12],
            enabled=True,
        )
    )
    test_db.commit()

    job_id = f"job_rt3_{uuid4().hex[:8]}"
    request = _make_analyze_request()

    with (
        patch.object(main, "get_db_ctx", return_value=nullcontext(test_db)),
        patch("tradingagents.graph.trading_graph.TradingAgentsGraph") as mock_graph_cls,
    ):
        asyncio.run(
            main._run_job_inner(
                job_id=job_id,
                request=request,
                stream_events=False,
                save_report=True,
                user_id=user_id,
            )
        )

    mock_graph_cls.assert_not_called()
    job = main._get_job(job_id)
    assert job["decision"] == "NO_TRADE"
    assert job["result"]["trade_action"] == "NO_TRADE"


# ---------------------------------------------------------------------------
# RT-4: Three-layer concatenation & atomic rollback
# ---------------------------------------------------------------------------

def test_RT4_three_layer_concatenation_and_atomic_rollback(test_db):
    user_id = _create_user(test_db, injection_enabled=True)

    # Initial valid rows
    svc.replace_custom_prompts(
        test_db,
        user_id,
        [{"target_type": "global", "target_key": "", "prompt_text": "初始合法提示词"}],
    )
    initial_rows = svc.list_custom_prompts(test_db, user_id)
    assert len(initial_rows) == 1
    assert initial_rows[0]["prompt_text"] == "初始合法提示词"

    # Case A: Global safe + Group arbiter violation -> Rejected, old rows kept
    with pytest.raises(ValueError, match="违反 E-02 证据独立性硬约束"):
        svc.replace_custom_prompts(
            test_db,
            user_id,
            [
                {"target_type": "global", "target_key": "", "prompt_text": "全局合法：注重基本面分析"},
                {"target_type": "group", "target_key": "arbiter", "prompt_text": "按分析师权重计票"},
            ],
        )

    rows_after_a = svc.list_custom_prompts(test_db, user_id)
    assert len(rows_after_a) == 1
    assert rows_after_a[0]["prompt_text"] == "初始合法提示词"

    # Case B: Global violation + Role bull safe -> Rejected, old rows kept
    with pytest.raises(ValueError, match="违反 E-02 证据独立性硬约束"):
        svc.replace_custom_prompts(
            test_db,
            user_id,
            [
                {"target_type": "global", "target_key": "", "prompt_text": "按分析师权重计票"},
                {"target_type": "role", "target_key": "bull_researcher", "prompt_text": "角色合法内容"},
            ],
        )

    rows_after_b = svc.list_custom_prompts(test_db, user_id)
    assert len(rows_after_b) == 1
    assert rows_after_b[0]["prompt_text"] == "初始合法提示词"


# ---------------------------------------------------------------------------
# RT-5: Non-manager role violation halts entire job before graph execution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target_role", ["bull_researcher", "bear_researcher", "trader", "risk_manager"])
def test_RT5_non_manager_role_violation_halts_entire_job(test_db, target_role):
    from api import main

    user_id = _create_user(test_db, injection_enabled=True)
    violating_text = "按分析师权重计票"
    test_db.add(
        UserCustomPromptDB(
            id=uuid4().hex,
            user_id=user_id,
            target_type="role",
            target_key=target_role,
            prompt_text=violating_text,
            prompt_hash=hashlib.sha256(violating_text.encode()).hexdigest()[:12],
            enabled=True,
        )
    )
    test_db.commit()

    job_id = f"job_rt5_{target_role}_{uuid4().hex[:8]}"
    request = _make_analyze_request()

    with (
        patch.object(main, "get_db_ctx", return_value=nullcontext(test_db)),
        patch("tradingagents.graph.trading_graph.TradingAgentsGraph") as mock_graph_cls,
    ):
        asyncio.run(
            main._run_job_inner(
                job_id=job_id,
                request=request,
                stream_events=False,
                save_report=True,
                user_id=user_id,
            )
        )

    mock_graph_cls.assert_not_called()
    job = main._get_job(job_id)
    assert job["decision"] == "NO_TRADE"
    assert job["result"]["trade_action"] == "NO_TRADE"
    assert job["result"]["prompt_guard_failure"]["role"] == target_role
    for rk in INJECTABLE_ROLES:
        assert job["result"]["custom_prompt_snapshot"]["roles"][rk]["injected"] is False


# ---------------------------------------------------------------------------
# RT-6: before_data and after_data placement intercepted
# ---------------------------------------------------------------------------

def test_RT6_both_placements_intercepted(test_db):
    from api import main

    user_id = _create_user(test_db, injection_enabled=True)
    violating_text = "按分析师权重计票"
    test_db.add(
        UserCustomPromptDB(
            id=uuid4().hex,
            user_id=user_id,
            target_type="global",
            target_key="",
            prompt_text=violating_text,
            prompt_hash=hashlib.sha256(violating_text.encode()).hexdigest()[:12],
            enabled=True,
        )
    )
    test_db.commit()

    for placement in ["before_data", "after_data"]:
        job_id = f"job_rt6_{placement}_{uuid4().hex[:8]}"
        request = _make_analyze_request()
        with (
            patch.object(main, "get_db_ctx", return_value=nullcontext(test_db)),
            patch("api.main.DEFAULT_PLACEMENT", placement),
            patch("tradingagents.graph.trading_graph.TradingAgentsGraph") as mock_graph_cls,
        ):
            asyncio.run(
                main._run_job_inner(
                    job_id=job_id,
                    request=request,
                    stream_events=False,
                    save_report=False,
                    user_id=user_id,
                )
            )
        mock_graph_cls.assert_not_called()
        job = main._get_job(job_id)
        assert job["decision"] == "NO_TRADE"


# ---------------------------------------------------------------------------
# RT-7: Switch off: zero resolver reads, zero linter calls, byte-identical
# ---------------------------------------------------------------------------

def test_RT7_switch_off_zero_reads_zero_linter(test_db):
    from api.main import _resolve_and_freeze_custom_prompts
    from tradingagents.agents.utils import prompt_injection

    user_id = _create_user(test_db, injection_enabled=False)
    # Put a violating prompt in DB
    violating_text = "按分析师权重计票"
    test_db.add(
        UserCustomPromptDB(
            id=uuid4().hex,
            user_id=user_id,
            target_type="global",
            target_key="",
            prompt_text=violating_text,
            prompt_hash=hashlib.sha256(violating_text.encode()).hexdigest()[:12],
            enabled=True,
        )
    )
    test_db.commit()

    with (
        patch.object(svc, "resolve_all_roles_prompts") as mock_resolve,
        patch.object(prompt_injection, "lint_custom_prompt") as mock_linter,
    ):
        bundle, enabled = _resolve_and_freeze_custom_prompts(test_db, user_id)

    assert enabled is False
    mock_resolve.assert_not_called()
    mock_linter.assert_not_called()
    assert all(not r["injected"] for r in bundle.values())
    assert all(r["resolved_text"] == "" for r in bundle.values())


# ---------------------------------------------------------------------------
# RT-8: Explicit negation ("不要/禁止按权重计票") not false positive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "不要按权重计票",
        "禁止按分析师权重计票",
        "严禁按权重计票",
        "不得对分析师加权",
        "切勿按独立票汇总",
    ],
)
def test_RT8_explicit_negation_not_false_positive(test_db, text):
    user_id = _create_user(test_db, injection_enabled=True)
    res = lint_custom_prompt(text)
    assert res.verdict == PromptGuardVerdict.SAFE_NEGATION_OR_UNRELATED

    # Must be accepted by save API
    saved = svc.replace_custom_prompts(
        test_db,
        user_id,
        [{"target_type": "global", "target_key": "", "prompt_text": text}],
    )
    assert len(saved) == 1
    assert saved[0]["prompt_text"] == text


# ---------------------------------------------------------------------------
# RT-9: Domain-safe weights not false positive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "仓位权重控制在 20% 以下",
        "参考沪深300指数权重进行资产配置",
        "采用多因子权重模型配置股票池",
        "组合等权重配置",
        "控制单行业权重不得超过 15%",
    ],
)
def test_RT9_domain_safe_weights_not_false_positive(test_db, text):
    user_id = _create_user(test_db, injection_enabled=True)
    res = lint_custom_prompt(text)
    assert res.verdict == PromptGuardVerdict.SAFE_NEGATION_OR_UNRELATED

    saved = svc.replace_custom_prompts(
        test_db,
        user_id,
        [{"target_type": "global", "target_key": "", "prompt_text": text}],
    )
    assert len(saved) == 1


# ---------------------------------------------------------------------------
# RT-10: Legacy DB rows not silently rewritten; runtime fail-closed
# ---------------------------------------------------------------------------

def test_RT10_legacy_rows_not_silently_rewritten_and_fails_closed(test_db):
    from api import main

    user_id = _create_user(test_db, injection_enabled=True)
    original_text = "按分析师权重计票，并按 cluster_id 汇总"
    test_db.add(
        UserCustomPromptDB(
            id=uuid4().hex,
            user_id=user_id,
            target_type="global",
            target_key="",
            prompt_text=original_text,
            prompt_hash=hashlib.sha256(original_text.encode()).hexdigest()[:12],
            enabled=True,
        )
    )
    test_db.commit()

    job_id = f"job_rt10_{uuid4().hex[:8]}"
    request = _make_analyze_request()

    with (
        patch.object(main, "get_db_ctx", return_value=nullcontext(test_db)),
        patch("tradingagents.graph.trading_graph.TradingAgentsGraph") as mock_graph_cls,
    ):
        asyncio.run(
            main._run_job_inner(
                job_id=job_id,
                request=request,
                stream_events=False,
                save_report=True,
                user_id=user_id,
            )
        )

    # Job failed closed
    job = main._get_job(job_id)
    assert job["decision"] == "NO_TRADE"

    # Database row was NOT altered, deleted, or silently sanitized
    row = test_db.query(UserCustomPromptDB).filter(UserCustomPromptDB.user_id == user_id).first()
    assert row is not None
    assert row.prompt_text == original_text


# ---------------------------------------------------------------------------
# RT-11: Blocked report readback and zero LLM calls
# ---------------------------------------------------------------------------

def test_RT11_blocked_report_readback_and_zero_llm_calls(test_db):
    from api import main

    user_id = _create_user(test_db, injection_enabled=True)
    violating_text = "按分析师权重计票"
    expected_hash = hashlib.sha256(violating_text.encode()).hexdigest()[:12]
    test_db.add(
        UserCustomPromptDB(
            id=uuid4().hex,
            user_id=user_id,
            target_type="global",
            target_key="",
            prompt_text=violating_text,
            prompt_hash=expected_hash,
            enabled=True,
        )
    )
    test_db.commit()

    job_id = f"job_rt11_{uuid4().hex[:8]}"
    request = _make_analyze_request()

    with (
        patch.object(main, "get_db_ctx", return_value=nullcontext(test_db)),
        patch("tradingagents.graph.trading_graph.TradingAgentsGraph") as mock_graph_cls,
    ):
        asyncio.run(
            main._run_job_inner(
                job_id=job_id,
                request=request,
                stream_events=False,
                save_report=True,
                user_id=user_id,
            )
        )

    # 0 LLM calls: graph never created
    mock_graph_cls.assert_not_called()

    # Read back report from database
    report = test_db.query(ReportDB).filter(ReportDB.id == job_id).first()
    assert report is not None
    assert report.status == "completed"
    assert report.decision == "NO_TRADE"
    assert report.trade_action == "NO_TRADE"
    assert report.analysis_status == "ABSTAIN"
    assert report.risk_status == "BLOCKED"

    result_data = report.result_data
    assert result_data["trade_action"] == "NO_TRADE"
    assert "e02_custom_prompt_guard_failed" in result_data["reason_codes"]
    assert f"prompt_hash:{expected_hash}" in result_data["reason_codes"]
    assert result_data["prompt_guard_failure"]["prompt_hash"] == expected_hash

    # Five roles all injected=False
    snapshot = result_data["custom_prompt_snapshot"]
    assert snapshot["enabled"] is True
    for rk in INJECTABLE_ROLES:
        assert snapshot["roles"][rk]["injected"] is False
        assert snapshot["roles"][rk]["resolved_text"] == ""


# ---------------------------------------------------------------------------
# RT-12: Legitimate custom_prompt regression
# ---------------------------------------------------------------------------

def test_RT12_legitimate_custom_prompt_regression(test_db):
    user_id = _create_user(test_db, injection_enabled=True)
    legit_text = "请在分析中优先关注 A 级证据，概率必须在 0.00-1.00 之间。"
    svc.replace_custom_prompts(
        test_db,
        user_id,
        [{"target_type": "global", "target_key": "", "prompt_text": legit_text}],
    )

    resolved = svc.resolve_all_roles_prompts(test_db, user_id)
    assert len(resolved) == 15
    for r in resolved:
        assert r["resolved_text"] == legit_text


# ---------------------------------------------------------------------------
# RT-13: Ambiguous text hierarchical handling
# ---------------------------------------------------------------------------

def test_RT13_ambiguous_text_hierarchical_handling(test_db):
    from api import main

    user_id = _create_user(test_db, injection_enabled=True)
    ambiguous_text = "综合各分析师观点进行加权研判，给出综合结论"

    # Layer 1: Save entry point rejects with 422 ValueError and rewrite advice
    with pytest.raises(ValueError, match="语义不确定"):
        svc.replace_custom_prompts(
            test_db,
            user_id,
            [{"target_type": "global", "target_key": "", "prompt_text": ambiguous_text}],
        )

    # Layer 2: Runtime entry point fails closed if existing in DB
    test_db.add(
        UserCustomPromptDB(
            id=uuid4().hex,
            user_id=user_id,
            target_type="global",
            target_key="",
            prompt_text=ambiguous_text,
            prompt_hash=hashlib.sha256(ambiguous_text.encode()).hexdigest()[:12],
            enabled=True,
        )
    )
    test_db.commit()

    job_id = f"job_rt13_{uuid4().hex[:8]}"
    request = _make_analyze_request()

    with (
        patch.object(main, "get_db_ctx", return_value=nullcontext(test_db)),
        patch("tradingagents.graph.trading_graph.TradingAgentsGraph") as mock_graph_cls,
    ):
        asyncio.run(
            main._run_job_inner(
                job_id=job_id,
                request=request,
                stream_events=False,
                save_report=True,
                user_id=user_id,
            )
        )

    mock_graph_cls.assert_not_called()
    job = main._get_job(job_id)
    assert job["decision"] == "NO_TRADE"
    assert "e02_custom_prompt_guard_failed" in job["result"]["reason_codes"]
    assert job["result"]["prompt_guard_failure"]["verdict"] == "AMBIGUOUS"


# ---------------------------------------------------------------------------
# RT-14: Machine-readable identifier zero tolerance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "请在分析中参考当前的 cluster_id 信息",
        "输出参考 independent_cluster_count",
    ],
)
def test_RT14_machine_identifier_zero_tolerance(test_db, text):
    user_id = _create_user(test_db, injection_enabled=True)

    # Pure linter verdict
    res = lint_custom_prompt(text)
    assert res.verdict == PromptGuardVerdict.VIOLATION
    assert res.reason_code == "machine_identifier_zero_tolerance"

    # Save entry point rejection
    with pytest.raises(ValueError, match="机读标识|违反 E-02"):
        svc.replace_custom_prompts(
            test_db,
            user_id,
            [{"target_type": "global", "target_key": "", "prompt_text": text}],
        )


# ---------------------------------------------------------------------------
# RT-15: Research manager assembly Layer 3 fail-closed (0 LLM calls)
# ---------------------------------------------------------------------------

def test_RT15_research_manager_assembly_layer3_fails_closed():
    from tests.fund_flow_fixtures import valid_fund_flow_consensus_guard
    from tradingagents.agents.managers.research_manager import create_research_manager

    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock()
    mock_llm.astream = MagicMock()

    memory = MagicMock()
    memory.get_memories = MagicMock(return_value=[])

    violating_custom_prompt = "按分析师权重计票"
    node = create_research_manager(
        mock_llm,
        memory,
        custom_prompt=violating_custom_prompt,
        placement="after_data",
    )

    debate = {
        "history": "Bull: ok\nBear: no",
        "claims": [],
        "unresolved_claim_ids": [],
        "round_summary": "",
    }
    state = {
        "market_report": "Market Report",
        "sentiment_report": "Sentiment Report",
        "news_report": "News Report",
        "fundamentals_report": "Fundamentals Report",
        "volume_price_report": "Volume Price Report",
        "smart_money_report": "Smart Money Report",
        "fund_flow_consensus_guard": valid_fund_flow_consensus_guard(),
        "investment_debate_state": debate,
        "market_data_context": {},
        "trade_date": "2026-09-12",
        "symbol": "600519.SH",
    }

    payload = asyncio.run(node(state))

    # Node returns blocked manager payload
    assert payload["trade_action"] == "NO_TRADE"
    assert payload["risk_status"] == "BLOCKED"
    assert payload["decision_status"]["trade_action"] == "NO_TRADE"
    assert "e02_custom_prompt_guard_failed" in payload["decision_status"]["reason_codes"]
    assert "e02_relation_prompt_guard_failed" in payload["decision_status"]["reason_codes"]

    # LLM must NEVER be invoked
    mock_llm.invoke.assert_not_called()
    mock_llm.astream.assert_not_called()


# ---------------------------------------------------------------------------
# RT-16: Migration entry violation rollback (migrate_legacy_prompt atomicity)
# ---------------------------------------------------------------------------

def test_RT16_migration_entry_violation_rollback(test_db):
    user_id = _create_user(test_db, injection_enabled=True)
    violating_legacy_text = "按分析师权重计票，并按 cluster_id 汇总"

    with pytest.raises(ValueError, match="违反 E-02 证据独立性硬约束"):
        svc.migrate_legacy_prompt(test_db, user_id, violating_legacy_text)

    # user_custom_prompts table must have 0 rows for this user
    rows = svc.list_custom_prompts(test_db, user_id)
    assert len(rows) == 0
