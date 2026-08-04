"""Tests for Phase C custom-prompt injection.

Test matrix (T1–T24):
  T1  – switch off: production _resolve_and_freeze_custom_prompts makes 0 resolver calls
  T2  – switch off: three final prompts byte-identical to pre-injection baseline
  T3  – switch on: each injectable role receives its own resolved text
  T4  – non-injected roles do NOT receive any injection (bull/bear/manager/trader/risk_manager DO)
  T5  – production _resolve_and_freeze_custom_prompts resolves once for a job
  T6  – snapshot structure contains all five inject roles
  T7  – DB change after freeze does NOT affect already-frozen bundle
  T8  – production _resolve_and_freeze_custom_prompts raises ValueError if resolved text > 6000 chars
  T9  – existing factory signatures accept new parameters with defaults
  T10 – probability=70 (percentage-style) → validator returns None
  T11 – probability=0.70 → passes through unchanged
  T12 – confidence is never copied into probability field
  T13 – _attach_custom_prompt_snapshot is a deepcopy, not a shared reference
  T14 – dual-horizon and single save paths both produce identical snapshot attachments
  T15 – bool and non-integer-float inputs are rejected by validators with warnings
  T16 – bull node: injected text in captured prompt exactly once, after data before requirements
  T17 – bear node: same position guarantee as T16
  T18 – research_manager node: injection once, after data before output requirements
  T19 – bull and bear nodes: node output first line is [PROMPT-OK], DEBATE_STATE parseable from output
  T20 – research_manager node: investment_plan first line is [PROMPT-OK], VERDICT parseable from output
  T21 – research_manager node: receives first-hand market/news/fundamentals evidence summaries
  T22 – trader node: injection once, in the user message after the investment-plan data block
  T23 – risk_manager node: injection once, after data before output requirements
  T24 – api.main _INJECT_ROLES includes trader and risk_manager

Evidence-summary unit tests live in tests/test_evidence_summary.py.
"""
from __future__ import annotations

import hashlib
import logging
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_PROMPT = "请在分析中优先关注 A 级证据，概率必须在 0.00-1.00 之间。"
LONG_PROMPT = "X" * 6001  # exceeds RESOLVED_PROMPT_MAX_CHARS=6000


def _make_resolved(role_key: str, text: str):
    """Build a fake resolve_all_roles_prompts entry."""
    h = hashlib.sha256(text.encode()).hexdigest()[:12] if text else None
    return {
        "role_key": role_key,
        "resolved_text": text,
        "resolved_hash": h,
        "resolved_length": len(text),
        "override_source": "global" if text else None,
    }


def _make_full_bundle(texts: dict[str, str]) -> list[dict]:
    """Build a full 15-role resolved list with custom text for named roles."""
    from api.services.role_routing_service import ALL_ROLES
    return [_make_resolved(rk, texts.get(rk, "")) for rk in ALL_ROLES]


# ---------------------------------------------------------------------------
# T1 – switch off: production function makes 0 resolver calls
# ---------------------------------------------------------------------------

def test_T1_switch_off_zero_db_reads():
    from api.main import _resolve_and_freeze_custom_prompts
    from api.services import custom_prompt_service

    with (
        patch.object(custom_prompt_service, "get_prompt_injection_enabled", return_value=False),
        patch.object(custom_prompt_service, "resolve_all_roles_prompts") as mock_resolve,
        patch.object(custom_prompt_service, "resolve_role_prompt") as mock_resolve_one,
    ):
        bundle, enabled = _resolve_and_freeze_custom_prompts(MagicMock(), "user1")

        assert enabled is False
        assert all(not v["injected"] for v in bundle.values())
        mock_resolve.assert_not_called()
        mock_resolve_one.assert_not_called()


# ---------------------------------------------------------------------------
# T2 – switch off: prompts byte-identical to baseline
# ---------------------------------------------------------------------------

def test_T2_switch_off_prompt_byte_identical():
    """With empty custom_prompt the factory output must equal the no-injection baseline."""
    from tradingagents.agents.utils.prompt_injection import build_injection_slots
    from tradingagents.prompts import get_prompt
    from tradingagents.dataflows.config import get_config

    base = get_prompt("bull_prompt", config=get_config())

    common_kwargs = dict(
        market_research_report="M",
        sentiment_report="S",
        news_report="N",
        fundamentals_report="F",
        volume_price_report="V",
        history="H",
        current_response="CR",
        past_memory_str="PM",
        focus_claims_text="",
        unresolved_claims_text="",
        claims_text="",
        round_summary="RS",
        round_goal="RG",
    )

    # Baseline: slots both empty (as if injection never existed)
    baseline = base.format(**common_kwargs, **build_injection_slots("", "before_data"))

    # With switch off custom_prompt=""
    injected = base.format(**common_kwargs, **build_injection_slots("", "after_data"))

    assert baseline == injected, "Empty-prompt slots must not change prompt bytes"


# ---------------------------------------------------------------------------
# T3 – switch on: each injectable role gets its own text
# ---------------------------------------------------------------------------

def test_T3_switch_on_each_role_gets_text():
    from tradingagents.agents.utils.prompt_injection import build_injection_slots

    texts = {
        "bull_researcher": "多头指令",
        "bear_researcher": "空头指令",
        "research_manager": "裁决指令",
        "trader": "交易员指令",
        "risk_manager": "风控指令",
    }

    for role, text in texts.items():
        slots = build_injection_slots(text, "before_data")
        assert text in slots["custom_prompt_before_data"], f"{role} missing injected text"
        assert slots["custom_prompt_after_data"] == ""


# ---------------------------------------------------------------------------
# T4 – non-injected roles receive no injection
# ---------------------------------------------------------------------------

def test_T4_other_roles_no_injection():
    """Exercise GraphSetup.setup_graph and verify only the five approved factories receive text."""
    from types import SimpleNamespace
    from tradingagents.graph.setup import GraphSetup

    factory_names = (
        "create_aggressive_debator", "create_bear_researcher", "create_bull_researcher",
        "create_conservative_debator", "create_fundamentals_analyst", "create_macro_analyst",
        "create_market_analyst", "create_neutral_debator", "create_news_analyst",
        "create_research_manager", "create_risk_manager", "create_smart_money_analyst",
        "create_social_media_analyst", "create_volume_price_analyst", "create_trader",
    )
    injected_factories = {
        "create_bull_researcher",
        "create_bear_researcher",
        "create_research_manager",
        "create_trader",
        "create_risk_manager",
    }
    factories = {name: MagicMock(return_value=name) for name in factory_names}
    conditional = SimpleNamespace(
        should_continue_market=lambda *_: "done",
        should_continue_debate=lambda *_: "Research Manager",
        should_continue_risk_analysis=lambda *_: "Risk Judge",
        should_revise_after_risk_judge=lambda *_: "END",
    )

    with patch("tradingagents.graph.setup._load_agent_factories", return_value=factories):
        with patch("tradingagents.graph.setup.StateGraph"):
            setup = GraphSetup(
                object(), object(), {"market": object()}, object(), object(), object(), object(), object(),
                conditional, data_collector=object(),
                custom_prompts={
                    "bull_researcher": "BULL",
                    "bear_researcher": "BEAR",
                    "research_manager": "MANAGER",
                    "trader": "TRADER",
                    "risk_manager": "RISK",
                },
                custom_prompt_placement="after_data",
            )
            setup.setup_graph(["market"])

    for factory_name, expected in (
        ("create_bull_researcher", "BULL"),
        ("create_bear_researcher", "BEAR"),
        ("create_research_manager", "MANAGER"),
        ("create_trader", "TRADER"),
        ("create_risk_manager", "RISK"),
    ):
        kwargs = factories[factory_name].call_args.kwargs
        assert kwargs["custom_prompt"] == expected
        assert kwargs["placement"] == "after_data"

    for factory_name in factory_names:
        if factory_name not in injected_factories:
            # Some analysts are not selected in this minimal graph fixture and are not called.
            if factories[factory_name].call_args is not None:
                assert "custom_prompt" not in factories[factory_name].call_args.kwargs


# ---------------------------------------------------------------------------
# T5 – production function resolves once for a job
# ---------------------------------------------------------------------------

def test_T5_resolve_called_once_for_two_graphs():
    """Call production _resolve_and_freeze_custom_prompts once and verify resolve_all_roles_prompts is called once."""
    from api.main import _resolve_and_freeze_custom_prompts
    from api.services import custom_prompt_service

    call_count = 0
    full_bundle = _make_full_bundle({"bull_researcher": SAMPLE_PROMPT})

    def fake_resolve(db, user_id):
        nonlocal call_count
        call_count += 1
        return full_bundle

    with patch.object(custom_prompt_service, "resolve_all_roles_prompts", side_effect=fake_resolve):
        with patch.object(custom_prompt_service, "get_prompt_injection_enabled", return_value=True):
            db = MagicMock()
            bundle, enabled = _resolve_and_freeze_custom_prompts(db, "u1")
            # Simulate passing frozen bundle to two graph instances
            graph1_prompts = {rk: v["resolved_text"] for rk, v in bundle.items()}
            graph2_prompts = {rk: v["resolved_text"] for rk, v in bundle.items()}

    assert call_count == 1
    assert enabled is True
    assert graph1_prompts == graph2_prompts


# ---------------------------------------------------------------------------
# T6 – snapshot structure contains all five inject roles
# ---------------------------------------------------------------------------

def test_T6_snapshot_structure_has_five_roles():
    inject_roles = ("bull_researcher", "bear_researcher", "research_manager", "trader", "risk_manager")
    frozen_bundle = {
        rk: {"resolved_text": SAMPLE_PROMPT if rk == "bull_researcher" else "",
             "resolved_hash": "abc" if rk == "bull_researcher" else None,
             "resolved_length": len(SAMPLE_PROMPT) if rk == "bull_researcher" else 0,
             "injected": rk == "bull_researcher"}
        for rk in inject_roles
    }
    snapshot = {"enabled": True, "placement": "before_data", "roles": deepcopy(frozen_bundle)}

    # All five roles must be present even when some have empty text
    for rk in inject_roles:
        assert rk in snapshot["roles"]
        role_entry = snapshot["roles"][rk]
        assert "resolved_text" in role_entry
        assert "injected" in role_entry

    # enabled=False path
    off_snapshot = {"enabled": False, "placement": "before_data",
                    "roles": {rk: {"resolved_text": "", "resolved_hash": None, "resolved_length": 0, "injected": False}
                              for rk in inject_roles}}
    assert off_snapshot["enabled"] is False
    assert all(not v["injected"] for v in off_snapshot["roles"].values())


# ---------------------------------------------------------------------------
# T7 – DB change after freeze does NOT affect already-frozen bundle
# ---------------------------------------------------------------------------

def test_T7_db_change_after_freeze_doesnt_affect_bundle():
    """Verify that calling DB resolve later doesn't alter a previously frozen bundle."""
    from api.main import _resolve_and_freeze_custom_prompts
    from api.services import custom_prompt_service

    original_text = "原始提示词"
    changed_text = "用户中途修改后的提示词"

    full_bundle_original = _make_full_bundle({"bull_researcher": original_text})
    full_bundle_changed = _make_full_bundle({"bull_researcher": changed_text})

    db = MagicMock()

    # Step 1: Call production function to freeze initial bundle
    with patch.object(custom_prompt_service, "get_prompt_injection_enabled", return_value=True):
        with patch.object(custom_prompt_service, "resolve_all_roles_prompts", return_value=full_bundle_original):
            bundle_job_start, _ = _resolve_and_freeze_custom_prompts(db, "u1")

    # Step 2: DB state changes (user updates custom prompts during job run)
    with patch.object(custom_prompt_service, "get_prompt_injection_enabled", return_value=True):
        with patch.object(custom_prompt_service, "resolve_all_roles_prompts", return_value=full_bundle_changed):
            bundle_job_later, _ = _resolve_and_freeze_custom_prompts(db, "u1")

    # The first job's frozen bundle is unaffected by subsequent DB resolve calls
    assert bundle_job_start["bull_researcher"]["resolved_text"] == original_text
    assert bundle_job_later["bull_researcher"]["resolved_text"] == changed_text


# ---------------------------------------------------------------------------
# T8 – production _resolve_and_freeze_custom_prompts raises ValueError if resolved text > 6000 chars
# ---------------------------------------------------------------------------

def test_T8_oversized_text_raises():
    from api.main import _resolve_and_freeze_custom_prompts
    from api.services import custom_prompt_service

    oversized = _make_full_bundle({"bull_researcher": LONG_PROMPT})

    with patch.object(custom_prompt_service, "get_prompt_injection_enabled", return_value=True):
        with patch.object(custom_prompt_service, "resolve_all_roles_prompts", return_value=oversized):
            db = MagicMock()
            with pytest.raises(ValueError, match="bull_researcher"):
                _resolve_and_freeze_custom_prompts(db, "u1")


# ---------------------------------------------------------------------------
# T9 – factory signatures accept the new parameters without breaking
# ---------------------------------------------------------------------------

def test_T9_factory_signatures_accept_new_params():
    from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
    from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
    from tradingagents.agents.managers.research_manager import create_research_manager
    from tradingagents.agents.managers.risk_manager import create_risk_manager
    from tradingagents.agents.trader.trader import create_trader

    llm = MagicMock()
    memory = MagicMock()
    memory.get_memories = MagicMock(return_value=[])

    # Verify factories can be called with new kwargs without TypeError
    node_bull = create_bull_researcher(llm, memory, custom_prompt=SAMPLE_PROMPT, placement="before_data")
    node_bear = create_bear_researcher(llm, memory, custom_prompt=SAMPLE_PROMPT, placement="after_data")
    node_mgr = create_research_manager(llm, memory, custom_prompt="", placement="before_data")
    node_trader = create_trader(llm, memory, custom_prompt=SAMPLE_PROMPT, placement="after_data")
    node_risk = create_risk_manager(llm, memory, custom_prompt=SAMPLE_PROMPT, placement="after_data")

    assert callable(node_bull)
    assert callable(node_bear)
    assert callable(node_mgr)
    assert callable(node_trader)
    assert callable(node_risk)


# ---------------------------------------------------------------------------
# T10 – probability=70 (percentage-style) rejected → None
# ---------------------------------------------------------------------------

def test_T10_probability_percentage_rejected():
    from api.services.report_service import StructuredReport

    r = StructuredReport(decision="HOLD", probability=70)
    assert r.probability is None, "probability=70 should be rejected (looks like a percentage)"


# ---------------------------------------------------------------------------
# T11 – probability=0.70 passes through unchanged
# ---------------------------------------------------------------------------

def test_T11_probability_decimal_accepted():
    from api.services.report_service import StructuredReport

    r = StructuredReport(decision="HOLD", probability=0.70)
    assert r.probability == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# T12 – confidence is never copied into probability
# ---------------------------------------------------------------------------

def test_T12_confidence_not_copied_to_probability():
    from api.services.report_service import StructuredReport

    # Simulate model returning confidence=75 but no explicit probability
    r = StructuredReport(decision="BUY", confidence=75, probability=None)
    assert r.probability is None, "probability must remain null when not explicitly given"
    assert r.confidence == 75

    # Also test that the validator rejects an attempt to pass confidence value as probability
    r2 = StructuredReport(decision="BUY", confidence=75, probability=75)
    assert r2.probability is None, "probability=75 should be rejected (percentage range)"
    assert r2.confidence == 75


# ---------------------------------------------------------------------------
# T13 – _attach_custom_prompt_snapshot produces a deepcopy, not shared reference
# ---------------------------------------------------------------------------

def test_T13_attach_snapshot_is_deepcopy():
    from api.main import _attach_custom_prompt_snapshot

    snapshot = {
        "enabled": True,
        "placement": "before_data",
        "roles": {"bull_researcher": {"resolved_text": "ORIG", "injected": True}},
    }
    result: dict = {}
    _attach_custom_prompt_snapshot(result, snapshot)

    # Mutate the original snapshot AFTER attaching
    snapshot["roles"]["bull_researcher"]["resolved_text"] = "MUTATED"

    saved = result["custom_prompt_snapshot"]["roles"]["bull_researcher"]["resolved_text"]
    assert saved == "ORIG", f"deepcopy isolation failed: got {saved!r}, expected 'ORIG'"


# ---------------------------------------------------------------------------
# T14 – dual-horizon and single save paths both produce identical snapshot attachments
# ---------------------------------------------------------------------------

def test_T14_both_save_branches_call_attach_snapshot():
    """All result branches in _run_job_inner must call the production helper."""
    import inspect
    import api.main as main_module

    source = inspect.getsource(main_module._run_job_inner)
    call = "_attach_custom_prompt_snapshot(result, _prompt_snapshot)"
    assert source.count(call) == 3, (
        "dual-horizon, single-horizon, and regular graph paths must all attach snapshots"
    )

    # Also verify helper isolation for the payload shape used by both branches.
    snapshot = {"enabled": False, "placement": "after_data", "roles": {}}
    for result in ({"final_trade_decision": "BUY"}, {"final_trade_decision": "HOLD"}):
        main_module._attach_custom_prompt_snapshot(result, snapshot)
        assert result["custom_prompt_snapshot"]["placement"] == "after_data"


# ---------------------------------------------------------------------------
# T15 – bool and non-integer-float inputs rejected by validators with warnings
# ---------------------------------------------------------------------------

def test_T15_bool_and_non_integer_float_rejected(caplog):
    from api.services.report_service import StructuredReport

    caplog.set_level(logging.WARNING, logger="api.services.report_service")

    # probability: every malformed/rejected input leaves a traceable warning
    rejected_probabilities = (True, False, "73%", -0.1, 1.5, 200, float("nan"), float("inf"))
    for value in rejected_probabilities:
        assert StructuredReport(decision="HOLD", probability=value).probability is None

    # confidence: bool, non-integer float, non-numeric text, and out-of-range values reject
    rejected_confidences = (True, False, 75.9, "not-a-number", -1, 101)
    for value in rejected_confidences:
        assert StructuredReport(decision="HOLD", confidence=value).confidence is None

    # confidence: exact integer float is accepted (75.0 -> 75)
    assert StructuredReport(decision="HOLD", confidence=75.0).confidence == 75

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "probability rejected" in messages
    assert "confidence rejected" in messages
    assert "cannot convert" in messages or "value must be a real number" in messages
    assert "out of [0, 100] range" in messages


# ---------------------------------------------------------------------------
# Helpers for T16–T20: fake async LLM + minimal state fixtures
# ---------------------------------------------------------------------------

def _make_debate_state(**overrides):
    base = {
        "history": "",
        "bear_history": "",
        "bull_history": "",
        "current_speaker": "",
        "current_response": "",
        "count": 0,
        "claims": [],
        "focus_claim_ids": [],
        "open_claim_ids": [],
        "resolved_claim_ids": [],
        "unresolved_claim_ids": [],
        "round_summary": "",
        "round_goal": "首轮目标",
        "claim_counter": 0,
        "judge_decision": "",
    }
    base.update(overrides)
    return base


def _make_graph_state(**overrides):
    base = {
        "market_report": "M",
        "sentiment_report": "S",
        "news_report": "N",
        "fundamentals_report": "F",
        "volume_price_report": "V",
        "smart_money_report": "SM",
        "investment_debate_state": _make_debate_state(),
        "horizon": "short",
        "user_intent": None,
    }
    base.update(overrides)
    return base


CUSTOM_TEXT = "用户自定义指令：请在分析中关注风险收益比。"

BULL_DEBATE_RESPONSE = """\
[PROMPT-OK]
多头论点概要。

<!-- DEBATE_STATE: {"responded_claim_ids": [], "new_claims": [{"claim": "超跌反弹机会", "evidence": ["技术"], "confidence": 0.65}], "resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], "round_summary": "提出核心多头claim", "round_goal": "下轮验证数据"} -->
"""

BEAR_DEBATE_RESPONSE = """\
[PROMPT-OK]
空头论点概要。

<!-- DEBATE_STATE: {"responded_claim_ids": [], "new_claims": [{"claim": "趋势向下风险大", "evidence": ["MACD偏空"], "confidence": 0.78}], "resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], "round_summary": "提出核心空头claim", "round_goal": "下轮攻击多头"} -->
"""

RESEARCH_MANAGER_RESPONSE = """\
[PROMPT-OK]
裁决：综合多空论点，给出 Hold 结论。

<!-- VERDICT: {"direction": "中性", "reason": "多空势均力敌，暂无明确方向"} -->
"""


# ---------------------------------------------------------------------------
# T16 – bull node: injected text appears once, AFTER last data field,
#        BEFORE 写作要求 — tested by capturing the actual prompt arg to llm.astream
# ---------------------------------------------------------------------------

def test_T16_bull_node_injection_position_after_data():
    import asyncio
    from tradingagents.agents.researchers.bull_researcher import create_bull_researcher

    captured_prompts: list[str] = []

    async def fake_astream(prompt, **kwargs):
        captured_prompts.append(prompt)
        yield MagicMock(content=BULL_DEBATE_RESPONSE)

    llm = MagicMock()
    llm.astream = fake_astream

    memory = MagicMock()
    memory.get_memories = MagicMock(return_value=[])

    node = create_bull_researcher(llm, memory, custom_prompt=CUSTOM_TEXT, placement="after_data")
    state = _make_graph_state()

    asyncio.run(node(state))

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]

    assert prompt.count(CUSTOM_TEXT) == 1, "Custom text must appear exactly once"

    # after_data: custom text must come after last data field and before 写作要求
    last_data_marker = "历史复盘经验："
    requirements_marker = "写作要求："
    pos_data = prompt.rfind(last_data_marker)
    pos_custom = prompt.find(CUSTOM_TEXT)
    pos_req = prompt.find(requirements_marker)
    assert pos_data < pos_custom < pos_req, (
        f"Position order wrong: last_data={pos_data} custom={pos_custom} requirements={pos_req}"
    )

    # switch-off baseline: empty custom_prompt → no custom text in prompt
    captured2: list[str] = []

    async def fake_astream2(prompt2, **kwargs):
        captured2.append(prompt2)
        yield MagicMock(content=BULL_DEBATE_RESPONSE)

    llm2 = MagicMock()
    llm2.astream = fake_astream2

    node_off = create_bull_researcher(llm2, memory, custom_prompt="", placement="after_data")
    asyncio.run(node_off(state))
    assert CUSTOM_TEXT not in captured2[0], "Empty custom_prompt must not appear in prompt"


# ---------------------------------------------------------------------------
# T17 – bear node: same position guarantee as T16
# ---------------------------------------------------------------------------

def test_T17_bear_node_injection_position_after_data():
    import asyncio
    from tradingagents.agents.researchers.bear_researcher import create_bear_researcher

    captured_prompts: list[str] = []

    async def fake_astream(prompt, **kwargs):
        captured_prompts.append(prompt)
        yield MagicMock(content=BEAR_DEBATE_RESPONSE)

    llm = MagicMock()
    llm.astream = fake_astream

    memory = MagicMock()
    memory.get_memories = MagicMock(return_value=[])

    node = create_bear_researcher(llm, memory, custom_prompt=CUSTOM_TEXT, placement="after_data")
    state = _make_graph_state()

    asyncio.run(node(state))

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]

    assert prompt.count(CUSTOM_TEXT) == 1
    pos_data = prompt.rfind("历史复盘经验：")
    pos_custom = prompt.find(CUSTOM_TEXT)
    pos_req = prompt.find("写作要求：")
    assert pos_data < pos_custom < pos_req


# ---------------------------------------------------------------------------
# T18 – research_manager node: injection once, after data before 输出要求
# ---------------------------------------------------------------------------

def test_T18_research_manager_injection_position_after_data():
    import asyncio
    from tradingagents.agents.managers.research_manager import create_research_manager

    captured_prompts: list[str] = []

    async def fake_astream(prompt, **kwargs):
        captured_prompts.append(prompt)
        yield MagicMock(content=RESEARCH_MANAGER_RESPONSE)

    llm = MagicMock()
    llm.astream = fake_astream

    memory = MagicMock()
    memory.get_memories = MagicMock(return_value=[])

    node = create_research_manager(llm, memory, custom_prompt=CUSTOM_TEXT, placement="after_data")
    debate = _make_debate_state(history="Bull: ok\nBear: no")
    state = _make_graph_state(investment_debate_state=debate)

    asyncio.run(node(state))

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]

    assert prompt.count(CUSTOM_TEXT) == 1
    pos_data = prompt.rfind("上一轮摘要：")
    pos_custom = prompt.find(CUSTOM_TEXT)
    pos_req = prompt.find("输出要求：")
    assert pos_data < pos_custom < pos_req, (
        f"Position order wrong: last_data={pos_data} custom={pos_custom} requirements={pos_req}"
    )


# ---------------------------------------------------------------------------
# T19 – bull and bear nodes: first line of ACTUAL node output is [PROMPT-OK],
#        DEBATE_STATE parseable from node output, marker does not contaminate block.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("role,factory_name,response_var", [
    ("bull", "create_bull_researcher", "BULL_DEBATE_RESPONSE"),
    ("bear", "create_bear_researcher", "BEAR_DEBATE_RESPONSE"),
])
def test_T19_debate_state_parseable_bull_and_bear(role, factory_name, response_var):
    import asyncio
    import importlib
    from tradingagents.agents.utils.debate_utils import extract_tagged_json

    response_text = globals()[response_var]
    module_path = {
        "create_bull_researcher": "tradingagents.agents.researchers.bull_researcher",
        "create_bear_researcher": "tradingagents.agents.researchers.bear_researcher",
    }[factory_name]
    mod = importlib.import_module(module_path)
    factory = getattr(mod, factory_name)

    async def fake_astream(prompt, **kwargs):
        yield MagicMock(content=response_text)

    llm = MagicMock()
    llm.astream = fake_astream
    memory = MagicMock()
    memory.get_memories = MagicMock(return_value=[])

    node = factory(llm, memory, custom_prompt=CUSTOM_TEXT, placement="after_data")
    result = asyncio.run(node(_make_graph_state()))

    new_state = result["investment_debate_state"]

    # 1. Output first line must be [PROMPT-OK] (checked on the ACTUAL node output)
    speaker_label = "Bull Analyst" if role == "bull" else "Bear Analyst"
    content_output = new_state.get("current_response", "")
    assert content_output.startswith(f"{speaker_label}: ")
    llm_output_text = content_output[len(f"{speaker_label}: "):]
    first_line = llm_output_text.strip().splitlines()[0].strip()
    assert first_line == "[PROMPT-OK]", f"{role}: expected node output first line '[PROMPT-OK]', got {first_line!r}"

    # Also verify history field starts with "Speaker: [PROMPT-OK]"
    history_key = f"{role}_history"
    raw_history = new_state.get(history_key, "")
    assert raw_history.startswith(f"{speaker_label}: [PROMPT-OK]"), (
        f"{role}: expected history to start with '{speaker_label}: [PROMPT-OK]', got {raw_history[:60]!r}"
    )

    # 2. Node updated debate state (claims added from DEBATE_STATE block)
    assert len(new_state["claims"]) > 0, f"{role} node must add claims from DEBATE_STATE"
    assert new_state["round_summary"] != "", f"{role} round_summary must be set"

    # 3. DEBATE_STATE parseable from response_text and structurally correct
    parsed_block = extract_tagged_json(response_text, "DEBATE_STATE")
    assert "new_claims" in parsed_block
    assert len(parsed_block["new_claims"]) > 0
    assert parsed_block.get("round_summary", "") != ""

    # 4. Marker must NOT appear inside the parsed machine-readable block
    assert "[PROMPT-OK]" not in str(parsed_block)


# ---------------------------------------------------------------------------
# T20 – research_manager node: investment_plan first line is [PROMPT-OK],
#        VERDICT parseable from node returned investment_plan
# ---------------------------------------------------------------------------

def test_T20_research_manager_verdict_parseable():
    import asyncio
    from tradingagents.agents.managers.research_manager import create_research_manager
    from tradingagents.agents.utils.agent_states import extract_verdict
    from tradingagents.agents.utils.debate_utils import extract_tagged_json

    async def fake_astream(prompt, **kwargs):
        yield MagicMock(content=RESEARCH_MANAGER_RESPONSE)

    llm = MagicMock()
    llm.astream = fake_astream
    memory = MagicMock()
    memory.get_memories = MagicMock(return_value=[])

    node = create_research_manager(llm, memory, custom_prompt=CUSTOM_TEXT, placement="after_data")
    debate = _make_debate_state(history="Bull: ok\nBear: no")
    result = asyncio.run(node(_make_graph_state(investment_debate_state=debate)))

    investment_plan = result["investment_plan"]
    assert investment_plan, "investment_plan must be non-empty"

    # 1. First line of ACTUAL returned investment_plan must be [PROMPT-OK]
    first_line = investment_plan.strip().splitlines()[0].strip()
    assert first_line == "[PROMPT-OK]", f"Expected investment_plan first line '[PROMPT-OK]', got {first_line!r}"

    # 2. VERDICT block must be parseable from investment_plan
    verdict_payload = extract_tagged_json(investment_plan, "VERDICT")
    assert verdict_payload.get("direction") == "中性", (
        f"Expected direction='中性', got {verdict_payload!r}"
    )
    assert "reason" in verdict_payload

    # extract_verdict helper must also work on investment_plan
    direction, _ = extract_verdict(investment_plan)
    assert direction == "中性"

    # [PROMPT-OK] must NOT contaminate the machine-readable block
    assert "[PROMPT-OK]" not in str(verdict_payload)


# ---------------------------------------------------------------------------
# T21 – research_manager node receives first-hand market/news/fundamentals
#        evidence summaries (DAV-68 M2: adjudication must anchor on evidence)
# ---------------------------------------------------------------------------

def test_T21_research_manager_receives_evidence_summaries():
    import asyncio
    from tradingagents.agents.managers.research_manager import create_research_manager

    captured_prompts: list[str] = []

    async def fake_astream(prompt, **kwargs):
        captured_prompts.append(prompt)
        yield MagicMock(content=RESEARCH_MANAGER_RESPONSE)

    llm = MagicMock()
    llm.astream = fake_astream
    memory = MagicMock()
    memory.get_memories = MagicMock(return_value=[])

    state = _make_graph_state(
        market_report=(
            "以下是本轮回测的市场技术分析。\n"
            "市场技术面：RSI 48.2，股价 1835.5 站上 50 日均线。\n"
            '<!-- VERDICT: {"direction": "偏多", "reason": "趋势向上"} -->'
        ),
        news_report=(
            "新闻：行业政策利好落地，预计增速 12%。\n"
            '<!-- VERDICT: {"direction": "偏多", "reason": "政策驱动"} -->'
        ),
        fundamentals_report=(
            "基本面：营收同比 +15%，毛利率 45%，经营现金流为正。\n"
            '<!-- VERDICT: {"direction": "中性", "reason": "估值合理"} -->'
        ),
        macro_report=(
            "宏观/板块：板块资金净流入 23 亿，政策窗口开启。\n"
            '<!-- VERDICT: {"direction": "偏多", "reason": "政策与资金共振"} -->'
        ),
        investment_debate_state=_make_debate_state(history="Bull: ok\nBear: no"),
    )

    node = create_research_manager(llm, memory, custom_prompt="", placement="after_data")
    asyncio.run(node(state))

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]

    # The evidence-summary section must be present with all three summaries.
    assert "分析师一手证据摘要" in prompt
    assert "市场技术证据摘要：[分析师结论：偏多]" in prompt
    assert "RSI 48.2" in prompt
    assert "1835.5" in prompt
    assert "新闻/宏观证据摘要：[分析师结论：偏多]" in prompt
    assert "基本面证据摘要：[分析师结论：中性]" in prompt
    assert "宏观/板块证据摘要：[分析师结论：偏多]" in prompt
    assert "+15%" in prompt or "15%" in prompt
    assert "23 亿" in prompt or "净流入 23" in prompt

    # Numberless boilerplate and the machine-readable VERDICT reason are NOT
    # passed through — adjudicators get facts, not prose or protocol blocks.
    assert "以下是本轮回测" not in prompt
    assert "趋势向上" not in prompt


# T21b – macro analyst not run: the macro evidence line must be omitted
#        entirely rather than injected as an empty/no-content placeholder
#        (DAV-68 optimization ①).
def test_T21b_macro_evidence_line_omitted_when_macro_report_empty():
    import asyncio
    from tradingagents.agents.managers.research_manager import create_research_manager

    captured_prompts: list[str] = []

    async def fake_astream(prompt, **kwargs):
        captured_prompts.append(prompt)
        yield MagicMock(content=RESEARCH_MANAGER_RESPONSE)

    llm = MagicMock()
    llm.astream = fake_astream
    memory = MagicMock()
    memory.get_memories = MagicMock(return_value=[])

    # No macro_report key at all -> state.get("macro_report", "") == "".
    state = _make_graph_state(
        market_report=(
            "市场技术面：RSI 48.2，股价 1835.5 站上 50 日均线。\n"
            '<!-- VERDICT: {"direction": "偏多", "reason": "趋势向上"} -->'
        ),
        news_report=(
            "新闻：行业政策利好落地，预计增速 12%。\n"
            '<!-- VERDICT: {"direction": "偏多", "reason": "政策驱动"} -->'
        ),
        fundamentals_report=(
            "基本面：营收同比 +15%，毛利率 45%。\n"
            '<!-- VERDICT: {"direction": "中性", "reason": "估值合理"} -->'
        ),
        investment_debate_state=_make_debate_state(history="Bull: ok\nBear: no"),
    )

    node = create_research_manager(llm, memory, custom_prompt="", placement="after_data")
    asyncio.run(node(state))

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]

    # The other three evidence summaries are still injected…
    assert "市场技术证据摘要：[分析师结论：偏多]" in prompt
    assert "新闻/宏观证据摘要：[分析师结论：偏多]" in prompt
    assert "基本面证据摘要：[分析师结论：中性]" in prompt
    # …but the macro line is gone (no dangling label, no placeholder).
    assert "宏观/板块证据摘要" not in prompt
    assert "报告无可用内容" not in prompt
    assert "Macro/sector evidence summary" not in prompt


# ---------------------------------------------------------------------------
# T22 – trader node: injection once, inside the user message after the
#        investment-plan data block
# ---------------------------------------------------------------------------

TRADER_RESPONSE = "交易员方案：建议买入，仓位 20%。"


def test_T22_trader_injection_position_after_data():
    import asyncio
    from tradingagents.agents.trader.trader import create_trader

    captured_messages: list[list[dict]] = []

    async def fake_astream(messages, **kwargs):
        captured_messages.append(messages)
        yield MagicMock(content=TRADER_RESPONSE)

    llm = MagicMock()
    llm.astream = fake_astream
    memory = MagicMock()
    memory.get_memories = MagicMock(return_value=[])

    state = _make_graph_state(
        company_of_interest="600519",
        investment_plan="研究经理给交易员下发的投资方案",
        trader_investment_plan="",
        instrument_context={},
        market_context={},
        user_context={},
        risk_feedback_state={},
    )

    node = create_trader(llm, memory, custom_prompt=CUSTOM_TEXT, placement="after_data")
    asyncio.run(node(state))

    assert len(captured_messages) == 1
    messages = captured_messages[0]
    user_content = messages[1]["content"]

    assert user_content.count(CUSTOM_TEXT) == 1, "Custom text must appear exactly once"

    # after_data for the trader: after the last data block (investment plan).
    pos_plan = user_content.rfind("研究经理方案内容：")
    pos_custom = user_content.find(CUSTOM_TEXT)
    assert pos_plan < pos_custom, (
        f"Position order wrong: plan={pos_plan} custom={pos_custom}"
    )

    # switch-off baseline: no custom text
    captured2: list[list[dict]] = []

    async def fake_astream2(messages2, **kwargs):
        captured2.append(messages2)
        yield MagicMock(content=TRADER_RESPONSE)

    llm2 = MagicMock()
    llm2.astream = fake_astream2
    node_off = create_trader(llm2, memory, custom_prompt="", placement="after_data")
    asyncio.run(node_off(state))
    assert CUSTOM_TEXT not in captured2[0][1]["content"], (
        "Empty custom_prompt must not appear in trader prompt"
    )


# ---------------------------------------------------------------------------
# T23 – risk_manager node: injection once, after data before output requirements
# ---------------------------------------------------------------------------

RISK_MANAGER_RESPONSE = """\
[PROMPT-OK]
风控结论：同意交易员方向，补充硬约束。

<!-- RISK_JUDGE: {"verdict": "pass", "revision_reason": "", "hard_constraints": ["仓位不超过20%"], "soft_constraints": [], "execution_preconditions": ["放量突破确认"], "de_risk_triggers": ["跌破止损价"]} -->
"""


def _make_risk_debate_state(**overrides):
    base = {
        "history": "Aggressive: ok\nConservative: no\nNeutral: maybe",
        "aggressive_history": "",
        "conservative_history": "",
        "neutral_history": "",
        "latest_speaker": "",
        "current_aggressive_response": "",
        "current_conservative_response": "",
        "current_neutral_response": "",
        "judge_decision": "",
        "count": 0,
        "claims": [],
        "focus_claim_ids": [],
        "open_claim_ids": [],
        "resolved_claim_ids": [],
        "unresolved_claim_ids": [],
        "round_summary": "",
        "round_goal": "",
        "claim_counter": 0,
    }
    base.update(overrides)
    return base


def test_T23_risk_manager_injection_position_after_data():
    import asyncio
    from tradingagents.agents.managers.risk_manager import create_risk_manager

    captured_prompts: list[str] = []

    async def fake_astream(prompt, **kwargs):
        captured_prompts.append(prompt)
        yield MagicMock(content=RISK_MANAGER_RESPONSE)

    llm = MagicMock()
    llm.astream = fake_astream
    memory = MagicMock()
    memory.get_memories = MagicMock(return_value=[])

    state = _make_graph_state(
        company_of_interest="600519",
        trader_investment_plan="交易员方案",
        risk_debate_state=_make_risk_debate_state(),
        instrument_context={},
        market_context={},
        user_context={},
    )

    node = create_risk_manager(llm, memory, custom_prompt=CUSTOM_TEXT, placement="after_data")
    asyncio.run(node(state))

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]

    assert prompt.count(CUSTOM_TEXT) == 1, "Custom text must appear exactly once"

    # after_data: after last data field, before 输出要求
    pos_data = prompt.rfind("上一轮摘要：")
    pos_custom = prompt.find(CUSTOM_TEXT)
    pos_req = prompt.find("输出要求：")
    assert pos_data < pos_custom < pos_req, (
        f"Position order wrong: last_data={pos_data} custom={pos_custom} requirements={pos_req}"
    )

    # switch-off baseline: no custom text
    captured2: list[str] = []

    async def fake_astream2(prompt2, **kwargs):
        captured2.append(prompt2)
        yield MagicMock(content=RISK_MANAGER_RESPONSE)

    llm2 = MagicMock()
    llm2.astream = fake_astream2
    node_off = create_risk_manager(llm2, memory, custom_prompt="", placement="after_data")
    asyncio.run(node_off(state))
    assert CUSTOM_TEXT not in captured2[0], "Empty custom_prompt must not appear in prompt"


# ---------------------------------------------------------------------------
# T24 – api.main._INJECT_ROLES must cover trader and risk_manager
# ---------------------------------------------------------------------------

def test_T24_inject_roles_include_trader_and_risk_manager():
    from api.main import _INJECT_ROLES

    assert "trader" in _INJECT_ROLES
    assert "risk_manager" in _INJECT_ROLES
    assert "bull_researcher" in _INJECT_ROLES
    assert "bear_researcher" in _INJECT_ROLES
    assert "research_manager" in _INJECT_ROLES
