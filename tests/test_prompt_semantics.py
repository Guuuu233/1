"""Deterministic red tests for the P5 prompt and structured-data contracts.

These tests deliberately use fixed prompt text, fake ``invoke``/``astream``
implementations, and in-memory SQLite.  They must not call a real model,
market-data provider, or network service.
"""
from __future__ import annotations

import asyncio
import json
import logging
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


TARGET_ROLES = ("bull_researcher", "bear_researcher", "research_manager")
LANGUAGES = ("zh", "en")
PLACEMENTS = ("before_data", "after_data")
CUSTOM_SENTINEL = "P5_FIXED_CUSTOM_SENTINEL_7f3a"


class _RecordingStreamLLM:
    """Small async fake that records the exact prompt supplied to ``astream``."""

    def __init__(self, response: str = "固定 fake 输出") -> None:
        self.response = response
        self.prompts: list[str] = []

    async def astream(self, prompt: str, **_kwargs):
        self.prompts.append(prompt)
        yield SimpleNamespace(content=self.response)


class _RecordingInvokeLLM:
    """Small sync fake used by structured extraction tests."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def invoke(self, messages, **_kwargs):
        self.prompts.append(messages[0].content)
        return SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False))


def _set_prompt_language(monkeypatch, language: str) -> None:
    from tradingagents.dataflows import config as config_module

    current = config_module.get_config()
    monkeypatch.setattr(
        config_module,
        "_config",
        {**current, "prompt_language": language},
    )


def _make_investment_debate_state(**overrides):
    state = {
        "history": "",
        "bull_history": "",
        "bear_history": "",
        "current_speaker": "",
        "current_response": "",
        "count": 0,
        "claims": [],
        "focus_claim_ids": [],
        "open_claim_ids": [],
        "resolved_claim_ids": [],
        "unresolved_claim_ids": [],
        "round_summary": "",
        "round_goal": "固定目标",
        "claim_counter": 0,
        "judge_decision": "",
    }
    state.update(overrides)
    return state


def _make_graph_state(**overrides):
    state = {
        "market_report": "固定市场报告",
        "sentiment_report": "固定情绪报告",
        "news_report": "固定新闻报告",
        "fundamentals_report": "固定基本面报告",
        "volume_price_report": "固定量价报告",
        "smart_money_report": "固定主力报告",
        "investment_debate_state": _make_investment_debate_state(),
        "horizon": "short",
        "user_intent": None,
    }
    state.update(overrides)
    return state


def _make_risk_state(**overrides):
    from tradingagents.agents.utils.debate_utils import build_empty_risk_debate_state

    state = build_empty_risk_debate_state()
    state.update({"round_summary": "原始摘要", "round_goal": "原始目标"})
    state.update(overrides)
    return state


def _machine_block(marker: str, payload: dict) -> str:
    return f"固定正文\n<!-- {marker}: {json.dumps(payload, ensure_ascii=False, allow_nan=True)} -->"


def _valid_machine_payload(*, confidence: float = 0.72, claim: str = "固定 claim") -> dict:
    return {
        "responded_claim_ids": [],
        "new_claims": [
            {
                "claim": claim,
                "evidence": ["固定证据"],
                "confidence": confidence,
                "target_claim_ids": [],
            }
        ],
        "resolved_claim_ids": [],
        "unresolved_claim_ids": [],
        "next_focus_claim_ids": [],
        "round_summary": "固定摘要",
        "round_goal": "固定目标",
    }


def _apply_machine_block(state: dict, raw_response: str, marker: str) -> dict:
    from tradingagents.agents.utils.debate_utils import update_debate_state_with_payload

    if marker == "DEBATE_STATE":
        kwargs = {
            "speaker_label": "Bull Analyst",
            "speaker_key": "Bull",
            "stance": "bullish",
            "history_key": "bull_history",
            "claim_prefix": "INV",
            "domain": "investment",
            "speaker_field": "current_speaker",
        }
    else:
        kwargs = {
            "speaker_label": "Aggressive Analyst",
            "speaker_key": "Aggressive",
            "stance": "aggressive",
            "history_key": "aggressive_history",
            "claim_prefix": "RISK",
            "domain": "risk",
            "speaker_field": "latest_speaker",
        }
    return update_debate_state_with_payload(
        state=state,
        raw_response=raw_response,
        marker=marker,
        store_current_response=True,
        **kwargs,
    )


def _canonical_state_snapshot(state: dict) -> dict:
    keys = (
        "claims",
        "claim_counter",
        "open_claim_ids",
        "resolved_claim_ids",
        "unresolved_claim_ids",
        "focus_claim_ids",
        "round_summary",
        "round_goal",
    )
    return {key: deepcopy(state.get(key)) for key in keys}


def _capture_factory_prompt(
    monkeypatch,
    role: str,
    language: str,
    *,
    custom_prompt: str,
    placement: str,
) -> str:
    _set_prompt_language(monkeypatch, language)

    factory_paths = {
        "bull_researcher": (
            "tradingagents.agents.researchers.bull_researcher",
            "create_bull_researcher",
        ),
        "bear_researcher": (
            "tradingagents.agents.researchers.bear_researcher",
            "create_bear_researcher",
        ),
        "research_manager": (
            "tradingagents.agents.managers.research_manager",
            "create_research_manager",
        ),
    }
    module_path, factory_name = factory_paths[role]
    module = __import__(module_path, fromlist=[factory_name])
    factory = getattr(module, factory_name)

    llm = _RecordingStreamLLM()
    memory = MagicMock()
    memory.get_memories.return_value = []
    node = factory(
        llm,
        memory,
        custom_prompt=custom_prompt,
        placement=placement,
    )
    asyncio.run(node(_make_graph_state()))
    assert len(llm.prompts) == 1
    return llm.prompts[0]


@pytest.mark.parametrize(
    "value, expected",
    [(0, 0), (1, 1), (100, 100), (75.0, 75)],
)
def test_report_confidence_accepts_integer_semantics(value, expected):
    from api.services.report_service import StructuredReport

    report = StructuredReport(decision="HOLD", confidence=value)
    assert report.confidence == expected


@pytest.mark.parametrize(
    "value",
    [True, False, 75.9, -1, 101, float("nan"), float("inf"), "not-a-number"],
)
def test_report_confidence_rejects_invalid_values_with_warning(caplog, value):
    from api.services.report_service import StructuredReport

    with caplog.at_level(logging.WARNING, logger="api.services.report_service"):
        report = StructuredReport(decision="HOLD", confidence=value)

    assert report.confidence is None
    assert any("confidence rejected" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("value, expected", [(0, 0.0), (1, 1.0), (0.70, 0.70)])
def test_report_probability_accepts_finite_unit_interval(value, expected):
    from api.services.report_service import StructuredReport

    report = StructuredReport(decision="HOLD", probability=value)
    assert report.probability == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    [True, False, -0.1, 1.01, 70, float("nan"), float("inf"), "70%"],
)
def test_report_probability_rejects_invalid_values_with_warning(caplog, value):
    from api.services.report_service import StructuredReport

    with caplog.at_level(logging.WARNING, logger="api.services.report_service"):
        report = StructuredReport(decision="HOLD", probability=value)

    assert report.probability is None
    assert any("probability rejected" in record.getMessage() for record in caplog.records)


def test_report_confidence_never_fills_probability():
    from api.services.report_service import StructuredReport

    report = StructuredReport(decision="BUY", confidence=75, probability=None)
    assert report.confidence == 75
    assert report.probability is None


@pytest.mark.parametrize("marker", ["DEBATE_STATE", "RISK_STATE"])
def test_claim_confidence_keeps_only_finite_unit_values(caplog, marker):
    invalid_claims = [
        {"claim": "缺失 confidence", "evidence": ["证据"]},
        {"claim": "越界 high", "evidence": ["证据"], "confidence": 1.01},
        {"claim": "越界 low", "evidence": ["证据"], "confidence": -0.1},
        {"claim": "NaN", "evidence": ["证据"], "confidence": float("nan")},
        {"claim": "Infinity", "evidence": ["证据"], "confidence": float("inf")},
        {"claim": "bool", "evidence": ["证据"], "confidence": True},
        {"claim": "text", "evidence": ["证据"], "confidence": "not-a-number"},
    ]
    payload = _valid_machine_payload(confidence=0, claim="合法 zero")
    payload["new_claims"].extend(
        [
            {"claim": "合法 middle", "evidence": ["证据"], "confidence": 0.72},
            {"claim": "合法 one", "evidence": ["证据"], "confidence": 1},
            *invalid_claims,
        ]
    )

    state = _make_investment_debate_state() if marker == "DEBATE_STATE" else _make_risk_state()
    with caplog.at_level(logging.WARNING):
        result = _apply_machine_block(state, _machine_block(marker, payload), marker)

    claims = {claim["claim"]: claim for claim in result["claims"]}
    assert claims["合法 zero"]["confidence"] == 0
    assert claims["合法 middle"]["confidence"] == pytest.approx(0.72)
    assert claims["合法 one"]["confidence"] == 1
    for invalid in invalid_claims:
        assert invalid["claim"] not in claims
    assert all(claim.get("confidence") != pytest.approx(0.6) for claim in claims.values())

    warning_messages = [record.getMessage() for record in caplog.records]
    assert warning_messages, "invalid claims need an observable parser warning"
    assert any("confidence" in message.lower() for message in warning_messages)


@pytest.mark.parametrize("marker", ["DEBATE_STATE", "RISK_STATE"])
@pytest.mark.parametrize("case", ["missing", "bad_json", "duplicate", "wrong_type"])
def test_bad_or_duplicate_machine_blocks_do_not_mutate_canonical_state(caplog, marker, case):
    payload = _valid_machine_payload(claim=f"{marker} claim")
    valid_block = _machine_block(marker, payload)
    if case == "missing":
        raw_response = "固定正文，没有机读块"
    elif case == "bad_json":
        raw_response = f"固定正文\n<!-- {marker}: {{\"new_claims\": [}} -->"
    elif case == "duplicate":
        raw_response = f"{valid_block}\n{valid_block}"
    else:
        raw_response = _machine_block(
            marker,
            {**payload, "new_claims": "必须是数组"},
        )

    state = _make_investment_debate_state() if marker == "DEBATE_STATE" else _make_risk_state()
    before = _canonical_state_snapshot(state)
    with caplog.at_level(logging.WARNING):
        result = _apply_machine_block(state, raw_response, marker)

    assert _canonical_state_snapshot(result) == before
    warning_messages = [record.getMessage() for record in caplog.records]
    assert any(marker in message for message in warning_messages)
    assert any(
        any(term in message.lower() for term in ("parse", "invalid", "duplicate", "missing", "重复", "失败"))
        for message in warning_messages
    )


@pytest.mark.parametrize("marker", ["DEBATE_STATE", "RISK_STATE"])
def test_valid_and_malformed_duplicate_machine_blocks_are_rejected(caplog, marker):
    payload = _valid_machine_payload(claim=f"{marker} canonical claim")
    valid_block = _machine_block(marker, payload)
    malformed_duplicate = (
        f"固定正文\n<!-- {marker} {json.dumps(payload, ensure_ascii=False, allow_nan=True)} -->"
    )
    raw_response = f"{valid_block}\n{malformed_duplicate}"

    state = _make_investment_debate_state() if marker == "DEBATE_STATE" else _make_risk_state()
    before = _canonical_state_snapshot(state)
    with caplog.at_level(logging.WARNING):
        result = _apply_machine_block(state, raw_response, marker)

    assert _canonical_state_snapshot(result) == before
    assert result["claims"] == before["claims"]
    warning_messages = [record.getMessage() for record in caplog.records]
    assert any(marker in message for message in warning_messages)
    assert any(
        "duplicate" in message.lower() and "malformed" in message.lower()
        for message in warning_messages
    )


@pytest.mark.parametrize("marker", ["DEBATE_STATE", "RISK_STATE"])
def test_unknown_machine_fields_are_ignored_but_warned(caplog, marker):
    payload = _valid_machine_payload()
    payload["unexpected_root"] = "do not persist"
    payload["new_claims"][0]["unexpected_claim_field"] = "do not persist"
    state = _make_investment_debate_state() if marker == "DEBATE_STATE" else _make_risk_state()

    with caplog.at_level(logging.WARNING):
        result = _apply_machine_block(state, _machine_block(marker, payload), marker)

    assert len(result["claims"]) == 1
    assert "unexpected_root" not in result
    assert "unexpected_claim_field" not in result["claims"][0]
    messages = [record.getMessage().lower() for record in caplog.records]
    assert any("unknown" in message or "extra" in message or "未知" in message for message in messages)


@pytest.mark.parametrize("fixture", [
    {
        "role": "bull_researcher",
        "language": "zh",
        "body": "主分析周期=短期；明确基准价=100；周期末高于100的上涨概率=0.70。",
        "expected": 0.70,
    },
    {
        "role": "bear_researcher",
        "language": "zh",
        "body": "主分析周期=短期；明确基准价=100；周期末高于100的上涨概率=0.30。",
        "expected": 0.30,
    },
    {
        "role": "bull_researcher",
        "language": "en",
        "body": "Primary horizon=short; benchmark price=100; probability of a higher end price=0.70.",
        "expected": 0.70,
    },
    {
        "role": "bear_researcher",
        "language": "en",
        "body": "Primary horizon=short; benchmark price=100; probability of a higher end price=0.30.",
        "expected": 0.30,
    },
])
def test_probability_fixture_is_upside_probability_not_bear_inversion(fixture):
    from api.services import report_service
    from tradingagents.prompts import get_prompt

    prompt_key = fixture["role"].replace("_researcher", "_prompt")
    template = get_prompt(
        prompt_key,
        config={"prompt_language": fixture["language"]},
    )

    if fixture["language"] == "zh":
        normalized = template.replace(" ", "")
        assert "上涨概率" in template
        assert any(term in template for term in ("主分析周期", "主周期"))
        assert "基准价" in template
        assert any(term in template for term in ("期末", "周期结束", "结束时"))
        assert any(
            term in normalized
            for term in (
                "不得反转",
                "不要反转",
                "不可反转",
                "不反转",
                "不得改成下跌概率",
                "不应改成下跌概率",
                "不是下跌概率",
                "只表示上涨概率",
                "不做1-p",
            )
        )
        assert "触发概率" not in template
    else:
        normalized = template.lower().replace(" ", "")
        assert any(term in normalized for term in ("upsideprobability", "probabilityofahigher", "probabilitythat"))
        assert any(term in normalized for term in ("primaryhorizon", "mainhorizon", "analysishorizon"))
        assert any(term in normalized for term in ("benchmarkprice", "baselineprice"))
        assert any(term in normalized for term in ("endof", "endprice", "higher"))
        assert any(
            term in normalized
            for term in (
                "donotinvert",
                "mustnotinvert",
                "neverinvert",
                "notadownsideprobability",
                "notdownsideprobability",
                "donotconverttodownside",
                "mustnotconvert",
                "donotuse1-p",
            )
        )

    fake_llm = _RecordingInvokeLLM(
        {"decision": "HOLD", "confidence": 50, "probability": fixture["expected"]}
    )
    fake_client = SimpleNamespace(get_llm=lambda: fake_llm)
    with patch("tradingagents.llm_clients.create_llm_client", return_value=fake_client):
        structured = report_service.extract_structured_data(
            final_trade_decision=fixture["body"],
            config={"llm_provider": "fake", "quick_think_llm": "fake"},
        )

    assert structured is not None
    assert structured.probability == pytest.approx(fixture["expected"])
    assert fixture["body"] in fake_llm.prompts[0]


@pytest.mark.parametrize("role", TARGET_ROLES)
@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_custom_prompt_on_injects_once_for_every_role_language_and_slot(
    monkeypatch, role, language, placement
):
    prompt = _capture_factory_prompt(
        monkeypatch,
        role,
        language,
        custom_prompt=CUSTOM_SENTINEL,
        placement=placement,
    )
    assert prompt.count(CUSTOM_SENTINEL) == 1


@pytest.mark.parametrize("role", TARGET_ROLES)
@pytest.mark.parametrize("language", LANGUAGES)
def test_custom_prompt_off_is_byte_identical_to_factory_legacy_baseline(monkeypatch, role, language):
    legacy_prompt = _capture_factory_prompt(
        monkeypatch,
        role,
        language,
        custom_prompt="",
        placement="after_data",
    )
    before_data_prompt = _capture_factory_prompt(
        monkeypatch,
        role,
        language,
        custom_prompt="",
        placement="before_data",
    )
    after_data_prompt = _capture_factory_prompt(
        monkeypatch,
        role,
        language,
        custom_prompt="",
        placement="after_data",
    )

    assert before_data_prompt == legacy_prompt
    assert after_data_prompt == legacy_prompt
    assert CUSTOM_SENTINEL not in legacy_prompt


def _make_sqlite_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from api.database import Base

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return session_factory()


@pytest.mark.parametrize(
    "override_name, value, persisted_name",
    [
        ("confidence_override", 101, "confidence"),
        ("confidence_override", 75.9, "confidence"),
        ("probability", 70, "probability"),
        ("probability", 1.01, "probability"),
        ("probability", float("nan"), "probability"),
    ],
)
def test_invalid_legacy_report_overrides_do_not_persist_canonical_values(
    override_name, value, persisted_name
):
    from api.database import ReportDB
    from api.services import report_service

    db = _make_sqlite_session()
    try:
        kwargs = {override_name: value}
        try:
            report = report_service.create_report(
                db=db,
                symbol="600519.SH",
                trade_date="2026-07-31",
                decision="HOLD",
                result_data={"final_trade_decision": "固定正文"},
                **kwargs,
            )
        except (TypeError, ValueError):
            assert db.query(ReportDB).count() == 0
            return

        assert getattr(report, persisted_name) is None
    finally:
        db.close()


def test_report_api_rejects_strict_boundary_bypasses_without_persisting_rows():
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    import api.main as main
    from api.database import Base, ReportDB

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = session_factory()

    def override_get_db():
        yield db

    current_user = SimpleNamespace(id="p5-api-user")
    main.app.dependency_overrides[main.get_db] = override_get_db
    main.app.dependency_overrides[main._require_api_user] = lambda: current_user
    client = TestClient(main.app, raise_server_exceptions=False)

    valid_payload = _valid_machine_payload()
    invalid_responses = [
        _machine_block("DEBATE_STATE", _valid_machine_payload(confidence=None)),
        _machine_block("RISK_STATE", _valid_machine_payload(confidence="0.7")),
        (
            "固定正文\n"
            f"<!-- DEBATE_STATE {json.dumps(valid_payload, ensure_ascii=False)} -->"
        ),
        _machine_block("RISK_STATE", valid_payload).removesuffix(" -->"),
    ]
    try:
        for index, final_trade_decision in enumerate(invalid_responses):
            response = client.post(
                "/v1/reports",
                json={
                    "symbol": f"60051{index:02d}.SH",
                    "trade_date": "2026-07-31",
                    "result_data": {"final_trade_decision": final_trade_decision},
                },
            )
            assert response.status_code == 422, response.text
            assert db.query(ReportDB).count() == 0

        valid_response = client.post(
            "/v1/reports",
            json={
                "symbol": "600519.SH",
                "trade_date": "2026-07-31",
                "result_data": {"final_trade_decision": _machine_block("DEBATE_STATE", valid_payload)},
            },
        )
        assert valid_response.status_code == 200, valid_response.text
        assert db.query(ReportDB).count() == 1
    finally:
        main.app.dependency_overrides.pop(main.get_db, None)
        main.app.dependency_overrides.pop(main._require_api_user, None)
        db.close()
        engine.dispose()


def test_unknown_structured_fields_are_not_canonical_but_body_is_preserved(caplog):
    from api.services import report_service

    body = "固定正文原文：保留 unrequested_body_field=secret，不要求它成为结构化字段。"
    payload = {
        "decision": "HOLD",
        "confidence": 50,
        "probability": None,
        "unexpected_field": "do not persist",
        "risks": [
            {
                "name": "固定风险",
                "level": "medium",
                "description": "固定说明",
                "unexpected_nested_field": "do not persist",
            }
        ],
    }
    fake_llm = _RecordingInvokeLLM(payload)
    fake_client = SimpleNamespace(get_llm=lambda: fake_llm)
    with (
        caplog.at_level(logging.WARNING, logger="api.services.report_service"),
        patch("tradingagents.llm_clients.create_llm_client", return_value=fake_client),
    ):
        structured = report_service.extract_structured_data(
            final_trade_decision=body,
            config={"llm_provider": "fake", "quick_think_llm": "fake"},
        )

    assert structured is not None
    canonical = structured.model_dump()
    assert "unexpected_field" not in canonical
    assert "unexpected_nested_field" not in canonical["risks"][0]
    assert any(
        term in record.getMessage().lower()
        for record in caplog.records
        for term in ("unknown", "extra", "未知")
    )

    result_data = {"final_trade_decision": body, "structured": canonical}
    resolved = report_service.resolve_report_fields(result_data=result_data)
    assert resolved["final_trade_decision"] == body

    db = _make_sqlite_session()
    try:
        report = report_service.create_report(
            db=db,
            symbol="600519.SH",
            trade_date="2026-07-31",
            decision="HOLD",
            result_data=result_data,
            risk_items=canonical["risks"],
        )
        assert report.result_data["final_trade_decision"] == body
        assert "unexpected_field" not in report.result_data["structured"]
        assert "unexpected_nested_field" not in report.risk_items[0]
    finally:
        db.close()
