"""Unit and golden regression tests for evidence verifier fairness and multi-line matching (DAV-338 / Lane C).

Tests:
1. Golden test set execution against tests/golden/evidence_sentences_20260823.json:
   - Positive cases (true facts in reports across single/multiple lines) evaluate to verified.
   - Negative cases (non-existent numbers or coincidental numbers without keyword support) evaluate to unsupported.
2. Granular tests for single-line vs multi-line aggregation.
3. Strict keyword guard to prevent coincidental numeric matches in unrelated contexts.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from tradingagents.agents.utils.evidence_verifier import (
    EvidenceFactualTruthEvaluator,
    STATUS_UNSUPPORTED,
    STATUS_VERIFIED,
    STATUS_CONTRADICTED,
)

GOLDEN_PATH = Path(__file__).parent / "golden" / "evidence_sentences_20260823.json"


def load_golden_cases():
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def evaluator():
    return EvidenceFactualTruthEvaluator()


def test_golden_dataset_exists_and_valid():
    cases = load_golden_cases()
    assert len(cases) >= 15
    pos_cases = [c for c in cases if c.get("is_positive")]
    neg_cases = [c for c in cases if not c.get("is_positive")]
    assert len(pos_cases) >= 12
    assert len(neg_cases) >= 1


@pytest.mark.parametrize("case", load_golden_cases(), ids=lambda c: f"{c['case_id']}_{c['symbol']}_{c['claim_id']}")
def test_golden_evidence_sentence_verdicts(case, evaluator):
    """Verify that each golden evidence sentence produces its expected verdict."""
    raw = case["raw_evidence"]
    seven_reports = case["seven_reports"]
    market_data_context = case.get("market_data_context")
    analysis_baseline_date = case.get("analysis_baseline_date")
    claim_id = case.get("claim_id")
    expected = case["expected_status"]

    res = evaluator.evaluate_single_evidence(
        raw_evidence=raw,
        seven_reports=seven_reports,
        market_data_context=market_data_context,
        analysis_baseline_date=analysis_baseline_date,
        claim_id=claim_id,
    )

    actual_status = res.get("status")
    assert actual_status == expected, (
        f"Case {case['case_id']} ({case['symbol']} {claim_id}) expected {expected}, got {actual_status}. "
        f"Details: {res.get('details')}. Raw evidence: {raw}"
    )
    if expected == STATUS_VERIFIED and case.get("is_positive"):
        # Details should explain single-line or multi_line_match
        assert "验证" in res.get("details", "") or "匹配" in res.get("details", "")


def test_negative_case_coincidental_number_without_keyword_rejected(evaluator):
    """Verify that a number appearing in an unrelated context without shared keywords remains unsupported."""
    seven_reports = {
        "volume_price_report": "- **成交量与量比**：8月21日成交 1.274 亿股，量比达 1.40；收盘位置 0.15。",
        "macro_report": "宏观报告内容，无相关财务数据。",
    }
    # Fabricated claim: quotes numbers 1.4 and 0.15 from volume_price_report but with revenue/margin keywords
    fake_evidence = "公司营收同比增长1.4倍且综合毛利率达到0.15"

    res = evaluator.evaluate_single_evidence(
        raw_evidence=fake_evidence,
        seven_reports=seven_reports,
        claim_id="FAKE-1",
    )
    assert res["status"] == STATUS_UNSUPPORTED
    assert "未在七份分析师报告" in res["details"]


def test_negative_case_nonexistent_derived_number_rejected(evaluator):
    """Verify that an evidence sentence containing a non-existent number is rejected."""
    seven_reports = {
        "news_report": "美的集团完成69.73亿元回购，回购均价79.79元/股。",
    }
    # 5.6% is not in the text
    ev_text = "69.73亿回购均价79.79元低于现价5.6%已充分定价"
    res = evaluator.evaluate_single_evidence(
        raw_evidence=ev_text,
        seven_reports=seven_reports,
        claim_id="INV-4",
    )
    assert res["status"] in (STATUS_UNSUPPORTED, STATUS_CONTRADICTED)


def test_debate_state_prompt_templates_symmetry():
    """Verify that both Bull and Bear DEBATE_STATE prompt examples are symmetric in resolved_claim_ids."""
    import re
    from tradingagents.prompts.zh import PROMPTS

    bull_p = PROMPTS["bull_prompt"]
    bear_p = PROMPTS["bear_prompt"]

    # Extract DEBATE_STATE block content
    pattern = re.compile(r"<!--\s*DEBATE_STATE:\s*(\{\{.*?\}\})\s*-->")
    bull_match = pattern.search(bull_p)
    bear_match = pattern.search(bear_p)

    assert bull_match is not None, "DEBATE_STATE block not found in bull_prompt"
    assert bear_match is not None, "DEBATE_STATE block not found in bear_prompt"

    # Both blocks should have `"resolved_claim_ids": []`
    assert '"resolved_claim_ids": []' in bull_match.group(1), f"Bull prompt resolved_claim_ids is not [] in: {bull_match.group(1)}"
    assert '"resolved_claim_ids": []' in bear_match.group(1), f"Bear prompt resolved_claim_ids is not [] in: {bear_match.group(1)}"


# ── DAV-595: Metric-Number Binding & Pseudo Contradiction Tests ──────────────


def test_reproduce_inv6_pseudo_contradiction_falsified_as_unsupported(evaluator):
    """INV-6 pseudo contradiction: '应收账款增长17.10%''营收增长3.55%' must NOT contradict '概率：25%'."""
    seven_reports = {
        "macro_report": (
            "| **乐观情景 (Bull Case)**<br>*(概率：25%)* | - 宏观逆周期货币与财政双发力；"
            "- 海外OBM出海营收加速上行；- 预期收益弹性空间：+8% 至 +15%。 |"
        )
    }
    raw_evidence = "2026Q2应收账款619.55亿元同比大增17.10%，严重背离营收3.55%的微弱增速"
    res = evaluator.evaluate_single_evidence(
        raw_evidence=raw_evidence,
        seven_reports=seven_reports,
        claim_id="INV-6",
    )
    assert res["status"] == STATUS_UNSUPPORTED
    assert res["status"] != STATUS_CONTRADICTED


def test_same_line_multi_metric_no_cross_binding_verified(evaluator):
    """Same-line multiple metrics must bind numbers to their specific metrics, preventing cross-value verification."""
    seven_reports = {
        "fundamentals_report": (
            "- **财务表现**：2026年Q2营业收入2600.42亿元，同比增长3.55%；综合毛利率达25.57%，净利率为9.81%。"
        )
    }
    # 正确匹配自身指标
    res_rev = evaluator.evaluate_single_evidence(
        raw_evidence="2026年Q2营业收入同比增长3.55%",
        seven_reports=seven_reports,
    )
    assert res_rev["status"] == STATUS_VERIFIED

    res_margin = evaluator.evaluate_single_evidence(
        raw_evidence="2026年Q2综合毛利率达25.57%",
        seven_reports=seven_reports,
    )
    assert res_margin["status"] == STATUS_VERIFIED


def test_same_line_multi_metric_cross_value_negative_not_verified(evaluator):
    """Negative test: Claiming gross margin is 3.55% or revenue growth is 25.57% must NOT be verified against the same line."""
    seven_reports = {
        "fundamentals_report": (
            "- **财务表现**：2026年Q2营业收入2600.42亿元，同比增长3.55%；综合毛利率达25.57%，净利率为9.81%。"
        )
    }
    # 串值负例 1: 将毛利率说成 3.55% (3.55% 属于营收，不属于毛利率)
    res_fake_margin = evaluator.evaluate_single_evidence(
        raw_evidence="2026年Q2综合毛利率为3.55%",
        seven_reports=seven_reports,
    )
    assert res_fake_margin["status"] == STATUS_CONTRADICTED

    # 串值负例 2: 将营收增速说成 25.57% (25.57% 属于毛利率，不属于营收)
    res_fake_rev = evaluator.evaluate_single_evidence(
        raw_evidence="2026年Q2营业收入同比增长25.57%",
        seven_reports=seven_reports,
    )
    assert res_fake_rev["status"] == STATUS_CONTRADICTED


def test_compound_sentence_atomic_evidence_splitting(evaluator):
    """Compound sentence split into atomic statements: each atomic fact must verify against its corresponding report line."""
    seven_reports = {
        "fundamentals_report": (
            "- **营业收入**：2026年Q2单季实现营业收入2600.42亿元，同比增长3.55%。\n"
            "- **营运指标**：2026Q2末应收账款为619.55亿元，同比增长17.10%。"
        )
    }
    compound_ev = "2026Q2应收账款619.55亿元同比大增17.10%，严重背离营收3.55%的微弱增速"
    res = evaluator.evaluate_single_evidence(
        raw_evidence=compound_ev,
        seven_reports=seven_reports,
    )
    assert res["status"] == STATUS_VERIFIED


def test_percentage_conflict_requires_same_metric_unit_period(evaluator):
    """Conflict detection must require matching metric name, unit, and period. Mismatched metric or period cannot trigger contradiction."""
    seven_reports = {
        "macro_report": "| **乐观情景** (概率：25%) | 预计2026Q3海外营收恢复增长 |",
    }
    # 概率 25% 不得与营收 3.55% 冲突
    res = evaluator.evaluate_single_evidence(
        raw_evidence="2026Q2营业收入同比增长3.55%",
        seven_reports=seven_reports,
    )
    assert res["status"] == STATUS_UNSUPPORTED
    assert res["status"] != STATUS_CONTRADICTED


def test_true_conflict_positive_case(evaluator):
    """True conflict positive case: when metric name, unit, and period match but numbers disagree, trigger CONTRADICTED."""
    seven_reports = {
        "fundamentals_report": "2026年Q2综合毛利率为25.57%，营业收入同比增长3.55%。",
    }
    res_conflict = evaluator.evaluate_single_evidence(
        raw_evidence="2026年Q2综合毛利率为15.00%",
        seven_reports=seven_reports,
    )
    assert res_conflict["status"] == STATUS_CONTRADICTED
    assert "毛利率" in res_conflict.get("details", "")


# =====================================================================
# E-03c Deterministic Verification Scenarios
# =====================================================================

def test_e03c_adopt_partial_reject_exact_coverage_boundaries(evaluator):
    """E-03c: Exact boundaries for adopt (100%), partial (>=67% mixed), and reject (<67% or conflict)."""
    from tradingagents.agents.utils.evidence_verifier import (
        aggregate_claim_evidence,
        DECISION_ADOPT,
        DECISION_PARTIAL,
        DECISION_REJECT,
    )

    # 1. 3/3 verified -> adopt
    s1 = aggregate_claim_evidence(
        claims=[{"claim_id": "C-1", "claim": "论点1", "evidence": ["E1", "E2", "E3"]}],
        claims_verification=[
            {"claim_id": "C-1", "raw": "E1", "status": "verified"},
            {"claim_id": "C-1", "raw": "E2", "status": "verified"},
            {"claim_id": "C-1", "raw": "E3", "status": "verified"},
        ],
    )["C-1"]
    assert s1["decision"] == DECISION_ADOPT
    assert s1["coverage"] == 1.0

    # 2. 2/3 verified (66.7%) -> partial
    s2 = aggregate_claim_evidence(
        claims=[{"claim_id": "C-2", "claim": "论点2", "evidence": ["E1", "E2", "E3"]}],
        claims_verification=[
            {"claim_id": "C-2", "raw": "E1", "status": "verified"},
            {"claim_id": "C-2", "raw": "E2", "status": "verified"},
            {"claim_id": "C-2", "raw": "E3", "status": "unsupported"},
        ],
    )["C-2"]
    assert s2["decision"] == DECISION_PARTIAL
    assert len(s2["verified_evidence"]) == 2
    assert s2["unsupported_evidence"] == ["E3"]
    assert s2["excluded_evidence"] == ["E3"]

    # 3. 1/2 verified (50%) -> reject (coverage < 67%)
    s3 = aggregate_claim_evidence(
        claims=[{"claim_id": "C-3", "claim": "论点3", "evidence": ["E1", "E2"]}],
        claims_verification=[
            {"claim_id": "C-3", "raw": "E1", "status": "verified"},
            {"claim_id": "C-3", "raw": "E2", "status": "unsupported"},
        ],
    )["C-3"]
    assert s3["decision"] == DECISION_REJECT
    assert "覆盖率不足" in s3["reason"]

    # 4. 0/2 verified -> reject (unsupported)
    s4 = aggregate_claim_evidence(
        claims=[{"claim_id": "C-4", "claim": "论点4", "evidence": ["E1", "E2"]}],
        claims_verification=[
            {"claim_id": "C-4", "raw": "E1", "status": "unsupported"},
            {"claim_id": "C-4", "raw": "E2", "status": "unsupported"},
        ],
    )["C-4"]
    assert s4["decision"] == DECISION_REJECT
    assert "未提供有效证据" in s4["reason"]

    # 5. Contradicted evidence -> reject even with verified items
    s5 = aggregate_claim_evidence(
        claims=[{"claim_id": "C-5", "claim": "论点5", "evidence": ["E1", "E2", "E3"]}],
        claims_verification=[
            {"claim_id": "C-5", "raw": "E1", "status": "verified"},
            {"claim_id": "C-5", "raw": "E2", "status": "verified"},
            {"claim_id": "C-5", "raw": "E3", "status": "contradicted"},
        ],
    )["C-5"]
    assert s5["decision"] == DECISION_REJECT
    assert "事实冲突" in s5["reason"]

    # 6. Source unavailable -> reject
    s6 = aggregate_claim_evidence(
        claims=[{"claim_id": "C-6", "claim": "论点6", "evidence": ["E1", "E2", "E3"]}],
        claims_verification=[
            {"claim_id": "C-6", "raw": "E1", "status": "verified"},
            {"claim_id": "C-6", "raw": "E2", "status": "verified"},
            {"claim_id": "C-6", "raw": "E3", "status": "source_unavailable", "is_fatal": True},
        ],
    )["C-6"]
    assert s6["decision"] == DECISION_REJECT
    assert "不可用数据源" in s6["reason"]


def test_e03c_pit_anti_lookahead_date_failure(evaluator):
    """E-03c: Anti-lookahead date check fails closed when evidence date exceeds baseline date."""
    seven_reports = {
        "market_report": "2026-08-20收盘价为1500元，放量突破均线。",
    }
    # 1. Past date matches -> verified
    res_past = evaluator.evaluate_single_evidence(
        raw_evidence="2026-08-20收盘价为1500元",
        seven_reports=seven_reports,
        analysis_baseline_date="2026-08-25",
    )
    assert res_past["status"] == STATUS_VERIFIED

    # 2. Future date without forward keywords -> contradicted (lookahead bias)
    res_future = evaluator.evaluate_single_evidence(
        raw_evidence="2026-09-02收盘价创出新高突破1600元",
        seven_reports=seven_reports,
        analysis_baseline_date="2026-08-25",
    )
    assert res_future["status"] == STATUS_CONTRADICTED
    assert "前视偏差" in res_future["details"]

    # 3. Future date with forward target/prediction keyword -> allowed to not be contradicted
    res_pred = evaluator.evaluate_single_evidence(
        raw_evidence="目标2026-09-02前收盘价有望冲击1600元",
        seven_reports=seven_reports,
        analysis_baseline_date="2026-08-25",
    )
    assert res_pred["status"] != STATUS_CONTRADICTED


def test_e03c_source_unavailable_provider_failure(evaluator):
    """E-03c: Citation of failed data source must be marked source_unavailable, not missing or absent in fact."""
    from tradingagents.agents.utils.evidence_verifier import STATUS_SOURCE_UNAVAILABLE

    market_data_context = {
        "data_failure_ledger": [
            {
                "source": "smart_money",
                "name": "主力资金数据流",
                "status": "failed",
                "provenance_status": "unverified",
            }
        ],
        "source_provenance": {
            "smart_money": {"status": "failed"},
        },
    }
    seven_reports = {
        "smart_money_report": "【数据获取失败】smart_money",
    }
    res = evaluator.evaluate_single_evidence(
        raw_evidence="smart_money主力资金净流入2.5亿元",
        seven_reports=seven_reports,
        market_data_context=market_data_context,
    )
    assert res["status"] == STATUS_SOURCE_UNAVAILABLE
    assert res["is_fatal"] is True
    assert "严重幻觉" in res["details"] or "不可用" in res["details"]


def test_e03c_valid_zero_and_decimal_zero_matching(evaluator):
    """E-03c: Zero values (0, 0.0, Decimal(0)) are valid financial metrics and must match reports."""
    seven_reports = {
        "fundamentals_report": "2026年Q2归母净利润同比增长0.0%，营业收入增长0.0%，资产负债率为0%。",
        "smart_money_report": "主力净流入0亿元，超大单净流出0万元。",
    }
    # 1. 0.0% match
    res_profit = evaluator.evaluate_single_evidence(
        raw_evidence="2026年Q2归母净利润同比增长0%",
        seven_reports=seven_reports,
    )
    assert res_profit["status"] == STATUS_VERIFIED

    # 2. 0亿元 match
    res_flow = evaluator.evaluate_single_evidence(
        raw_evidence="主力净流入0亿元",
        seven_reports=seven_reports,
    )
    assert res_flow["status"] == STATUS_VERIFIED

    # 3. Disagreeing with 0% triggers contradiction
    res_diff = evaluator.evaluate_single_evidence(
        raw_evidence="2026年Q2归母净利润同比增长10.0%",
        seven_reports=seven_reports,
    )
    assert res_diff["status"] == STATUS_CONTRADICTED


def test_e03c_observation_and_hypothesis_never_upgraded_to_adopt():
    """E-03c: Observation and hypothesis claims stay in audited partial state and never upgrade to adopt."""
    from tradingagents.agents.utils.evidence_verifier import (
        aggregate_claim_evidence,
        is_observation_or_hypothesis_claim,
        DECISION_PARTIAL,
        DECISION_ADOPT,
    )

    claims = [
        {"claim_id": "CLM-1", "claim": "主力大单净买入", "evidence": ["净流入1亿"], "is_observation": True},
        {"claim_id": "CLM-2", "claim": "【假设】若下周放量突破3000点将加速上涨", "evidence": ["突破60日线"]},
        {"claim_id": "CLM-3", "claim": "题材情绪向好", "claim_type": "hypothesis", "evidence": ["政策支持"]},
        {"claim_id": "CLM-4", "claim": "确凿基本面事实", "claim_type": "fact", "evidence": ["营收2000亿"]},
    ]
    assert is_observation_or_hypothesis_claim(claims[0]) is True
    assert is_observation_or_hypothesis_claim(claims[1]) is True
    assert is_observation_or_hypothesis_claim(claims[2]) is True
    assert is_observation_or_hypothesis_claim(claims[3]) is False

    ver_items = [
        {"claim_id": "CLM-1", "raw": "净流入1亿", "status": "verified"},
        {"claim_id": "CLM-2", "raw": "突破60日线", "status": "verified"},
        {"claim_id": "CLM-3", "raw": "政策支持", "status": "verified"},
        {"claim_id": "CLM-4", "raw": "营收2000亿", "status": "verified"},
    ]
    summary = aggregate_claim_evidence(claims=claims, claims_verification=ver_items)

    assert summary["CLM-1"]["decision"] == DECISION_PARTIAL
    assert summary["CLM-2"]["decision"] == DECISION_PARTIAL
    assert summary["CLM-3"]["decision"] == DECISION_PARTIAL
    assert summary["CLM-4"]["decision"] == DECISION_ADOPT


def test_e03c_preserves_raw_evidence_source_and_state_change_reasons():
    """E-03c: Verifier retains raw evidence, sources, and state change reasons for auditability."""
    from tradingagents.agents.utils.evidence_verifier import aggregate_claim_evidence

    claim = {
        "claim_id": "INV-AUDIT",
        "speaker": "Bull Analyst",
        "speaker_key": "Bull",
        "stance": "bullish",
        "claim": "量价综合立论",
        "evidence": ["放量突破", "虚假指标", "冲突数据"],
    }
    ver_items = [
        {"claim_id": "INV-AUDIT", "raw": "放量突破", "status": "verified", "matched_role": "market_report"},
        {"claim_id": "INV-AUDIT", "raw": "虚假指标", "status": "unsupported"},
        {"claim_id": "INV-AUDIT", "raw": "冲突数据", "status": "contradicted"},
    ]
    summary = aggregate_claim_evidence(claims=[claim], claims_verification=ver_items)
    s = summary["INV-AUDIT"]

    assert s["verified_evidence"] == ["放量突破"]
    assert s["unsupported_evidence"] == ["虚假指标"]
    assert s["contradicted_evidence"] == ["冲突数据"]
    assert "冲突数据" in s["excluded_evidence"]
    assert "虚假指标" in s["excluded_evidence"]
    assert "存在 1 条与报告事实冲突/前视偏差证据" in s["reason"]
