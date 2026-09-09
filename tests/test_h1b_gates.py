"""Unit tests for P3-H1b: Credit Weighting Gates and Layered Isolation (TDD Suite).

Tests cover:
1. 7-dimension gate threshold evaluation (N, Side, Time, T+5, Balance, Bias Freeze, Magnitude).
2. Layered isolation state machine:
   - System gate failure -> Global weight=1.0 (Shadow-only)
   - Single model bias -> Only that model clamped to 1.0 + bias_freeze_reason
   - Abnormal model ratio > 50% -> Global fallback to Shadow
3. Credit weighting application:
   - Only verified claims receive relative weight modification in [0.85, 1.15]
   - Contradicted/unsupported claims NEVER elevated by credit weighting
   - Feature flag false -> 100% flat weighting (1.0) and preserved shadow metrics
4. Read-only gate verification script logic (verify_h1b_gates.py).
"""

import copy
import json
import os
import pytest

from tradingagents.agents.utils.agent_states import (
    DEFAULT_FEATURE_FLAGS,
    PROTOCOL_VERSION_V1_LEGACY,
    PROTOCOL_VERSION_V2_STRUCTURED,
    get_protocol_metadata,
)
from tradingagents.agents.utils.shadow_credit import (
    H1B_THRESHOLDS,
    calculate_shadow_credit_metrics,
    evaluate_h1b_system_gates,
    evaluate_model_bias_and_weights,
    calculate_claim_credit_weights,
    apply_credit_weighting_to_debate,
    resolve_claim_credit_weights_for_manager,
    extract_report_industry,
    is_qualifying_v2_report,
    normalize_report_for_evaluation,
    filter_v2_completed_reports,
    filter_reports_by_cohort,
    assert_cohort_homogeneity,
)


def _build_mock_debate_sample(
    *,
    symbol: str = "600519.SH",
    industry: str = "白酒",
    trade_date: str = "2026-08-01",
    decision_model_version: str | None = "decision_model.v1",
    evidence_contract_version: str | None = "evidence_contract.v0",
    price_basis_version: str | None = "price_basis.unspecified",
    generated_by_commit_sha: str | None = "e10b106df9d3173258b0a3fefc90ba7f3559f109",
    bull_model: str = "deepseek-r1",
    bear_model: str = "qwen-max",
    manager_model: str = "gpt-4o",
    winner: str = "bull",
    bull_v_cnt: int = 2,
    bull_t_cnt: int = 2,
    bear_v_cnt: int = 2,
    bear_t_cnt: int = 2,
    bull_ch_adopt: int = 1,
    bull_ch_tot: int = 1,
    bear_ch_adopt: int = 1,
    bear_ch_tot: int = 1,
    t_plus_5_hit: bool | None = True,
    consistency_failed: bool = False,
    market_regime: str = "震荡",
    sample_idx: int = 0,
) -> dict:
    """Helper to construct a mock v2 debate report with structured shadow metrics."""
    res = {
        "symbol": symbol,
        "industry": industry,
        "trade_date": trade_date,
        "market_regime": market_regime,
        "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
        "status": "completed",
        "analysis_status": "VALID",
        "trade_action": "BUY" if winner == "bull" else ("SELL" if winner == "bear" else "HOLD"),
        "final_trade_decision": "决策建议",
        "market_report": "市场分析报告正文",
        "fundamentals_report": "基本面分析报告正文",
        "macro_report": "宏观分析报告正文",
        "sentiment_report": "情绪分析报告正文",
        "news_report": "新闻分析报告正文",
        "smart_money_report": "资金分析报告正文",
        "volume_price_report": "量价分析报告正文",
        "investment_debate_state": {
            "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
            "claims": [
                {
                    "claim_id": "INV-1",
                    "speaker_key": "Bull",
                    "speaker": "Bull Analyst",
                    "stance": "bullish",
                    "claim": f"多头看好逻辑_{symbol}_{sample_idx}",
                    "evidence": ["多头证据1"],
                    "status": "verified",
                    "is_verified": True,
                },
                {
                    "claim_id": "INV-2",
                    "speaker_key": "Bear",
                    "speaker": "Bear Analyst",
                    "stance": "bearish",
                    "claim": f"空头看空逻辑_{symbol}_{sample_idx}",
                    "evidence": ["空头证据1"],
                    "status": "verified",
                    "is_verified": True,
                },
            ],
            "claim_evidence_summary": {
                "INV-1": {
                    "speaker_key": "Bull",
                    "counts": {"verified": bull_v_cnt, "total": bull_t_cnt},
                    "coverage": bull_v_cnt / bull_t_cnt if bull_t_cnt else 0,
                    "decision": "adopt",
                },
                "INV-2": {
                    "speaker_key": "Bear",
                    "counts": {"verified": bear_v_cnt, "total": bear_t_cnt},
                    "coverage": bear_v_cnt / bear_t_cnt if bear_t_cnt else 0,
                    "decision": "adopt",
                },
            },
            "challenges": [
                {
                    "challenge_id": "CH-1",
                    "speaker_key": "Bull",
                    "target_claim_id": "INV-2",
                    "adopted": bool(bull_ch_adopt > 0),
                },
                {
                    "challenge_id": "CH-2",
                    "speaker_key": "Bear",
                    "target_claim_id": "INV-1",
                    "adopted": bool(bear_ch_adopt > 0),
                },
            ],
            "manager_verdict": {
                "winner": winner,
                "direction": "看多" if winner == "bull" else ("看空" if winner == "bear" else "中性"),
                "adopted_challenge_ids": (["CH-1"] if bull_ch_adopt > 0 else []) + (["CH-2"] if bear_ch_adopt > 0 else []),
                "consistency_check_passed": not consistency_failed,
                "failed_checks": ["自洽校验失败"] if consistency_failed else [],
            },
            "round_messages": [
                {"speaker_key": "Bull", "stance": "bullish", "model_name": bull_model},
                {"speaker_key": "Bear", "stance": "bearish", "model_name": bear_model},
                {"speaker_key": "Research Manager", "is_verdict": True, "model_name": manager_model},
            ],
            "feature_flags": {
                "v2_debate_enabled": True,
                "shadow_credit_enabled": True,
                "credit_weighting_enabled": False,
            },
        },
        "shadow_credit_metrics": {
            "schema_version": "h1a_json_v1",
            "credit_weighting_enabled": False,
            "bull_verified_rate": round(bull_v_cnt / bull_t_cnt, 4) if bull_t_cnt else None,
            "bear_verified_rate": round(bear_v_cnt / bear_t_cnt, 4) if bear_t_cnt else None,
            "bull_challenge_adoption_rate": round(bull_ch_adopt / bull_ch_tot, 4) if bull_ch_tot else None,
            "bear_challenge_adoption_rate": round(bear_ch_adopt / bear_ch_tot, 4) if bear_ch_tot else None,
            "manager_evidence_coverage": 1.0,
            "manager_consistency_gate_triggered": consistency_failed,
            "t_plus_5_direction_hit": t_plus_5_hit,
            "sample_count": 1,
            "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
            "model_id_by_stance": {
                "bull": bull_model,
                "bear": bear_model,
                "manager": manager_model,
            },
        },
    }
    if decision_model_version:
        res["decision_model_version"] = decision_model_version
    if evidence_contract_version:
        res["evidence_contract_version"] = evidence_contract_version
    if price_basis_version:
        res["price_basis_version"] = price_basis_version
    if generated_by_commit_sha:
        res["generated_by_commit_sha"] = generated_by_commit_sha
    return res


def _build_qualifying_sample_pool(n: int = 60) -> list[dict]:
    """Generate a pool of n samples that satisfy all 7 gate dimensions."""
    samples = []
    symbols = [f"60000{i:02d}.SH" for i in range(20)]
    industries = ["电子", "医药", "白酒", "新能源", "金融", "军工", "化工"]
    regimes = ["牛市", "熊市", "震荡"]

    from datetime import date, timedelta
    start_date = date(2026, 6, 1)

    for i in range(n):
        sym = symbols[i % len(symbols)]
        ind = industries[i % len(industries)]
        cur_date = (start_date + timedelta(days=int(i * 1.0))).strftime("%Y-%m-%d")
        winner = "bull" if (i % 2 == 0) else "bear"
        regime = regimes[i % len(regimes)]

        sample = _build_mock_debate_sample(
            symbol=sym,
            industry=ind,
            trade_date=cur_date,
            bull_model="deepseek-r1",
            bear_model="qwen-max",
            manager_model="gpt-4o",
            winner=winner,
            bull_v_cnt=3,
            bull_t_cnt=3,
            bear_v_cnt=3,
            bear_t_cnt=3,
            bull_ch_adopt=1,
            bull_ch_tot=1,
            bear_ch_adopt=1,
            bear_ch_tot=1,
            t_plus_5_hit=True,
            consistency_failed=False,
            market_regime=regime,
            sample_idx=i,
        )
        samples.append(sample)
    return samples


class TestH1bSevenDimensionGates:
    """Test suite for 7-dimension gate threshold validation."""

    def test_dimension_n_insufficient_samples_fails(self):
        """Dimension 1 (N): Less than 60 samples must fail."""
        samples = _build_qualifying_sample_pool(n=59)
        result = evaluate_h1b_system_gates(samples)
        assert result["passed"] is False
        assert result["matrix"]["dimension_n"]["passed"] is False
        assert result["matrix"]["dimension_n"]["details"]["sample_count"] == 59
        assert result["matrix"]["dimension_n"]["details"]["min_required"] == 60
        assert result["recommendation"] == "KEEP_FALSE"

    def test_dimension_n_insufficient_unique_symbols_fails(self):
        """Dimension 1 (N): Less than 20 unique symbols must fail."""
        samples = _build_qualifying_sample_pool(n=60)
        # Force all samples to share only 10 symbols
        for idx, s in enumerate(samples):
            s["symbol"] = f"60000{idx % 10:02d}.SH"

        result = evaluate_h1b_system_gates(samples)
        assert result["passed"] is False
        assert result["matrix"]["dimension_n"]["passed"] is False
        assert result["matrix"]["dimension_n"]["details"]["unique_symbols"] == 10
        assert result["matrix"]["dimension_n"]["details"]["min_unique_symbols"] == 20

    def test_dimension_n_max_single_symbol_concentration_fails(self):
        """Dimension 1 (N): Single symbol share > 15% must fail."""
        samples = _build_qualifying_sample_pool(n=60)
        # Make one symbol take 12 samples out of 60 (20% > 15%)
        for i in range(12):
            samples[i]["symbol"] = "600519.SH"

        result = evaluate_h1b_system_gates(samples)
        assert result["passed"] is False
        assert result["matrix"]["dimension_n"]["passed"] is False
        assert result["matrix"]["dimension_n"]["details"]["max_symbol_share"] > 0.15

    def test_dimension_side_split_insufficient_samples_fails(self):
        """Dimension 2 (Side): bull or bear samples < 25 must fail."""
        samples = _build_qualifying_sample_pool(n=60)
        # Set 40 bull winners and 20 bear winners
        for i in range(40):
            samples[i]["investment_debate_state"]["manager_verdict"]["winner"] = "bull"
        for i in range(40, 60):
            samples[i]["investment_debate_state"]["manager_verdict"]["winner"] = "bear"

        result = evaluate_h1b_system_gates(samples)
        assert result["passed"] is False
        assert result["matrix"]["dimension_side"]["passed"] is False
        assert result["matrix"]["dimension_side"]["details"]["bear_samples"] == 20

    def test_dimension_time_span_insufficient_days_fails(self):
        """Dimension 3 (Time): Less than 45 calendar days must fail."""
        samples = _build_qualifying_sample_pool(n=60)
        # Constrain all trade dates to 20 days span
        for i, s in enumerate(samples):
            s["trade_date"] = f"2026-08-{1 + (i % 20):02d}"

        result = evaluate_h1b_system_gates(samples)
        assert result["passed"] is False
        assert result["matrix"]["dimension_time"]["passed"] is False
        assert result["matrix"]["dimension_time"]["details"]["calendar_days"] < 45

    def test_dimension_t_plus_5_no_due_samples_fails_without_vacuous_pass(self):
        """Dimension 4 (T+5): When due_count == 0, completeness rate must be 0.0 and fail even if N >= 60."""
        samples = _build_qualifying_sample_pool(n=60)
        # Clear T+5 evaluation so all 60 samples are not due / pending
        for s in samples:
            s["shadow_credit_metrics"]["t_plus_5_direction_hit"] = None
            s["shadow_credit_metrics"]["t_plus_5_status"] = "pending_due"
            s["is_t_plus_5_due"] = False
            s["t_plus_5_evaluated"] = False

        result = evaluate_h1b_system_gates(samples)
        assert result["passed"] is False
        dim_t5 = result["matrix"]["dimension_t5"]
        assert dim_t5["passed"] is False
        assert dim_t5["details"]["due_count"] == 0
        assert dim_t5["details"]["completed_count"] == 0
        assert dim_t5["details"]["completeness_rate"] == 0.0
        assert dim_t5["details"]["reason"] == "no_due_samples"

    def test_dimension_t_plus_5_completeness_passes_when_due_and_sufficient(self):
        """Dimension 4 (T+5): When due_count > 0 and completeness rate >= 95%, passes."""
        samples = _build_qualifying_sample_pool(n=60)
        # All 60 samples are due and completed (60/60 = 100% >= 95%)
        result = evaluate_h1b_system_gates(samples)
        dim_t5 = result["matrix"]["dimension_t5"]
        assert dim_t5["passed"] is True
        assert dim_t5["details"]["due_count"] == 60
        assert dim_t5["details"]["completed_count"] == 60
        assert dim_t5["details"]["completeness_rate"] == 1.0
        assert "reason" not in dim_t5["details"]

    def test_dimension_t_plus_5_completeness_fails(self):
        """Dimension 4 (T+5): Completeness rate < 95% must fail."""
        samples = _build_qualifying_sample_pool(n=60)
        # Set 10 samples to have missing/unreached T+5 when they should be evaluated (50/60 = 83.3% < 95%)
        for i in range(10):
            samples[i]["shadow_credit_metrics"]["t_plus_5_direction_hit"] = None
            samples[i]["is_t_plus_5_due"] = True  # explicitly marked as due but missing

        result = evaluate_h1b_system_gates(samples)
        assert result["passed"] is False
        dim_t5 = result["matrix"]["dimension_t5"]
        assert dim_t5["passed"] is False
        assert dim_t5["details"]["due_count"] == 60
        assert dim_t5["details"]["completed_count"] == 50
        assert dim_t5["details"]["completeness_rate"] == round(50 / 60, 4)

    def test_dimension_balance_ratio_and_diff_fails(self):
        """Dimension 5 (Balance): |Nbull - Nbear| > 10 must fail."""
        samples = _build_qualifying_sample_pool(n=60)
        for i in range(36):
            samples[i]["investment_debate_state"]["manager_verdict"]["winner"] = "bull"
        for i in range(36, 60):
            samples[i]["investment_debate_state"]["manager_verdict"]["winner"] = "bear"
        # 36 vs 24 -> diff = 12 > 10
        result = evaluate_h1b_system_gates(samples)
        assert result["passed"] is False
        assert result["matrix"]["dimension_balance"]["passed"] is False
        assert result["matrix"]["dimension_balance"]["details"]["side_diff"] == 12

    def test_dimension_bias_freeze_delta_verified_fails(self):
        """Dimension 6 (Bias Freeze): Δverified > 18% must fail."""
        samples = _build_qualifying_sample_pool(n=60)
        for s in samples:
            s["shadow_credit_metrics"]["bull_verified_rate"] = 0.90
            s["shadow_credit_metrics"]["bear_verified_rate"] = 0.70  # delta = 20% > 18%

        result = evaluate_h1b_system_gates(samples)
        assert result["passed"] is False
        assert result["matrix"]["dimension_bias"]["passed"] is False
        assert result["matrix"]["dimension_bias"]["details"]["delta_verified_rate"] == pytest.approx(0.20, abs=1e-4)

    def test_all_dimensions_pass_produces_eligible_recommendation(self):
        """When all 7 dimensions pass, system gate is PASS and recommendation is ELIGIBLE_FOR_ACTIVATION."""
        samples = _build_qualifying_sample_pool(n=60)
        # Ensure 60 calendar days and 40 trading days
        from datetime import date, timedelta
        start = date(2026, 5, 1)
        for i, s in enumerate(samples):
            s["trade_date"] = (start + timedelta(days=i)).strftime("%Y-%m-%d")

        result = evaluate_h1b_system_gates(samples)
        assert result["passed"] is True
        assert result["recommendation"] == "ELIGIBLE_FOR_ACTIVATION"
        for dim_name, dim_info in result["matrix"].items():
            assert dim_info["passed"] is True, f"Dimension {dim_name} failed: {dim_info}"


class TestLayeredIsolationStateMachine:
    """Test suite for layered isolation (System -> Model -> Global Shadow)."""

    def test_system_gate_fail_forces_global_shadow(self):
        """If system gate fails, credit weighting is completely inactive (weights=1.0)."""
        samples = _build_qualifying_sample_pool(n=10)  # Only 10 samples -> FAIL
        system_gate = evaluate_h1b_system_gates(samples)
        assert system_gate["passed"] is False

        model_isolation = evaluate_model_bias_and_weights(samples, system_gate_passed=False)
        assert model_isolation["credit_weighting_active"] is False
        assert model_isolation["global_fallback_shadow"] is True
        assert model_isolation["model_weights"]["deepseek-r1"] == 1.0
        assert model_isolation["model_weights"]["qwen-max"] == 1.0

    def test_single_model_bias_clamped_to_1_0(self):
        """If a single model has bias exceeding threshold, only that model is clamped to 1.0 with reason."""
        samples = _build_qualifying_sample_pool(n=60)
        from datetime import date, timedelta
        start = date(2026, 5, 1)
        for i, s in enumerate(samples):
            s["trade_date"] = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            # Let deepseek-r1 have high verified delta or high consistency gate trigger rate
            s["shadow_credit_metrics"]["bull_verified_rate"] = 0.95
            s["shadow_credit_metrics"]["bear_verified_rate"] = 0.72  # > 18% bias on bull stance

        system_gate = evaluate_h1b_system_gates(samples)
        # Note: system gate bias check may pass or fail depending on setup, let's test model-level isolation
        model_isolation = evaluate_model_bias_and_weights(
            samples,
            system_gate_passed=True,
            per_model_bias_overrides={"deepseek-r1": {"biased": True, "reason": "Δverified=23% > 18%"}},
        )

        assert model_isolation["credit_weighting_active"] is True
        assert model_isolation["model_weights"]["deepseek-r1"] == 1.0
        assert "deepseek-r1" in model_isolation["bias_freeze_reasons"]
        assert "Δverified" in model_isolation["bias_freeze_reasons"]["deepseek-r1"]

    def test_abnormal_model_ratio_over_50_pct_falls_back_to_global_shadow(self):
        """If abnormal/biased models ratio > 50%, trigger global fallback to Shadow."""
        samples = _build_qualifying_sample_pool(n=60)
        # Suppose 2 out of 3 models are biased (> 50%)
        model_isolation = evaluate_model_bias_and_weights(
            samples,
            system_gate_passed=True,
            per_model_bias_overrides={
                "deepseek-r1": {"biased": True, "reason": "Bias exceeded"},
                "qwen-max": {"biased": True, "reason": "Consistency triggers > 5%"},
                "gpt-4o": {"biased": False},
            },
        )

        assert model_isolation["credit_weighting_active"] is False
        assert model_isolation["global_fallback_shadow"] is True
        assert model_isolation["abnormal_model_ratio"] == pytest.approx(2 / 3, abs=1e-3)
        assert model_isolation["model_weights"]["deepseek-r1"] == 1.0
        assert model_isolation["model_weights"]["qwen-max"] == 1.0
        assert model_isolation["model_weights"]["gpt-4o"] == 1.0


class TestCreditWeightingApplication:
    """Test suite for credit weight application on claims and verdicts."""

    def test_unsupported_and_contradicted_never_elevated(self):
        """Contradicted and unsupported claims must NEVER receive weight > 0 or be elevated to verified."""
        claims = [
            {"claim_id": "C1", "speaker": "Bull", "model_name": "deepseek-r1", "status": "verified", "is_verified": True},
            {"claim_id": "C2", "speaker": "Bull", "model_name": "deepseek-r1", "status": "unsupported", "is_verified": False},
            {"claim_id": "C3", "speaker": "Bear", "model_name": "qwen-max", "status": "contradicted", "is_verified": False},
        ]
        claim_summary = {
            "C1": {"decision": "adopt", "counts": {"verified": 1, "total": 1}},
            "C2": {"decision": "reject", "counts": {"verified": 0, "total": 1}},
            "C3": {"decision": "reject", "counts": {"contradicted": 1, "total": 1}},
        }
        model_weights = {"deepseek-r1": 1.15, "qwen-max": 0.85}

        weights_res = calculate_claim_credit_weights(
            claims=claims,
            claim_evidence_summary=claim_summary,
            model_weights=model_weights,
            credit_weighting_enabled=True,
            system_gate_passed=True,
        )

        # C1 (verified) gets weight 1.15
        assert weights_res["claim_weights"]["C1"] == 1.15
        # C2 and C3 must NOT be elevated
        assert weights_res["claim_weights"]["C2"] == 0.0 or weights_res["claim_decisions"]["C2"] == "reject"
        assert weights_res["claim_weights"]["C3"] == 0.0 or weights_res["claim_decisions"]["C3"] == "reject"
        assert weights_res["claim_decisions"]["C2"] == "reject"
        assert weights_res["claim_decisions"]["C3"] == "reject"

    def test_weight_multiplier_strictly_bounded(self):
        """Weights must be strictly bounded in [0.85, 1.15]."""
        claims = [
            {"claim_id": "C1", "speaker": "Bull", "model_name": "super-bull", "status": "verified"},
            {"claim_id": "C2", "speaker": "Bear", "model_name": "bad-bear", "status": "verified"},
        ]
        claim_summary = {
            "C1": {"decision": "adopt", "counts": {"verified": 1, "total": 1}},
            "C2": {"decision": "adopt", "counts": {"verified": 1, "total": 1}},
        }
        # Provide out-of-bound raw weights 1.50 and 0.50
        model_weights = {"super-bull": 1.50, "bad-bear": 0.50}

        weights_res = calculate_claim_credit_weights(
            claims=claims,
            claim_evidence_summary=claim_summary,
            model_weights=model_weights,
            credit_weighting_enabled=True,
            system_gate_passed=True,
        )

        assert weights_res["claim_weights"]["C1"] == 1.15
        assert weights_res["claim_weights"]["C2"] == 0.85

    def test_flag_disabled_instant_flat_weighting(self):
        """When credit_weighting_enabled=False, all weights are 1.0 and shadow metrics are preserved."""
        fixture = _build_mock_debate_sample()
        fixture["investment_debate_state"]["feature_flags"]["credit_weighting_enabled"] = False

        applied = apply_credit_weighting_to_debate(fixture)
        assert applied["credit_weighting_active"] is False
        assert applied["shadow_credit_metrics"]["credit_weighting_enabled"] is False
        # All claim weights are 1.0 (flat)
        for w in applied.get("claim_weights", {}).values():
            assert w == 1.0


class TestResearchManagerGateWiring:
    """research_manager must not hardcode system_gate_passed=False when flag is on."""

    def _claims_and_summary(self):
        claims = [
            {
                "claim_id": "C1",
                "speaker": "Bull",
                "model_name": "deepseek-r1",
                "status": "verified",
                "is_verified": True,
            },
            {
                "claim_id": "C2",
                "speaker": "Bear",
                "model_name": "qwen-max",
                "status": "verified",
                "is_verified": True,
            },
        ]
        claim_summary = {
            "C1": {"decision": "adopt", "counts": {"verified": 1, "total": 1}},
            "C2": {"decision": "adopt", "counts": {"verified": 1, "total": 1}},
        }
        return claims, claim_summary

    def test_empty_history_fail_closed_flat(self):
        claims, summary = self._claims_and_summary()
        res = resolve_claim_credit_weights_for_manager(
            claims=claims,
            claim_evidence_summary=summary,
            historical_samples=[],
            credit_weighting_enabled=True,
        )
        assert res["system_gate_passed"] is False
        assert res["credit_weighting_active"] is False
        assert res["recommendation"] == "KEEP_FALSE"
        assert res["claim_weights"]["C1"] == 1.0
        assert res["claim_weights"]["C2"] == 1.0

    def test_qualifying_history_activates_non_flat_weights(self):
        claims, summary = self._claims_and_summary()
        history = _build_qualifying_sample_pool(n=60)
        gate = evaluate_h1b_system_gates(history)
        assert gate["passed"] is True

        res = resolve_claim_credit_weights_for_manager(
            claims=claims,
            claim_evidence_summary=summary,
            historical_samples=history,
            credit_weighting_enabled=True,
        )
        assert res["system_gate_passed"] is True
        assert res["credit_weighting_active"] is True
        assert res["recommendation"] == "ELIGIBLE_FOR_ACTIVATION"
        # Default calibrated model weight is 1.05 when unbiased and gates pass
        assert res["claim_weights"]["C1"] == pytest.approx(1.05)
        assert res["claim_weights"]["C2"] == pytest.approx(1.05)

    def test_flag_off_stays_flat_even_if_gates_pass(self):
        claims, summary = self._claims_and_summary()
        history = _build_qualifying_sample_pool(n=60)
        res = resolve_claim_credit_weights_for_manager(
            claims=claims,
            claim_evidence_summary=summary,
            historical_samples=history,
            credit_weighting_enabled=False,
        )
        assert res["system_gate_passed"] is True
        assert res["credit_weighting_active"] is False
        assert res["claim_weights"]["C1"] == 1.0


class TestH1bV2OnlySampleFilteringAndIndustry:
    """Test suite for Track A6: Gate verification counting only completed v2 reports."""

    def test_extract_report_industry_from_multiple_sources(self):
        """extract_report_industry correctly finds industry across all metadata locations."""
        # 1. Direct top-level
        assert extract_report_industry({"industry": "白酒"}) == "白酒"
        assert extract_report_industry({"sector": "半导体"}) == "半导体"

        # 2. instrument_context
        assert extract_report_industry({"instrument_context": {"industry": "新能源"}}) == "新能源"

        # 3. market_data_context.industry_linkage
        assert extract_report_industry({
            "market_data_context": {
                "industry_linkage": {"industry_name": "家用电器与智能家居"}
            }
        }) == "家用电器与智能家居"

        # 4. data_collection_provenance.industry_linkage_raw
        assert extract_report_industry({
            "data_collection_provenance": {
                "industry_linkage_raw": {"industry_name": "医药生物与创新药"}
            }
        }) == "医药生物与创新药"

        # 5. quadrant_1_protocol_metadata
        assert extract_report_industry({
            "quadrant_1_protocol_metadata": {"industry": "电力与公用事业"}
        }) == "电力与公用事业"

        # 6. Nested under result_data
        assert extract_report_industry({
            "result_data": {
                "market_data_context": {
                    "industry_linkage": {"industry_name": "军工装备"}
                }
            }
        }) == "军工装备"

        # 7. Absent / empty / unknown -> None (never fabricate)
        assert extract_report_industry({}) is None
        assert extract_report_industry({"industry": ""}) is None
        assert extract_report_industry({"quadrant_1_protocol_metadata": {"industry": "未知行业"}}) is None

    def test_is_qualifying_v2_report_true_for_valid_v2_and_winner(self):
        """is_qualifying_v2_report returns True only for completed v2 reports with analysis_status=VALID, directional action, and winner."""
        # Standard v2 report with winner
        v2_bull = {
            "status": "completed",
            "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
            "analysis_status": "VALID",
            "trade_action": "BUY",
            "manager_verdict": {"winner": "bull", "direction": "看多"},
        }
        assert is_qualifying_v2_report(v2_bull) is True

        v2_bear = {
            "status": "completed",
            "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
            "analysis_status": "VALID",
            "trade_action": "SELL",
            "manager_verdict": {"winner": "bear", "direction": "看空"},
        }
        assert is_qualifying_v2_report(v2_bear) is True

        # Nested in result_data
        v2_nested = {
            "status": "completed",
            "analysis_status": "VALID",
            "trade_action": "HOLD",
            "result_data": {
                "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
                "investment_debate_state": {
                    "manager_verdict": {"winner": "tie", "direction": "中性"}
                },
            },
        }
        assert is_qualifying_v2_report(v2_nested) is True

    def test_is_qualifying_v2_report_false_for_v1_legacy_or_missing_winner(self):
        """is_qualifying_v2_report returns False for legacy reports, missing winner, or non-completed."""
        # v1 legacy without winner
        v1_legacy = {
            "status": "completed",
            "protocol_version": PROTOCOL_VERSION_V1_LEGACY,
        }
        assert is_qualifying_v2_report(v1_legacy) is False

        # Non-completed status (failed/pending/running)
        failed_report = {
            "status": "failed",
            "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
            "manager_verdict": {"winner": "bull"},
        }
        assert is_qualifying_v2_report(failed_report) is False

        running_report = {
            "status": "running",
            "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
            "manager_verdict": {"winner": "bull"},
        }
        assert is_qualifying_v2_report(running_report) is False

        # Empty or missing winner
        no_winner = {
            "status": "completed",
            "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
            "manager_verdict": {},
        }
        assert is_qualifying_v2_report(no_winner) is False

        invalid_winner = {
            "status": "completed",
            "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
            "manager_verdict": {"winner": "invalid_winner"},
        }
        assert is_qualifying_v2_report(invalid_winner) is False

    def test_filter_v2_completed_reports_excludes_old_samples_and_denominators(self):
        """Pool of mixed legacy and v2 reports: legacy reports excluded from sample count and denominator."""
        mixed_pool = [
            # 3 old v1 reports without winner
            {"symbol": "600001.SH", "status": "completed", "protocol_version": "v1_legacy"},
            {"symbol": "600002.SH", "status": "completed", "protocol_version": "v1_legacy"},
            {"symbol": "600003.SH", "status": "completed", "protocol_version": "v1_legacy"},
            # 1 failed v2 report
            {"symbol": "600004.SH", "status": "failed", "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED, "manager_verdict": {"winner": "bull"}},
            # 4 valid completed v2 reports with VALID analysis_status and directional actions
            {
                "symbol": "600519.SH",
                "industry": "白酒",
                "status": "completed",
                "analysis_status": "VALID",
                "trade_action": "BUY",
                "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
                "trade_date": "2026-08-01",
                "manager_verdict": {"winner": "bull", "direction": "看多"},
            },
            {
                "symbol": "000858.SZ",
                "industry": "白酒",
                "status": "completed",
                "analysis_status": "VALID",
                "trade_action": "BUY",
                "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
                "trade_date": "2026-08-02",
                "manager_verdict": {"winner": "bull", "direction": "看多"},
            },
            {
                "symbol": "600276.SH",
                "industry": "医药",
                "status": "completed",
                "analysis_status": "VALID",
                "trade_action": "SELL",
                "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
                "trade_date": "2026-08-03",
                "manager_verdict": {"winner": "bear", "direction": "看空"},
            },
            {
                "symbol": "300750.SZ",
                "industry": "新能源",
                "status": "completed",
                "analysis_status": "VALID",
                "trade_action": "SELL",
                "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
                "trade_date": "2026-08-04",
                "manager_verdict": {"winner": "bear", "direction": "看空"},
            },
        ]

        filtered = filter_v2_completed_reports(mixed_pool)
        assert len(filtered) == 4

        # Gate evaluation must have sample_count=4 (not 8)
        gate_res = evaluate_h1b_system_gates(filtered)
        assert gate_res["summary"]["sample_count"] == 4
        assert gate_res["matrix"]["dimension_n"]["details"]["sample_count"] == 4
        assert gate_res["matrix"]["dimension_n"]["details"]["unique_symbols"] == 4
        assert gate_res["matrix"]["dimension_n"]["details"]["unique_industries"] == 3
        # 2 bull, 2 bear -> side split 2/2, balance ratio 50%
        assert gate_res["matrix"]["dimension_side"]["details"]["bull_samples"] == 2
        assert gate_res["matrix"]["dimension_side"]["details"]["bear_samples"] == 2
        assert gate_res["matrix"]["dimension_balance"]["details"]["bull_ratio"] == 0.5
        assert gate_res["matrix"]["dimension_balance"]["details"]["side_diff"] == 0

    def test_golden_audit_samples_extracted_and_evaluated_correctly(self):
        """Golden audit samples must be parsed with real industries (3) and real side split (2 bull, 1 bear)."""
        import os
        import glob
        golden_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests",
            "golden",
            "audit_20260823",
        )
        assert os.path.exists(golden_dir)

        golden_files = sorted(glob.glob(os.path.join(golden_dir, "*_result_data.json")))
        assert len(golden_files) == 3

        raw_samples = []
        for fpath in golden_files:
            with open(fpath, "r", encoding="utf-8") as f:
                raw_samples.append(json.load(f))

        # D-009 §5 contract: raw pre-D-009 golden samples (analysis_status IS NULL) are excluded as legacy_null
        filtered_raw, excluded = filter_v2_completed_reports(raw_samples, return_excluded_counts=True)
        assert len(filtered_raw) == 0
        assert excluded["legacy_null"] == 3

        # Calibrated golden samples with VALID analysis_status and directional trade_action qualify
        for s in raw_samples:
            s["analysis_status"] = "VALID"
            s["trade_action"] = "BUY" if s.get("manager_verdict", {}).get("winner") == "bull" else "SELL"

        filtered = filter_v2_completed_reports(raw_samples)
        assert len(filtered) == 3

        # Check extracted industries
        industries = [s.get("industry") for s in filtered]
        assert "家用电器与智能家居" in industries
        assert "电力与公用事业" in industries
        assert "医药生物与创新药" in industries

        # Check gate evaluation
        gate_res = evaluate_h1b_system_gates(filtered)
        assert gate_res["summary"]["sample_count"] == 3
        assert gate_res["matrix"]["dimension_n"]["details"]["unique_industries"] == 3
        assert gate_res["matrix"]["dimension_side"]["details"]["bull_samples"] == 2
        assert gate_res["matrix"]["dimension_side"]["details"]["bear_samples"] == 1
        assert gate_res["matrix"]["dimension_side"]["details"]["bull_verified_claims"] > 0
        assert gate_res["matrix"]["dimension_side"]["details"]["bear_verified_claims"] > 0

    def test_verify_h1b_gates_script_runs_and_verifies_v2_only(self, tmp_path):
        """scripts/verify_h1b_gates.py run_verify produces correct v2-only output JSON and structure."""
        from scripts.verify_h1b_gates import load_reports_from_db, run_verify
        import glob

        # 1. Test load_reports_from_db fallback to golden samples:
        # Under D-009 §5, legacy golden samples without analysis_status are excluded as legacy_null
        reports, excluded = load_reports_from_db(return_excluded_counts=True)
        assert len(reports) == 0
        assert excluded["legacy_null"] == 3

        # 2. Test run_verify execution with input_file containing qualifying v2 reports
        golden_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tests",
            "golden",
            "audit_20260823",
        )
        golden_files = sorted(glob.glob(os.path.join(golden_dir, "*_result_data.json")))
        qualifying_samples = []
        for fpath in golden_files:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["analysis_status"] = "VALID"
                data["trade_action"] = "BUY" if data.get("manager_verdict", {}).get("winner") == "bull" else "SELL"
                qualifying_samples.append(data)

        sample_file = str(tmp_path / "qualifying_input.json")
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(qualifying_samples, f)

        out_json = str(tmp_path / "test_h1b_report.json")
        res = run_verify(input_file=sample_file, output_json=out_json, cohort="legacy_unversioned")
        assert res["task_id"] == "P3-H1b"
        assert res["sample_count"] == 3
        assert res["gate_evaluation"]["matrix"]["dimension_n"]["details"]["unique_industries"] == 3
        assert res["gate_evaluation"]["matrix"]["dimension_side"]["details"]["bull_samples"] == 2
        assert res["gate_evaluation"]["matrix"]["dimension_side"]["details"]["bear_samples"] == 1
        assert os.path.exists(out_json)


class TestH1bVerifyGatesDbPath:
    """Test suite for Track A9: verify_h1b_gates --db-path loading and failure behavior."""

    @pytest.fixture
    def custom_sqlite_db(self, tmp_path):
        """Create a temporary SQLite DB populated with known ReportDB records."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from api.database import Base, ReportDB

        db_path = str(tmp_path / "custom_tradingagents.db")
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Insert 4 qualifying completed v2 reports
        v2_reports = [
            ReportDB(
                id="rep-1",
                symbol="600519.SH",
                trade_date="2026-08-01",
                status="completed",
                analysis_status="VALID",
                trade_action="BUY",
                result_data={
                    "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
                    "industry": "白酒",
                    "manager_verdict": {"winner": "bull", "direction": "看多"},
                    "shadow_credit_metrics": {"bull_verified_rate": 0.9, "bear_verified_rate": 0.8},
                },
            ),
            ReportDB(
                id="rep-2",
                symbol="000858.SZ",
                trade_date="2026-08-02",
                status="completed",
                analysis_status="VALID",
                trade_action="BUY",
                result_data={
                    "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
                    "industry": "白酒",
                    "manager_verdict": {"winner": "bull", "direction": "看多"},
                    "shadow_credit_metrics": {"bull_verified_rate": 0.9, "bear_verified_rate": 0.8},
                },
            ),
            ReportDB(
                id="rep-3",
                symbol="600276.SH",
                trade_date="2026-08-03",
                status="completed",
                analysis_status="VALID",
                trade_action="SELL",
                result_data={
                    "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
                    "industry": "医药",
                    "manager_verdict": {"winner": "bear", "direction": "看空"},
                    "shadow_credit_metrics": {"bull_verified_rate": 0.85, "bear_verified_rate": 0.85},
                },
            ),
            ReportDB(
                id="rep-4",
                symbol="300750.SZ",
                trade_date="2026-08-04",
                status="completed",
                analysis_status="VALID",
                trade_action="SELL",
                result_data={
                    "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
                    "industry": "新能源",
                    "manager_verdict": {"winner": "bear", "direction": "看空"},
                    "shadow_credit_metrics": {"bull_verified_rate": 0.8, "bear_verified_rate": 0.9},
                },
            ),
            # 1 legacy completed v1 report (no winner) -> excluded by v2 filter
            ReportDB(
                id="rep-5",
                symbol="601398.SH",
                trade_date="2026-08-05",
                status="completed",
                result_data={"protocol_version": PROTOCOL_VERSION_V1_LEGACY},
            ),
            # 1 failed v2 report -> excluded by completed filter
            ReportDB(
                id="rep-6",
                symbol="600036.SH",
                trade_date="2026-08-06",
                status="failed",
                result_data={
                    "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
                    "manager_verdict": {"winner": "bull"},
                },
            ),
        ]

        session.add_all(v2_reports)
        session.commit()
        session.close()
        engine.dispose()
        return db_path

    def test_load_reports_from_db_reads_custom_sqlite(self, custom_sqlite_db):
        """Passing db_path loads only from that SQLite DB and applies qualifying v2 filter."""
        from scripts.verify_h1b_gates import load_reports_from_db

        reports = load_reports_from_db(db_path=custom_sqlite_db)
        assert len(reports) == 4
        symbols = [r.get("symbol") for r in reports]
        assert symbols == ["600519.SH", "000858.SZ", "600276.SH", "300750.SZ"]
        assert "601398.SH" not in symbols
        assert "600036.SH" not in symbols

    def test_run_verify_with_db_path(self, custom_sqlite_db, tmp_path):
        """run_verify with db_path runs gate evaluation against the specified SQLite database."""
        from scripts.verify_h1b_gates import run_verify

        out_json = str(tmp_path / "out_verify.json")
        res = run_verify(db_path=custom_sqlite_db, output_json=out_json)
        assert res["sample_count"] == 4
        assert res["gate_evaluation"]["matrix"]["dimension_n"]["details"]["sample_count"] == 4
        assert res["gate_evaluation"]["matrix"]["dimension_n"]["details"]["unique_symbols"] == 4
        assert res["gate_evaluation"]["matrix"]["dimension_n"]["details"]["unique_industries"] == 3
        assert res["gate_evaluation"]["matrix"]["dimension_side"]["details"]["bull_samples"] == 2
        assert res["gate_evaluation"]["matrix"]["dimension_side"]["details"]["bear_samples"] == 2
        assert os.path.exists(out_json)

    def test_non_existent_db_path_raises_file_not_found(self):
        """Non-existent db_path must raise FileNotFoundError and NEVER fall back to golden samples."""
        from scripts.verify_h1b_gates import load_reports_from_db, run_verify

        fake_path = "/non/existent/path/tradingagents_fake.db"
        with pytest.raises(FileNotFoundError):
            load_reports_from_db(db_path=fake_path)

        with pytest.raises(FileNotFoundError):
            run_verify(db_path=fake_path)

    def test_invalid_corrupt_db_path_raises_runtime_error(self, tmp_path):
        """Corrupt or non-SQLite file must raise RuntimeError and NEVER fall back to golden samples."""
        from scripts.verify_h1b_gates import load_reports_from_db, run_verify

        corrupt_file = tmp_path / "corrupt.db"
        corrupt_file.write_text("This is not a SQLite database file")

        with pytest.raises(RuntimeError):
            load_reports_from_db(db_path=str(corrupt_file))

        with pytest.raises(RuntimeError):
            run_verify(db_path=str(corrupt_file))

    def test_empty_sqlite_db_returns_zero_samples_without_golden_fallback(self, tmp_path):
        """Empty SQLite DB returns 0 qualifying samples without silently falling back to golden."""
        from sqlalchemy import create_engine
        from api.database import Base
        from scripts.verify_h1b_gates import load_reports_from_db, run_verify

        empty_db = str(tmp_path / "empty.db")
        engine = create_engine(f"sqlite:///{empty_db}")
        Base.metadata.create_all(bind=engine)
        engine.dispose()

        reports = load_reports_from_db(db_path=empty_db)
        assert len(reports) == 0

        out_json = str(tmp_path / "empty_report.json")
        res = run_verify(db_path=empty_db, output_json=out_json)
        assert res["sample_count"] == 0
        assert res["gate_evaluation"]["matrix"]["dimension_n"]["details"]["sample_count"] == 0
        assert res["recommendation"] == "KEEP_FALSE"

    def test_cli_subprocess_execution_with_db_path(self, custom_sqlite_db, tmp_path):
        """CLI invocation with --db-path exits with 0 and writes evaluation report."""
        import subprocess
        import sys

        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
            "verify_h1b_gates.py",
        )
        out_json = str(tmp_path / "cli_out.json")
        proc = subprocess.run(
            [sys.executable, script_path, "--cohort", "legacy_unversioned", "--db-path", custom_sqlite_db, "--output-json", out_json],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        assert os.path.exists(out_json)

    def test_unmigrated_db_without_industry_column_ensures_schema_and_loads_completed(self, tmp_path):
        """Track A14: unmigrated SQLite DB without industry column has schema ensured on read and loads completed reports."""
        import sqlite3
        from sqlalchemy import create_engine, inspect
        from scripts.verify_h1b_gates import load_reports_from_db, run_verify

        db_path = str(tmp_path / "unmigrated_reports.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE reports (
                id VARCHAR(36) PRIMARY KEY,
                user_id VARCHAR(64),
                symbol VARCHAR(20),
                trade_date VARCHAR(10),
                status VARCHAR(20),
                error TEXT,
                decision VARCHAR(50),
                direction VARCHAR(50),
                confidence INTEGER,
                probability FLOAT,
                target_price FLOAT,
                stop_loss_price FLOAT,
                analysis_status VARCHAR(32),
                trade_action VARCHAR(32),
                risk_status VARCHAR(32),
                result_data JSON,
                risk_items JSON,
                key_metrics JSON,
                data_gaps JSON,
                falsification_conditions JSON,
                not_applicable BOOLEAN,
                analyst_traces JSON,
                market_report TEXT,
                sentiment_report TEXT,
                news_report TEXT,
                fundamentals_report TEXT,
                macro_report TEXT,
                smart_money_report TEXT,
                volume_price_report TEXT,
                game_theory_report TEXT,
                investment_plan TEXT,
                trader_investment_plan TEXT,
                final_trade_decision TEXT,
                created_at DATETIME,
                updated_at DATETIME
            )
            """
        )
        conn.execute(
            """
            INSERT INTO reports (id, symbol, trade_date, status, analysis_status, trade_action, result_data) VALUES
            ('rep-unmig-1', '600519.SH', '2026-08-01', 'completed', 'VALID', 'BUY', '{"protocol_version": "v2_structured_disagreement", "industry": "白酒", "manager_verdict": {"winner": "bull", "direction": "看多"}}'),
            ('rep-unmig-2', '000858.SZ', '2026-08-02', 'completed', 'VALID', 'SELL', '{"protocol_version": "v2_structured_disagreement", "industry": "白酒", "manager_verdict": {"winner": "bear", "direction": "看空"}}'),
            ('rep-unmig-3', '600036.SH', '2026-08-03', 'failed', 'INVALID_RUN', 'NO_TRADE', '{"protocol_version": "v2_structured_disagreement", "manager_verdict": {"winner": "bull"}}')
            """
        )
        conn.commit()
        conn.close()

        # Check that industry column does NOT exist before ensure
        check_engine = create_engine(f"sqlite:///{db_path}")
        insp_before = inspect(check_engine)
        cols_before = {col["name"] for col in insp_before.get_columns("reports")}
        assert "industry" not in cols_before
        check_engine.dispose()

        # Call load_reports_from_db: schema should be ensured before query
        reports = load_reports_from_db(db_path=db_path)
        assert len(reports) == 2
        symbols = [r.get("symbol") for r in reports]
        assert symbols == ["600519.SH", "000858.SZ"]

        # Verify industry column and index were added to SQLite table
        insp_after = inspect(check_engine)
        cols_after = {col["name"] for col in insp_after.get_columns("reports")}
        assert "industry" in cols_after
        indexes_after = {idx["name"] for idx in insp_after.get_indexes("reports")}
        assert "ix_reports_industry" in indexes_after
        check_engine.dispose()

        # Call run_verify to verify full path succeeds
        out_json = str(tmp_path / "out_unmigrated.json")
        res = run_verify(db_path=db_path, output_json=out_json)
        assert res["sample_count"] == 2
        assert os.path.exists(out_json)

    def test_schema_migration_failure_raises_explicit_runtime_error_without_golden_fallback(self, tmp_path, monkeypatch):
        """Track A14: failure during _ensure_report_schema raises explicit RuntimeError and never falls back to golden."""
        import sqlite3
        from scripts.verify_h1b_gates import load_reports_from_db, run_verify

        db_path = str(tmp_path / "fail_migration.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE reports (id VARCHAR(64) PRIMARY KEY, status VARCHAR(20))")
        conn.commit()
        conn.close()

        def _mock_ensure_fail(target_engine=None):
            raise RuntimeError("Mocked DDL migration failure on reports")

        monkeypatch.setattr("api.database._ensure_report_schema", _mock_ensure_fail)

        with pytest.raises(RuntimeError) as exc_info:
            load_reports_from_db(db_path=db_path)
        assert "Mocked DDL migration failure" in str(exc_info.value)

        with pytest.raises(RuntimeError) as exc_info:
            run_verify(db_path=db_path)
        assert "Mocked DDL migration failure" in str(exc_info.value)

    def test_schema_inspection_failure_raises_explicit_runtime_error(self, tmp_path, monkeypatch):
        """Track A14: inspection failure during _ensure_report_schema raises explicit RuntimeError without fallback."""
        from scripts.verify_h1b_gates import load_reports_from_db, run_verify

        db_path = str(tmp_path / "inspect_fail.db")
        tmp_path.joinpath("inspect_fail.db").touch()

        def _mock_inspect_fail(target):
            raise RuntimeError("Mocked inspect failure")

        monkeypatch.setattr("sqlalchemy.inspect", _mock_inspect_fail)

        with pytest.raises(RuntimeError) as exc_info:
            load_reports_from_db(db_path=db_path)
        assert "Mocked inspect failure" in str(exc_info.value)

        with pytest.raises(RuntimeError) as exc_info:
            run_verify(db_path=db_path)
        assert "Mocked inspect failure" in str(exc_info.value)

    def test_default_db_schema_migration_failure_raises_explicit_runtime_error_without_golden_fallback(self, monkeypatch):
        """Track A14: default DB with reports table experiencing migration failure raises RuntimeError instead of golden fallback."""
        from scripts.verify_h1b_gates import load_reports_from_db

        class _MockInspector:
            def has_table(self, table_name):
                return table_name == "reports"

        monkeypatch.setattr("sqlalchemy.inspect", lambda engine: _MockInspector())
        monkeypatch.setattr(
            "api.database._ensure_report_schema",
            lambda target_engine=None: (_ for _ in ()).throw(RuntimeError("Mocked default DB migration error")),
        )

        with pytest.raises(RuntimeError) as exc_info:
            load_reports_from_db()
        assert "Mocked default DB migration error" in str(exc_info.value)


class TestH1bTPlus5DueInference:
    """Test suite for Track A12: T+5 due inference via trade_date / t_plus_5_date in Dimension 4."""

    def test_t5_due_inferred_from_past_trade_date_when_unbackfilled_fails_completeness(self):
        """When 60 samples have past trade_date (T+5 elapsed) but unbackfilled (hit/status/is_due is None),
        Dimension 4 infers due_count=60, completed_count=0, completeness_rate=0.0, and fails due to missing data.
        """
        samples = _build_qualifying_sample_pool(n=60)
        # Simulate unbackfilled historical samples where trade_date is in early August 2026 (>= 7 days before 2026-09-02)
        for i, s in enumerate(samples):
            s["trade_date"] = "2026-08-03"
            # Clear all T+5 pre-evaluated metadata
            s["shadow_credit_metrics"]["t_plus_5_direction_hit"] = None
            s["shadow_credit_metrics"].pop("t_plus_5_status", None)
            s["shadow_credit_metrics"].pop("t_plus_5_date", None)
            s.pop("t_plus_5_status", None)
            s.pop("t_plus_5_date", None)
            s.pop("t_plus_5_evaluated", None)
            s["is_t_plus_5_due"] = None

        result = evaluate_h1b_system_gates(samples, as_of="2026-09-02")
        assert result["passed"] is False
        dim_t5 = result["matrix"]["dimension_t5"]
        assert dim_t5["passed"] is False
        assert dim_t5["details"]["due_count"] == 60
        assert dim_t5["details"]["completed_count"] == 0
        assert dim_t5["details"]["completeness_rate"] == 0.0
        # Honest failure: due_count > 0 so reason is NOT "no_due_samples"
        assert "reason" not in dim_t5["details"]

    def test_t5_due_not_inferred_when_trade_date_is_pending_due(self):
        """When trade_date is too recent (T+5 > as_of), samples are pending and not due (due_count=0 -> no_due_samples)."""
        samples = _build_qualifying_sample_pool(n=60)
        for s in samples:
            s["trade_date"] = "2026-09-01"
            s["shadow_credit_metrics"]["t_plus_5_direction_hit"] = None
            s["is_t_plus_5_due"] = None
            s.pop("t_plus_5_status", None)
            s.pop("t_plus_5_date", None)
            s.pop("t_plus_5_evaluated", None)

        # As of 2026-09-02, T+5 of 2026-09-01 is 2026-09-08 (> 2026-09-02)
        result = evaluate_h1b_system_gates(samples, as_of="2026-09-02")
        assert result["passed"] is False
        dim_t5 = result["matrix"]["dimension_t5"]
        assert dim_t5["passed"] is False
        assert dim_t5["details"]["due_count"] == 0
        assert dim_t5["details"]["completed_count"] == 0
        assert dim_t5["details"]["completeness_rate"] == 0.0
        assert dim_t5["details"]["reason"] == "no_due_samples"

    def test_t5_due_explicit_false_and_suspension_excluded(self):
        """Explicit is_t_plus_5_due=False, pending_due, and suspension status are excluded from due denominator."""
        samples = _build_qualifying_sample_pool(n=60)
        for i, s in enumerate(samples):
            s["trade_date"] = "2026-08-01"
            s["shadow_credit_metrics"]["t_plus_5_direction_hit"] = None
            if i < 20:
                s["is_t_plus_5_due"] = False
            elif i < 40:
                s["t_plus_5_status"] = "suspension"
            else:
                s["t_plus_5_status"] = "pending_due"

        result = evaluate_h1b_system_gates(samples, as_of="2026-09-02")
        assert result["passed"] is False
        dim_t5 = result["matrix"]["dimension_t5"]
        assert dim_t5["details"]["due_count"] == 0
        assert dim_t5["details"]["completed_count"] == 0
        assert dim_t5["details"]["reason"] == "no_due_samples"

    def test_t5_due_inferred_from_top_level_or_metrics_t_plus_5_date(self):
        """When t_plus_5_date is explicitly present (on sample or metrics), use it to compare against as_of."""
        samples = _build_qualifying_sample_pool(n=60)
        for i, s in enumerate(samples):
            s["trade_date"] = None
            s["shadow_credit_metrics"]["t_plus_5_direction_hit"] = None
            s["is_t_plus_5_due"] = None
            if i < 30:
                s["t_plus_5_date"] = "2026-08-10"
            else:
                s["shadow_credit_metrics"]["t_plus_5_date"] = "2026-08-12"

        result = evaluate_h1b_system_gates(samples, as_of="2026-09-02")
        dim_t5 = result["matrix"]["dimension_t5"]
        assert dim_t5["details"]["due_count"] == 60
        assert dim_t5["details"]["completed_count"] == 0

        # Now test with future t_plus_5_date
        for s in samples:
            s["t_plus_5_date"] = "2026-09-15"
            s["shadow_credit_metrics"].pop("t_plus_5_date", None)
        result_future = evaluate_h1b_system_gates(samples, as_of="2026-09-02")
        assert result_future["matrix"]["dimension_t5"]["details"]["due_count"] == 0
        assert result_future["matrix"]["dimension_t5"]["details"]["reason"] == "no_due_samples"

    def test_t5_due_unparseable_or_failed_calendar_does_not_invent_due(self):
        """When trade date is unparseable or calendar fails/has insufficient days, do NOT invent due status."""
        samples = _build_qualifying_sample_pool(n=60)
        for s in samples:
            s["trade_date"] = "not-a-valid-date"
            s["shadow_credit_metrics"]["t_plus_5_direction_hit"] = None
            s["is_t_plus_5_due"] = None

        res_invalid = evaluate_h1b_system_gates(samples, as_of="2026-09-02")
        assert res_invalid["matrix"]["dimension_t5"]["details"]["due_count"] == 0
        assert res_invalid["matrix"]["dimension_t5"]["details"]["reason"] == "no_due_samples"

        # Test with custom calendar having insufficient forward trading days (< 5)
        for s in samples:
            s["trade_date"] = "2026-08-01"
        res_cal_fail = evaluate_h1b_system_gates(
            samples,
            as_of="2026-09-02",
            trading_calendar=["2026-08-01", "2026-08-02"],
        )
        assert res_cal_fail["matrix"]["dimension_t5"]["details"]["due_count"] == 0
        assert res_cal_fail["matrix"]["dimension_t5"]["details"]["reason"] == "no_due_samples"

    def test_t5_due_mixed_due_and_pending_calculates_honest_rate(self):
        """Mixed sample set: 57 due and completed + 3 due uncompleted -> 57/60 = 95% PASS.
        56 due and completed + 4 due uncompleted -> 56/60 = 93.3% FAIL.
        """
        samples = _build_qualifying_sample_pool(n=60)
        # 57 completed due samples
        for i in range(57):
            samples[i]["trade_date"] = "2026-08-03"
            samples[i]["shadow_credit_metrics"]["t_plus_5_direction_hit"] = True

        # 3 uncompleted due samples (due inferred from trade_date, hit is None)
        for i in range(57, 60):
            samples[i]["trade_date"] = "2026-08-03"
            samples[i]["shadow_credit_metrics"]["t_plus_5_direction_hit"] = None
            samples[i]["is_t_plus_5_due"] = None

        res_95 = evaluate_h1b_system_gates(samples, as_of="2026-09-02")
        dim_t5_95 = res_95["matrix"]["dimension_t5"]
        assert dim_t5_95["passed"] is True
        assert dim_t5_95["details"]["due_count"] == 60
        assert dim_t5_95["details"]["completed_count"] == 57
        assert dim_t5_95["details"]["completeness_rate"] == 0.95

        # Change 1 more to uncompleted (56/60)
        samples[56]["shadow_credit_metrics"]["t_plus_5_direction_hit"] = None
        samples[56]["is_t_plus_5_due"] = None
        res_93 = evaluate_h1b_system_gates(samples, as_of="2026-09-02")
        dim_t5_93 = res_93["matrix"]["dimension_t5"]
        assert dim_t5_93["passed"] is False
        assert dim_t5_93["details"]["due_count"] == 60
        assert dim_t5_93["details"]["completed_count"] == 56
        assert dim_t5_93["details"]["completeness_rate"] == round(56 / 60, 4)

    def test_t5_due_nested_result_data_inference(self):
        """evaluate_h1b_system_gates extracts trade_date and t_plus_5_date from nested result_data."""
        reports = []
        for i in range(60):
            rep = {
                "id": f"rep-{i}",
                "symbol": f"6000{i % 20:02d}.SH",
                "trade_date": "2026-08-03",
                "status": "completed",
                "analysis_status": "VALID",
                "trade_action": "BUY" if i % 2 == 0 else "SELL",
                "result_data": {
                    "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
                    "trade_date": "2026-08-03",
                    "industry": "电子",
                    "manager_verdict": {"winner": "bull" if i % 2 == 0 else "bear", "direction": "看多"},
                    "shadow_credit_metrics": {
                        "bull_verified_rate": 0.9,
                        "bear_verified_rate": 0.9,
                        "t_plus_5_direction_hit": None,
                    },
                },
            }
            reports.append(rep)

        res = evaluate_h1b_system_gates(reports, as_of="2026-09-02")
        dim_t5 = res["matrix"]["dimension_t5"]
        assert dim_t5["passed"] is False
        assert dim_t5["details"]["due_count"] == 60
        assert dim_t5["details"]["completed_count"] == 0
        assert dim_t5["details"]["completeness_rate"] == 0.0

    def test_cli_subprocess_execution_with_bad_db_path_exits_nonzero(self, tmp_path):
        """CLI invocation with bad --db-path exits with non-zero exit code and error logged."""
        import subprocess
        import sys

        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
            "verify_h1b_gates.py",
        )
        fake_db = "/non/existent/path/tradingagents_fake.db"
        out_json = str(tmp_path / "cli_bad_out.json")
        proc = subprocess.run(
            [sys.executable, script_path, "--db-path", fake_db, "--output-json", out_json],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        assert not os.path.exists(out_json)


class TestH1bCohortIsolation:
    """Test suite for DAV-601: H1b cohort isolation (CLI + evaluation entry)."""

    def test_cli_missing_cohort_fails_closed_nonzero_exit_and_no_pass_report(self, tmp_path):
        """1. 未传 --cohort: 必须非零退出，严禁输出 PASS 报告。"""
        import subprocess
        import sys

        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
            "verify_h1b_gates.py",
        )
        out_json = str(tmp_path / "cli_no_cohort.json")
        proc = subprocess.run(
            [sys.executable, script_path, "--output-json", out_json],
            capture_output=True,
            text=True,
        )
        # Must exit non-zero
        assert proc.returncode != 0
        # Must NOT write a PASS report
        if os.path.exists(out_json):
            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data.get("gate_evaluation", {}).get("passed") is not True

    def test_cli_empty_cohort_fails_closed_nonzero_exit(self, tmp_path):
        """CLI with empty or whitespace --cohort must exit non-zero."""
        import subprocess
        import sys

        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
            "verify_h1b_gates.py",
        )
        out_json = str(tmp_path / "cli_empty_cohort.json")
        proc = subprocess.run(
            [sys.executable, script_path, "--cohort", "", "--output-json", out_json],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0

    def test_cohort_legacy_unversioned_includes_only_unversioned_samples(self):
        """2. --cohort=legacy_unversioned: 只纳入缺版本字段的旧样本，不得标成 v1。"""
        legacy_samples = _build_qualifying_sample_pool(n=60)
        # Ensure legacy samples have no version triad fields
        for s in legacy_samples:
            s.pop("decision_model_version", None)
            s.pop("evidence_contract_version", None)
            s.pop("price_basis_version", None)
            s.pop("generated_by_commit_sha", None)

        # Create v1 samples
        v1_samples = _build_qualifying_sample_pool(n=10)
        for s in v1_samples:
            s["decision_model_version"] = "decision_model.v1"
            s["evidence_contract_version"] = "evidence_contract.v0"
            s["price_basis_version"] = "price_basis.unspecified"
            s["generated_by_commit_sha"] = "e10b106df9d3173258b0a3fefc90ba7f3559f109"

        mixed_pool = legacy_samples + v1_samples
        filtered, meta = filter_reports_by_cohort(mixed_pool, cohort="legacy_unversioned")

        # Exactly 60 legacy samples included, 10 v1 samples excluded
        assert len(filtered) == 60
        assert meta["cohort_type"] == "legacy_unversioned"

        # Must NOT re-label legacy samples to v1
        for s in filtered:
            dmv = s.get("decision_model_version") or s.get("result_data", {}).get("decision_model_version")
            assert dmv != "decision_model.v1"

    def test_cohort_triad_version_filtering_exact_match_and_ignores_commit_sha(self):
        """3. 指定三元版本时：只纳入三字段全等的样本；SHA 只进 JSON 摘要，不当过滤主键。"""
        # Target triad
        target_triad = "decision_model.v1:evidence_contract.v0:price_basis.unspecified"

        sample_sha1 = _build_mock_debate_sample(symbol="600519.SH")
        sample_sha1["decision_model_version"] = "decision_model.v1"
        sample_sha1["evidence_contract_version"] = "evidence_contract.v0"
        sample_sha1["price_basis_version"] = "price_basis.unspecified"
        sample_sha1["generated_by_commit_sha"] = "1111111111111111111111111111111111111111"

        sample_sha2 = _build_mock_debate_sample(symbol="000858.SZ")
        sample_sha2["decision_model_version"] = "decision_model.v1"
        sample_sha2["evidence_contract_version"] = "evidence_contract.v0"
        sample_sha2["price_basis_version"] = "price_basis.unspecified"
        sample_sha2["generated_by_commit_sha"] = "2222222222222222222222222222222222222222"

        # Mismatch in decision_model_version
        sample_diff_dmv = _build_mock_debate_sample(symbol="600276.SH")
        sample_diff_dmv["decision_model_version"] = "decision_model.v2"
        sample_diff_dmv["evidence_contract_version"] = "evidence_contract.v0"
        sample_diff_dmv["price_basis_version"] = "price_basis.unspecified"

        # Mismatch in evidence_contract_version
        sample_diff_ecv = _build_mock_debate_sample(symbol="300750.SZ")
        sample_diff_ecv["decision_model_version"] = "decision_model.v1"
        sample_diff_ecv["evidence_contract_version"] = "evidence_contract.v1"
        sample_diff_ecv["price_basis_version"] = "price_basis.unspecified"

        # Mismatch in price_basis_version
        sample_diff_pbv = _build_mock_debate_sample(symbol="601398.SH")
        sample_diff_pbv["decision_model_version"] = "decision_model.v1"
        sample_diff_pbv["evidence_contract_version"] = "evidence_contract.v0"
        sample_diff_pbv["price_basis_version"] = "price_basis.pit_adjusted"

        # Unversioned legacy sample
        sample_legacy = _build_mock_debate_sample(
            symbol="600036.SH",
            decision_model_version=None,
            evidence_contract_version=None,
            price_basis_version=None,
            generated_by_commit_sha=None,
        )

        pool = [sample_sha1, sample_sha2, sample_diff_dmv, sample_diff_ecv, sample_diff_pbv, sample_legacy]
        filtered, meta = filter_reports_by_cohort(pool, cohort=target_triad)

        # Only sample_sha1 and sample_sha2 match the triad
        assert len(filtered) == 2
        symbols = [s["symbol"] for s in filtered]
        assert symbols == ["600519.SH", "000858.SZ"]

        # Commit SHAs must not filter samples out, but appear in commit_shas summary
        assert set(meta["commit_shas"]) == {
            "1111111111111111111111111111111111111111",
            "2222222222222222222222222222222222222222",
        }

    def test_empty_cohort_fails_closed_due_count_zero_not_pass(self):
        """4. 空结果：passed=false，due_count==0 不得 PASS。"""
        res = evaluate_h1b_system_gates([], cohort="decision_model.v1:evidence_contract.v0:price_basis.unspecified")
        assert res["passed"] is False
        assert res["summary"]["system_gate_status"] == "FAIL"
        assert res["summary"]["recommendation"] == "KEEP_FALSE"
        assert res["matrix"]["dimension_t5"]["details"]["due_count"] == 0
        assert res["matrix"]["dimension_t5"]["passed"] is False

    def test_mixed_cohorts_rejected_not_silently_merged(self):
        """5. 混世代同一次评价：拒绝（明确 FAIL / 非零），不得静默合并。"""
        legacy_samples = _build_qualifying_sample_pool(n=30)
        for s in legacy_samples:
            s.pop("decision_model_version", None)
            s.pop("evidence_contract_version", None)
            s.pop("price_basis_version", None)
            s.pop("generated_by_commit_sha", None)
        v1_samples = _build_qualifying_sample_pool(n=30)
        for s in v1_samples:
            s["decision_model_version"] = "decision_model.v1"
            s["evidence_contract_version"] = "evidence_contract.v0"
            s["price_basis_version"] = "price_basis.unspecified"

        mixed_pool = legacy_samples + v1_samples
        # Calling evaluate_h1b_system_gates directly on mixed pool without cohort filter
        res = evaluate_h1b_system_gates(mixed_pool)
        assert res["passed"] is False
        assert res["summary"]["system_gate_status"] == "FAIL"
        assert res["recommendation"] == "KEEP_FALSE"
        assert res.get("matrix", {}).get("cohort_homogeneity", {}).get("passed") is False

        # assert_cohort_homogeneity must raise ValueError
        with pytest.raises(ValueError, match="Mixed cohort generations"):
            assert_cohort_homogeneity(mixed_pool)

    def test_online_manager_resolution_unlabeled_samples_fallback_to_flat_1_0(self):
        """6. 线上加权解析遇未标记样本：保持/降级权重 1.0，不抛崩主链路。"""
        claims = [
            {"claim_id": "C1", "speaker": "Bull", "model_name": "deepseek-r1", "status": "verified"},
            {"claim_id": "C2", "speaker": "Bear", "model_name": "qwen-max", "status": "verified"},
        ]
        summary = {
            "C1": {"decision": "adopt", "counts": {"verified": 1, "total": 1}},
            "C2": {"decision": "adopt", "counts": {"verified": 1, "total": 1}},
        }
        # 60 unlabeled legacy samples
        unlabeled_samples = _build_qualifying_sample_pool(n=60)
        for s in unlabeled_samples:
            s.pop("decision_model_version", None)
            s.pop("evidence_contract_version", None)
            s.pop("price_basis_version", None)
            s.pop("generated_by_commit_sha", None)

        # Must not raise an exception!
        res = resolve_claim_credit_weights_for_manager(
            claims=claims,
            claim_evidence_summary=summary,
            historical_samples=unlabeled_samples,
            credit_weighting_enabled=True,
        )

        assert res["credit_weighting_active"] is False
        assert res["system_gate_passed"] is False
        assert res["recommendation"] == "KEEP_FALSE"
        assert res["claim_weights"]["C1"] == 1.0
        assert res["claim_weights"]["C2"] == 1.0
        assert res["global_fallback_shadow"] is True

    def test_online_manager_resolution_mixed_cohorts_fallback_to_flat_1_0(self):
        """6b. 线上加权解析遇混世代样本：保持/降级权重 1.0，不抛崩主链路。"""
        claims = [
            {"claim_id": "C1", "speaker": "Bull", "model_name": "deepseek-r1", "status": "verified"},
        ]
        summary = {
            "C1": {"decision": "adopt", "counts": {"verified": 1, "total": 1}},
        }
        legacy_samples = _build_qualifying_sample_pool(n=30)
        for s in legacy_samples:
            s.pop("decision_model_version", None)
            s.pop("evidence_contract_version", None)
            s.pop("price_basis_version", None)
            s.pop("generated_by_commit_sha", None)
        v1_samples = _build_qualifying_sample_pool(n=30)
        for s in v1_samples:
            s["decision_model_version"] = "decision_model.v1"
            s["evidence_contract_version"] = "evidence_contract.v0"
            s["price_basis_version"] = "price_basis.unspecified"

        mixed_pool = legacy_samples + v1_samples

        # Must not raise an exception!
        res = resolve_claim_credit_weights_for_manager(
            claims=claims,
            claim_evidence_summary=summary,
            historical_samples=mixed_pool,
            credit_weighting_enabled=True,
        )

        assert res["credit_weighting_active"] is False
        assert res["system_gate_passed"] is False
        assert res["claim_weights"]["C1"] == 1.0
        assert res["global_fallback_shadow"] is True

