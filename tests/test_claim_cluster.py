"""Unit and integration tests for P0-4b: Claim evidence clustering and vote deduplication.

Covers:
1. Foxconn Nail Case: 3 price-derived reports (Market, Volume-Price, Smart Money) referencing
   the same close price shock, volume, and daily return collapse into 1 single cluster.
   Directional weight must use independent_cluster_count (1 cluster), NOT analyst_count (3 votes, 60%).
2. Independent Fundamentals: 4th report referencing verifiable revenue / net profit forms a 2nd cluster.
3. Unsupported / Narrative claim: Claim with no clusterable observable data points gets status unsupported
   and is excluded from independent_cluster_count and verified_evidence_count.
4. Same Cluster Same Direction Vote Ceiling: Multiple same-direction claims in the same cluster cast at most 1 vote.
5. Deterministic Cluster ID: Stable hash / ID generation is reproducible and lockable.
6. Prompt & Manager State Integration: Prompt specifies cluster deduplication while retaining DAV-336 7 analysts.
"""
from __future__ import annotations

import asyncio
import json
import pytest

from tradingagents.agents.utils.claim_cluster import (
    CLUSTER_TYPE_FUNDAMENTALS,
    CLUSTER_TYPE_MACRO_POLICY,
    CLUSTER_TYPE_PRICE_SHOCK,
    CLUSTER_TYPE_SENTIMENT_NEWS,
    CLUSTER_TYPE_UNSUPPORTED,
    assign_claim_cluster,
    compute_cluster_id,
    extract_evidence_ids,
    format_claim_cluster_summary_for_prompt,
    tally_cluster_votes,
)
from tradingagents.prompts import get_prompt
from tradingagents.prompts.catalog import ZH_PROMPTS, EN_PROMPTS
from tradingagents.agents.utils.evidence_relations import (
    EvidenceRelation,
    EvidenceRelationGraph,
    RelationType,
)


class TestClaimClusterCore:
    """Core clustering and deterministic ID tests."""

    def test_cluster_id_is_stable_and_deterministic(self):
        """Cluster ID must be deterministic, reproducible, and lockable across runs."""
        cid1 = compute_cluster_id(
            cluster_type=CLUSTER_TYPE_PRICE_SHOCK,
            symbol="601138.SH",
            date="2026-08-22",
        )
        cid2 = compute_cluster_id(
            cluster_type=CLUSTER_TYPE_PRICE_SHOCK,
            symbol="601138.SH",
            date="2026-08-22",
        )
        assert cid1 == cid2
        assert "price_shock" in cid1 or "601138" in cid1

        cid_fund = compute_cluster_id(
            cluster_type=CLUSTER_TYPE_FUNDAMENTALS,
            symbol="601138.SH",
            date="2026-Q2",
        )
        assert cid1 != cid_fund

    def test_foxconn_nail_case_three_price_derived_reports_collapse_to_one_cluster(self):
        """Acceptance Nail: 3 price-derived reports (Market, Volume-Price, Smart Money)

        on 工业富联 referencing same close price, volume, and daily pct change must form
        exactly 1 cluster. When evaluated alongside 2 bear reports (forming 1 bear cluster),
        the bull weight must be 1/(1+1)=50%, NOT 3/5=60%.
        """
        symbol = "601138.SH"
        trade_date = "2026-08-22"

        # 3 Bull analysts quoting price-derived metrics for 工业富联 on 2026-08-22
        market_claim = {
            "claim_id": "INV-1",
            "speaker": "Market Analyst",
            "speaker_key": "market",
            "stance": "bullish",
            "claim": "工业富联放量突破22.5元收盘价阻力位",
            "evidence": ["收盘价22.5元", "突破20日均线"],
            "confidence": 0.85,
        }
        volume_price_claim = {
            "claim_id": "INV-2",
            "speaker": "Volume Price Analyst",
            "speaker_key": "volume_price",
            "stance": "bullish",
            "claim": "成交量放大1.5倍且当日涨幅5.2%",
            "evidence": ["成交量放大1.5倍", "当日涨幅5.2%"],
            "confidence": 0.88,
        }
        smart_money_claim = {
            "claim_id": "INV-3",
            "speaker": "Smart Money Analyst",
            "speaker_key": "smart_money",
            "stance": "bullish",
            "claim": "主力资金放量净流入且长阳收盘",
            "evidence": ["主力资金净流入5.2亿元", "收盘价22.5元"],
            "confidence": 0.80,
        }

        # 2 Bear analysts quoting a distinct fundamental risk cluster
        bear_fund_claim = {
            "claim_id": "INV-4",
            "speaker": "Fundamentals Analyst",
            "speaker_key": "fundamentals",
            "stance": "bearish",
            "claim": "应收账款周转率下降且毛利率承压",
            "evidence": ["应收账款周转天数增加15天", "毛利率同比下滑2.1%"],
            "confidence": 0.75,
        }
        bear_macro_claim = {
            "claim_id": "INV-5",
            "speaker": "Macro Analyst",
            "speaker_key": "macro",
            "stance": "bearish",
            "claim": "行业产能扩张导致毛利率下滑",
            "evidence": ["毛利率同比下滑2.1%"],
            "confidence": 0.70,
        }

        claims = [
            market_claim,
            volume_price_claim,
            smart_money_claim,
            bear_fund_claim,
            bear_macro_claim,
        ]

        metrics = tally_cluster_votes(
            claims=claims,
            symbol=symbol,
            trade_date=trade_date,
        )

        # 1. analyst_count must be 5 (explanatory only)
        assert metrics["analyst_count"] == 5

        # 2. Bull side: 3 price-derived reports collapse into 1 cluster
        assert metrics["direction_cluster_counts"]["bull"] == 1
        assert metrics["bull_cluster_count"] == 1

        # 3. Bear side: 2 fundamental reports collapse into 1 cluster
        assert metrics["direction_cluster_counts"]["bear"] == 1
        assert metrics["bear_cluster_count"] == 1

        # 4. Total independent clusters must be 2 (price_shock + fundamentals)
        assert metrics["independent_cluster_count"] == 2

        # 5. Bull direction weight MUST be 1 / 2 = 50%, strictly NOT 3 / 5 = 60%!
        assert metrics["cluster_weights"]["bull"] == 0.50
        assert metrics["cluster_weights"]["bull"] != 0.60
        assert metrics["cluster_weights"]["bear"] == 0.50

    def test_independent_fundamentals_forms_distinct_cluster(self):
        """4th report referencing verifiable revenue / profit forms an independent cluster."""
        symbol = "601138.SH"
        trade_date = "2026-08-22"

        # 3 price-derived reports
        c1 = {
            "claim_id": "INV-1",
            "speaker": "Market Analyst",
            "stance": "bullish",
            "claim": "突破22.5元收盘价阻力位",
            "evidence": ["收盘价22.5元", "日涨幅5.2%"],
        }
        c2 = {
            "claim_id": "INV-2",
            "speaker": "Volume Price Analyst",
            "stance": "bullish",
            "claim": "放量成交量放大1.5倍",
            "evidence": ["成交量放大1.5倍"],
        }
        c3 = {
            "claim_id": "INV-3",
            "speaker": "Smart Money Analyst",
            "stance": "bullish",
            "claim": "主力资金净流入",
            "evidence": ["主力资金净流入5.2亿元"],
        }

        # 4th report with independent fundamental revenue/profit
        c4 = {
            "claim_id": "INV-4",
            "speaker": "Fundamentals Analyst",
            "stance": "bullish",
            "claim": "中报营收同比增长30%且净利润超预期",
            "evidence": ["营收同比增长30%", "净利润50亿元"],
        }

        metrics = tally_cluster_votes(
            claims=[c1, c2, c3, c4],
            symbol=symbol,
            trade_date=trade_date,
        )

        assert metrics["analyst_count"] == 4
        assert metrics["independent_cluster_count"] == 2
        assert metrics["bull_cluster_count"] == 2
        # Verified evidence count covers all valid evidence items across the clusters
        assert metrics["verified_evidence_count"] >= 4

    def test_unsupported_narrative_claim_excluded_from_clusters(self):
        """Claim with only narrative / no clusterable data points is unsupported and excluded."""
        symbol = "601138.SH"
        trade_date = "2026-08-22"

        valid_claim = {
            "claim_id": "INV-1",
            "speaker": "Market Analyst",
            "stance": "bullish",
            "claim": "收盘价站上22.5元",
            "evidence": ["收盘价22.5元"],
        }

        narrative_claim = {
            "claim_id": "INV-2",
            "speaker": "Retail Debater",
            "stance": "bullish",
            "claim": "主力意图明显，后市坚定看多，走势非常好看，强烈推荐",
            "evidence": ["团队实力强大", "后市可期"],
        }

        empty_ev_claim = {
            "claim_id": "INV-3",
            "speaker": "Casual Debater",
            "stance": "bearish",
            "claim": "感觉要跌",
            "evidence": [],
        }

        enriched_narrative = assign_claim_cluster(
            narrative_claim,
            symbol=symbol,
            date=trade_date,
        )
        assert enriched_narrative["cluster_status"] == "unsupported"
        assert enriched_narrative["cluster_id"] is None
        assert len(enriched_narrative["evidence_ids"]) == 0

        metrics = tally_cluster_votes(
            claims=[valid_claim, narrative_claim, empty_ev_claim],
            symbol=symbol,
            trade_date=trade_date,
        )

        assert metrics["independent_cluster_count"] == 1
        assert metrics["bull_cluster_count"] == 1
        assert metrics["bear_cluster_count"] == 0
        assert "INV-2" in metrics["unsupported_claim_ids"]
        assert "INV-3" in metrics["unsupported_claim_ids"]

    def test_same_cluster_same_direction_at_most_one_vote(self):
        """Multiple claims in the same cluster on the same direction yield at most 1 vote."""
        symbol = "601138.SH"
        trade_date = "2026-08-22"

        claims = [
            {
                "claim_id": f"INV-{i}",
                "speaker": f"Analyst {i}",
                "stance": "bullish",
                "claim": f"价格冲击观点 {i}",
                "evidence": ["收盘价22.5元", "成交量放大1.5倍"],
            }
            for i in range(1, 6)
        ]

        metrics = tally_cluster_votes(
            claims=claims,
            symbol=symbol,
            trade_date=trade_date,
        )

        assert metrics["analyst_count"] == 5
        assert metrics["independent_cluster_count"] == 1
        assert metrics["bull_cluster_count"] == 1
        assert metrics["cluster_weights"]["bull"] == 1.0

    def test_verified_evidence_count_with_claims_verification(self):
        """verified_evidence_count must only count verified evidence items.

        Contradicted, unsupported, and source_unavailable items must be excluded.
        """
        symbol = "601138.SH"
        trade_date = "2026-08-22"

        c1 = {
            "claim_id": "INV-1",
            "speaker": "Bull Analyst 1",
            "stance": "bullish",
            "claim": "收盘价放量突破阻力位",
            "evidence": ["收盘价22.5元", "突破20日均线"],
        }
        c2 = {
            "claim_id": "INV-2",
            "speaker": "Bull Analyst 2",
            "stance": "bullish",
            "claim": "成交量放大且主力资金流入",
            "evidence": ["成交量放大1.5倍", "当日涨幅5.2%"],
        }

        # Case 1: Partial verification
        # INV-1: '收盘价22.5元' verified, '突破20日均线' contradicted
        # INV-2: '成交量放大1.5倍' unsupported, '当日涨幅5.2%' verified
        claims_verification = [
            {"claim_id": "INV-1", "raw": "收盘价22.5元", "status": "verified"},
            {"claim_id": "INV-1", "raw": "突破20日均线", "status": "contradicted"},
            {"claim_id": "INV-2", "raw": "成交量放大1.5倍", "status": "unsupported"},
            {"claim_id": "INV-2", "raw": "当日涨幅5.2%", "status": "verified"},
        ]

        metrics = tally_cluster_votes(
            claims=[c1, c2],
            claims_verification=claims_verification,
            symbol=symbol,
            trade_date=trade_date,
        )

        assert metrics["analyst_count"] == 2
        assert metrics["independent_cluster_count"] == 1
        # Exactly 2 verified items ('收盘价22.5元' and '当日涨幅5.2%'); contradicted and unsupported excluded
        assert metrics["verified_evidence_count"] == 2

        # Case 2: All contradicted / unsupported -> verified_evidence_count must be 0
        all_failed_verification = [
            {"claim_id": "INV-1", "raw": "收盘价22.5元", "status": "contradicted"},
            {"claim_id": "INV-1", "raw": "突破20日均线", "status": "contradicted"},
            {"claim_id": "INV-2", "raw": "成交量放大1.5倍", "status": "unsupported"},
            {"claim_id": "INV-2", "raw": "当日涨幅5.2%", "status": "source_unavailable"},
        ]

        failed_metrics = tally_cluster_votes(
            claims=[c1, c2],
            claims_verification=all_failed_verification,
            symbol=symbol,
            trade_date=trade_date,
        )
        assert failed_metrics["verified_evidence_count"] == 0


class TestPromptSemanticsRegression:
    """Prompt requirement tests for P0-4b."""

    def test_chinese_prompt_cluster_dedup_and_dav336_seven_analysts(self):
        """Chinese prompt must retain all 7 analysts individually (DAV-336) and specify cluster dedup."""
        prompt = ZH_PROMPTS["research_manager_prompt"]

        # 1. All 7 analysts must be listed
        analysts = [
            "宏观板块",
            "市场（技术面）",
            "舆情（情绪）",
            "新闻",
            "基本面",
            "主力资金",
            "量价",
        ]
        for analyst in analysts:
            assert analyst in prompt, f"Missing analyst in prompt: {analyst}"

        # 2. Cluster deduplication requirement
        assert "cluster_id" in prompt or "cluster" in prompt
        assert "analyst_count" in prompt or "人头" in prompt
        assert "独立权重" in prompt or "去重" in prompt

    def test_english_prompt_cluster_tally_replaced(self):
        """English prompt must not say 'Tally analyst verdicts and compute bull/bear ratio'."""
        prompt = EN_PROMPTS["research_manager_prompt"]
        assert "Tally analyst verdicts and compute bull/bear ratio" not in prompt
        assert "cluster" in prompt.lower()


_MANAGER_LLM_RESPONSE = (
    "研究总监正式报告正文\n\n"
    '<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "看多", "reason": "突破确认", "position_pct": 30, "entry": "22.5", "target": "25.0", "stop_loss": "21.0", "upside": 11.0, "downside": 6.0, "odds": 1.8, "adopted_claim_ids": ["INV-1"], "partially_adopted_claims": [], "rejected_claim_ids": [], "excluded_evidence": [], "dispute_map": []} -->'
    '\n<!-- VERDICT: {"direction": "看多", "reason": "突破确认"} -->'
)


def _research_manager_state() -> dict:
    """Seven-report manager state whose debate already passed the pre-gate."""
    from tests.test_debate_b3_protocol import _build_v2_challenge_completed_state

    inv_debate_state = _build_v2_challenge_completed_state()
    inv_debate_state["tiebreak_skipped"] = True
    return {
        "symbol": "601138.SH",
        "trade_date": "2026-08-22",
        "market_data_context": {
            "symbol": "601138.SH",
            "analysis_baseline_date": "2026-08-22",
            "trade_date": "2026-08-22",
            "source_provenance": {
                "stock_data": {"status": "available", "as_of": "2026-08-22"},
            },
        },
        "investment_debate_state": inv_debate_state,
        "macro_report": "宏观板块报告：政策支持行业发展。",
        "market_report": "市场技术报告：收盘价22.5元，突破阻力位。",
        "sentiment_report": "舆情报告：情绪乐观偏多。",
        "news_report": "新闻报告：行业订单增加。",
        "fundamentals_report": "基本面报告：前三季度扣非净利15.2亿元同比增25%。",
        "volume_price_report": "量价报告：放量突破20日均线。",
        "smart_money_report": "主力资金报告：东财主力净流入1.29亿元。",
        "fund_flow_consensus_guard": {
            "blocked": False,
            "direction_allowed": True,
            "status": "selected",
        },
    }


def _run_research_manager(state: dict) -> tuple[dict, list[str]]:
    """Run the real manager node with a streaming stub LLM; return (payload, prompts sent)."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from tradingagents.agents.managers.research_manager import create_research_manager

    captured_prompts: list[str] = []

    async def _fake_stream(prompt: str):
        captured_prompts.append(prompt)
        yield SimpleNamespace(content=_MANAGER_LLM_RESPONSE)

    mock_llm = MagicMock()
    mock_llm.astream = _fake_stream
    mock_memory = MagicMock()
    mock_memory.get_memories.return_value = []
    result = asyncio.run(create_research_manager(mock_llm, mock_memory)(state))
    return result, captured_prompts


class TestIntegrationDebateAndResearchManager:
    """Integration tests verifying debate state normalization and research manager integration."""

    def test_debate_state_claim_normalization_preserves_evidence_and_cluster_ids(self):
        from tradingagents.agents.utils.debate_utils import update_debate_state_with_payload

        state = {
            "claims": [],
            "open_claim_ids": [],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": [],
            "round_messages": [],
            "count": 0,
            "claim_counter": 0,
            "history": "",
        }

        # Simulate debater output with machine block
        raw_response = (
            "立论发言正文\n\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": [], "new_claims": [{"claim": "放量突破收盘价阻力位", "evidence": ["收盘价22.5元", "成交量放大1.5倍"], "confidence": 0.85, "battlefield": "price_volume", "target_claim_ids": []}], "resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], "round_summary": "首轮立论", "round_goal": "建立核心优势"} -->'
        )

        new_state = update_debate_state_with_payload(
            state=state,
            raw_response=raw_response,
            speaker_key="Bull",
            speaker_label="Bull Analyst",
            stance="bullish",
            history_key="bull_history",
            marker="DEBATE_STATE",
            claim_prefix="INV",
            domain="investment",
            speaker_field="current_speaker",
        )

        assert len(new_state["claims"]) == 1
        claim = new_state["claims"][0]
        assert claim["claim_id"] == "INV-1"
        assert "cluster_id" in claim
        assert claim["cluster_id"] is not None
        assert "price_shock" in claim["cluster_id"]
        assert len(claim["evidence_ids"]) == 2

    def test_research_manager_node_produces_claim_cluster_metrics(self):
        result, captured_prompts = _run_research_manager(_research_manager_state())
        debate_state = result["investment_debate_state"]

        assert "claim_cluster_metrics" in debate_state
        metrics = debate_state["claim_cluster_metrics"]
        # No E-01 relation graph is present in this legacy fixture.  The manager
        # must not reinterpret keyword clusters as independent support.
        assert metrics["independent_cluster_count"] == 0
        assert metrics["relation_graph_status"] == "pending"
        assert metrics["independence_status"] == "UNKNOWN"
        assert metrics["pending_claim_ids"]
        assert debate_state["independent_cluster_count"] == 0
        assert debate_state["analyst_count"] >= 1
        assert debate_state["verified_evidence_count"] >= 1

        # Manager prompt must have received claim_cluster_metrics (analyst_count, independent_cluster_count, verified_evidence_count)
        assert len(captured_prompts) == 1
        prompt_text = captured_prompts[0]
        assert "analyst_count" in prompt_text
        assert "independent_cluster_count" in prompt_text
        assert "verified_evidence_count" in prompt_text
        assert "E-02" in prompt_text
        assert "PENDING" in prompt_text
        assert "按 cluster_id 去重计票" not in prompt_text
        for legacy_weighting in ("动态加权", "动态赋予权重", "verdict 与权重", "高权重"):
            assert legacy_weighting not in prompt_text
        assert "E-02 关系贡献硬闸：状态=PENDING" in prompt_text


class TestRelationAwareClaimMetrics:
    """E-02 production metric adapter: explicit relations only, unknown otherwise."""

    @staticmethod
    def _claims():
        return [
            {
                "claim_id": claim_id,
                "speaker": f"Analyst {claim_id}",
                "stance": "bullish",
                "claim": f"收盘价22.5元观察 {claim_id}",
                "evidence": ["收盘价22.5元"],
            }
            for claim_id in ("A", "B", "C", "D")
        ]

    def test_missing_relation_graph_is_pending_not_keyword_independent(self):
        metrics = tally_cluster_votes(
            claims=self._claims(),
            symbol="601138.SH",
            trade_date="2026-08-22",
            relation_graph=[],
            relation_graph_status="pending",
            relation_graph_reason="E-01 relation graph unavailable",
        )

        assert metrics["relation_graph_status"] == "pending"
        assert metrics["independence_status"] == "UNKNOWN"
        assert metrics["global_contribution_cap"] == 1
        assert metrics["independent_cluster_count"] == 0
        assert metrics["pending_claim_ids"] == ["A", "B", "C", "D"]
        summary = format_claim_cluster_summary_for_prompt(metrics)
        assert "PENDING" in summary
        assert "UNKNOWN" in summary
        assert "E-01 relation graph unavailable" in summary

    def test_explicit_relations_bound_contribution_and_preserve_audit(self):
        graph = EvidenceRelationGraph.from_relations([
            EvidenceRelation(
                "A",
                RelationType.SOURCE_REPETITION,
                "B",
                metadata={"canonical_event_id": "cninfo:evt-1", "why": "same official event"},
            ),
            EvidenceRelation(
                "C",
                RelationType.DERIVED_OBSERVATION,
                "D",
                metadata={"why": "explicit derived observation"},
            ),
        ])

        metrics = tally_cluster_votes(
            claims=self._claims(),
            symbol="601138.SH",
            trade_date="2026-08-22",
            relation_graph=graph,
            relation_graph_status="available",
        )

        assert metrics["relation_graph_status"] == "available"
        assert metrics["independence_status"] == "UNKNOWN"
        assert metrics["global_contribution_cap"] == 1
        assert metrics["folded_component_count"] == 2
        assert metrics["independent_cluster_count"] == 1
        assert metrics["pending_claim_ids"] == []
        assert [
            tuple(component["member_claim_ids"])
            for component in metrics["relation_audit"]["folded_components"]
        ] == [("A", "B"), ("C", "D")]
        assert metrics["relation_audit"]["raw_relations"][0]["metadata"]["why"] == "same official event"
        assert len(metrics["relation_audit"]["audit_edges"]) == 2

    def test_invalid_relation_graph_is_pending_and_has_error_audit(self):
        metrics = tally_cluster_votes(
            claims=self._claims(),
            relation_graph=[EvidenceRelation("A", RelationType.SOURCE_REPETITION, "GHOST")],
            relation_graph_status="available",
        )

        assert metrics["relation_graph_status"] == "invalid"
        assert metrics["independence_status"] == "UNKNOWN"
        assert metrics["independent_cluster_count"] == 0
        assert metrics["pending_claim_ids"] == ["A", "B", "C", "D"]
        assert metrics["relation_audit"]["error"]["reason"] == "DANGLING_REFERENCE"
        assert metrics["relation_audit"]["error"]["stage"] == "validate"

        auto_inferred_metrics = tally_cluster_votes(
            claims=self._claims(),
            relation_graph=[
                EvidenceRelation(
                    "A",
                    RelationType.SOURCE_REPETITION,
                    "B",
                    metadata={"auto_inferred": True},
                )
            ],
            relation_graph_status="available",
        )
        assert auto_inferred_metrics["relation_graph_status"] == "invalid"
        assert auto_inferred_metrics["independent_cluster_count"] == 0
        assert auto_inferred_metrics["relation_audit"]["error"]["reason"] == "UNSUPPORTED_INFERENCE"
        assert [
            item["reason"] for item in auto_inferred_metrics["relation_audit"]["error"]["validation_results"]
        ] == ["UNSUPPORTED_INFERENCE"]

        contradictory_metrics = tally_cluster_votes(
            claims=self._claims(),
            relation_graph=[
                EvidenceRelation("A", RelationType.SUPPORTS, "B"),
                EvidenceRelation("A", RelationType.REFUTES, "B"),
            ],
            relation_graph_status="available",
        )
        assert contradictory_metrics["relation_graph_status"] == "invalid"
        assert contradictory_metrics["relation_audit"]["error"]["reason"] == "CONTRADICTORY_RELATION"

    def test_serialized_payload_is_coerced_or_rejected_with_raw_payload(self):
        serialized = {
            "relations": [
                {
                    "source_id": "A",
                    "relation_type": "SOURCE_REPETITION",
                    "target_id": "B",
                    "metadata": {"canonical_event_id": "cninfo:evt-1"},
                }
            ]
        }
        metrics = tally_cluster_votes(
            claims=self._claims(),
            relation_graph=serialized,
            relation_graph_status="available",
        )
        assert metrics["relation_graph_status"] == "available"
        assert metrics["folded_component_count"] == 1
        assert metrics["relation_audit"]["raw_relations"][0]["metadata"] == {"canonical_event_id": "cninfo:evt-1"}
        assert metrics["relation_audit"]["raw_payload"] is None
        assert "error" not in metrics["relation_audit"]

        malformed = [{"source_id": "A", "target_id": "B"}, float("nan")]
        rejected = tally_cluster_votes(
            claims=self._claims(),
            relation_graph=malformed,
            relation_graph_status="available",
            relation_graph_reason="explicit E-01 relation graph supplied at state.evidence_relations",
        )
        audit = rejected["relation_audit"]
        assert rejected["relation_graph_status"] == "invalid"
        assert rejected["independent_cluster_count"] == 0
        assert rejected["pending_claim_ids"] == ["A", "B", "C", "D"]
        assert rejected["relation_graph_reason"].startswith(
            "explicit E-01 relation graph supplied at state.evidence_relations; rejected: "
        )
        assert audit["raw_payload"] == [{"source_id": "A", "target_id": "B"}, "nan"]
        assert audit["raw_relations"] == []
        assert audit["error"]["type"] == "ValueError"
        assert audit["error"]["stage"] == "deserialize"
        assert "Missing required field 'relation_type'" in audit["error"]["message"]
        json.dumps(audit, allow_nan=False)


class TestResearchManagerRelationGraphWiring:
    """Manager path: graph discovery, seam coercion, state propagation, and prompt status."""

    def test_serialized_graph_is_folded_and_audited_through_manager(self):
        state = _research_manager_state()
        state["investment_debate_state"]["evidence_relation_graph"] = {
            "version": "v1",
            "relations": [
                {
                    "source_id": "INV-1",
                    "relation_type": "SOURCE_REPETITION",
                    "target_id": "INV-2",
                    "metadata": {"canonical_event_id": "cninfo:evt-1", "why": "same official event"},
                }
            ],
        }

        result, captured_prompts = _run_research_manager(state)
        debate_state = result["investment_debate_state"]
        metrics = debate_state["claim_cluster_metrics"]
        audit = debate_state["evidence_relation_reduction"]

        assert metrics["relation_graph_status"] == "available"
        assert metrics["relation_graph_reason"] == (
            "explicit E-01 relation graph supplied at investment_debate_state.evidence_relation_graph"
        )
        assert metrics["independence_status"] == "UNKNOWN"
        assert metrics["folded_component_count"] == 1
        assert metrics["independent_cluster_count"] == 1
        assert {"INV-1", "INV-2"}.isdisjoint(metrics["pending_claim_ids"])
        assert audit["folded_components"][0]["member_claim_ids"] == ["INV-1", "INV-2"]
        assert audit["raw_relations"][0]["metadata"]["why"] == "same official event"
        assert audit["raw_payload"] is None
        assert "error" not in audit
        assert result["manager_verdict"]["evidence_relation_status"] == "available"
        assert "E-02 关系贡献硬闸：状态=AVAILABLE" in captured_prompts[0]

    def test_auto_inferred_graph_is_rejected_and_prompt_status_follows_metrics(self):
        state = _research_manager_state()
        state["evidence_relations"] = [
            {
                "source_id": "INV-1",
                "relation_type": "SOURCE_REPETITION",
                "target_id": "INV-2",
                "metadata": {"auto_inferred": True},
            }
        ]

        result, captured_prompts = _run_research_manager(state)
        debate_state = result["investment_debate_state"]
        metrics = debate_state["claim_cluster_metrics"]
        error = debate_state["evidence_relation_reduction"]["error"]

        assert metrics["relation_graph_status"] == "invalid"
        assert metrics["independent_cluster_count"] == 0
        assert metrics["folded_component_count"] == 0
        assert error["stage"] == "validate"
        assert error["reason"] == "UNSUPPORTED_INFERENCE"
        assert [item["reason"] for item in error["validation_results"]] == ["UNSUPPORTED_INFERENCE"]
        assert debate_state["evidence_relation_reduction"]["raw_relations"][0]["metadata"] == {"auto_inferred": True}
        assert result["manager_verdict"]["evidence_relation_status"] == "invalid"
        # Discovery only saw a supplied graph; the prompt must carry the seam's final status.
        assert "E-02 关系贡献硬闸：状态=INVALID" in captured_prompts[0]
        assert "状态=AVAILABLE" not in captured_prompts[0]

    def test_malformed_graph_keeps_raw_payload_and_parse_error_in_state(self):
        state = _research_manager_state()
        malformed = {
            "relations": [
                {"source_id": "INV-1", "relation_type": "SAME_TOPIC_GUESS", "target_id": "INV-2"},
            ]
        }
        state["market_data_context"]["evidence_relation_graph"] = malformed

        result, captured_prompts = _run_research_manager(state)
        debate_state = result["investment_debate_state"]
        metrics = debate_state["claim_cluster_metrics"]
        audit = debate_state["evidence_relation_reduction"]

        assert metrics["relation_graph_status"] == "invalid"
        assert metrics["independent_cluster_count"] == 0
        assert metrics["pending_claim_ids"] == ["INV-1", "INV-2", "INV-3", "INV-4", "INV-5", "INV-6"]
        assert metrics["relation_graph_reason"].startswith(
            "explicit E-01 relation graph supplied at market_data_context.evidence_relation_graph; rejected: "
        )
        assert audit["raw_payload"] == malformed
        assert audit["raw_relations"] == []
        assert audit["error"]["type"] == "ValueError"
        assert audit["error"]["stage"] == "deserialize"
        assert "Unknown relation_type 'SAME_TOPIC_GUESS'" in audit["error"]["message"]
        assert "E-02 关系贡献硬闸：状态=INVALID" in captured_prompts[0]

    def test_blocked_manager_path_propagates_relation_rejection_audit(self):
        state = _research_manager_state()
        state["fund_flow_consensus_guard"] = {"blocked": True, "direction_allowed": False, "status": "blocked"}
        state["evidence_relation_graph"] = {"relations": "INV-1->INV-2"}

        result, captured_prompts = _run_research_manager(state)
        audit = result["investment_debate_state"]["evidence_relation_reduction"]

        assert captured_prompts == []
        assert result["trade_action"] == "NO_TRADE"
        assert result["manager_verdict"]["evidence_relation_status"] == "invalid"
        assert audit["raw_payload"] == {"relations": "INV-1->INV-2"}
        assert audit["error"]["type"] == "TypeError"
        assert audit["error"]["stage"] == "deserialize"

    def test_prompt_template_drift_fails_closed_before_llm(self, monkeypatch):
        from tradingagents.agents.managers import research_manager

        drifted = ZH_PROMPTS["research_manager_prompt"].replace("与动态加权权重：", "与动态权重：")
        monkeypatch.setattr(research_manager, "get_prompt", lambda key, config=None: drifted)

        result, captured_prompts = _run_research_manager(_research_manager_state())
        verdict = result["manager_verdict"]

        assert captured_prompts == []
        assert result["trade_action"] == "NO_TRADE"
        assert verdict["consistency_check_passed"] is False
        assert any("E-02 关系提示词守卫未通过" in check for check in verdict["failed_checks"])
        assert result["investment_debate_state"]["claim_cluster_metrics"]["relation_graph_status"] == "pending"


class TestRelationPromptGuard:
    """E-02 manager prompt guard: exact rewrite of legacy weighting, fail closed on drift."""

    _LEGACY_WEIGHTING = {
        "zh": ("动态加权", "动态赋予权重", "verdict 与权重", "按 cluster_id 去重计票", "高权重", "权重高", "加权技术"),
        "en": ("Dynamically weight", "cluster-based directional weight", "deduplicating claims by cluster_id", "Primary weight on"),
    }

    @pytest.mark.parametrize("language, prompts", [("zh", ZH_PROMPTS), ("en", EN_PROMPTS)])
    def test_real_template_loses_every_weighting_instruction(self, language, prompts):
        from string import Formatter
        from tradingagents.agents.managers.research_manager import _apply_relation_prompt_guard

        template = prompts["research_manager_prompt"]
        guarded = _apply_relation_prompt_guard(template, language)

        for marker in self._LEGACY_WEIGHTING[language]:
            assert marker in template
            assert marker not in guarded

        def fields(text):
            return sorted({name for _, name, _, _ in Formatter().parse(text) if name})

        assert fields(guarded) == fields(template)
        if language == "zh":
            for analyst in ("宏观板块", "市场（技术面）", "舆情（情绪）", "新闻", "基本面", "主力资金", "量价"):
                assert f"{analyst}：verdict 与证据贡献状态" in guarded
            assert "短线视角（short）" in guarded and "中线视角（medium）" in guarded
        else:
            assert "Short-term research horizon" in guarded
            assert guarded.count("Action basis: Primary adjudication focus on") == 2

    @pytest.mark.parametrize(
        "language, mutate, expected",
        [
            ("zh", lambda t: t.replace("与动态加权权重：", "与动态权重："), "matched 0 time"),
            ("zh", lambda t: t + "\n补充：按分析师权重汇总。", "survived"),
            ("en", lambda t: t.replace("Primary weight on ", "Main weight on ", 1), "matched 1 time"),
            ("en", lambda t: t + "\nWeight bullish clusters higher.", "survived"),
        ],
    )
    def test_template_drift_fails_closed(self, language, mutate, expected):
        from tradingagents.agents.managers.research_manager import (
            RelationPromptGuardError,
            _apply_relation_prompt_guard,
        )

        prompts = EN_PROMPTS if language == "en" else ZH_PROMPTS
        with pytest.raises(RelationPromptGuardError, match=expected):
            _apply_relation_prompt_guard(mutate(prompts["research_manager_prompt"]), language)


class TestEvidenceReducerExportsRegression:
    """Regression assertions ensuring E-02 pure reducer types and API are exported and backwards-compatible."""

    def test_e02_reducer_exports_and_contract_compatibility(self):
        """claim_cluster module must export E-02 reducer types and pure function."""
        from tradingagents.agents.utils.claim_cluster import (
            EvidenceReductionError,
            EvidenceReductionResult,
            FoldedComponent,
            IndependenceStatus,
            ReducerFailReason,
            reduce_evidence_claims,
        )

        assert issubclass(EvidenceReductionError, ValueError)
        assert issubclass(IndependenceStatus, str)
        assert issubclass(ReducerFailReason, str)
        assert IndependenceStatus.UNKNOWN == "UNKNOWN"

        # Baseline call: empty claims produce cap=0 and empty result
        empty_res = reduce_evidence_claims([], [])
        assert isinstance(empty_res, EvidenceReductionResult)
        assert empty_res.global_contribution_cap == 0
        assert empty_res.independence_status == IndependenceStatus.UNKNOWN
        assert empty_res.folded_components == ()
        assert empty_res.unconnected_claim_ids == ()
        assert empty_res.audit_edges == ()

        # Baseline call: single claim produces cap=1 and unconnected_claim_ids
        single_res = reduce_evidence_claims(["INV-1"], [])
        assert single_res.global_contribution_cap == 1
        assert single_res.unconnected_claim_ids == ("INV-1",)
        assert single_res.folded_components == ()
