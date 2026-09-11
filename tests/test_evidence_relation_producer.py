"""Unit, integration, and red-team tests for E-01 Producer (Claim-to-Claim path).

Covers:
- RT-1: Shared canonical_event_id + distinct evidence_ids -> SOURCE_REPETITION edge -> folds to 1 component.
- RT-2: Provable derivation (target_claim_ids) -> DERIVED_OBSERVATION edge -> folds to 1 component.
- RT-3: Dangling reference endpoint (not in claim_ids universe) -> DANGLING_REFERENCE fail-closed.
- RT-4: Self-loop edge (source == target) -> SELF_LOOP fail-closed.
- RT-5: Cycle in derivation relations (A->B->A) -> CYCLE_DETECTED fail-closed.
- RT-6: Unprovable relation (similarity / keywords / LLM guess) -> No edge generated; status remains pending/unknown.
- RT-7: Ambiguous multi-source relation graphs -> RELATION_GRAPH_STATUS_INVALID, never picks one.
- RT-8: End-to-end: valid graph produces available; invalid/ambiguous/unknown inputs remain invalid/pending.
- Real analysis reproduction: available when relations exist, pending when not established; 11 audit keys preserved.
"""
from __future__ import annotations

import pytest

from tradingagents.agents.utils.claim_cluster import (
    EvidenceReductionError,
    IndependenceStatus,
    reduce_evidence_claims,
    _reduce_supplied_relation_graph,
)
from tradingagents.agents.utils.debate_utils import (
    build_debate_claim_relation_graph,
    update_debate_state_with_payload,
)
from tradingagents.agents.utils.evidence_relations import (
    EvidenceRelation,
    EvidenceRelationGraph,
    FailClosedReason,
    RelationType,
    build_canonical_source_repetition,
    validate_relation,
    validate_relation_graph,
)
from tradingagents.agents.managers.research_manager import (
    RELATION_GRAPH_STATUS_AVAILABLE,
    RELATION_GRAPH_STATUS_INVALID,
    RELATION_GRAPH_STATUS_PENDING,
    _resolve_relation_graph_context,
)
from tests.test_claim_cluster import (
    _research_manager_state,
    _run_research_manager,
)


class TestRedTeamScenarios:
    """Rigorous verification of red_team_scenarios RT-1 through RT-8."""

    def test_rt1_canonical_source_repetition_different_evidence_ids_folds_component(self):
        """RT-1: 两 claim 共享同一非空 canonical_event_id 且 evidence_id 互异 -> 产出 SOURCE_REPETITION 边并折叠。"""
        claim1 = {
            "claim_id": "INV-1",
            "speaker": "Bull Analyst",
            "stance": "bullish",
            "claim": "公司获得重大中标合同",
            "evidence": ["中标金额15亿元"],
            "confidence": 0.85,
            "canonical_event_id": "cninfo:1200000001",
            "evidence_id": "ev_ann_001",
        }
        claim2 = {
            "claim_id": "INV-2",
            "speaker": "Bear Analyst",
            "stance": "bearish",
            "claim": "重大中标项目存在毛利率偏低风险",
            "evidence": ["历史同类项目毛利仅8%"],
            "confidence": 0.75,
            "canonical_event_id": "cninfo:1200000001",
            "evidence_id": "ev_ann_002",
        }

        # 1. build_canonical_source_repetition 直接作用于两条 claim
        rel = build_canonical_source_repetition(claim1, claim2)
        assert rel is not None
        assert rel.relation_type == RelationType.SOURCE_REPETITION
        assert rel.source_id == "INV-1"
        assert rel.target_id == "INV-2"
        assert rel.metadata["canonical_event_id"] == "cninfo:1200000001"

        # 2. build_debate_claim_relation_graph 生成图
        graph = build_debate_claim_relation_graph([claim1, claim2])
        assert len(graph.relations) == 1
        assert graph.relations[0].dedupe_key == ("INV-1", "SOURCE_REPETITION", "INV-2")

        # 3. 下游 reduce_evidence_claims 验证折叠
        res = reduce_evidence_claims(["INV-1", "INV-2"], graph)
        assert len(res.folded_components) == 1
        comp = res.folded_components[0]
        assert comp.member_claim_ids == ("INV-1", "INV-2")
        assert comp.derived_terminal_ids == ()
        assert res.global_contribution_cap == 1
        assert res.independence_status == IndependenceStatus.UNKNOWN

        # 4. 边界防御测试
        # 4a. 相同 evidence_id -> 必须为 None，不得建边
        claim_same_eid = dict(claim2, evidence_id="ev_ann_001")
        assert build_canonical_source_repetition(claim1, claim_same_eid) is None

        # 4b. 不同 canonical_event_id -> 必须为 None
        claim_diff_cid = dict(claim2, canonical_event_id="cninfo:9999999999")
        assert build_canonical_source_repetition(claim1, claim_diff_cid) is None

        # 4c. 缺少 canonical_event_id -> 必须为 None
        claim_no_cid = dict(claim2, canonical_event_id=None)
        assert build_canonical_source_repetition(claim1, claim_no_cid) is None

        # 4d. 缺少 evidence_id -> 必须为 None
        claim_no_eid = dict(claim2, evidence_id=None, evidence_ids=[])
        assert build_canonical_source_repetition(claim1, claim_no_eid) is None

    def test_rt2_derived_observation_explicit_derivation_folds_component(self):
        """RT-2: 同一观测派生的两条 claim（派生关系可证） -> 产出 DERIVED_OBSERVATION 边并折叠。"""
        # INV-1 为基础观测，INV-2 与 INV-3 均派生自 INV-1
        claim1 = {
            "claim_id": "INV-1",
            "speaker": "Bull Analyst",
            "stance": "bullish",
            "claim": "全资子公司收到中标通知书金额15亿元",
            "evidence": ["中标通知书编号ZB-2026-001"],
            "confidence": 0.90,
            "target_claim_ids": [],
            "stage": "opening",
        }
        claim2 = {
            "claim_id": "INV-2",
            "speaker": "Bull Analyst",
            "stance": "bullish",
            "claim": "该15亿中标预计增厚当期净利润约1.2亿元",
            "evidence": ["行业平均净利率8%测算"],
            "confidence": 0.80,
            "target_claim_ids": ["INV-1"],
            "stage": "challenge",
        }
        claim3 = {
            "claim_id": "INV-3",
            "speaker": "Bear Analyst",
            "stance": "bearish",
            "claim": "该中标合同垫资周期长或将加大应收账款周转压力",
            "evidence": ["招标公告付款条件为分期支付"],
            "confidence": 0.75,
            "target_claim_ids": ["INV-1"],
            "stage": "challenge",
        }

        graph = build_debate_claim_relation_graph([claim1, claim2, claim3])
        assert len(graph.relations) == 2

        # 验证有向边 source(derived) -> target(base)
        rel_types = {r.dedupe_key for r in graph.relations}
        assert ("INV-2", "DERIVED_OBSERVATION", "INV-1") in rel_types
        assert ("INV-3", "DERIVED_OBSERVATION", "INV-1") in rel_types

        # 验证下游 reducer 将派生自同一观测的 claims 折叠为单组件
        res = reduce_evidence_claims(["INV-1", "INV-2", "INV-3"], graph)
        assert len(res.folded_components) == 1
        comp = res.folded_components[0]
        assert comp.member_claim_ids == ("INV-1", "INV-2", "INV-3")
        assert comp.derived_terminal_ids == ("INV-1",)
        assert res.global_contribution_cap == 1
        assert res.unconnected_claim_ids == ()

    def test_rt3_dangling_reference_endpoint_fails_closed(self):
        """RT-3: 关系端点引用了不在 claim_ids 宇宙内的 id -> 必须 DANGLING_REFERENCE fail-closed。"""
        claim_ids = ["INV-1", "INV-2"]
        # 端点引用了幽灵节点 INV-999
        bad_rel = EvidenceRelation(
            source_id="INV-1",
            relation_type=RelationType.DERIVED_OBSERVATION,
            target_id="INV-999",
            metadata={"producer": "debate_protocol_v2"},
        )
        graph = EvidenceRelationGraph.from_relations([bad_rel])

        # 1. validate_relation_graph 必须识别断链
        valid, results = validate_relation_graph(graph, set(claim_ids))
        assert valid is False
        assert any(r.reason == FailClosedReason.DANGLING_REFERENCE for r in results)

        # 2. _reduce_supplied_relation_graph 拒绝折叠并记录审计
        res, raw_rels, rejection = _reduce_supplied_relation_graph(claim_ids, graph)
        assert res is None
        assert rejection is not None
        assert rejection.stage == "validate"
        assert rejection.error.error_reason == FailClosedReason.DANGLING_REFERENCE

        # 3. 验证 producer 不会静默丢弃该断链边，而是真实暴露给下游校验
        claim_with_dangling = {
            "claim_id": "INV-1",
            "claim": "立论1",
            "evidence": ["证据1"],
            "confidence": 0.8,
            "target_claim_ids": ["INV-999"],
        }
        prod_graph = build_debate_claim_relation_graph([claim_with_dangling])
        assert any(r.target_id == "INV-999" for r in prod_graph.relations)
        v_res, _ = validate_relation_graph(prod_graph, {"INV-1"})
        assert v_res is False

    def test_rt4_self_loop_fails_closed(self):
        """RT-4: 自环边（source == target） -> 必须 fail-closed 拒绝。"""
        rel = EvidenceRelation(
            source_id="INV-1",
            relation_type=RelationType.DERIVED_OBSERVATION,
            target_id="INV-1",
            metadata={"producer": "debate_protocol_v2"},
        )
        # 1. validate_relation 校验自环
        val = validate_relation(rel, {"INV-1"})
        assert val.valid is False
        assert val.reason == FailClosedReason.SELF_LOOP

        # 2. validate_relation_graph 校验整图自环
        graph = EvidenceRelationGraph.from_relations([rel])
        v_graph, results = validate_relation_graph(graph, {"INV-1"})
        assert v_graph is False
        assert any(r.reason == FailClosedReason.SELF_LOOP for r in results)

        # 3. seam 拒绝折叠
        res, _, rejection = _reduce_supplied_relation_graph(["INV-1"], graph)
        assert res is None
        assert rejection is not None
        assert rejection.stage == "validate"
        assert rejection.error.error_reason == FailClosedReason.SELF_LOOP

    def test_rt5_cycle_detected_fails_closed(self):
        """RT-5: 关系图存在环（A->B->A） -> 必须 fail-closed 拒绝，不得侥幸通过。"""
        claim_ids = ["INV-1", "INV-2"]
        r1 = EvidenceRelation("INV-1", RelationType.DERIVED_OBSERVATION, "INV-2")
        r2 = EvidenceRelation("INV-2", RelationType.DERIVED_OBSERVATION, "INV-1")
        graph = EvidenceRelationGraph.from_relations([r1, r2])

        # 1. 图校验直接查出环路
        v_graph, results = validate_relation_graph(graph, set(claim_ids))
        assert v_graph is False
        assert any(r.reason == FailClosedReason.CYCLE_DETECTED for r in results)

        # 2. reduce_evidence_claims 自身也必须抛出 CYCLE_DETECTED 异常
        with pytest.raises(EvidenceReductionError) as exc_info:
            reduce_evidence_claims(claim_ids, graph)
        assert exc_info.value.error_reason == FailClosedReason.CYCLE_DETECTED

        # 3. 三节点传递环 (A->B->C->A)
        c_claim_ids = ["INV-1", "INV-2", "INV-3"]
        c1 = EvidenceRelation("INV-1", RelationType.DERIVED_OBSERVATION, "INV-2")
        c2 = EvidenceRelation("INV-2", RelationType.DERIVED_OBSERVATION, "INV-3")
        c3 = EvidenceRelation("INV-3", RelationType.DERIVED_OBSERVATION, "INV-1")
        c_graph = EvidenceRelationGraph.from_relations([c1, c2, c3])
        v_c_graph, c_results = validate_relation_graph(c_graph, set(c_claim_ids))
        assert v_c_graph is False
        assert any(r.reason == FailClosedReason.CYCLE_DETECTED for r in c_results)

    def test_rt6_unprovable_relations_strictly_forbids_edges(self):
        """RT-6: 关系无法证明（标题相似 / 关键词重合 / LLM推断） -> 不得生成边；未知必须显式 unknown。"""
        # 两条文本高度相似但无法可证关系的立论
        claim1 = {
            "claim_id": "INV-1",
            "claim": "主力超大单今日净流入2.5亿元，资金加速进场",
            "evidence": ["东财资金流监控数据"],
            "confidence": 0.85,
            "target_claim_ids": [],
        }
        claim2 = {
            "claim_id": "INV-2",
            "claim": "主力大单今日大幅净流入2.48亿元，机构买盘强劲",
            "evidence": ["同花顺主力跟踪数据"],
            "confidence": 0.83,
            "target_claim_ids": [],
        }

        # 1. 生产者不得基于文本相似度建边
        graph = build_debate_claim_relation_graph([claim1, claim2])
        assert len(graph.relations) == 0

        # 2. 封禁任何 auto_inferred 或 similarity_score 注入
        forbidden_rel = EvidenceRelation(
            "INV-1",
            RelationType.SOURCE_REPETITION,
            "INV-2",
            metadata={"auto_inferred": True, "similarity_score": 0.98},
        )
        val = validate_relation(forbidden_rel, {"INV-1", "INV-2"})
        assert val.valid is False
        assert val.reason == FailClosedReason.UNSUPPORTED_INFERENCE

    def test_rt7_multi_source_ambiguity_returns_invalid_never_chooses(self):
        """RT-7: 多来源关系图歧义（state 与 debate_state 各有一份且不一致） -> 返回 invalid，不得择优。"""
        graph_a = {
            "relations": [
                {
                    "source_id": "INV-1",
                    "relation_type": "SOURCE_REPETITION",
                    "target_id": "INV-2",
                    "metadata": {"canonical_event_id": "cninfo:evt-A"},
                }
            ]
        }
        graph_b = {
            "relations": [
                {
                    "source_id": "INV-3",
                    "relation_type": "DERIVED_OBSERVATION",
                    "target_id": "INV-4",
                    "metadata": {"derivation_source": "target_claim_ids"},
                }
            ]
        }

        state = {
            "evidence_relation_graph": graph_a,
            "investment_debate_state": {
                "evidence_relation_graph": graph_b,
            },
        }

        raw_graph, status, reason = _resolve_relation_graph_context(
            state,
            state["investment_debate_state"],
        )

        assert status == RELATION_GRAPH_STATUS_INVALID
        assert "multiple E-01 relation graph payloads supplied" in reason
        assert isinstance(raw_graph, dict)
        assert "state.evidence_relation_graph" in raw_graph
        assert "investment_debate_state.evidence_relation_graph" in raw_graph

    def test_rt8_end_to_end_valid_vs_invalid_and_pending_gates(self):
        """RT-8: 合法图输入下产出 available；RT-3~RT-7 的未知/歧义/断链输入必须继续为 pending/invalid。"""
        # 1. 合法 fixture: 产出 available 且有效贡献上限生效
        state_valid = _research_manager_state()
        state_valid["investment_debate_state"]["evidence_relation_graph"] = {
            "version": "v1",
            "relations": [
                {
                    "source_id": "INV-1",
                    "relation_type": "SOURCE_REPETITION",
                    "target_id": "INV-2",
                    "metadata": {"canonical_event_id": "cninfo:evt-001"},
                }
            ],
        }
        res_valid, prompts_valid = _run_research_manager(state_valid)
        metrics_valid = res_valid["investment_debate_state"]["claim_cluster_metrics"]
        audit_valid = res_valid["investment_debate_state"]["evidence_relation_reduction"]

        assert metrics_valid["relation_graph_status"] == RELATION_GRAPH_STATUS_AVAILABLE
        assert metrics_valid["independent_cluster_count"] == 1
        assert metrics_valid["effective_contribution_count"] == 1
        assert metrics_valid["folded_component_count"] == 1
        assert res_valid["manager_verdict"]["evidence_relation_status"] == "available"
        assert "E-02 关系贡献硬闸：状态=AVAILABLE" in prompts_valid[0]

        # 检查 11 个审计键齐全
        expected_keys = {
            "status", "reason", "raw_relations", "raw_payload", "audit_edges",
            "ignored_relations", "folded_components", "unconnected_claim_ids",
            "pending_claims", "independence_status", "global_contribution_cap",
        }
        assert expected_keys.issubset(set(audit_valid.keys()))

        # 2. RT-3 断链端点输入: 必须为 invalid
        state_rt3 = _research_manager_state()
        state_rt3["investment_debate_state"]["evidence_relation_graph"] = {
            "version": "v1",
            "relations": [
                {
                    "source_id": "INV-1",
                    "relation_type": "DERIVED_OBSERVATION",
                    "target_id": "INV-GHOST",
                }
            ],
        }
        res_rt3, _ = _run_research_manager(state_rt3)
        assert res_rt3["investment_debate_state"]["claim_cluster_metrics"]["relation_graph_status"] == RELATION_GRAPH_STATUS_INVALID
        assert res_rt3["investment_debate_state"]["claim_cluster_metrics"]["independent_cluster_count"] == 0

        # 3. RT-4 自环输入: 必须为 invalid
        state_rt4 = _research_manager_state()
        state_rt4["investment_debate_state"]["evidence_relation_graph"] = {
            "version": "v1",
            "relations": [
                {
                    "source_id": "INV-1",
                    "relation_type": "DERIVED_OBSERVATION",
                    "target_id": "INV-1",
                }
            ],
        }
        res_rt4, _ = _run_research_manager(state_rt4)
        assert res_rt4["investment_debate_state"]["claim_cluster_metrics"]["relation_graph_status"] == RELATION_GRAPH_STATUS_INVALID

        # 4. RT-5 环路输入: 必须为 invalid
        state_rt5 = _research_manager_state()
        state_rt5["investment_debate_state"]["evidence_relation_graph"] = {
            "version": "v1",
            "relations": [
                {"source_id": "INV-1", "relation_type": "DERIVED_OBSERVATION", "target_id": "INV-2"},
                {"source_id": "INV-2", "relation_type": "DERIVED_OBSERVATION", "target_id": "INV-1"},
            ],
        }
        res_rt5, _ = _run_research_manager(state_rt5)
        assert res_rt5["investment_debate_state"]["claim_cluster_metrics"]["relation_graph_status"] == RELATION_GRAPH_STATUS_INVALID

        # 5. RT-6 关系未知/无图: 必须为 pending
        state_rt6 = _research_manager_state()
        # 无 evidence_relation_graph
        state_rt6["investment_debate_state"].pop("evidence_relation_graph", None)
        state_rt6.pop("evidence_relation_graph", None)
        res_rt6, _ = _run_research_manager(state_rt6)
        assert res_rt6["investment_debate_state"]["claim_cluster_metrics"]["relation_graph_status"] == RELATION_GRAPH_STATUS_PENDING
        assert res_rt6["investment_debate_state"]["claim_cluster_metrics"]["independent_cluster_count"] == 0

        # 6. RT-7 歧义输入: 必须为 invalid
        state_rt7 = _research_manager_state()
        state_rt7["evidence_relation_graph"] = state_valid["investment_debate_state"]["evidence_relation_graph"]
        state_rt7["investment_debate_state"]["evidence_relation_graph"] = state_rt3["investment_debate_state"]["evidence_relation_graph"]
        res_rt7, _ = _run_research_manager(state_rt7)
        assert res_rt7["investment_debate_state"]["claim_cluster_metrics"]["relation_graph_status"] == RELATION_GRAPH_STATUS_INVALID

    def test_rt9_supports_and_refutes_valid_e01_but_never_folded(self):
        """RT-9: 可证明的非折叠关系（SUPPORTS / REFUTES / 跨立场）通过 E-01 校验，但下游 reducer 绝不折叠。"""
        claim_ids = ["INV-1", "INV-2", "INV-3"]
        # INV-2 supports INV-1; INV-3 refutes INV-1
        rel_sup = EvidenceRelation(
            source_id="INV-2",
            relation_type=RelationType.SUPPORTS,
            target_id="INV-1",
            metadata={"citation_role": "premise"},
        )
        rel_ref = EvidenceRelation(
            source_id="INV-3",
            relation_type=RelationType.REFUTES,
            target_id="INV-1",
            metadata={"rationale": "falsified assumption"},
        )
        graph = EvidenceRelationGraph.from_relations([rel_sup, rel_ref])

        # 1. E-01 单边与整图校验必须 PASS
        valid, results = validate_relation_graph(graph, set(claim_ids))
        assert valid is True
        assert results == []

        # 2. 下游 reduce_evidence_claims 绝不将 SUPPORTS/REFUTES 折叠进 FoldedComponent
        res = reduce_evidence_claims(claim_ids, graph)
        assert len(res.folded_components) == 0
        assert set(res.unconnected_claim_ids) == {"INV-1", "INV-2", "INV-3"}
        assert res.independence_status == IndependenceStatus.UNKNOWN

        # 3. 通过 seam 审计验证: SUPPORTS 与 REFUTES 明确被分类至 ignored_relations
        _, _, rejection = _reduce_supplied_relation_graph(claim_ids, graph)
        assert rejection is None
        state = _research_manager_state()
        state["investment_debate_state"]["evidence_relation_graph"] = graph.to_dict()
        res_mgr, _ = _run_research_manager(state)
        audit = res_mgr["investment_debate_state"]["evidence_relation_reduction"]
        assert audit["folded_components"] == []
        ignored_types = [r["relation_type"] for r in audit["ignored_relations"]]
        assert "SUPPORTS" in ignored_types
        assert "REFUTES" in ignored_types

    def test_rt10_same_container_dual_keys_and_multi_container_ambiguity(self):
        """RT-10: 同容器双键冲突与多容器歧义注入 -> 严格返回 invalid，保留全部歧义源审计，不得择优。"""
        graph_payload_1 = {
            "version": "v1",
            "relations": [
                {
                    "source_id": "INV-1",
                    "relation_type": "SOURCE_REPETITION",
                    "target_id": "INV-2",
                    "metadata": {"canonical_event_id": "cninfo:evt-1"},
                }
            ],
        }
        graph_payload_2 = {
            "version": "v1",
            "relations": [
                {
                    "source_id": "INV-3",
                    "relation_type": "DERIVED_OBSERVATION",
                    "target_id": "INV-4",
                }
            ],
        }

        # 1. 同容器双键冲突: investment_debate_state 同时提供 evidence_relation_graph 与 evidence_relations
        state_same_container = _research_manager_state()
        state_same_container["investment_debate_state"]["evidence_relation_graph"] = graph_payload_1
        state_same_container["investment_debate_state"]["evidence_relations"] = graph_payload_2
        raw_graph, status, reason = _resolve_relation_graph_context(
            state_same_container,
            state_same_container["investment_debate_state"],
        )
        assert status == RELATION_GRAPH_STATUS_INVALID
        assert "multiple E-01 relation graph payloads supplied" in reason
        assert "investment_debate_state.evidence_relation_graph" in raw_graph
        assert "investment_debate_state.evidence_relations" in raw_graph

        # 2. 同容器双键冲突: state 同时提供双键
        state_root = _research_manager_state()
        state_root["evidence_relation_graph"] = graph_payload_1
        state_root["evidence_relations"] = graph_payload_2
        raw_root, status_root, _ = _resolve_relation_graph_context(
            state_root,
            state_root["investment_debate_state"],
        )
        assert status_root == RELATION_GRAPH_STATUS_INVALID
        assert "state.evidence_relation_graph" in raw_root
        assert "state.evidence_relations" in raw_root

        # 3. 跨容器歧义 (枚举 4 个候选容器中的任意两处，如 market_data_context 与 event_coverage)
        state_cross = _research_manager_state()
        state_cross["market_data_context"]["evidence_relation_graph"] = graph_payload_1
        state_cross["event_coverage"] = {"evidence_relations": graph_payload_2}
        raw_cross, status_cross, _ = _resolve_relation_graph_context(
            state_cross,
            state_cross["investment_debate_state"],
        )
        assert status_cross == RELATION_GRAPH_STATUS_INVALID
        assert "market_data_context.evidence_relation_graph" in raw_cross
        assert "event_coverage.evidence_relations" in raw_cross

        # 4. 端到端跑 manager，验证 invalid 审计保留全部歧义载荷且不折叠
        res_mgr, _ = _run_research_manager(state_same_container)
        metrics = res_mgr["investment_debate_state"]["claim_cluster_metrics"]
        audit = res_mgr["investment_debate_state"]["evidence_relation_reduction"]
        assert metrics["relation_graph_status"] == RELATION_GRAPH_STATUS_INVALID
        assert metrics["independent_cluster_count"] == 0
        assert audit["raw_payload"] == raw_graph

    def test_rt11_contradictory_relation_and_lookahead_violation_fail_closed(self):
        """RT-11: 互斥边矛盾（同一有向边并存 SUPPORTS 与 REFUTES）与前视违规（时间晚于 baseline_date）必须 fail-closed。"""
        claim_ids = ["INV-1", "INV-2"]

        # 1. 互斥边矛盾: 同一有向边 INV-1 -> INV-2 并存 SUPPORTS 与 REFUTES
        r_sup = EvidenceRelation("INV-1", RelationType.SUPPORTS, "INV-2")
        r_ref = EvidenceRelation("INV-1", RelationType.REFUTES, "INV-2")
        graph_contra = EvidenceRelationGraph.from_relations([r_sup, r_ref])

        valid, results = validate_relation_graph(graph_contra, set(claim_ids))
        assert valid is False
        assert any(r.reason == FailClosedReason.CONTRADICTORY_RELATION for r in results)

        res, _, rejection = _reduce_supplied_relation_graph(claim_ids, graph_contra)
        assert res is None
        assert rejection is not None
        assert rejection.error.error_reason == FailClosedReason.CONTRADICTORY_RELATION

        # 2. 时间前视违规: source_ctx 时间晚于 baseline_date (2026-08-25 > 2026-08-22)
        r_norm = EvidenceRelation("INV-1", RelationType.DERIVED_OBSERVATION, "INV-2")
        val_lookahead = validate_relation(
            r_norm,
            known_node_ids=set(claim_ids),
            source_ctx={"ann_date": "2026-08-25"},
            baseline_date="2026-08-22",
        )
        assert val_lookahead.valid is False
        assert val_lookahead.reason == FailClosedReason.LOOKAHEAD_VIOLATION


class TestDebateProducerLifecycle:
    """Tests verifying debate lifecycle state updating and relation graph production."""

    def test_update_debate_state_produces_graph_when_relations_exist(self):
        """Debate state settlement produces relation graph when challenge claims target opening claims."""
        # 1. 模拟开篇两轮立论
        init_state = {
            "claims": [],
            "open_claim_ids": [],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": [],
            "round_messages": [],
            "count": 0,
            "claim_counter": 0,
            "history": "",
        }
        raw_msg1 = (
            "多头立论正文\n\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": [], "new_claims": [{"claim": "中标15亿公告发布", "evidence": ["公告编号001"], "confidence": 0.85, "battlefield": "fundamentals", "target_claim_ids": []}], "resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], "round_summary": "立论1", "round_goal": "建立优势"} -->'
        )
        state1 = update_debate_state_with_payload(
            state=init_state,
            raw_response=raw_msg1,
            speaker_key="Bull",
            speaker_label="Bull Analyst",
            stance="bullish",
            history_key="bull_history",
            marker="DEBATE_STATE",
            claim_prefix="INV",
            domain="investment",
            speaker_field="current_speaker",
        )
        assert len(state1["claims"]) == 1
        # 开篇无 target_claim_ids，关系图为 None
        assert state1.get("evidence_relation_graph") is None

        # 2. 空头质询并引用 INV-1
        raw_msg2 = (
            "空头反驳正文\n\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-1"], "new_claims": [{"claim": "中标项目垫资周期过长", "evidence": ["垫资公告条款"], "confidence": 0.80, "battlefield": "fundamentals", "target_claim_ids": ["INV-1"]}], "resolved_claim_ids": [], "unresolved_claim_ids": ["INV-1"], "next_focus_claim_ids": [], "round_summary": "质询1", "round_goal": "暴露风险"} -->'
        )
        state2 = update_debate_state_with_payload(
            state=state1,
            raw_response=raw_msg2,
            speaker_key="Bear",
            speaker_label="Bear Analyst",
            stance="bearish",
            history_key="bear_history",
            marker="DEBATE_STATE",
            claim_prefix="INV",
            domain="investment",
            speaker_field="current_speaker",
        )
        assert len(state2["claims"]) == 2
        # 质询阶段产生了 target_claim_ids: ["INV-1"]，关系图必须自动产出！
        graph_dict = state2.get("evidence_relation_graph")
        assert graph_dict is not None
        assert graph_dict["version"] == "v1"
        assert len(graph_dict["relations"]) == 1
        rel = graph_dict["relations"][0]
        assert rel["source_id"] == "INV-2"
        assert rel["relation_type"] == "DERIVED_OBSERVATION"
        assert rel["target_id"] == "INV-1"

        # 3. 传入 research_manager 完整运行（配合合规的 4 轮已完结辩论）
        from tests.test_debate_b3_protocol import _build_v2_challenge_completed_state
        inv_state = _build_v2_challenge_completed_state()
        inv_state["tiebreak_skipped"] = True
        # 加入 state2 产出的关系图与派生 claims
        inv_state["claims"].extend(state2["claims"])
        inv_state["evidence_relation_graph"] = state2["evidence_relation_graph"]

        mgr_state = _research_manager_state()
        mgr_state["investment_debate_state"] = inv_state
        result, captured_prompts = _run_research_manager(mgr_state)
        metrics = result["investment_debate_state"]["claim_cluster_metrics"]
        assert metrics["relation_graph_status"] == RELATION_GRAPH_STATUS_AVAILABLE
        assert metrics["independent_cluster_count"] == 1
        assert metrics["effective_contribution_count"] == 1
        assert len(captured_prompts) == 1
        assert "E-02 关系贡献硬闸：状态=AVAILABLE" in captured_prompts[0]

    def test_unstructured_response_fallback_preserves_evidence_relation_graph(self):
        """回退分支约束: 当解析失败进入 _record_unstructured_response 时，先前已建立的关系图不得被篡改或清空。"""
        from tradingagents.agents.utils.debate_utils import _record_unstructured_response
        prior_graph = {
            "version": "v1",
            "relations": [
                {
                    "source_id": "INV-2",
                    "relation_type": "DERIVED_OBSERVATION",
                    "target_id": "INV-1",
                    "metadata": {"producer": "debate_protocol_v2"},
                }
            ],
        }
        state_with_graph = {
            "count": 2,
            "claims": [{"claim_id": "INV-1"}, {"claim_id": "INV-2"}],
            "evidence_relation_graph": prior_graph,
            "round_messages": [],
            "attempts": [],
        }

        # 模拟模型输出无法解析为机读块，触发 _record_unstructured_response
        new_state = _record_unstructured_response(
            state=state_with_graph,
            raw_response="模型回答无法提取标签",
            speaker_label="Bull Analyst",
            speaker_key="Bull",
            history_key="bull_history",
            speaker_field="current_speaker",
            store_current_response=True,
            parse_status="missing",
            error_detail="未找到机读块",
            domain="investment",
        )

        assert new_state["blocked"] is True
        assert new_state["parse_status"] == "missing"
        # 核心断言：原有的 evidence_relation_graph 完好无损保留，未被清空或覆盖
        assert new_state.get("evidence_relation_graph") == prior_graph

    def test_early_return_path_preserves_relation_audit_encapsulation(self):
        """早退路径: INVALID/ABSTAIN 场景（_blocked_manager_payload）同样合规封装关系图审计数据。"""
        state = _research_manager_state()
        # 注入合法关系图
        state["investment_debate_state"]["evidence_relation_graph"] = {
            "version": "v1",
            "relations": [
                {
                    "source_id": "INV-1",
                    "relation_type": "SOURCE_REPETITION",
                    "target_id": "INV-2",
                    "metadata": {"canonical_event_id": "cninfo:evt-100"},
                }
            ],
        }
        # 触发资金流 guard 阻断（走早期 ABSTAIN 分支）
        state["fund_flow_consensus_guard"] = {
            "blocked": True,
            "direction_allowed": False,
            "status": "blocked",
        }

        result, captured_prompts = _run_research_manager(state)
        assert captured_prompts == []
        assert result["trade_action"] == "NO_TRADE"
        debate_state = result["investment_debate_state"]
        metrics = debate_state["claim_cluster_metrics"]
        audit = debate_state["evidence_relation_reduction"]

        # 即使早退，关系图仍然合规执行了反序列化与折叠审计，并正确记录审计字段
        assert metrics["relation_graph_status"] == RELATION_GRAPH_STATUS_AVAILABLE
        assert metrics["independent_cluster_count"] == 1
        assert audit["folded_components"][0]["member_claim_ids"] == ["INV-1", "INV-2"]
        assert TestRealAnalysisReproduction.REQUIRED_11_AUDIT_KEYS.issubset(set(audit.keys()))


class TestRealAnalysisReproduction:
    """Acceptance Nail 2 & 3: Real analysis reproduction with available vs pending and 11 audit keys."""

    REQUIRED_11_AUDIT_KEYS = frozenset({
        "status",
        "reason",
        "raw_relations",
        "raw_payload",
        "audit_edges",
        "ignored_relations",
        "folded_components",
        "unconnected_claim_ids",
        "pending_claims",
        "independence_status",
        "global_contribution_cap",
    })

    def test_reproduction_available_when_relations_exist(self):
        """合法证据关系存在时：evidence_relation_status 为 available，independent_cluster_count / effective_contribution_count 为非零值，11 审计键齐全。"""
        from tests.test_debate_b3_protocol import _build_v2_challenge_completed_state
        inv_state = _build_v2_challenge_completed_state()
        inv_state["tiebreak_skipped"] = True

        # 添加包含合法派生关系的 claims: INV-7 派生自 INV-1
        derived_claim = {
            "claim_id": "INV-7",
            "speaker": "Bear Analyst",
            "speaker_key": "Bear",
            "stance": "bearish",
            "claim": "主力买单中对冲盘占比较高，实际多头动能存疑",
            "evidence": ["高频量化对冲委托比达35%"],
            "confidence": 0.82,
            "target_claim_ids": ["INV-1"],
            "stage": "challenge",
        }
        inv_state["claims"].append(derived_claim)

        # 运行 producer 生成关系图并写入 debate_state
        graph = build_debate_claim_relation_graph(inv_state["claims"])
        inv_state["evidence_relation_graph"] = graph.to_dict()

        # 运行真实分析 (Research Manager 节点)
        state = _research_manager_state()
        state["investment_debate_state"] = inv_state
        result, captured_prompts = _run_research_manager(state)

        debate_state = result["investment_debate_state"]
        metrics = debate_state["claim_cluster_metrics"]
        reduction_audit = debate_state["evidence_relation_reduction"]

        # 断言 1: evidence_relation_status 为 available
        assert debate_state["evidence_relation_status"] == "available"
        assert result["manager_verdict"]["evidence_relation_status"] == "available"
        assert metrics["relation_graph_status"] == "available"

        # 断言 2: independent_cluster_count 与 effective_contribution_count 为非零值
        assert metrics["independent_cluster_count"] > 0
        assert metrics["effective_contribution_count"] > 0
        assert metrics["folded_component_count"] >= 1

        # 断言 3: 审计键齐全 (11 个审计键不得减少)
        audit_keys = set(reduction_audit.keys())
        assert self.REQUIRED_11_AUDIT_KEYS.issubset(audit_keys)
        assert len(audit_keys) >= 11

    def test_reproduction_pending_when_relations_do_not_exist(self):
        """证据关系不成立时：evidence_relation_status 仍为 pending，independent_cluster_count 为 0，11 审计键齐全。"""
        from tests.test_debate_b3_protocol import _build_v2_challenge_completed_state
        inv_state = _build_v2_challenge_completed_state()
        inv_state["tiebreak_skipped"] = True

        # 确保所有 claims 均无可证关系 (无 target_claim_ids, 无 canonical_event_id)
        for c in inv_state["claims"]:
            c["target_claim_ids"] = []
            c.pop("canonical_event_id", None)
            c.pop("evidence_id", None)
        inv_state["evidence_relation_graph"] = None

        # 运行真实分析 (Research Manager 节点)
        state = _research_manager_state()
        state["investment_debate_state"] = inv_state
        result, captured_prompts = _run_research_manager(state)

        debate_state = result["investment_debate_state"]
        metrics = debate_state["claim_cluster_metrics"]
        reduction_audit = debate_state["evidence_relation_reduction"]

        # 断言 1: 证据关系不成立时仍为 pending
        assert debate_state["evidence_relation_status"] == "pending"
        assert result["manager_verdict"]["evidence_relation_status"] == "pending"
        assert metrics["relation_graph_status"] == "pending"

        # 断言 2: independent_cluster_count 与 effective_contribution_count 归零
        assert metrics["independent_cluster_count"] == 0
        assert metrics["effective_contribution_count"] == 0
        assert metrics["folded_component_count"] == 0

        # 断言 3: 审计键齐全 (11 个审计键不得减少)
        audit_keys = set(reduction_audit.keys())
        assert self.REQUIRED_11_AUDIT_KEYS.issubset(audit_keys)
        assert len(audit_keys) >= 11
