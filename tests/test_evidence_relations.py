"""Unit tests for evidence relation types, graph container, validation rules, and cycle detection.

Covers:
- DAV-741 14 core scenarios:
  1. Valid SOURCE_REPETITION with identical canonical_event_id
  2. Valid DERIVED_OBSERVATION (explicit caller-provided)
  3. Valid SUPPORTS relation (explicit caller-provided)
  4. Valid REFUTES relation (explicit caller-provided)
  5. Idempotent duplicate edge insertion
  6. Serialization and deserialization roundtrip
  7. Rejection of self-loop (SELF_LOOP)
  8. Rejection of dangling reference (DANGLING_REFERENCE)
  9. Rejection of contradictory relations (CONTRADICTORY_RELATION)
  10. Rejection of cyclic derivation relations (CYCLE_DETECTED)
  11. Rejection of lookahead violation (LOOKAHEAD_VIOLATION)
  12. Rejection of unhashed content for canonical derivation (CONTENT_UNHASHED)
  13. Strict forbidding of CollateralRecord.canonical_event_id
  14. Rejection of unsupported auto-inference (UNSUPPORTED_INFERENCE)
- Additional required edge cases:
  15. Rejection of empty, whitespace, or non-string endpoint IDs
  16. Rejection of non-RelationType and unknown relation type deserialization without silent upgrade
  17. Metadata deep freeze and nested immutability (MappingProxyType, tuples)
  18. Rejection of NaN and Infinity in metadata
  19. build_canonical_source_repetition edge cases (identical evidence_id, missing evidence_id, missing canonical_id, whitespace trim)
  20. Malformed, missing, and future dates fail-closed (MALFORMED_TIMESTAMP, MISSING_TIMESTAMP, LOOKAHEAD_VIOLATION)
  21. Deterministic first-seen metadata semantics for duplicate edges
  22. EvidenceRelation is unhashable and cannot be placed in a set
"""
from __future__ import annotations

from datetime import date
import json
from types import MappingProxyType
import pytest

from tradingagents.agents.utils.evidence_relations import (
    EvidenceRelation,
    EvidenceRelationGraph,
    FailClosedReason,
    RelationType,
    ValidationResult,
    build_canonical_source_repetition,
    detect_relation_cycles,
    validate_relation,
    validate_relation_graph,
)
from tradingagents.dataflows.news_event_evidence import CollateralRecord


# ============================================================================
# DAV-741 14 Core Scenarios
# ============================================================================

def test_valid_source_repetition_with_identical_canonical_id():
    """1. 正例：两端具备相同官方 canonical_event_id，build_canonical_source_repetition 建边成功且校验通过。"""
    ev1 = {
        "evidence_id": "ev_cninfo_rep_1",
        "canonical_event_id": "cninfo:1200000001",
    }
    ev2 = {
        "evidence_id": "ev_cninfo_rep_2",
        "canonical_event_id": "cninfo:1200000001",
    }
    rel = build_canonical_source_repetition(ev1, ev2)
    assert rel is not None
    assert rel.source_id == "ev_cninfo_rep_1"
    assert rel.target_id == "ev_cninfo_rep_2"
    assert rel.relation_type == RelationType.SOURCE_REPETITION
    assert rel.metadata["canonical_event_id"] == "cninfo:1200000001"

    known_nodes = {"ev_cninfo_rep_1", "ev_cninfo_rep_2"}
    val = validate_relation(rel, known_nodes)
    assert val.valid is True
    assert val.reason is None


def test_valid_derived_observation_explicit_caller_provided():
    """2. 正例：调用方显式提供的 DERIVED_OBSERVATION 边经合法性校验通过。"""
    rel = EvidenceRelation(
        source_id="ev_disclosure_001",
        relation_type=RelationType.DERIVED_OBSERVATION,
        target_id="ev_collateral_forecast_001",
        metadata={"rationale": "officially disclosed forecast verified with actuals"},
    )
    known_nodes = {"ev_disclosure_001", "ev_collateral_forecast_001"}
    val = validate_relation(rel, known_nodes)
    assert val.valid is True
    assert val.reason is None


def test_valid_supports_relation_explicit_caller_provided():
    """3. 正例：辩论开篇协议显式绑定的 SUPPORTS 边经合法性校验通过。"""
    rel = EvidenceRelation(
        source_id="ev_revenue_surge",
        relation_type=RelationType.SUPPORTS,
        target_id="bull-1",
        metadata={"citation_role": "premise"},
    )
    known_nodes = {"ev_revenue_surge", "bull-1"}
    val = validate_relation(rel, known_nodes)
    assert val.valid is True
    assert val.reason is None


def test_valid_refutes_relation_explicit_caller_provided():
    """4. 正例：辩论质询协议显式声明的 REFUTES 边经合法性校验通过。"""
    rel = EvidenceRelation(
        source_id="ev_accounting_restatement",
        relation_type=RelationType.REFUTES,
        target_id="bull-1",
        metadata={"challenge_type": "material_distortion"},
    )
    known_nodes = {"ev_accounting_restatement", "bull-1"}
    val = validate_relation(rel, known_nodes)
    assert val.valid is True
    assert val.reason is None


def test_idempotent_duplicate_edge_insertion():
    """5. 正例：相同 dedupe_key 重复插入图容器，自动幂等去重为 1 条边。"""
    rel1 = EvidenceRelation(
        source_id="ev_source_a",
        relation_type=RelationType.SUPPORTS,
        target_id="claim_target_b",
        metadata={"source": "first"},
    )
    rel2 = EvidenceRelation(
        source_id="ev_source_a",
        relation_type=RelationType.SUPPORTS,
        target_id="claim_target_b",
        metadata={"source": "second_duplicate"},
    )
    graph = EvidenceRelationGraph.from_relations([rel1, rel2])
    assert len(graph.relations) == 1
    assert graph.relations[0].source_id == "ev_source_a"
    assert graph.relations[0].target_id == "claim_target_b"
    assert graph.relations[0].relation_type == RelationType.SUPPORTS


def test_serialization_and_deserialization_roundtrip():
    """6. 正例：to_dict -> JSON -> from_dict 双向无损，from_dict(None)/from_dict({}) 安全回退空图。"""
    rel1 = EvidenceRelation(
        source_id="ev_1",
        relation_type=RelationType.SOURCE_REPETITION,
        target_id="ev_2",
        metadata={"canonical_event_id": "cninfo:999", "tags": ["verified", "official"]},
    )
    rel2 = EvidenceRelation(
        source_id="ev_1",
        relation_type=RelationType.SUPPORTS,
        target_id="claim_1",
        metadata={"weight": 1.0},
    )
    graph = EvidenceRelationGraph.from_relations([rel1, rel2])

    serialized = graph.to_dict()
    assert isinstance(serialized, dict)
    assert serialized["version"] == "v1"
    assert len(serialized["relations"]) == 2

    # Verify JSON standard roundtrip
    json_str = json.dumps(serialized)
    parsed = json.loads(json_str)

    restored_graph = EvidenceRelationGraph.from_dict(parsed)
    assert len(restored_graph.relations) == 2
    assert restored_graph.relations[0].source_id == "ev_1"
    assert restored_graph.relations[0].target_id == "ev_2"
    assert restored_graph.relations[0].relation_type == RelationType.SOURCE_REPETITION
    assert restored_graph.relations[0].metadata["canonical_event_id"] == "cninfo:999"
    assert restored_graph.relations[0].metadata["tags"] == ("verified", "official")

    # Safe fallbacks for empty / None inputs
    assert EvidenceRelationGraph.from_dict(None).relations == ()
    assert EvidenceRelationGraph.from_dict({}).relations == ()
    assert EvidenceRelationGraph.from_dict({"relations": None}).relations == ()


def test_reject_self_loop():
    """7. 反例：source_id == target_id 被拒绝，返回 SELF_LOOP。"""
    rel = EvidenceRelation(
        source_id="ev_same_node",
        relation_type=RelationType.SUPPORTS,
        target_id="ev_same_node",
    )
    known_nodes = {"ev_same_node"}
    val = validate_relation(rel, known_nodes)
    assert val.valid is False
    assert val.reason == FailClosedReason.SELF_LOOP
    assert "Self-loop forbidden" in val.message


def test_reject_dangling_reference_missing_target():
    """8. 反例：节点不在 known_node_ids 被拒绝，返回 DANGLING_REFERENCE。"""
    rel = EvidenceRelation(
        source_id="ev_known",
        relation_type=RelationType.SUPPORTS,
        target_id="ev_unknown_ghost",
    )
    known_nodes = {"ev_known"}
    val = validate_relation(rel, known_nodes)
    assert val.valid is False
    assert val.reason == FailClosedReason.DANGLING_REFERENCE
    assert "ev_unknown_ghost" in val.message


def test_reject_contradictory_relations_supports_and_refutes():
    """9. 反例：同一有向边并存支持与反驳被拒绝，返回 CONTRADICTORY_RELATION。"""
    rel_sup = EvidenceRelation(
        source_id="ev_premise",
        relation_type=RelationType.SUPPORTS,
        target_id="claim_alpha",
    )
    rel_ref = EvidenceRelation(
        source_id="ev_premise",
        relation_type=RelationType.REFUTES,
        target_id="claim_alpha",
    )
    # Both in the graph
    graph = EvidenceRelationGraph(relations=(rel_sup, rel_ref))
    known_nodes = {"ev_premise", "claim_alpha"}
    valid, results = validate_relation_graph(graph, known_nodes)
    assert valid is False
    assert any(r.reason == FailClosedReason.CONTRADICTORY_RELATION for r in results)


def test_reject_cyclic_derived_observation():
    """10. 反例：A -> B -> C -> A 衍生时序环路被染色法检出，返回 CYCLE_DETECTED。"""
    r1 = EvidenceRelation("node_a", RelationType.DERIVED_OBSERVATION, "node_b")
    r2 = EvidenceRelation("node_b", RelationType.DERIVED_OBSERVATION, "node_c")
    r3 = EvidenceRelation("node_c", RelationType.DERIVED_OBSERVATION, "node_a")

    graph = EvidenceRelationGraph(relations=(r1, r2, r3))
    known_nodes = {"node_a", "node_b", "node_c"}

    # Direct cycle detector test
    cycles = detect_relation_cycles(graph.relations)
    assert len(cycles) > 0
    assert cycles[0] == ["node_a", "node_b", "node_c", "node_a"]

    # Graph validation test
    valid, results = validate_relation_graph(graph, known_nodes)
    assert valid is False
    assert any(r.reason == FailClosedReason.CYCLE_DETECTED for r in results)


def test_reject_lookahead_violation_future_evidence():
    """11. 反例：证据发布时间晚于基准日被拒绝，返回 LOOKAHEAD_VIOLATION。"""
    rel = EvidenceRelation(
        source_id="ev_future",
        relation_type=RelationType.SUPPORTS,
        target_id="claim_1",
    )
    known_nodes = {"ev_future", "claim_1"}
    source_ctx = {"published_at": "2026-06-15"}
    baseline = date(2026, 1, 1)

    val = validate_relation(
        rel,
        known_nodes,
        source_ctx=source_ctx,
        baseline_date=baseline,
    )
    assert val.valid is False
    assert val.reason == FailClosedReason.LOOKAHEAD_VIOLATION
    assert "after baseline date" in val.message


def test_reject_unhashed_content_for_canonical_derivation():
    """12. 反例：规范源公告缺失 content_hash 被拒绝，返回 CONTENT_UNHASHED。"""
    rel = EvidenceRelation(
        source_id="ev_canonical_without_hash",
        relation_type=RelationType.DERIVED_OBSERVATION,
        target_id="ev_derived_1",
    )
    known_nodes = {"ev_canonical_without_hash", "ev_derived_1"}
    source_ctx = {
        "is_canonical": True,
        "content_hash": "",  # Empty hash
    }
    val = validate_relation(rel, known_nodes, source_ctx=source_ctx)
    assert val.valid is False
    assert val.reason == FailClosedReason.CONTENT_UNHASHED


def test_strictly_forbid_collateral_canonical_event_id():
    """13. 反例：CollateralRecord 被赋予非空 canonical_event_id 时立即抛出 ValueError。"""
    with pytest.raises(ValueError, match="CollateralRecord strictly forbids canonical_event_id; MUST be None"):
        CollateralRecord(
            symbol="000001.SZ",
            ann_date="2026-01-01",
            source_type="forecast",
            collateral_id="tushare:forecast:000001.SZ:2026-01-01",
            canonical_event_id="cninfo:123456",
        )


def test_reject_auto_inference_from_similarity_or_verifier_status_or_soft_collateral():
    """14. 反例：(a)相似度阈值拒建边；(b)VERIFIED 状态拒建支持边；(c)旁证软对齐拒自动建边，均返回 UNSUPPORTED_INFERENCE。"""
    known_nodes = {"node_1", "node_2"}

    # (a) Attempt auto-inference from similarity score
    rel_sim = EvidenceRelation(
        source_id="node_1",
        relation_type=RelationType.SUPPORTS,
        target_id="node_2",
        metadata={"inference_source": "similarity", "similarity_score": 0.88},
    )
    val_sim = validate_relation(rel_sim, known_nodes)
    assert val_sim.valid is False
    assert val_sim.reason == FailClosedReason.UNSUPPORTED_INFERENCE

    # (b) Attempt auto-inference from verifier status
    rel_ver = EvidenceRelation(
        source_id="node_1",
        relation_type=RelationType.SUPPORTS,
        target_id="node_2",
        metadata={"verifier_status": "STATUS_VERIFIED"},
    )
    val_ver = validate_relation(rel_ver, known_nodes)
    assert val_ver.valid is False
    assert val_ver.reason == FailClosedReason.UNSUPPORTED_INFERENCE

    # (c) Attempt auto-inference from soft collateral alignment
    rel_soft = EvidenceRelation(
        source_id="node_1",
        relation_type=RelationType.DERIVED_OBSERVATION,
        target_id="node_2",
        metadata={"inference_source": "soft_alignment", "auto_inferred": True},
    )
    val_soft = validate_relation(rel_soft, known_nodes)
    assert val_soft.valid is False
    assert val_soft.reason == FailClosedReason.UNSUPPORTED_INFERENCE


# ============================================================================
# Additional Required Edge Cases
# ============================================================================

def test_reject_empty_or_non_string_endpoint_ids():
    """15. 额外覆盖：空 ID、空白字符 ID、非字符串 ID 严格拒绝。"""
    # Empty string
    with pytest.raises(ValueError, match="cannot be empty"):
        EvidenceRelation("", RelationType.SUPPORTS, "target_1")

    with pytest.raises(ValueError, match="cannot be empty"):
        EvidenceRelation("source_1", RelationType.SUPPORTS, "   ")

    # Non-string endpoint
    with pytest.raises(TypeError, match="must be strings"):
        EvidenceRelation(12345, RelationType.SUPPORTS, "target_1")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be strings"):
        EvidenceRelation("source_1", RelationType.SUPPORTS, None)  # type: ignore[arg-type]


def test_reject_unknown_relation_type_and_no_silent_upgrade():
    """16. 额外覆盖：非 RelationType 实例报错，未知 relation_type 字符串反序列化时禁止静默升级。"""
    # Non-RelationType constructor argument
    with pytest.raises(TypeError, match="relation_type must be an instance of RelationType"):
        EvidenceRelation("src", "SUPPORTS", "tgt")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="relation_type must be an instance of RelationType"):
        EvidenceRelation("src", 999, "tgt")  # type: ignore[arg-type]

    # Deserialization of unknown relation type must raise ValueError and NOT fall back
    with pytest.raises(ValueError, match="Unknown relation_type 'UNKNOWN_CUSTOM_TYPE' cannot be deserialized"):
        EvidenceRelation.from_dict({
            "source_id": "src",
            "relation_type": "UNKNOWN_CUSTOM_TYPE",
            "target_id": "tgt",
        })


def test_metadata_deep_freeze_and_immutability():
    """17. 额外覆盖：metadata 嵌套可变对象深冻结，运行时修改被底层拦截，原始对象解耦。"""
    inner_dict = {"param": "value", "count": 10}
    inner_list = ["item1", {"nested_key": 42}]
    input_meta = {
        "config": inner_dict,
        "items": inner_list,
    }

    rel = EvidenceRelation(
        source_id="src_1",
        relation_type=RelationType.DERIVED_OBSERVATION,
        target_id="tgt_1",
        metadata=input_meta,
    )

    # 1. Top-level metadata is MappingProxyType
    assert isinstance(rel.metadata, MappingProxyType)

    # 2. Nested dict is MappingProxyType
    assert isinstance(rel.metadata["config"], MappingProxyType)
    with pytest.raises(TypeError, match="'mappingproxy' object does not support item assignment"):
        rel.metadata["config"]["param"] = "mutated"  # type: ignore[index]

    # 3. Nested list is tuple
    assert isinstance(rel.metadata["items"], tuple)
    with pytest.raises(TypeError, match="'tuple' object does not support item assignment"):
        rel.metadata["items"][0] = "mutated"  # type: ignore[index]

    # 4. Nested dict inside list is MappingProxyType
    assert isinstance(rel.metadata["items"][1], MappingProxyType)
    with pytest.raises(TypeError, match="'mappingproxy' object does not support item assignment"):
        rel.metadata["items"][1]["nested_key"] = 999  # type: ignore[index]

    # 5. Modifying original input objects does not affect the frozen relation
    inner_dict["param"] = "tampered_original"
    inner_list.append("tampered_element")
    assert rel.metadata["config"]["param"] == "value"
    assert len(rel.metadata["items"]) == 2

    # 6. Non-string keys rejected
    with pytest.raises(TypeError, match="metadata keys must be strings"):
        EvidenceRelation("src_1", RelationType.SUPPORTS, "tgt_1", metadata={100: "numeric_key"})  # type: ignore[dict-item]


def test_reject_metadata_nan_and_inf():
    """18. 额外覆盖：metadata 中包含 NaN、+Infinity、-Infinity 均抛出 ValueError。"""
    with pytest.raises(ValueError, match="cannot contain NaN/Infinity"):
        EvidenceRelation("src_1", RelationType.SUPPORTS, "tgt_1", metadata={"score": float("nan")})

    with pytest.raises(ValueError, match="cannot contain NaN/Infinity"):
        EvidenceRelation("src_1", RelationType.SUPPORTS, "tgt_1", metadata={"score": float("inf")})

    with pytest.raises(ValueError, match="cannot contain NaN/Infinity"):
        EvidenceRelation("src_1", RelationType.SUPPORTS, "tgt_1", metadata={"nested": {"score": float("-inf")}})


def test_canonical_source_repetition_builder_edge_cases():
    """19. 额外覆盖：同 canonical ID 但相同/缺失 evidence ID，空白字符 trim 等边界测试。"""
    # Case 1: Identical evidence_id -> must return None (distinct non-empty required)
    ev_same_eid_1 = {"canonical_event_id": "cninfo:123", "evidence_id": "ev_shared"}
    ev_same_eid_2 = {"canonical_event_id": "cninfo:123", "evidence_id": "ev_shared"}
    assert build_canonical_source_repetition(ev_same_eid_1, ev_same_eid_2) is None

    # Case 2: Missing evidence_id on one or both -> must return None
    ev_missing_eid_1 = {"canonical_event_id": "cninfo:123", "evidence_id": ""}
    ev_missing_eid_2 = {"canonical_event_id": "cninfo:123", "evidence_id": "ev_valid"}
    assert build_canonical_source_repetition(ev_missing_eid_1, ev_missing_eid_2) is None
    assert build_canonical_source_repetition(ev_missing_eid_2, ev_missing_eid_1) is None

    ev_no_eid_1 = {"canonical_event_id": "cninfo:123"}
    assert build_canonical_source_repetition(ev_no_eid_1, ev_missing_eid_2) is None

    # Case 3: Missing / empty canonical_event_id -> must return None
    ev_no_cid = {"canonical_event_id": "", "evidence_id": "ev_1"}
    ev_cid = {"canonical_event_id": "cninfo:123", "evidence_id": "ev_2"}
    assert build_canonical_source_repetition(ev_no_cid, ev_cid) is None

    # Case 4: Mismatched canonical_event_id -> must return None
    ev_mismatch = {"canonical_event_id": "cninfo:456", "evidence_id": "ev_2"}
    assert build_canonical_source_repetition(ev_cid, ev_mismatch) is None

    # Case 5: None input objects -> must return None
    assert build_canonical_source_repetition(None, ev_cid) is None
    assert build_canonical_source_repetition(ev_cid, None) is None

    # Case 6: Whitespace-padded matching canonical_event_id -> should match after trim
    ev_padded_1 = {"canonical_event_id": "  cninfo:123  ", "evidence_id": "ev_1"}
    ev_padded_2 = {"canonical_event_id": "cninfo:123", "evidence_id": "ev_2"}
    rel = build_canonical_source_repetition(ev_padded_1, ev_padded_2)
    assert rel is not None
    assert rel.metadata["canonical_event_id"] == "cninfo:123"


def test_malformed_and_missing_and_future_dates_fail_closed():
    """20. 额外覆盖：畸形/缺失/未来日期全链路 fail-closed 拦截。"""
    rel = EvidenceRelation("src_ev", RelationType.SUPPORTS, "tgt_ev")
    known_nodes = {"src_ev", "tgt_ev"}

    # 1. Malformed baseline_date
    val_bad_base = validate_relation(rel, known_nodes, baseline_date="invalid-date")
    assert val_bad_base.valid is False
    assert val_bad_base.reason == FailClosedReason.MALFORMED_TIMESTAMP

    val_bad_base_type = validate_relation(rel, known_nodes, baseline_date=12345)  # type: ignore[arg-type]
    assert val_bad_base_type.valid is False
    assert val_bad_base_type.reason == FailClosedReason.MALFORMED_TIMESTAMP

    # 2. Missing timestamp when checking against baseline_date
    ctx_missing_ts = {"some_other_field": "foo"}
    val_missing_ts = validate_relation(
        rel,
        known_nodes,
        source_ctx=ctx_missing_ts,
        baseline_date="2026-01-01",
    )
    assert val_missing_ts.valid is False
    assert val_missing_ts.reason == FailClosedReason.MISSING_TIMESTAMP

    # 3. Malformed timestamp in context
    ctx_malformed_ts = {"published_at": "2026-99-99"}
    val_malformed_ts = validate_relation(
        rel,
        known_nodes,
        source_ctx=ctx_malformed_ts,
        baseline_date="2026-01-01",
    )
    assert val_malformed_ts.valid is False
    assert val_malformed_ts.reason == FailClosedReason.MALFORMED_TIMESTAMP

    # 4. Valid past / exact date succeeds
    ctx_valid_past = {"published_at": "2025-12-31"}
    val_valid_past = validate_relation(
        rel,
        known_nodes,
        source_ctx=ctx_valid_past,
        baseline_date="2026-01-01",
    )
    assert val_valid_past.valid is True


def test_duplicate_edge_deterministic_first_seen_metadata():
    """21. 额外覆盖：重复边 metadata 冲突的确定性首见语义，第一条被保留。"""
    edge_1 = EvidenceRelation(
        source_id="src_node",
        relation_type=RelationType.REVISES,
        target_id="tgt_node",
        metadata={"version": 1, "note": "first_seen_content"},
    )
    edge_2 = EvidenceRelation(
        source_id="src_node",
        relation_type=RelationType.REVISES,
        target_id="tgt_node",
        metadata={"version": 2, "note": "conflicting_second_content"},
    )

    graph = EvidenceRelationGraph.from_relations([edge_1, edge_2])
    assert len(graph.relations) == 1
    # First-seen semantics: edge_1 is preserved intact
    assert graph.relations[0].metadata["version"] == 1
    assert graph.relations[0].metadata["note"] == "first_seen_content"


def test_relation_is_unhashable_and_cannot_be_in_set():
    """22. 额外覆盖：EvidenceRelation 显式禁止哈希，无法存入 set，保障不可变集合安全。"""
    rel = EvidenceRelation("src", RelationType.SUPPORTS, "tgt", metadata={"a": 1})
    with pytest.raises(TypeError, match="unhashable type: 'EvidenceRelation'"):
        hash(rel)

    s = set()
    with pytest.raises(TypeError, match="unhashable type: 'EvidenceRelation'"):
        s.add(rel)


def test_frozen_metadata_can_be_reused_to_construct_new_relation():
    """已冻结的 metadata 必须能再次用于构造 relation。

    __post_init__ 深冻结后 metadata 是 MappingProxyType/tuple，本身不是 json
    可序列化类型。若 JSON 安全校验直接作用于入参，则任何以既有 relation 的
    metadata 组合新 relation 的路径都会被误判为「非 JSON 安全」而失败。
    """
    src = EvidenceRelation("a", RelationType.SUPPORTS, "b", {"k": {"n": 1}, "lst": [1, 2]})

    reused = EvidenceRelation("c", RelationType.SUPPORTS, "d", src.metadata)
    assert reused.metadata["k"]["n"] == 1
    assert reused.metadata["lst"] == (1, 2)
    assert isinstance(reused.metadata, MappingProxyType)

    # from_dict 收到冻结 metadata 同样不得报错
    via_dict = EvidenceRelation.from_dict(
        {"source_id": "e", "relation_type": "SUPPORTS", "target_id": "f", "metadata": src.metadata}
    )
    assert via_dict.to_dict()["metadata"] == {"k": {"n": 1}, "lst": [1, 2]}


def test_json_safety_rejections_survive_thaw_normalization():
    """解冻归一不得放宽 JSON 安全校验：NaN/Inf 与不可序列化对象仍须拒绝。"""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            EvidenceRelation("a", RelationType.SUPPORTS, "b", {"x": bad})

    with pytest.raises(TypeError):
        EvidenceRelation("a", RelationType.SUPPORTS, "b", {"x": object()})

    # 嵌套层同样拒绝
    with pytest.raises(ValueError):
        EvidenceRelation("a", RelationType.SUPPORTS, "b", {"outer": {"inner": float("nan")}})
