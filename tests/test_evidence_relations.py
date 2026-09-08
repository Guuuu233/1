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

from datetime import date, datetime
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


def test_datetime_baseline_fails_closed_not_raises():
    """baseline_date 传 datetime 必须走 fail-closed，不得抛异常穿透调用方。

    datetime 是 date 的子类；若判型顺序把 datetime 归入 date 分支而不归一，
    末尾 `dt > b_dt` 会拿 date 与 datetime 相比并抛 TypeError，使护栏在
    调用方看来是崩溃而非 ValidationResult。
    """
    rel = EvidenceRelation("a", RelationType.SUPPORTS, "b")
    ids = {"a", "b"}

    # 同日：不算前视，必须返回 valid 的 ValidationResult
    res = validate_relation(
        rel, ids, {"published_at": "2026-09-08"}, {"published_at": "2026-09-08"},
        baseline_date=datetime(2026, 9, 8, 23, 59, 59),
    )
    assert isinstance(res, ValidationResult)
    assert res.valid is True

    # 证据日期晚于 baseline：必须 fail-closed 为 LOOKAHEAD_VIOLATION，而不是抛错
    res2 = validate_relation(
        rel, ids, {"published_at": "2026-09-09"}, {"published_at": "2026-09-08"},
        baseline_date=datetime(2026, 9, 8, 10, 0, 0),
    )
    assert res2.valid is False
    assert res2.reason == FailClosedReason.LOOKAHEAD_VIOLATION

    # datetime 与等价 date 的判定必须一致
    res3 = validate_relation(
        rel, ids, {"published_at": "2026-09-09"}, {"published_at": "2026-09-08"},
        baseline_date=date(2026, 9, 8),
    )
    assert res3.reason == res2.reason


def test_present_but_empty_timestamp_fails_closed_not_substituted():
    """时间字段「存在但为空」是畸形数据，不得静默换用下一个字段。

    原实现用 `or` 链取时间戳，published_at="" 会静默回落到 trade_date，
    让交易日冒充发布时间并通过校验。这属 D-008 禁止的时间语义互换，
    与 DAV-719 判过的 or 吃掉合法假值是同一模式。
    """
    rel = EvidenceRelation("a", RelationType.SUPPORTS, "b")
    ids = {"a", "b"}

    for empty in ("", "   ", None):
        res = validate_relation(
            rel, ids,
            {"published_at": empty, "trade_date": "2026-09-01"},
            {"published_at": "2026-09-01"},
            baseline_date="2026-09-08",
        )
        assert res.valid is False, f"published_at={empty!r} 不得被 trade_date 顶替"
        assert res.reason == FailClosedReason.MALFORMED_TIMESTAMP
        assert "published_at" in res.message


def test_timestamp_field_priority_and_semantics_are_explicit():
    """字段优先级按声明顺序，且实际采用的时间语义须出现在前视消息中。"""
    rel = EvidenceRelation("a", RelationType.SUPPORTS, "b")
    ids = {"a", "b"}

    # 只声明 trade_date 时才用 trade_date，且前视消息标明来源字段
    res = validate_relation(
        rel, ids, {"trade_date": "2026-09-09"}, {"trade_date": "2026-09-01"},
        baseline_date="2026-09-08",
    )
    assert res.reason == FailClosedReason.LOOKAHEAD_VIOLATION
    assert "trade_date" in res.message

    # published_at 存在且合法时优先于 trade_date
    res2 = validate_relation(
        rel, ids,
        {"published_at": "2026-09-09", "trade_date": "2026-09-01"},
        {"published_at": "2026-09-01"},
        baseline_date="2026-09-08",
    )
    assert res2.reason == FailClosedReason.LOOKAHEAD_VIOLATION
    assert "published_at" in res2.message

    # 一个时间字段都没有仍是 MISSING_TIMESTAMP
    res3 = validate_relation(
        rel, ids, {"other": "x"}, {"published_at": "2026-09-01"},
        baseline_date="2026-09-08",
    )
    assert res3.reason == FailClosedReason.MISSING_TIMESTAMP


# ============================================================================
# Knife 1: EvidenceRelationGraph.from_dict 顶层类型严格校验
# ============================================================================

@pytest.mark.parametrize("bad_input", [
    [],
    "bad",
    123,
    True,
    False,
    [{"source_id": "a", "relation_type": "SUPPORTS", "target_id": "b"}],
    ("a", "b"),
])
def test_from_dict_non_mapping_top_level_raises_type_error(bad_input):
    """任何非 Mapping 顶层输入抛 TypeError，不得静默变空图。"""
    with pytest.raises(TypeError, match="EvidenceRelationGraph.from_dict requires a mapping or None"):
        EvidenceRelationGraph.from_dict(bad_input)  # type: ignore[arg-type]


@pytest.mark.parametrize("empty_mapping", [
    None,
    {},
    {"relations": None},
    MappingProxyType({}),
    MappingProxyType({"relations": None}),
])
def test_from_dict_empty_or_none_mapping_preserves_empty_graph(empty_mapping):
    """保留 None / 空 Mapping / relations=None 的既有空图语义。"""
    graph = EvidenceRelationGraph.from_dict(empty_mapping)
    assert graph.relations == ()


# ============================================================================
# Knife 2: 完整时间字符串验证与对称 fail-closed
# ============================================================================

@pytest.mark.parametrize("malformed_ts", [
    "2026-09-08junk",
    "2026-09-08T99:99:99",
    "2026-99-99",
    "2026-09-08 99:99:99",
    "2026-09-08T",
    "2026-09-08-extra",
    "invalid",
    "",
    "   ",
])
def test_malformed_timestamp_strings_fail_closed_symmetrically(malformed_ts):
    """非法时间、任意 junk 尾缀、空白均返回 MALFORMED_TIMESTAMP，baseline 与 ctx 行为对称。"""
    rel = EvidenceRelation("a", RelationType.SUPPORTS, "b")
    ids = {"a", "b"}

    # 1. baseline_date 侧拦截
    res_base = validate_relation(
        rel, ids,
        source_ctx={"published_at": "2026-09-01"},
        target_ctx={"published_at": "2026-09-01"},
        baseline_date=malformed_ts,
    )
    assert res_base.valid is False
    assert res_base.reason == FailClosedReason.MALFORMED_TIMESTAMP

    # 2. source_ctx 侧拦截
    res_src = validate_relation(
        rel, ids,
        source_ctx={"published_at": malformed_ts},
        target_ctx={"published_at": "2026-09-01"},
        baseline_date="2026-09-08",
    )
    assert res_src.valid is False
    assert res_src.reason == FailClosedReason.MALFORMED_TIMESTAMP

    # 3. target_ctx 侧拦截
    res_tgt = validate_relation(
        rel, ids,
        source_ctx={"published_at": "2026-09-01"},
        target_ctx={"published_at": malformed_ts},
        baseline_date="2026-09-08",
    )
    assert res_tgt.valid is False
    assert res_tgt.reason == FailClosedReason.MALFORMED_TIMESTAMP


@pytest.mark.parametrize("valid_iso_str,expected_date", [
    ("2026-09-08", date(2026, 9, 8)),
    ("2026-09-08T12:34:56", date(2026, 9, 8)),
    ("2026-09-08T12:34:56Z", date(2026, 9, 8)),
    ("2026-09-08T12:34:56+00:00", date(2026, 9, 8)),
    ("2026-09-08T12:34:56+08:00", date(2026, 9, 8)),
    ("2026-09-08 12:34:56", date(2026, 9, 8)),
    ("2026-09-08 12:34:56.123456", date(2026, 9, 8)),
    ("2026-09-08T12:34:56.123456Z", date(2026, 9, 8)),
])
def test_valid_iso_date_and_datetime_strings_accepted_symmetrically(valid_iso_str, expected_date):
    """明确接受完整 ISO date 和合法 ISO datetime，对称支持 baseline 与 ctx。"""
    rel = EvidenceRelation("a", RelationType.SUPPORTS, "b")
    ids = {"a", "b"}

    # 当证据与基准日同一天（均为 valid_iso_str），应当验证通过
    res = validate_relation(
        rel, ids,
        source_ctx={"published_at": valid_iso_str},
        target_ctx={"trade_date": valid_iso_str},
        baseline_date=valid_iso_str,
    )
    assert res.valid is True
    assert res.reason is None

    # baseline 设为前一天，证据设为 valid_iso_str，必须判定为前视穿透 LOOKAHEAD_VIOLATION
    res_lookahead = validate_relation(
        rel, ids,
        source_ctx={"published_at": valid_iso_str},
        target_ctx={"trade_date": "2026-09-07"},
        baseline_date="2026-09-07",
    )
    assert res_lookahead.valid is False
    assert res_lookahead.reason == FailClosedReason.LOOKAHEAD_VIOLATION


# ============================================================================
# Knife 3: detect_relation_cycles 显式空容器检测零种关系
# ============================================================================

@pytest.mark.parametrize("empty_target_types", [
    set(),
    [],
    (),
    frozenset(),
])
def test_detect_relation_cycles_empty_target_types_detects_zero_cycles(empty_target_types):
    """只有 target_types is None 使用默认；显式空容器必须检测零种关系并返回空列表。"""
    r1 = EvidenceRelation("node_x", RelationType.DERIVED_OBSERVATION, "node_y")
    r2 = EvidenceRelation("node_y", RelationType.DERIVED_OBSERVATION, "node_x")
    relations = [r1, r2]

    # 1. 默认 target_types=None 检测出衍生时序环路
    assert len(detect_relation_cycles(relations, target_types=None)) > 0

    # 2. 显式空容器必须检测 0 种关系，返回空列表 []
    assert detect_relation_cycles(relations, target_types=empty_target_types) == []


# ============================================================================
# Knife 4: _deep_thaw_mapping 递归 thaw list 与 tuple 嵌套 metadata
# ============================================================================

@pytest.mark.parametrize("container_builder", [
    lambda proxy: [proxy],
    lambda proxy: (proxy,),
    lambda proxy: [{"nested_proxy": proxy}],
    lambda proxy: ({"nested_proxy": proxy},),
    lambda proxy: [[proxy]],
    lambda proxy: [(proxy,)],
])
def test_deep_thaw_mapping_supports_list_and_tuple_nested_mappingproxy(container_builder):
    """对 list 与 tuple 均递归 thaw，确保任意 JSON-safe list/tuple 中嵌套既有 frozen metadata 可再次构造并无损 roundtrip。"""
    src = EvidenceRelation("src", RelationType.SUPPORTS, "tgt", metadata={"k": "v", "num": 42})
    assert isinstance(src.metadata, MappingProxyType)

    nested_meta = {"items": container_builder(src.metadata)}
    rel = EvidenceRelation("a", RelationType.SUPPORTS, "b", metadata=nested_meta)

    # 验证能无损 to_dict 并进行标准 JSON 序列化和反序列化往返
    d = rel.to_dict()
    json_bytes = json.dumps(d, allow_nan=False)
    parsed = json.loads(json_bytes)
    restored = EvidenceRelation.from_dict(parsed)
    assert restored.to_dict() == d


@pytest.mark.parametrize("bad_nested_meta,expected_exc", [
    ({"items": [float("nan")]}, ValueError),
    ({"items": [float("inf")]}, ValueError),
    ({"items": [float("-inf")]}, ValueError),
    ({"items": (float("nan"),)}, ValueError),
    ({"items": [{123: "non_str_key"}]}, TypeError),
    ({"items": [object()]}, TypeError),
    ({"items": ([object()],)}, TypeError),
])
def test_deep_thaw_mapping_still_rejects_nan_inf_and_unserializable(bad_nested_meta, expected_exc):
    """NaN/Inf、非字符串 key、真正不可序列化对象在 list/tuple 嵌套中仍被严格拒绝。"""
    with pytest.raises(expected_exc):
        EvidenceRelation("a", RelationType.SUPPORTS, "b", metadata=bad_nested_meta)
