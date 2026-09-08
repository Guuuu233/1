"""Evidence relation types, graph container, validation rules, and cycle detection.

Implements E-01 minimal infrastructure contract:
- Immutable relation structures and JSON-safe deeply immutable metadata.
- Closed-universe node validation, self-loop prevention, contradiction checks.
- DFS-based cycle detection for temporal/derivation DAG edges.
- Explicit caller-provided relations for supports, refutes, and derived observations.
- Sole automatic builder strictly limited to shared non-empty canonical_event_id.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
import json
from types import MappingProxyType
from typing import Any, Container, Iterable, Mapping, Sequence


class RelationType(str, Enum):
    """Supported edge relation types in the evidence relation graph."""
    SOURCE_REPETITION = "SOURCE_REPETITION"      # 同源复述（仅官方公告相同规范ID）
    DERIVED_OBSERVATION = "DERIVED_OBSERVATION"  # 衍生观察（旁证/派生事实，显式提供）
    SUPPORTS = "SUPPORTS"                        # 逻辑支持（论点与论据，显式提供）
    REFUTES = "REFUTES"                          # 逻辑反驳（质询对抗，显式提供）
    REVISES = "REVISES"                          # 修正替代（更正公告，显式提供）


class FailClosedReason(str, Enum):
    """Fail-closed rejection reasons for relation and graph validation."""
    SELF_LOOP = "SELF_LOOP"
    DANGLING_REFERENCE = "DANGLING_REFERENCE"
    CONTRADICTORY_RELATION = "CONTRADICTORY_RELATION"
    CYCLE_DETECTED = "CYCLE_DETECTED"
    LOOKAHEAD_VIOLATION = "LOOKAHEAD_VIOLATION"
    MALFORMED_TIMESTAMP = "MALFORMED_TIMESTAMP"
    MISSING_TIMESTAMP = "MISSING_TIMESTAMP"
    CONTENT_UNHASHED = "CONTENT_UNHASHED"
    UNSUPPORTED_INFERENCE = "UNSUPPORTED_INFERENCE"


def _check_string_keys(val: Any) -> None:
    """Recursively ensure all mapping keys are strings."""
    if isinstance(val, Mapping):
        for k, v in val.items():
            if not isinstance(k, str):
                raise TypeError(f"metadata keys must be strings, got key {k!r} of type {type(k).__name__}")
            _check_string_keys(v)
    elif isinstance(val, (list, tuple)):
        for item in val:
            _check_string_keys(item)


def _deep_freeze_mapping(val: Any) -> Any:
    """Recursively freeze dicts to MappingProxyType and lists/tuples to immutable tuples."""
    if isinstance(val, Mapping):
        return MappingProxyType({k: _deep_freeze_mapping(v) for k, v in val.items()})
    elif isinstance(val, (list, tuple)):
        return tuple(_deep_freeze_mapping(item) for item in val)
    return val


def _deep_thaw_mapping(val: Any) -> Any:
    """Recursively thaw MappingProxyType to dict and tuple to list for JSON export."""
    if isinstance(val, (Mapping, MappingProxyType)):
        return {k: _deep_thaw_mapping(v) for k, v in val.items()}
    elif isinstance(val, tuple):
        return [_deep_thaw_mapping(item) for item in val]
    return val


@dataclass(frozen=True)
class EvidenceRelation:
    """Immutable directed relation edge between evidence/claim nodes."""
    source_id: str
    relation_type: RelationType
    target_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    __hash__ = None  # 显式禁止哈希，杜绝 unhashable dict 碰撞与危险的 set 混用

    def __post_init__(self):
        # 1. 校验 source_id 和 target_id 为非空字符串
        if not isinstance(self.source_id, str) or not isinstance(self.target_id, str):
            raise TypeError("source_id and target_id must be strings")
        if not self.source_id.strip() or not self.target_id.strip():
            raise ValueError("source_id and target_id cannot be empty")

        # 2. 校验 relation_type 必须是 RelationType 实例
        if not isinstance(self.relation_type, RelationType):
            raise TypeError(f"relation_type must be an instance of RelationType, got {type(self.relation_type).__name__}")

        # 3. 校验 metadata 必须是 Mapping 且键全为 string
        if not isinstance(self.metadata, Mapping):
            raise TypeError(f"metadata must be a Mapping, got {type(self.metadata).__name__}")
        _check_string_keys(self.metadata)

        # 4. 先解冻成纯 dict/list 再校验 JSON 安全性（allow_nan=False 拒绝 NaN/Infinity）。
        #    冻结产物 MappingProxyType/tuple 不是 json 可序列化类型，若直接对入参 dumps，
        #    会把「已冻结的合法 metadata」误判为「非 JSON 安全」，使 relation 无法由既有
        #    relation 的 metadata 组合，from_dict 收到冻结映射时同样报错。
        thawed = _deep_thaw_mapping(self.metadata)
        try:
            json.dumps(thawed, allow_nan=False)
        except (TypeError, ValueError, OverflowError) as e:
            if isinstance(e, TypeError):
                raise TypeError(f"EvidenceRelation.metadata must be JSON-safe: {e}") from e
            raise ValueError(f"EvidenceRelation.metadata must be JSON-safe and cannot contain NaN/Infinity: {e}") from e

        # 5. 递归深度冻结包装
        object.__setattr__(self, "metadata", _deep_freeze_mapping(thawed))

    @property
    def dedupe_key(self) -> tuple[str, str, str]:
        """Canonical triple key for idempotent deduplication."""
        rel_val = self.relation_type.value if isinstance(self.relation_type, RelationType) else str(self.relation_type)
        return (self.source_id, rel_val, self.target_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to lossless JSON-serializable dictionary."""
        rel_val = self.relation_type.value if isinstance(self.relation_type, RelationType) else str(self.relation_type)
        return {
            "source_id": self.source_id,
            "relation_type": rel_val,
            "target_id": self.target_id,
            "metadata": _deep_thaw_mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceRelation:
        """Create instance from JSON dictionary. Never silently upgrade unknown relation types."""
        if not isinstance(data, Mapping):
            raise TypeError(f"EvidenceRelation.from_dict requires a mapping, got {type(data).__name__}")
        for field_name in ("source_id", "relation_type", "target_id"):
            if field_name not in data:
                raise ValueError(f"Missing required field '{field_name}' in relation dict")
        raw_rel = data["relation_type"]
        if isinstance(raw_rel, RelationType):
            rel_type = raw_rel
        elif isinstance(raw_rel, str):
            try:
                rel_type = RelationType(raw_rel)
            except ValueError as e:
                raise ValueError(f"Unknown relation_type '{raw_rel}' cannot be deserialized: {e}") from e
        else:
            raise TypeError(f"relation_type must be str or RelationType, got {type(raw_rel).__name__}")

        raw_meta = data.get("metadata")
        if raw_meta is None:
            metadata: dict[str, Any] = {}
        elif isinstance(raw_meta, Mapping):
            metadata = dict(raw_meta)
        else:
            raise TypeError(f"metadata must be a mapping, got {type(raw_meta).__name__}")

        return cls(
            source_id=data["source_id"],
            relation_type=rel_type,
            target_id=data["target_id"],
            metadata=metadata,
        )


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of relation or graph validation rule checks."""
    valid: bool
    reason: FailClosedReason | None = None
    message: str = ""


@dataclass(frozen=True)
class EvidenceRelationGraph:
    """Immutable container of validated evidence relations with ordered tuple storage."""
    relations: tuple[EvidenceRelation, ...] = ()
    version: str = "v1"

    def __post_init__(self):
        if not isinstance(self.relations, (tuple, list)):
            raise TypeError(f"relations must be a sequence of EvidenceRelation, got {type(self.relations).__name__}")
        deduped: dict[tuple[str, str, str], EvidenceRelation] = {}
        for r in self.relations:
            if not isinstance(r, EvidenceRelation):
                raise TypeError(f"Expected EvidenceRelation, got {type(r).__name__}")
            if r.dedupe_key not in deduped:
                deduped[r.dedupe_key] = r
        object.__setattr__(self, "relations", tuple(deduped.values()))
        if not isinstance(self.version, str):
            object.__setattr__(self, "version", str(self.version))

    @classmethod
    def from_relations(cls, relations: Iterable[EvidenceRelation]) -> EvidenceRelationGraph:
        """Construct graph with idempotent deduplication based on dedupe_key (first-seen semantics)."""
        return cls(relations=tuple(relations), version="v1")

    def to_dict(self) -> dict[str, Any]:
        """Lossless JSON serialization of the relation graph."""
        return {
            "version": self.version,
            "relations": [r.to_dict() for r in self.relations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EvidenceRelationGraph:
        """Robust deserialization with safe fallback for None/empty inputs."""
        if not data or not isinstance(data, Mapping):
            return cls()
        version = str(data.get("version") or "v1")
        raw_relations = data.get("relations")
        if raw_relations is None:
            return cls(relations=(), version=version)
        if not isinstance(raw_relations, (list, tuple)):
            raise TypeError(f"relations in dict must be a list or tuple, got {type(raw_relations).__name__}")
        relations = [EvidenceRelation.from_dict(r) for r in raw_relations]
        return cls(
            relations=tuple(relations),
            version=version,
        )


def validate_relation(
    relation: EvidenceRelation,
    known_node_ids: Container[str],
    source_ctx: Mapping[str, Any] | None = None,
    target_ctx: Mapping[str, Any] | None = None,
    baseline_date: date | str | None = None,
) -> ValidationResult:
    """Validate a single relation edge against domain invariants."""
    # 1. 检查自环
    if relation.source_id == relation.target_id:
        return ValidationResult(
            valid=False,
            reason=FailClosedReason.SELF_LOOP,
            message=f"Self-loop forbidden: source and target are identical ({relation.source_id})",
        )

    # 2. 检查闭包宇宙（防断链）
    missing = []
    if relation.source_id not in known_node_ids:
        missing.append(relation.source_id)
    if relation.target_id not in known_node_ids:
        missing.append(relation.target_id)
    if missing:
        return ValidationResult(
            valid=False,
            reason=FailClosedReason.DANGLING_REFERENCE,
            message=f"Dangling reference to unknown node(s): {missing}",
        )

    # 3. 检查时间前视穿透与畸形日期 fail-closed
    if baseline_date is not None:
        b_dt: date
        if isinstance(baseline_date, datetime):
            # datetime 是 date 的子类。若不先归一到 date，b_dt 会保持 datetime，
            # 而 ctx 侧时间戳已归一为 date，末尾 `dt > b_dt` 就变成 date 与 datetime
            # 相比并抛 TypeError——护栏由 fail-closed 退化成异常穿透调用方。
            # 判型顺序须与下方 ctx 时间戳分支保持一致。
            b_dt = baseline_date.date()
        elif isinstance(baseline_date, date):
            b_dt = baseline_date
        elif isinstance(baseline_date, str):
            try:
                b_dt = datetime.strptime(baseline_date.strip()[:10], "%Y-%m-%d").date()
            except (ValueError, TypeError):
                return ValidationResult(
                    valid=False,
                    reason=FailClosedReason.MALFORMED_TIMESTAMP,
                    message=f"Malformed baseline_date: {baseline_date!r}",
                )
        else:
            return ValidationResult(
                valid=False,
                reason=FailClosedReason.MALFORMED_TIMESTAMP,
                message=f"baseline_date must be date or str, got {type(baseline_date).__name__}",
            )

        for ctx, name in [(source_ctx, "source"), (target_ctx, "target")]:
            if ctx is not None:
                ts = ctx.get("published_at") or ctx.get("ann_date") or ctx.get("timestamp") or ctx.get("trade_date")
                if not ts:
                    return ValidationResult(
                        valid=False,
                        reason=FailClosedReason.MISSING_TIMESTAMP,
                        message=f"{name} context is missing timestamp",
                    )
                dt: date
                if isinstance(ts, datetime):
                    dt = ts.date()
                elif isinstance(ts, date):
                    dt = ts
                elif isinstance(ts, str):
                    try:
                        dt = datetime.strptime(ts.strip()[:10], "%Y-%m-%d").date()
                    except (ValueError, TypeError):
                        return ValidationResult(
                            valid=False,
                            reason=FailClosedReason.MALFORMED_TIMESTAMP,
                            message=f"{name} context contains malformed timestamp: {ts!r}",
                        )
                else:
                    return ValidationResult(
                        valid=False,
                        reason=FailClosedReason.MALFORMED_TIMESTAMP,
                        message=f"{name} context timestamp has invalid type: {type(ts).__name__}",
                    )

                if dt > b_dt:
                    return ValidationResult(
                        valid=False,
                        reason=FailClosedReason.LOOKAHEAD_VIOLATION,
                        message=f"{name} date {dt} is after baseline date {b_dt}",
                    )

    # 4. 检查规范源内容哈希有效性
    if source_ctx and source_ctx.get("is_canonical"):
        content_hash = source_ctx.get("content_hash")
        if not content_hash or not str(content_hash).strip():
            return ValidationResult(
                valid=False,
                reason=FailClosedReason.CONTENT_UNHASHED,
                message="Canonical source must have non-empty content_hash",
            )

    # 5. 检查禁止的自动启发式推断（相似度、verifier 状态或软对齐伪因果）
    if relation.metadata.get("auto_inferred") or relation.metadata.get("inference_source") in {"similarity", "verifier_status", "soft_alignment"}:
        return ValidationResult(
            valid=False,
            reason=FailClosedReason.UNSUPPORTED_INFERENCE,
            message="Automatic inference from similarity, verifier status, or soft collateral alignment is strictly forbidden",
        )
    if "similarity_score" in relation.metadata or "verifier_status" in relation.metadata:
        return ValidationResult(
            valid=False,
            reason=FailClosedReason.UNSUPPORTED_INFERENCE,
            message="Automatic inference from similarity score or verifier status is forbidden",
        )

    return ValidationResult(valid=True)


def detect_relation_cycles(
    relations: Sequence[EvidenceRelation],
    target_types: Container[RelationType] | None = None,
) -> list[list[str]]:
    """Detect cycles within hierarchical/derivation relation subgraphs using DFS 3-color."""
    types = target_types or {RelationType.DERIVED_OBSERVATION, RelationType.REVISES}
    adj: dict[str, list[str]] = {}
    for r in relations:
        if r.relation_type in types:
            adj.setdefault(r.source_id, []).append(r.target_id)

    visited: dict[str, int] = {}  # 0: white (unvisited), 1: gray (visiting), 2: black (visited)
    cycles: list[list[str]] = []

    def dfs(node: str, path: list[str]):
        visited[node] = 1
        path.append(node)
        for neighbor in adj.get(node, []):
            state = visited.get(neighbor, 0)
            if state == 1:
                idx = path.index(neighbor)
                cycles.append(path[idx:] + [neighbor])
            elif state == 0:
                dfs(neighbor, path)
        path.pop()
        visited[node] = 2

    for n in list(adj.keys()):
        if visited.get(n, 0) == 0:
            dfs(n, [])
    return cycles


def validate_relation_graph(
    graph: EvidenceRelationGraph,
    known_node_ids: Container[str],
) -> tuple[bool, list[ValidationResult]]:
    """Validate entire graph: single-edge checks, contradiction checks, and cycle detection."""
    results: list[ValidationResult] = []
    directed_relations: dict[tuple[str, str], set[RelationType]] = {}

    for r in graph.relations:
        res = validate_relation(r, known_node_ids)
        if not res.valid:
            results.append(res)
        edge = (r.source_id, r.target_id)
        directed_relations.setdefault(edge, set()).add(r.relation_type)

    # 互斥关系检查（同一有向边禁止并存 SUPPORTS 与 REFUTES）
    for (src, tgt), rel_types in directed_relations.items():
        if RelationType.SUPPORTS in rel_types and RelationType.REFUTES in rel_types:
            results.append(ValidationResult(
                valid=False,
                reason=FailClosedReason.CONTRADICTORY_RELATION,
                message=f"Contradictory SUPPORTS and REFUTES detected on directed edge ({src} -> {tgt})",
            ))

    # DAG 环路检测
    cycles = detect_relation_cycles(graph.relations)
    if cycles:
        results.append(ValidationResult(
            valid=False,
            reason=FailClosedReason.CYCLE_DETECTED,
            message=f"Cycle(s) detected in derivation relations: {cycles}",
        ))

    return len(results) == 0, results


def build_canonical_source_repetition(ev1: Any, ev2: Any) -> EvidenceRelation | None:
    """Sole automatic builder: builds SOURCE_REPETITION edge if and only if both evidences share identical non-empty canonical_event_id and distinct non-empty evidence_ids."""
    if ev1 is None or ev2 is None:
        return None

    id1 = getattr(ev1, "canonical_event_id", None)
    if id1 is None and isinstance(ev1, Mapping):
        id1 = ev1.get("canonical_event_id")

    id2 = getattr(ev2, "canonical_event_id", None)
    if id2 is None and isinstance(ev2, Mapping):
        id2 = ev2.get("canonical_event_id")

    if id1 is None or id2 is None:
        return None

    c_id1 = str(id1).strip()
    c_id2 = str(id2).strip()
    if not c_id1 or not c_id2 or c_id1 != c_id2:
        return None

    e1 = getattr(ev1, "evidence_id", None)
    if e1 is None and isinstance(ev1, Mapping):
        e1 = ev1.get("evidence_id")

    e2 = getattr(ev2, "evidence_id", None)
    if e2 is None and isinstance(ev2, Mapping):
        e2 = ev2.get("evidence_id")

    if e1 is None or e2 is None:
        return None

    s_e1 = str(e1).strip()
    s_e2 = str(e2).strip()
    if not s_e1 or not s_e2 or s_e1 == s_e2:
        return None

    return EvidenceRelation(
        source_id=s_e1,
        relation_type=RelationType.SOURCE_REPETITION,
        target_id=s_e2,
        metadata={"canonical_event_id": c_id1},
    )
