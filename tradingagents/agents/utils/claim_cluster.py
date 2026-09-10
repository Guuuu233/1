from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import logging
import math
import re
from typing import Any, Mapping, Sequence, Tuple, Union

from tradingagents.agents.utils.debate_utils import normalize_text
from tradingagents.agents.utils.evidence_relations import (
    EvidenceRelation,
    EvidenceRelationGraph,
    FailClosedReason,
    RelationType,
    ValidationResult,
    validate_relation_graph,
)

logger = logging.getLogger(__name__)

CLUSTER_TYPE_PRICE_SHOCK = "price_shock"
CLUSTER_TYPE_FUNDAMENTALS = "fundamentals"
CLUSTER_TYPE_MACRO_POLICY = "macro_policy"
CLUSTER_TYPE_SENTIMENT_NEWS = "sentiment_news"
CLUSTER_TYPE_UNSUPPORTED = "unsupported"

_PRICE_SHOCK_KEYWORDS = frozenset({
    "收盘", "收盘价", "开盘", "开盘价", "最高价", "最低价", "现价", "股价", "价格",
    "突破", "跌破", "阻力位", "支撑位", "均线", "ma5", "ma10", "ma20", "ma60",
    "ma", "ema", "vwma", "vwap", "k线", "实体", "阳线", "阴线", "跳空", "缺口",
    "涨幅", "跌幅", "涨跌幅", "涨停", "跌停", "当日涨幅", "当日跌幅", "振幅", "涨跌", "冲高", "回落", "连板",
    "成交量", "成交额", "换手率", "换手", "量比", "放量", "缩量", "地量", "天量", "量价", "量能",
    "macd", "kdj", "rsi", "boll", "布林", "atr", "dif", "dea", "金叉", "死叉", "顶背离", "底背离", "超买", "超卖", "多头排列", "空头排列",
    "主力资金", "主力净流入", "主力净额", "资金净流入", "资金流出", "主力流出", "超大单", "大单", "中单", "小单", "全单",
    "主力净买入", "主力建仓", "主力吸筹", "主力出货", "北向资金", "两融", "融资买入", "资金面", "长阳",
})

_FUNDAMENTALS_KEYWORDS = frozenset({
    "营收", "营业收入", "收入", "净利润", "扣非净利润", "扣非", "归母净利润", "利润", "毛利", "毛利率", "净利率",
    "roe", "roa", "eps", "每股收益", "资产负债率", "负债率", "现金流", "自由现金流", "fcf", "经营性现金流",
    "资本开支", "capex", "研发费用", "研发投入", "研发", "在手订单", "订单", "存货", "应收账款", "周转率", "周转天数",
    "产能", "产能利用率", "减值", "商誉", "分红", "股息率", "股息", "pe", "pb", "ps", "估值", "市盈率", "市净率",
    "净资产", "总资产", "中报", "年报", "季报", "一季报", "三季报", "业绩预告", "业绩", "财报",
})

_MACRO_POLICY_KEYWORDS = frozenset({
    "降息", "降准", "加息", "央行", "lpr", "m2", "cpi", "ppi", "gdp", "财政赤字", "赤字", "汇率",
    "美联储", "关税", "补贴", "以旧换新", "政策红利", "货币政策", "产业政策", "外需", "内需", "美债",
    "美债收益率", "美元", "原油", "油价", "黄金", "大宗商品", "铜价", "宏观", "货币",
})

_SENTIMENT_NEWS_KEYWORDS = frozenset({
    "舆情", "情绪", "散户情绪", "看多占比", "看空占比", "机构调研", "行业新政", "诉讼", "重组", "获批",
    "临床", "专利", "合作", "license-out", "突发事件", "传闻", "舆论", "热搜", "新闻",
})

_DATE_PATTERN = re.compile(r"\b(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?|\d{4}Q[1-4]|\d{4}H[12]|\d{4}年报|\d{4}中报)\b")
_SYMBOL_PATTERN = re.compile(r"\b(\d{6})(?:\.(?:SH|SZ|BJ))?\b", re.IGNORECASE)
_NUMERIC_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:%|亿元|万元|万手|手|元|bp|倍)?")


def normalize_symbol(symbol: str | None) -> str:
    """Extract and normalize 6-digit stock symbol or entity string."""
    if not symbol:
        return "default_symbol"
    text = str(symbol).strip()
    match = _SYMBOL_PATTERN.search(text)
    if match:
        return match.group(1)
    cleaned = re.sub(r"[^\w一-鿿]+", "", text)
    return cleaned or "default_symbol"


def normalize_date_period(date_str: str | None) -> str:
    """Normalize date or period string."""
    if not date_str:
        return "default_date"
    text = str(date_str).strip()
    match = _DATE_PATTERN.search(text)
    if match:
        raw_match = match.group(1)
        return raw_match.replace("/", "-").replace("年", "-").replace("月", "-").replace("日", "")
    return text.replace("/", "-")


def compute_cluster_id(
    cluster_type: str,
    symbol: str | None = None,
    date: str | None = None,
) -> str:
    """Compute deterministic, stable hash-based cluster identifier."""
    norm_type = (cluster_type or CLUSTER_TYPE_UNSUPPORTED).strip().lower()
    if norm_type == CLUSTER_TYPE_UNSUPPORTED:
        return ""
    norm_sym = normalize_symbol(symbol)
    norm_date = normalize_date_period(date)
    raw_key = f"{norm_type}:{norm_sym}:{norm_date}"
    hash_str = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:12]
    return f"cluster_{norm_type}_{hash_str}"


def classify_evidence_text(
    evidence_text: str,
    fallback_symbol: str | None = None,
    fallback_date: str | None = None,
) -> tuple[str, str, str]:
    """Classify evidence string into domain cluster type and extract symbol/date context."""
    text = str(evidence_text or "").strip()
    if not text:
        return CLUSTER_TYPE_UNSUPPORTED, "", ""

    lower_text = text.lower()
    sym_match = _SYMBOL_PATTERN.search(text)
    detected_symbol = sym_match.group(1) if sym_match else normalize_symbol(fallback_symbol)

    date_match = _DATE_PATTERN.search(text)
    detected_date = date_match.group(1) if date_match else normalize_date_period(fallback_date)

    # Check keyword memberships
    has_numbers = bool(_NUMERIC_PATTERN.search(text))
    has_price = any(kw in lower_text for kw in _PRICE_SHOCK_KEYWORDS)
    has_fund = any(kw in lower_text for kw in _FUNDAMENTALS_KEYWORDS)
    has_macro = any(kw in lower_text for kw in _MACRO_POLICY_KEYWORDS)
    has_sentiment = any(kw in lower_text for kw in _SENTIMENT_NEWS_KEYWORDS)

    if not (has_price or has_fund or has_macro or has_sentiment or has_numbers):
        return CLUSTER_TYPE_UNSUPPORTED, detected_symbol, detected_date

    if has_price:
        return CLUSTER_TYPE_PRICE_SHOCK, detected_symbol, detected_date
    if has_fund:
        return CLUSTER_TYPE_FUNDAMENTALS, detected_symbol, detected_date
    if has_macro:
        return CLUSTER_TYPE_MACRO_POLICY, detected_symbol, detected_date
    if has_sentiment:
        return CLUSTER_TYPE_SENTIMENT_NEWS, detected_symbol, detected_date

    # If only numbers present without specific keywords
    return CLUSTER_TYPE_PRICE_SHOCK, detected_symbol, detected_date


def compute_evidence_id(
    cluster_id: str,
    raw_evidence: str,
    index: int = 1,
) -> str:
    """Generate deterministic evidence ID bound to cluster and normalized text."""
    norm_ev = normalize_text(raw_evidence)
    ev_hash = hashlib.sha256(f"{cluster_id}:{norm_ev}".encode("utf-8")).hexdigest()[:10]
    return f"ev_{ev_hash}_{index}"


def extract_evidence_ids(
    evidence_items: Sequence[Any] | None,
    cluster_id: str,
) -> list[str]:
    """Extract deterministic evidence IDs for a list of evidence strings."""
    if not evidence_items or not cluster_id:
        return []
    ids: list[str] = []
    for idx, item in enumerate(evidence_items, start=1):
        text = str(item or "").strip()
        if text:
            ids.append(compute_evidence_id(cluster_id, text, idx))
    return ids


def _build_verification_index(
    claims_verification: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[str], set[str], bool]:
    if claims_verification is None:
        return set(), set(), set(), set(), False

    verified_cid_raw: set[tuple[str, str]] = set()
    verified_cid_norm: set[tuple[str, str]] = set()
    verified_raw: set[str] = set()
    verified_norm: set[str] = set()

    if isinstance(claims_verification, Sequence):
        for item in claims_verification:
            if not isinstance(item, Mapping):
                continue
            st = str(item.get("status") or "").strip().lower()
            is_fatal = bool(item.get("is_fatal", False))
            if st in {"verified", "pass", "ok"} and not is_fatal and st not in {
                "contradicted",
                "unsupported",
                "source_unavailable",
                "failed",
                "unavailable",
                "error",
                "missing",
            }:
                cid = str(item.get("claim_id") or "").strip()
                raw = str(item.get("raw") or item.get("evidence") or "").strip()
                if raw:
                    norm = normalize_text(raw)
                    if cid:
                        verified_cid_raw.add((cid, raw))
                        verified_cid_norm.add((cid, norm))
                    verified_raw.add(raw)
                    verified_norm.add(norm)
    elif isinstance(claims_verification, Mapping):
        for cid, info in claims_verification.items():
            if isinstance(info, Mapping):
                for raw in (info.get("verified_evidence") or []):
                    r_str = str(raw).strip()
                    if r_str:
                        norm = normalize_text(r_str)
                        verified_cid_raw.add((str(cid).strip(), r_str))
                        verified_cid_norm.add((str(cid).strip(), norm))
                        verified_raw.add(r_str)
                        verified_norm.add(norm)

    return verified_cid_raw, verified_cid_norm, verified_raw, verified_norm, True


def _is_evidence_verified(
    claim_id: str | None,
    raw_evidence: str,
    ver_index: tuple[set[tuple[str, str]], set[tuple[str, str]], set[str], set[str], bool],
) -> bool:
    v_cid_raw, v_cid_norm, v_raw, v_norm, is_active = ver_index
    if not is_active:
        return True
    raw_str = str(raw_evidence).strip()
    if not raw_str:
        return False
    norm_str = normalize_text(raw_str)
    cid_str = str(claim_id or "").strip()
    if cid_str:
        if (cid_str, raw_str) in v_cid_raw or (cid_str, norm_str) in v_cid_norm:
            return True
    return raw_str in v_raw or norm_str in v_norm


def assign_claim_cluster(
    claim: Mapping[str, Any],
    symbol: str | None = None,
    date: str | None = None,
    claims_verification: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assign deterministic cluster_id and evidence_ids to a claim object."""
    claim_dict = dict(claim)
    claim_id = str(claim_dict.get("claim_id") or "").strip()
    evidence_list = claim_dict.get("evidence") or []
    if isinstance(evidence_list, str):
        evidence_list = [evidence_list]
    valid_evidence = [str(e).strip() for e in evidence_list if str(e).strip()]

    ver_index = _build_verification_index(claims_verification)
    verified_evidence = [e for e in valid_evidence if _is_evidence_verified(claim_id, e, ver_index)]

    if not valid_evidence:
        claim_dict["cluster_id"] = None
        claim_dict["evidence_ids"] = []
        claim_dict["verified_evidence"] = []
        claim_dict["verified_evidence_ids"] = []
        claim_dict["cluster_status"] = "unsupported"
        return claim_dict

    cluster_votes: dict[str, int] = {}
    cluster_info: dict[str, tuple[str, str]] = {}
    for ev in valid_evidence:
        ctype, sym, dt = classify_evidence_text(ev, fallback_symbol=symbol, fallback_date=date)
        if ctype != CLUSTER_TYPE_UNSUPPORTED:
            cluster_votes[ctype] = cluster_votes.get(ctype, 0) + 1
            cluster_info[ctype] = (sym, dt)

    if not cluster_votes:
        claim_dict["cluster_id"] = None
        claim_dict["evidence_ids"] = []
        claim_dict["verified_evidence"] = []
        claim_dict["verified_evidence_ids"] = []
        claim_dict["cluster_status"] = "unsupported"
        return claim_dict

    primary_type = max(cluster_votes.items(), key=lambda x: x[1])[0]
    sym, dt = cluster_info[primary_type]
    cid = compute_cluster_id(primary_type, symbol=sym, date=dt)
    ev_ids = extract_evidence_ids(valid_evidence, cid)
    ver_ev_ids = extract_evidence_ids(verified_evidence, cid)

    claim_dict["cluster_id"] = cid
    claim_dict["evidence_ids"] = ev_ids
    claim_dict["verified_evidence"] = verified_evidence
    claim_dict["verified_evidence_ids"] = ver_ev_ids
    claim_dict["cluster_type"] = primary_type
    claim_dict["cluster_status"] = "clustered"
    return claim_dict


def cluster_claims(
    claims: Sequence[Mapping[str, Any]],
    symbol: str | None = None,
    date: str | None = None,
    claims_verification: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Enrich all claims with deterministic cluster_id and evidence_ids."""
    return [
        assign_claim_cluster(c, symbol=symbol, date=date, claims_verification=claims_verification)
        for c in claims
    ]


def _normalize_stance(raw_stance: str | None, speaker: str | None = None) -> str:
    s = (raw_stance or "").strip().lower()
    if s in {"bull", "bullish", "long", "buy", "多", "多头", "看多", "偏多"}:
        return "bull"
    if s in {"bear", "bearish", "short", "sell", "空", "空头", "看空", "偏空"}:
        return "bear"
    sp = (speaker or "").strip().lower()
    if "bull" in sp:
        return "bull"
    if "bear" in sp:
        return "bear"
    return "neutral"


def _extract_analyst_count(
    claims: Sequence[Mapping[str, Any]],
    reports: Mapping[str, str] | None = None,
) -> int:
    speakers = set()
    for c in claims:
        sp = str(c.get("speaker") or c.get("speaker_key") or "").strip()
        if sp:
            speakers.add(sp)
    if reports:
        for rk, rbody in reports.items():
            if str(rbody or "").strip():
                speakers.add(rk)
    return len(speakers) if speakers else len(claims)


def _build_clusters_map(
    enriched_claims: Sequence[Mapping[str, Any]],
    claims_verification_active: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    clusters_map: dict[str, dict[str, Any]] = {}
    unsupported_ids: list[str] = []

    for c in enriched_claims:
        cid = c.get("cluster_id")
        claim_id = str(c.get("claim_id") or "").strip()
        if not cid or c.get("cluster_status") == "unsupported":
            if claim_id:
                unsupported_ids.append(claim_id)
            continue

        stance = _normalize_stance(c.get("stance"), c.get("speaker") or c.get("speaker_key"))
        if cid not in clusters_map:
            clusters_map[cid] = {
                "cluster_id": cid,
                "cluster_type": c.get("cluster_type", "unknown"),
                "claims": [],
                "speakers": set(),
                "stances": set(),
                "evidence_ids": set(),
                "verified_evidence_ids": set(),
                "direction_votes": {"bull": 0, "bear": 0, "neutral": 0},
            }
        cluster_entry = clusters_map[cid]
        if claim_id:
            cluster_entry["claims"].append(claim_id)
        sp = str(c.get("speaker") or c.get("speaker_key") or "").strip()
        if sp:
            cluster_entry["speakers"].add(sp)
        cluster_entry["stances"].add(stance)
        for evid in (c.get("evidence_ids") or []):
            cluster_entry["evidence_ids"].add(evid)
        for evid in (c.get("verified_evidence_ids") or []):
            cluster_entry["verified_evidence_ids"].add(evid)

        # Every cluster casts at most ONE vote per direction
        cluster_entry["direction_votes"][stance] = 1

    return clusters_map, unsupported_ids


RELATION_GRAPH_STATUS_AVAILABLE = "available"
RELATION_GRAPH_STATUS_PENDING = "pending"
RELATION_GRAPH_STATUS_INVALID = "invalid"
_RELATION_FOLDING_TYPES = frozenset({
    RelationType.SOURCE_REPETITION,
    RelationType.DERIVED_OBSERVATION,
})
_RAW_PAYLOAD_MAX_DEPTH = 20
_RELATION_GRAPH_MAPPING_KEYS = frozenset({"version", "relations"})


@dataclass(frozen=True)
class _RelationGraphRejection:
    """Why a supplied relation graph was not folded; serialized into relation_audit.error."""
    stage: str                                          # status / deserialize / validate / reduce
    error: BaseException
    validation_results: Tuple[ValidationResult, ...] = ()
    raw_payload: Any = None                             # JSON-safe copy when deserialization failed


def _relation_sequence(relation_graph: Any) -> list[EvidenceRelation]:
    if isinstance(relation_graph, EvidenceRelationGraph):
        return list(relation_graph.relations)
    if isinstance(relation_graph, (list, tuple)):
        return [item for item in relation_graph if isinstance(item, EvidenceRelation)]
    return []


def _coerce_relation_graph(raw_graph: Any) -> EvidenceRelationGraph:
    """Deserialize only an explicitly supplied E-01 graph representation."""
    if isinstance(raw_graph, EvidenceRelationGraph):
        return raw_graph
    if isinstance(raw_graph, Mapping):
        # EvidenceRelationGraph.from_dict falls back to an empty graph for {} and for a
        # missing/None "relations" entry.  At the E-02 seam that fallback would report a
        # malformed payload as an available empty graph without an error audit, so the
        # serialized graph shape is required explicitly here.
        unexpected = sorted(str(key) for key in raw_graph if key not in _RELATION_GRAPH_MAPPING_KEYS)
        if unexpected:
            raise ValueError(
                f"evidence relation graph mapping has unexpected key(s) {unexpected}; "
                "expected 'relations' and optional 'version'"
            )
        if "relations" not in raw_graph:
            raise ValueError("evidence relation graph mapping is missing required key 'relations'")
        relations = raw_graph["relations"]
        if not isinstance(relations, (list, tuple)):
            raise TypeError(
                f"evidence relation graph 'relations' must be a list or tuple, got {type(relations).__name__}"
            )
        return EvidenceRelationGraph.from_dict(raw_graph)
    if isinstance(raw_graph, (list, tuple)):
        raw_relations = list(raw_graph)
        if all(isinstance(item, EvidenceRelation) for item in raw_relations):
            return EvidenceRelationGraph.from_relations(raw_relations)
        return EvidenceRelationGraph.from_dict({"relations": raw_relations})
    raise TypeError(
        "evidence relation graph must be EvidenceRelationGraph, a graph mapping, "
        f"or an explicit relation sequence, got {type(raw_graph).__name__}"
    )


def _json_safe_payload(value: Any, depth: int = 0) -> Any:
    """Copy a rejected relation payload into JSON-safe audit data without reinterpreting it."""
    if isinstance(value, (EvidenceRelation, EvidenceRelationGraph)):
        return value.to_dict()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if depth < _RAW_PAYLOAD_MAX_DEPTH:
        if isinstance(value, Mapping):
            return {str(key): _json_safe_payload(item, depth + 1) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe_payload(item, depth + 1) for item in value]
    return repr(value)


def _reduce_supplied_relation_graph(
    claim_ids: list[str],
    relation_graph: Any,
) -> tuple[EvidenceReductionResult | None, list[EvidenceRelation], _RelationGraphRejection | None]:
    """Deserialize, E-01-validate, then reduce a supplied graph; a rejected graph is never folded."""
    if relation_graph is None:
        return None, [], _RelationGraphRejection("deserialize", TypeError("available relation graph is missing"))
    try:
        graph = _coerce_relation_graph(relation_graph)
    except (TypeError, ValueError) as exc:
        return None, [], _RelationGraphRejection(
            "deserialize",
            exc,
            raw_payload=_json_safe_payload(relation_graph),
        )

    raw_relations = _relation_sequence(relation_graph) or list(graph.relations)
    try:
        graph_valid, validation_results = validate_relation_graph(graph, set(claim_ids))
    except (TypeError, ValueError) as exc:
        return None, raw_relations, _RelationGraphRejection("validate", exc)
    if not graph_valid:
        details = "; ".join(item.message or str(item.reason) for item in validation_results)
        error = EvidenceReductionError(
            error_reason=validation_results[0].reason or FailClosedReason.UNSUPPORTED_INFERENCE,
            message=f"E-01 relation graph validation failed: {details}",
        )
        return None, raw_relations, _RelationGraphRejection("validate", error, tuple(validation_results))

    try:
        return reduce_evidence_claims(claim_ids, graph), raw_relations, None
    except (EvidenceReductionError, TypeError, ValueError) as exc:
        return None, raw_relations, _RelationGraphRejection("reduce", exc)


def _serialize_relation_reduction(
    result: EvidenceReductionResult | None,
    raw_relations: Sequence[EvidenceRelation],
    status: str,
    reason: str,
    pending_claim_ids: Sequence[str],
    rejection: _RelationGraphRejection | None = None,
) -> dict[str, Any]:
    """Serialize reducer output without turning audit data into a voting signal."""
    folded_components: list[dict[str, Any]] = []
    audit_edges: list[dict[str, Any]] = []
    unconnected_claim_ids: list[str] = []
    independence_status = IndependenceStatus.UNKNOWN.value
    contribution_cap = 0

    if result is not None:
        unconnected_claim_ids = list(result.unconnected_claim_ids)
        independence_status = result.independence_status.value
        contribution_cap = result.global_contribution_cap
        audit_edges = [edge.to_dict() for edge in result.audit_edges]
        for component in result.folded_components:
            component_edges = [edge.to_dict() for edge in component.audit_edges]
            folded_components.append({
                "component_id": component.component_id,
                "member_claim_ids": list(component.member_claim_ids),
                "derived_terminal_ids": list(component.derived_terminal_ids),
                "audit_edges": component_edges,
                "reason": (
                    "explicit E-01 SOURCE_REPETITION/DERIVED_OBSERVATION relation(s) "
                    "connect these claims; this is not an independence proof"
                ),
            })

    raw_edge_dicts = [edge.to_dict() for edge in raw_relations]
    ignored_relations = [
        edge.to_dict()
        for edge in raw_relations
        if edge.relation_type not in _RELATION_FOLDING_TYPES
    ]
    pending_audit = [
        {
            "claim_id": claim_id,
            "status": "pending",
            "reason": (
                "no explicit E-01 SOURCE_REPETITION or DERIVED_OBSERVATION "
                "edge proves a contribution relationship; independence is UNKNOWN"
            ),
        }
        for claim_id in pending_claim_ids
    ]

    audit: dict[str, Any] = {
        "status": status,
        "reason": reason,
        "raw_relations": raw_edge_dicts,
        "raw_payload": rejection.raw_payload if rejection is not None else None,
        "audit_edges": audit_edges,
        "ignored_relations": ignored_relations,
        "folded_components": folded_components,
        "unconnected_claim_ids": unconnected_claim_ids,
        "pending_claims": pending_audit,
        "independence_status": independence_status,
        "global_contribution_cap": contribution_cap,
    }
    if rejection is not None:
        error_reason = getattr(rejection.error, "error_reason", None)
        audit["error"] = {
            "type": type(rejection.error).__name__,
            "stage": rejection.stage,
            "reason": getattr(error_reason, "value", None) or str(error_reason or "INVALID_RELATION_GRAPH"),
            "message": str(rejection.error),
            "validation_results": [
                {"reason": getattr(item.reason, "value", None), "message": item.message}
                for item in rejection.validation_results
            ],
        }
    return audit


def _apply_relation_reduction(
    metrics: dict[str, Any],
    claims: Sequence[Mapping[str, Any]],
    relation_graph: Any,
    relation_graph_status: str,
    relation_graph_reason: str,
) -> dict[str, Any]:
    """Replace keyword-derived contribution fields with the E-02 conservative seam."""
    status = str(relation_graph_status or RELATION_GRAPH_STATUS_PENDING).strip().lower()
    reason = str(relation_graph_reason or "").strip()
    claim_ids = sorted({
        str(claim.get("claim_id") or "").strip()
        for claim in claims
        if str(claim.get("claim_id") or "").strip()
    })
    raw_relations = _relation_sequence(relation_graph)
    result: EvidenceReductionResult | None = None
    rejection: _RelationGraphRejection | None = None

    if status == RELATION_GRAPH_STATUS_AVAILABLE:
        result, raw_relations, rejection = _reduce_supplied_relation_graph(claim_ids, relation_graph)
        if rejection is not None:
            reason = f"{reason}; rejected: {rejection.error}" if reason else str(rejection.error)
    elif status == RELATION_GRAPH_STATUS_PENDING:
        # An absent graph is represented explicitly by an empty relation input;
        # no edge is invented and every claim stays pending/unknown.
        result = reduce_evidence_claims(claim_ids, [])
        reason = reason or "E-01 relation graph is unavailable; contribution remains pending"
    elif status == RELATION_GRAPH_STATUS_INVALID:
        reason = reason or "E-01 relation graph is invalid; contribution remains pending"
        rejection = _RelationGraphRejection(
            "status",
            ValueError(reason),
            raw_payload=None if relation_graph is None else _json_safe_payload(relation_graph),
        )
    else:
        unsupported = ValueError(f"Unsupported relation_graph_status: {status!r}")
        reason = reason or str(unsupported)
        rejection = _RelationGraphRejection(
            "status",
            unsupported,
            raw_payload=None if relation_graph is None else _json_safe_payload(relation_graph),
        )

    if rejection is not None:
        status = RELATION_GRAPH_STATUS_INVALID
        logger.warning(
            "[claim_cluster] E-01 relation graph not folded (%s stage): %s",
            rejection.stage,
            rejection.error,
        )

    if status == RELATION_GRAPH_STATUS_AVAILABLE and result is not None:
        pending_claim_ids = list(result.unconnected_claim_ids)
    else:
        pending_claim_ids = claim_ids

    effective_contribution_count = int(
        status == RELATION_GRAPH_STATUS_AVAILABLE
        and result is not None
        and bool(result.folded_components)
        and result.global_contribution_cap > 0
    )
    legacy_metrics = dict(metrics)
    relation_audit = _serialize_relation_reduction(
        result=result,
        raw_relations=raw_relations,
        status=status,
        reason=reason,
        pending_claim_ids=pending_claim_ids,
        rejection=rejection,
    )

    metrics.update({
        # Keep the historical field names, but make them conservative and
        # explicitly unrelated to keyword-cluster voting.
        "independent_cluster_count": effective_contribution_count,
        "bull_cluster_count": 0,
        "bear_cluster_count": 0,
        "neutral_cluster_count": 0,
        "direction_cluster_counts": {"bull": 0, "bear": 0, "neutral": 0},
        "cluster_weights": {"bull": 0.0, "bear": 0.0, "neutral": 0.0},
        "relation_graph_status": status,
        "relation_graph_reason": reason,
        "independence_status": relation_audit["independence_status"],
        "global_contribution_cap": relation_audit["global_contribution_cap"],
        "folded_component_count": len(relation_audit["folded_components"]),
        "effective_contribution_count": effective_contribution_count,
        "pending_claim_ids": pending_claim_ids,
        "relation_audit": relation_audit,
        "legacy_keyword_metrics": legacy_metrics,
    })
    return metrics


def tally_cluster_votes(
    claims: Sequence[Mapping[str, Any]] | None = None,
    reports: Mapping[str, str] | None = None,
    claims_verification: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    symbol: str | None = None,
    trade_date: str | None = None,
    horizon: str | None = None,
    relation_graph: Any = None,
    relation_graph_status: str | None = None,
    relation_graph_reason: str | None = None,
) -> dict[str, Any]:
    """Tally legacy explanatory metrics and optionally apply the E-02 seam.

    The optional relation arguments are deliberately opt-in so existing callers
    retain their explanatory keyword metrics. The research manager always passes
    an explicit status, including ``pending`` when the graph is absent.
    ``relation_graph`` may be an ``EvidenceRelationGraph``, a serialized graph
    mapping, or an explicit relation sequence. A payload that cannot be
    deserialized or fails E-01 validation is never folded; ``relation_audit``
    keeps its JSON-safe ``raw_payload`` and a structured ``error``.
    """
    claims_list = list(claims or [])
    enriched_claims = cluster_claims(
        claims_list,
        symbol=symbol,
        date=trade_date,
        claims_verification=claims_verification,
    )

    analyst_count = _extract_analyst_count(claims_list, reports)
    has_ver = claims_verification is not None
    clusters_map, unsupported_ids = _build_clusters_map(
        enriched_claims,
        claims_verification_active=has_ver,
    )

    bull_cluster_count = sum(1 for cl in clusters_map.values() if cl["direction_votes"].get("bull"))
    bear_cluster_count = sum(1 for cl in clusters_map.values() if cl["direction_votes"].get("bear"))
    neutral_cluster_count = sum(1 for cl in clusters_map.values() if cl["direction_votes"].get("neutral"))
    independent_cluster_count = len(clusters_map)

    if has_ver:
        verified_evidence_count = sum(len(cl["verified_evidence_ids"]) for cl in clusters_map.values())
    else:
        verified_evidence_count = sum(len(cl["evidence_ids"]) for cl in clusters_map.values())

    active_dir_total = bull_cluster_count + bear_cluster_count
    if active_dir_total > 0:
        bull_weight = round(bull_cluster_count / active_dir_total, 4)
        bear_weight = round(bear_cluster_count / active_dir_total, 4)
    else:
        bull_weight = 0.0
        bear_weight = 0.0

    metrics = {
        "analyst_count": analyst_count,
        "independent_cluster_count": independent_cluster_count,
        "verified_evidence_count": verified_evidence_count,
        "bull_cluster_count": bull_cluster_count,
        "bear_cluster_count": bear_cluster_count,
        "neutral_cluster_count": neutral_cluster_count,
        "direction_cluster_counts": {
            "bull": bull_cluster_count,
            "bear": bear_cluster_count,
            "neutral": neutral_cluster_count,
        },
        "cluster_weights": {
            "bull": bull_weight,
            "bear": bear_weight,
            "neutral": 1.0 - (bull_weight + bear_weight) if active_dir_total == 0 else 0.0,
        },
        "unsupported_claim_ids": unsupported_ids,
        "clusters": [
            {
                "cluster_id": cl["cluster_id"],
                "cluster_type": cl["cluster_type"],
                "claims": cl["claims"],
                "speakers": sorted(cl["speakers"]),
                "stances": sorted(cl["stances"]),
                "evidence_count": len(cl["verified_evidence_ids"]) if has_ver else len(cl["evidence_ids"]),
                "direction_votes": cl["direction_votes"],
            }
            for cl in clusters_map.values()
        ],
    }
    if relation_graph_status is not None or relation_graph is not None:
        return _apply_relation_reduction(
            metrics,
            claims_list,
            relation_graph,
            relation_graph_status or RELATION_GRAPH_STATUS_AVAILABLE,
            relation_graph_reason or "",
        )
    return metrics


def _format_relation_summary_for_prompt(
    metrics: Mapping[str, Any],
    language: str,
) -> str:
    audit = metrics.get("relation_audit") or {}
    status = str(metrics.get("relation_graph_status") or audit.get("status") or "pending").upper()
    reason = str(metrics.get("relation_graph_reason") or audit.get("reason") or "").strip()
    independence = str(metrics.get("independence_status") or "UNKNOWN")
    cap = metrics.get("global_contribution_cap", 0)
    effective = metrics.get("effective_contribution_count", 0)
    analyst_count = metrics.get("analyst_count", 0)
    verified_evidence_count = metrics.get("verified_evidence_count", 0)
    pending = metrics.get("pending_claim_ids") or audit.get("unconnected_claim_ids") or []
    folded = audit.get("folded_components") or []
    raw_relations = audit.get("raw_relations") or []
    ignored = audit.get("ignored_relations") or []

    if language == "en":
        lines = [
            "### Evidence Relation Reduction (E-02; no independent-vote inference)",
            f"- E-01 relation graph status: {status} ({reason})",
            f"- analyst_count: {analyst_count} (explanatory context only; not an independence proof)",
            f"- verified_evidence_count: {verified_evidence_count} (factual verification only; not a voting weight)",
            f"- independence_status: {independence}; global_contribution_cap: {cap}; effective contribution slots (independent_cluster_count compatibility field): {effective}",
            f"- pending/unknown claim IDs: {', '.join(map(str, pending)) or 'none'}",
            f"- folded relation components: {len(folded)} (relation grouping only; not proof of mutual independence)",
        ]
        for component in folded:
            members = ", ".join(component.get("member_claim_ids") or []) or "none"
            terminals = ", ".join(component.get("derived_terminal_ids") or []) or "not applicable"
            lines.append(f"  * component={component.get('component_id')}, members=[{members}], derived_terminals=[{terminals}]")
            lines.append(f"    reason: {component.get('reason', '')}")
        if raw_relations:
            lines.append("- raw relation audit (original references retained):")
            lines.extend(f"  * {json.dumps(edge, ensure_ascii=False, sort_keys=True)}" for edge in raw_relations)
        if ignored:
            lines.append("- relations not used for folding (no same-fact inference):")
            lines.extend(f"  * {json.dumps(edge, ensure_ascii=False, sort_keys=True)}" for edge in ignored)
        return "\n".join(lines)

    lines = [
        "### 证据关系折叠与贡献约束 (E-02；禁止推断独立票)",
        f"- E-01关系图状态: {status}（{reason}）",
        f"- 参与分析师人数 (analyst_count): {analyst_count}（仅作解释性参考，不证明独立性）",
        f"- 真实核验有效证据数 (verified_evidence_count): {verified_evidence_count}（仅作事实核验，不是投票权重）",
        f"- 独立性证明 (independence_status): {independence}；全局贡献上限 (global_contribution_cap): {cap}；当前有效贡献槽（兼容字段 independent_cluster_count）: {effective}",
        f"- 待补关系/未知 claim: {', '.join(map(str, pending)) or '无'}",
        f"- 已证明折叠组数: {len(folded)}（仅表示关系分组，不证明组间相互独立）",
    ]
    for component in folded:
        members = ", ".join(component.get("member_claim_ids") or []) or "无"
        terminals = ", ".join(component.get("derived_terminal_ids") or []) or "不适用"
        lines.append(f"  * 组件={component.get('component_id')}，成员=[{members}]，派生终点=[{terminals}]")
        lines.append(f"    合并原因: {component.get('reason', '')}")
    if raw_relations:
        lines.append("- 原始关系审计（保留原始引用与 metadata）:")
        lines.extend(f"  * {json.dumps(edge, ensure_ascii=False, sort_keys=True)}" for edge in raw_relations)
    if ignored:
        lines.append("- 未用于折叠的关系（不得据此推断同事实/独立性）:")
        lines.extend(f"  * {json.dumps(edge, ensure_ascii=False, sort_keys=True)}" for edge in ignored)
    return "\n".join(lines)


def format_claim_cluster_summary_for_prompt(
    metrics: Mapping[str, Any] | None,
    language: str = "zh",
) -> str:
    """Format a concise, human/LLM-consumable summary of claim evidence cluster metrics."""
    if not metrics:
        return ""
    if metrics.get("relation_graph_status"):
        return _format_relation_summary_for_prompt(metrics, language)
    analyst_count = metrics.get("analyst_count", 0)
    independent_cluster_count = metrics.get("independent_cluster_count", 0)
    verified_evidence_count = metrics.get("verified_evidence_count", 0)
    bull_clusters = metrics.get("bull_cluster_count", 0)
    bear_clusters = metrics.get("bear_cluster_count", 0)
    neutral_clusters = metrics.get("neutral_cluster_count", 0)
    cluster_weights = metrics.get("cluster_weights", {}) or {}
    bull_weight = cluster_weights.get("bull", 0.0)
    bear_weight = cluster_weights.get("bear", 0.0)

    if language == "en":
        lines = [
            "### Claim Evidence Cluster Metrics (Deduplication Summary)",
            f"- analyst_count: {analyst_count} (explanatory context only; do NOT use directly as voting weight)",
            f"- independent_cluster_count: {independent_cluster_count} (bull clusters={bull_clusters}, bear clusters={bear_clusters}, neutral={neutral_clusters})",
            f"- verified_evidence_count: {verified_evidence_count} (only factual verified evidence counted)",
            f"- directional_cluster_weights: Bull={bull_weight:.1%}, Bear={bear_weight:.1%}",
        ]
        clusters = metrics.get("clusters") or []
        if clusters:
            lines.append("- evidence_clusters:")
            for cl in clusters:
                c_id = cl.get("cluster_id")
                c_type = cl.get("cluster_type")
                claims_str = ", ".join(cl.get("claims") or []) or "none"
                speakers_str = ", ".join(cl.get("speakers") or []) or "none"
                votes = cl.get("direction_votes") or {}
                lines.append(f"  * [{c_id}] type={c_type}, claims=[{claims_str}], speakers=[{speakers_str}], votes={votes}")
        return "\n".join(lines)

    lines = [
        "### Claim 证据簇与去重计票全景 (claim_cluster_metrics)",
        f"- 参与分析师人数 (analyst_count): {analyst_count}（仅作解释性参考，严禁直接作为多空投票权重）",
        f"- 独立证据簇数 (independent_cluster_count): {independent_cluster_count}（多头独立簇={bull_clusters}, 空头独立簇={bear_clusters}, 中性={neutral_clusters}）",
        f"- 真实核验有效证据数 (verified_evidence_count): {verified_evidence_count}（仅统计经核验通过证据，矛盾/未支撑项不计入）",
        f"- 证据簇方向权重 (cluster_weights): 多头权重={bull_weight:.1%}, 空头权重={bear_weight:.1%}",
    ]
    clusters = metrics.get("clusters") or []
    if clusters:
        lines.append("- 独立证据簇明细:")
        for cl in clusters:
            c_id = cl.get("cluster_id")
            c_type = cl.get("cluster_type")
            claims_str = ", ".join(cl.get("claims") or []) or "无"
            speakers_str = ", ".join(cl.get("speakers") or []) or "无"
            votes = cl.get("direction_votes") or {}
            lines.append(f"  * 【{c_id}】 类型={c_type} | 关联 claim=[{claims_str}] | 发言分析师=[{speakers_str}] | 方向投票={votes}")
    return "\n".join(lines)


# ==============================================================================
# E-02: 证据关系 Reducer 折叠与贡献上限（纯函数）
# 冻结契约 v4 (DAV-772 第四版微修订)：
# 1. SOURCE_REPETITION 作为无向等价关系做连通分量折叠；规范化键为 (min(u,v), rep, max(u,v))；不报 cycle。
# 2. DERIVED_OBSERVATION 保持 source_id(derived) -> target_id(base)，仅此子图参与 cycle 与 terminal 分析。
# 3. 纯 repetition 组件 derived_terminal_ids=() 并标注“不适用派生终点”。
# 4. component_id 采用无歧义 Canonical JSON 编码防碰撞后 SHA-256 计算。
# 5. 错误分类 100% 复用 E-01 FailClosedReason 与 ValidationResult；专有异常仅新增 MALFORMED_CLAIM_SEQUENCE / AMBIGUOUS_DERIVED_TERMINAL。
# 6. 未连接声明严禁称作独立声明；independence_status 恒为 UNKNOWN，global_contribution_cap 恒为 1（0 声明为 0）。
# 7. 严格输入置换不变性。
# ==============================================================================


class IndependenceStatus(str, Enum):
    """全局独立性证明状态（路径 A 下多节点间独立性无法证明）。"""
    UNKNOWN = "UNKNOWN"


class ReducerFailReason(str, Enum):
    """Reducer 专用防御异常原因（不重名新增，共享图错误直接复用 FailClosedReason）。"""
    MALFORMED_CLAIM_SEQUENCE = "MALFORMED_CLAIM_SEQUENCE"      # claim_ids 非序列、含非字符串、空白字符或 None/dict/list 等畸形元素
    AMBIGUOUS_DERIVED_TERMINAL = "AMBIGUOUS_DERIVED_TERMINAL"  # 要求严格单派生终点但存在多终点或纯 repetition 无派生终点


class EvidenceReductionError(ValueError):
    """非法输入整图拒绝异常容器（复用 E-01 FailClosedReason 与 ReducerFailReason）。"""

    def __init__(
        self,
        error_reason: Union[FailClosedReason, ReducerFailReason],
        message: str,
        affected_nodes: Tuple[str, ...] = (),
        affected_edges: Tuple[Tuple[str, str], ...] = (),
    ) -> None:
        reason_val = error_reason.value if hasattr(error_reason, "value") else str(error_reason)
        super().__init__(f"[{reason_val}] {message}")
        self.error_reason = error_reason
        self.message = message
        self.affected_nodes = tuple(affected_nodes)
        self.affected_edges = tuple(affected_edges)


@dataclass(frozen=True)
class FoldedComponent:
    """已证明折叠组件纯数据结构。"""
    component_id: str                              # 基于严格排序成员无歧义 Canonical JSON 编码经 SHA-256 截断生成的确定性 ID
    member_claim_ids: Tuple[str, ...]              # 组件内全部成员声明 ID（ASCII 严格升序）
    derived_terminal_ids: Tuple[str, ...]          # 仅基于 DERIVED_OBSERVATION 出边计算的终点元组（ASCII 严格升序；纯 repetition 组件恒为 ()，标注不适用派生终点）
    audit_edges: Tuple[EvidenceRelation, ...]      # 组件内包含的规范化审计边（去重并稳定排序）

    @property
    def is_ambiguous_derived_terminal(self) -> bool:
        """是否存在多终点或缺失派生终点的语义歧义。"""
        return len(self.derived_terminal_ids) != 1

    def get_single_derived_terminal(self) -> str:
        """严格单派生终点提取器：多终点或纯 repetition 无派生终点时立即 Fail-Closed 抛出 AMBIGUOUS_DERIVED_TERMINAL。"""
        if len(self.derived_terminal_ids) != 1:
            raise EvidenceReductionError(
                error_reason=ReducerFailReason.AMBIGUOUS_DERIVED_TERMINAL,
                message=f"Component {self.component_id} has ambiguous or empty derived terminals: {self.derived_terminal_ids}",
                affected_nodes=self.derived_terminal_ids,
                affected_edges=tuple((e.source_id, e.target_id) for e in self.audit_edges),
            )
        return self.derived_terminal_ids[0]


@dataclass(frozen=True)
class EvidenceReductionResult:
    """纯函数最终输出契约：只报告可证明折叠事实，不推断独立性，不输出独立投票。"""
    folded_components: Tuple[FoldedComponent, ...]  # 已证明折叠组件列表（按 component_id 升序排列）
    unconnected_claim_ids: Tuple[str, ...]          # 未被折叠边连接的声明列表（ASCII 严格升序，严禁称作独立声明）
    audit_edges: Tuple[EvidenceRelation, ...]       # 全图参与折叠的规范化审计边（全量去重排序）
    independence_status: IndependenceStatus         # 恒为 IndependenceStatus.UNKNOWN
    global_contribution_cap: int                    # 全局下游贡献上限（0 个声明为 0；>=1 个声明恒为 1）


def reduce_evidence_claims(
    claim_ids: Sequence[str],
    relations: Sequence[EvidenceRelation] | EvidenceRelationGraph,
) -> EvidenceReductionResult:
    """纯函数：基于显式可证明证据关系执行确定性同源等价折叠与派生 DAG 分流分析。

    核心约束：
    1. SOURCE_REPETITION 作为无向等价关系处理，反向重复与三角同源幂等合并，不参与有向环检测；
    2. DERIVED_OBSERVATION 保持 source_id(derived) -> target_id(base)，仅此子图进行有向环检测与派生终点分析；
    3. 组件由 repetition 与 derived 的无向投影共同形成；纯 repetition 组件 derived_terminal_ids=()；
    4. component_id 使用 Canonical JSON 编码防碰撞后哈希；
    5. unconnected_claim_ids 严禁称作独立声明，全图独立性恒为 UNKNOWN，全局下游贡献上限 global_contribution_cap 恒为 1（0 声明为 0）；
    6. 共享图错误直接复用 E-01 FailClosedReason (SELF_LOOP, DANGLING_REFERENCE, CYCLE_DETECTED)；
    7. 输入排列不影响输出结果（严格置换不变性）。
    """
    # Phase 1: Input Validation & Fail-Closed Defense
    if claim_ids is None or not isinstance(claim_ids, (list, tuple)):
        raise EvidenceReductionError(
            error_reason=ReducerFailReason.MALFORMED_CLAIM_SEQUENCE,
            message="claim_ids must be a sequence (list or tuple)",
        )
    for idx, item in enumerate(claim_ids):
        if not isinstance(item, str) or not item.strip():
            raise EvidenceReductionError(
                error_reason=ReducerFailReason.MALFORMED_CLAIM_SEQUENCE,
                message=f"Invalid claim ID at index {idx}: {item!r}. Must be a non-empty string.",
            )

    if relations is None:
        raise TypeError("relations must be Sequence[EvidenceRelation] or EvidenceRelationGraph, got None")
    if isinstance(relations, EvidenceRelationGraph):
        rel_seq = relations.relations
    elif isinstance(relations, (list, tuple)):
        rel_seq = relations
    else:
        raise TypeError(
            f"relations must be Sequence[EvidenceRelation] or EvidenceRelationGraph, got {type(relations).__name__}"
        )

    for idx, rel in enumerate(rel_seq):
        if not isinstance(rel, EvidenceRelation):
            raise TypeError(f"relations element at index {idx} must be EvidenceRelation, got {type(rel).__name__}")

    unique_claim_ids = sorted(set(claim_ids))
    if len(unique_claim_ids) == 0:
        if len(rel_seq) == 0:
            return EvidenceReductionResult(
                folded_components=(),
                unconnected_claim_ids=(),
                audit_edges=(),
                independence_status=IndependenceStatus.UNKNOWN,
                global_contribution_cap=0,
            )

    # Sort relations for deterministic, permutation-invariant evaluation
    sorted_rel_seq = sorted(
        rel_seq,
        key=lambda r: (
            r.relation_type.value,
            r.source_id,
            r.target_id,
            json.dumps(r.to_dict(), sort_keys=True, ensure_ascii=False),
        ),
    )

    # Phase 2: Edge Categorization & Endpoint Normalization (Fail-Closed)
    repetition_edges_map: dict[tuple[str, str, str], EvidenceRelation] = {}
    derived_edges_map: dict[tuple[str, str, str], EvidenceRelation] = {}

    for rel in sorted_rel_seq:
        # Dangling reference check (E-01 FailClosedReason.DANGLING_REFERENCE)
        if rel.source_id not in unique_claim_ids or rel.target_id not in unique_claim_ids:
            missing = tuple(sorted([n for n in (rel.source_id, rel.target_id) if n not in unique_claim_ids]))
            raise EvidenceReductionError(
                error_reason=FailClosedReason.DANGLING_REFERENCE,
                message=f"Dangling edge: {rel.source_id}->{rel.target_id} references unknown node(s): {missing}",
                affected_nodes=missing,
                affected_edges=((rel.source_id, rel.target_id),),
            )

        # Self-loop check (E-01 FailClosedReason.SELF_LOOP)
        if rel.source_id == rel.target_id:
            raise EvidenceReductionError(
                error_reason=FailClosedReason.SELF_LOOP,
                message=f"Self-loop: {rel.source_id}",
                affected_nodes=(rel.source_id,),
                affected_edges=((rel.source_id, rel.target_id),),
            )

        if rel.relation_type == RelationType.SOURCE_REPETITION:
            u, v = sorted([rel.source_id, rel.target_id])
            rep_key = (u, RelationType.SOURCE_REPETITION.value, v)
            if rep_key not in repetition_edges_map:
                norm_rel = EvidenceRelation(
                    source_id=u,
                    relation_type=RelationType.SOURCE_REPETITION,
                    target_id=v,
                    metadata=dict(rel.to_dict()["metadata"]),
                )
                repetition_edges_map[rep_key] = norm_rel
        elif rel.relation_type == RelationType.DERIVED_OBSERVATION:
            derived_key = (rel.source_id, RelationType.DERIVED_OBSERVATION.value, rel.target_id)
            if derived_key not in derived_edges_map:
                derived_edges_map[derived_key] = rel

    # Phase 3: Directed Cycle Detection on DERIVED_OBSERVATION Subgraph Only (Fail-Closed)
    adj_derived: dict[str, list[str]] = {u: [] for u in unique_claim_ids}
    for rel in derived_edges_map.values():
        adj_derived[rel.source_id].append(rel.target_id)
    for u in unique_claim_ids:
        adj_derived[u].sort()

    color: dict[str, int] = {u: 0 for u in unique_claim_ids}
    path: list[str] = []

    def dfs(node: str) -> None:
        color[node] = 1
        path.append(node)
        for nbr in adj_derived[node]:
            if color[nbr] == 1:
                idx = path.index(nbr)
                cycle_nodes = path[idx:] + [nbr]
                cycle_edges = tuple((cycle_nodes[i], cycle_nodes[i + 1]) for i in range(len(cycle_nodes) - 1))
                raise EvidenceReductionError(
                    error_reason=FailClosedReason.CYCLE_DETECTED,
                    message=f"Cycle in derived subgraph: {'->'.join(cycle_nodes)}",
                    affected_nodes=tuple(cycle_nodes),
                    affected_edges=cycle_edges,
                )
            elif color[nbr] == 0:
                dfs(nbr)
        path.pop()
        color[node] = 2

    for u in unique_claim_ids:
        if color[u] == 0:
            dfs(u)

    # Phase 4: Connected Component Partitioning (Joint Undirected Projection)
    undirected_adj: dict[str, set[str]] = {u: set() for u in unique_claim_ids}
    for (u, _, v) in repetition_edges_map.keys():
        undirected_adj[u].add(v)
        undirected_adj[v].add(u)
    for rel in derived_edges_map.values():
        undirected_adj[rel.source_id].add(rel.target_id)
        undirected_adj[rel.target_id].add(rel.source_id)

    visited: set[str] = set()
    raw_components: list[list[str]] = []
    for u in unique_claim_ids:
        if u not in visited:
            comp: list[str] = []
            queue: list[str] = [u]
            visited.add(u)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for nbr in sorted(undirected_adj[curr]):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            raw_components.append(sorted(comp))

    # Phase 5: Component Assembly & Canonical JSON ID Generation
    folded_components: list[FoldedComponent] = []
    unconnected_claims: list[str] = []

    for comp in raw_components:
        comp_set = set(comp)
        if len(comp) == 1 and len(undirected_adj[comp[0]]) == 0:
            unconnected_claims.append(comp[0])
            continue

        comp_derived = [
            rel for rel in derived_edges_map.values()
            if rel.source_id in comp_set and rel.target_id in comp_set
        ]
        if len(comp_derived) == 0:
            terminals: Tuple[str, ...] = ()
        else:
            cand_nodes = set(rel.source_id for rel in comp_derived) | set(rel.target_id for rel in comp_derived)
            terminals = tuple(sorted([
                n for n in cand_nodes
                if len([rel for rel in comp_derived if rel.source_id == n]) == 0
            ]))

        comp_audit = sorted(
            [rel for rel in repetition_edges_map.values() if rel.source_id in comp_set and rel.target_id in comp_set]
            + comp_derived,
            key=lambda r: (r.relation_type.value, r.source_id, r.target_id),
        )

        canonical_payload = json.dumps(comp, ensure_ascii=False, separators=(',', ':'))
        comp_hash = hashlib.sha256(canonical_payload.encode('utf-8')).hexdigest()[:12]
        comp_id = f"comp_{comp[0]}_{comp_hash}"

        folded_components.append(FoldedComponent(
            component_id=comp_id,
            member_claim_ids=tuple(comp),
            derived_terminal_ids=terminals,
            audit_edges=tuple(comp_audit),
        ))

    # Phase 6: Output Assembly (Strict Invariants)
    folded_components.sort(key=lambda c: c.component_id)
    unconnected_claims.sort()
    all_audit_edges = sorted(
        list(repetition_edges_map.values()) + list(derived_edges_map.values()),
        key=lambda r: (r.relation_type.value, r.source_id, r.target_id),
    )
    return EvidenceReductionResult(
        folded_components=tuple(folded_components),
        unconnected_claim_ids=tuple(unconnected_claims),
        audit_edges=tuple(all_audit_edges),
        independence_status=IndependenceStatus.UNKNOWN,
        global_contribution_cap=1,
    )
