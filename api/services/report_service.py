"""Report service for database operations."""

import json
import json_repair
import logging
import math
import re
from numbers import Real

logger = logging.getLogger(__name__)
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Iterable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import func
from sqlalchemy.orm import Session, load_only

from api.database import ReportDB
from tradingagents.llm_clients.thinking_cleaner import clean_report_result_data


REPORT_SUMMARY_COLUMNS = (
    ReportDB.id,
    ReportDB.user_id,
    ReportDB.symbol,
    ReportDB.trade_date,
    ReportDB.status,
    ReportDB.error,
    ReportDB.decision,
    ReportDB.direction,
    ReportDB.confidence,
    ReportDB.probability,
    ReportDB.target_price,
    ReportDB.stop_loss_price,
    ReportDB.risk_items,
    ReportDB.key_metrics,
    ReportDB.data_gaps,
    ReportDB.falsification_conditions,
    ReportDB.not_applicable,
    ReportDB.analyst_traces,
    ReportDB.created_at,
    ReportDB.updated_at,
)

ACTIVE_REPORT_STATUSES = ("pending", "running")
STALE_REPORT_ERROR_MESSAGE = "分析任务已中断，请重新发起分析"


# ─── Structured extraction schemas ───────────────────────────────────────────

_RISK_ITEM_FIELDS = frozenset(("name", "level", "description"))
_KEY_METRIC_FIELDS = frozenset(("name", "value", "status"))
_STRUCTURED_REPORT_FIELDS = frozenset(
    (
        "decision",
        "confidence",
        "probability",
        "target_price",
        "stop_loss_price",
        "risks",
        "key_metrics",
        "data_gaps",
        "falsification_conditions",
        "not_applicable",
    )
)

_REPORT_MACHINE_BLOCK_TAGS = ("DEBATE_STATE", "RISK_STATE")
_REPORT_MACHINE_LIST_FIELDS = (
    "responded_claim_ids",
    "new_claims",
    "resolved_claim_ids",
    "unresolved_claim_ids",
    "next_focus_claim_ids",
)
_REPORT_MACHINE_TEXT_FIELDS = ("round_summary", "round_goal")
_LEGAL_DECISION_ALIASES = {
    "BUY": "BUY",
    "SELL": "SELL",
    "HOLD": "HOLD",
    "增持": "BUY",
    "买入": "BUY",
    "看多": "BUY",
    "减持": "SELL",
    "卖出": "SELL",
    "看空": "SELL",
    "持有": "HOLD",
    "中性": "HOLD",
}


def _warn_unknown_fields(model_name: str, values: Any, allowed_fields: frozenset[str]) -> Any:
    if not isinstance(values, dict):
        return values
    unknown_fields = sorted(str(key) for key in values if key not in allowed_fields)
    if unknown_fields:
        logger.warning(
            "[report_service] unknown structured fields ignored for %s: %s",
            model_name,
            ", ".join(unknown_fields),
        )
    return values


def _coerce_probability_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        logger.warning(
            "[report_service] probability rejected: value must be a real number, got %r",
            value,
        )
        return None
    try:
        probability = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        logger.warning(
            "[report_service] probability rejected: cannot convert %r to float",
            value,
        )
        return None
    if not math.isfinite(probability):
        logger.warning("[report_service] probability rejected: value must be finite, got %r", value)
        return None
    if not 0.0 <= probability <= 1.0:
        logger.warning(
            "[report_service] probability rejected: %s is outside the finite [0, 1] range",
            probability,
        )
        return None
    return probability


def _coerce_confidence_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        logger.warning(
            "[report_service] confidence rejected: value must be a real number, got %r",
            value,
        )
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        logger.warning(
            "[report_service] confidence rejected: cannot convert %r to int",
            value,
        )
        return None
    if not math.isfinite(numeric):
        logger.warning("[report_service] confidence rejected: value must be finite, got %r", value)
        return None
    if not numeric.is_integer():
        logger.warning("[report_service] confidence rejected: %r is not an integer", value)
        return None
    confidence = int(numeric)
    if not 0 <= confidence <= 100:
        logger.warning(
            "[report_service] confidence rejected: %d is out of [0, 100] range",
            confidence,
        )
        return None
    return confidence


def _canonicalize_structured_items(items: Any, schema, field_name: str) -> Optional[List[dict]]:
    if items is None:
        return None
    if not isinstance(items, list):
        logger.warning("[report_service] %s rejected: structured field must be an array", field_name)
        return []

    canonical_items: list[dict] = []
    for index, item in enumerate(items):
        try:
            model = item if isinstance(item, schema) else schema(**item)
            canonical_items.append(model.model_dump())
        except (TypeError, ValueError) as exc:
            logger.warning(
                "[report_service] %s item %d rejected: %s",
                field_name,
                index,
                exc,
            )
    return canonical_items


class RiskItemSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="风险名称，15字以内")
    level: str = Field("medium", description="风险等级")
    description: str = Field("", description="一句话说明，30字以内")

    @model_validator(mode="before")
    @classmethod
    def _warn_unknown_fields(cls, values):
        return _warn_unknown_fields("risk item", values, _RISK_ITEM_FIELDS)

    @field_validator("level", mode="before")
    @classmethod
    def _coerce_level(cls, v):
        if isinstance(v, str) and v.lower() in ("high", "medium", "low"):
            return v.lower()
        return "medium"


class KeyMetricSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="指标名称，如 PE、ROE、营收增速")
    value: str = Field(..., description="指标值，包含单位，如 28.5x、15.2%")
    status: str = Field("neutral", description="优劣判断")

    @model_validator(mode="before")
    @classmethod
    def _warn_unknown_fields(cls, values):
        return _warn_unknown_fields("key metric", values, _KEY_METRIC_FIELDS)

    @field_validator("value", mode="before")
    @classmethod
    def _coerce_value(cls, v):
        # LLM 可能返回数字而非字符串
        return str(v) if not isinstance(v, str) else v

    @field_validator("status", mode="before")
    @classmethod
    def _coerce_status(cls, v):
        if isinstance(v, str) and v.lower() in ("good", "neutral", "bad"):
            return v.lower()
        return "neutral"


class StructuredReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    decision: Optional[str] = Field(None, description="交易决策关键词：BUY/SELL/HOLD/增持/减持/持有")
    confidence: Optional[int] = Field(None, description="整体置信度 0-100")
    probability: Optional[float] = Field(None, description="报告明确给出的上涨概率")
    target_price: Optional[float] = Field(None, description="目标价（数字，无单位）")
    stop_loss_price: Optional[float] = Field(None, description="止损价（数字，无单位）")
    risks: List[RiskItemSchema] = Field(default_factory=list, description="主要风险，最多5条")
    key_metrics: List[KeyMetricSchema] = Field(default_factory=list, description="关键指标，最多6条")
    data_gaps: List[str] = Field(default_factory=list, description="报告明确列出的数据缺口")
    falsification_conditions: List[str] = Field(default_factory=list, description="报告明确列出的证伪条件")
    not_applicable: bool = Field(False, description="本分析框架是否明确不适用")

    @model_validator(mode="before")
    @classmethod
    def _warn_unknown_fields(cls, values):
        return _warn_unknown_fields("report", values, _STRUCTURED_REPORT_FIELDS)

    @field_validator("data_gaps", "falsification_conditions", mode="before")
    @classmethod
    def _coerce_string_list(cls, v):
        return [] if v is None else v

    @field_validator("decision", mode="before")
    @classmethod
    def _coerce_decision(cls, value):
        if value is None:
            return None
        normalized = str(value).strip()
        canonical = _LEGAL_DECISION_ALIASES.get(normalized)
        if canonical is None:
            canonical = _LEGAL_DECISION_ALIASES.get(normalized.upper())
        if canonical is None:
            logger.warning("[report_service] decision rejected: illegal structured value %r", value)
            return None
        return canonical

    @field_validator("not_applicable", mode="before")
    @classmethod
    def _coerce_not_applicable(cls, v):
        return False if v is None else v

    @field_validator("probability", mode="before")
    @classmethod
    def _coerce_probability(cls, v):
        return _coerce_probability_value(v)

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, v):
        return _coerce_confidence_value(v)

    @field_validator("target_price", "stop_loss_price", mode="before")
    @classmethod
    def _coerce_price(cls, v):
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, Real):
            logger.warning(
                "[report_service] price rejected: value must be a real number, got %r",
                v,
            )
            return None
        try:
            price = float(v)
        except (TypeError, ValueError, OverflowError) as exc:
            logger.warning(
                "[report_service] price rejected: cannot convert %r to float",
                v,
            )
            return None
        if not math.isfinite(price):
            logger.warning("[report_service] price rejected: value must be finite, got %r", v)
            return None
        if price < 0:
            logger.warning("[report_service] price rejected: value must be non-negative, got %r", v)
            return None
        return price


def _strict_unit_interval(value: Any, field_name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite number in [0, 1]")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a finite number in [0, 1]") from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field_name} must be a finite number in [0, 1]")
    return numeric


def _strict_claim_confidence(value: Any) -> float:
    if value is None or isinstance(value, (bool, str, bytes, bytearray)):
        raise ValueError("claim confidence must be a finite number in [0, 1]")
    return _strict_unit_interval(value, "claim confidence")


def _iter_report_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_report_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_report_strings(child)


def _validate_report_machine_payload(payload: Dict[str, Any], tag: str) -> None:
    allowed_fields = (*_REPORT_MACHINE_LIST_FIELDS, *_REPORT_MACHINE_TEXT_FIELDS)
    unknown_fields = sorted(str(key) for key in payload if key not in allowed_fields)
    if unknown_fields:
        logger.warning(
            "[report_service] unknown machine fields ignored for %s: %s",
            tag,
            ", ".join(unknown_fields),
        )

    for field_name in _REPORT_MACHINE_LIST_FIELDS:
        value = payload.get(field_name)
        if value is not None and not isinstance(value, list):
            raise ValueError(f"{tag} machine field {field_name} must be an array")
    for field_name in _REPORT_MACHINE_TEXT_FIELDS:
        value = payload.get(field_name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{tag} machine field {field_name} must be a string")

    claims = payload.get("new_claims") or []
    for index, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict):
            raise ValueError(f"{tag} claim {index} must be an object")
        if not isinstance(claim.get("claim"), str) or not claim["claim"].strip():
            raise ValueError(f"{tag} claim {index} must contain non-empty claim text")
        evidence = claim.get("evidence")
        if evidence is not None and not isinstance(evidence, list):
            raise ValueError(f"{tag} claim {index} evidence must be an array")
        target_claim_ids = claim.get("target_claim_ids")
        if target_claim_ids is not None and not isinstance(target_claim_ids, list):
            raise ValueError(f"{tag} claim {index} target_claim_ids must be an array")
        if "confidence" not in claim:
            raise ValueError(f"{tag} claim {index} confidence is required")
        _strict_claim_confidence(claim.get("confidence"))
        claim_unknown_fields = sorted(
            str(key) for key in claim if key not in ("claim", "evidence", "confidence", "target_claim_ids")
        )
        if claim_unknown_fields:
            logger.warning(
                "[report_service] unknown machine claim fields ignored for %s claim %d: %s",
                tag,
                index,
                ", ".join(claim_unknown_fields),
            )


def validate_report_machine_blocks(result_data: Optional[Dict[str, Any]]) -> None:
    """Validate every embedded machine block before a report is persisted."""
    if result_data is None:
        return
    if not isinstance(result_data, dict):
        raise ValueError("result_data must be an object")

    for text in _iter_report_strings(result_data):
        for tag in _REPORT_MACHINE_BLOCK_TAGS:
            openings = list(re.finditer(rf"<!--\s*{re.escape(tag)}\b", text))
            if not openings:
                continue
            if len(openings) > 1:
                raise ValueError(f"{tag} machine block must not be duplicated")
            marker_suffix = text[openings[0].end():].lstrip()
            if not marker_suffix.startswith(":"):
                raise ValueError(f"{tag} machine block must use ':' after the marker")
            payload_text = marker_suffix[1:]
            closing_index = payload_text.find("-->")
            if closing_index < 0:
                raise ValueError(f"{tag} machine block is truncated")
            try:
                payload = json.loads(payload_text[:closing_index].strip())
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"{tag} machine block contains invalid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{tag} machine block must contain an object")
            _validate_report_machine_payload(payload, tag)


def canonicalize_report_result_data(
    result_data: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the only result shape allowed at API and ReportDB boundaries.

    This function deliberately raises for malformed report objects or machine
    blocks.  Unknown structured fields are handled by ``StructuredReport``
    with warnings and are omitted from the canonical nested value.
    """
    if result_data is None:
        return None
    if not isinstance(result_data, dict):
        raise ValueError("result_data must be an object")

    validate_report_machine_blocks(result_data)
    canonical_data = dict(result_data)
    if "structured" not in canonical_data:
        return canonical_data

    structured = canonical_data.get("structured")
    if isinstance(structured, StructuredReport):
        canonical_data["structured"] = structured.model_dump()
        return canonical_data
    if not isinstance(structured, dict):
        raise ValueError("structured report must be an object")
    canonical_data["structured"] = StructuredReport(**structured).model_dump()
    return canonical_data


def extract_structured_data(
    final_trade_decision: str,
    fundamentals_report: str = "",
    config: Optional[Dict[str, Any]] = None,
) -> Optional[StructuredReport]:
    """Use LLM structured output to extract key data from report text."""
    if not final_trade_decision:
        return None
    if config is None:
        from tradingagents.default_config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG

    try:
        from langchain_core.messages import HumanMessage
        from tradingagents.llm_clients import create_llm_client

        client = create_llm_client(
            provider=config.get("llm_provider", "openai"),
            model=config.get("quick_think_llm", "gpt-4o-mini"),
            base_url=config.get("backend_url"),
            api_key=config.get("api_key"),
        )
        llm = client.get_llm()

        prompt = (
            "请从以下投资分析报告中提取结构化信息，并以 JSON 格式返回。\n\n"
            f"【最终交易决策】\n{final_trade_decision[:3000]}\n\n"
            f"【基本面报告摘要】\n{fundamentals_report[:1000]}\n\n"
            "提取要求（请确保输出为有效的 JSON 对象，不要包裹在 markdown 代码块中）：\n"
            "1. decision：决策方向关键词（BUY/SELL/HOLD 或 增持/减持/持有）\n"
            "2. confidence：整体置信度（0-100 整数），若文中未明确给出则为 null；"
            "禁止把 confidence 换算为 probability 或代填 probability 字段\n"
            "3. target_price / stop_loss_price：纯数字，若未提及则为 null\n"
            "4. risks：最多5条主要风险，每条包含名称（15字内）、等级（high/medium/low）、一句话说明\n"
            "5. key_metrics：最多6条关键财务/估值指标，每条包含名称、值（含单位）、优劣（good/neutral/bad）\n"
            "6. probability：在报告对应的主分析周期内，期末价格高于分析基准价的概率；"
            "必须是 0.00–1.00 之间的小数（不是百分比整数）；"
            "报告未明确给出则为 null；禁止用 confidence 换算或代填；"
            "双周期报告仅取主周期的 probability\n"
            "7. data_gaps：报告明确列出的数据缺口字符串数组；未提及则为 []\n"
            "8. falsification_conditions：报告明确列出的证伪条件字符串数组；未提及则为 []\n"
            "9. not_applicable：报告明确表示本分析框架不适用时为 true，否则为 false"
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = json_repair.loads(raw)
        result = StructuredReport(**parsed)
        return result
    except Exception as e:
        logger.warning(f"LLM structured extraction failed: {e}")
        if 'raw' in locals():
            logger.warning(f"Raw LLM output:\n{raw}")
        return None


# ─── Fallback regex extraction (used when LLM extraction unavailable) ─────────

def _extract_confidence_regex(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    for pattern in (r'置信度[:：]\s*(\d+)%', r'confidence[:：]\s*(\d+)%'):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            v = int(m.group(1))
            return v if 0 <= v <= 100 else None
    return None


def _extract_price_regex(text: Optional[str], price_type: str = "target") -> Optional[float]:
    if not text:
        return None
    if price_type == "target":
        patterns = [
            r'目标价[:：]\s*[¥$]?\s*(\d+\.?\d*)',
            r'目标价格[:：]\s*[¥$]?\s*(\d+\.?\d*)',
            r'target[:：]\s*[¥$]?\s*(\d+\.?\d*)',
        ]
    else:
        patterns = [
            r'止损价[:：]\s*[¥$]?\s*(\d+\.?\d*)',
            r'止损价格[:：]\s*[¥$]?\s*(\d+\.?\d*)',
            r'stop[-\s_]?loss[:：]\s*[¥$]?\s*(\d+\.?\d*)',
        ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def _extract_verdict(text: Optional[str]) -> Optional[Dict[str, str]]:
    if not text:
        return None
    match = re.search(r"<!--\s*VERDICT:\s*(\{.*?\})\s*-->", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    try:
        # Clean potential newlines or invisible characters common in LLM outputs
        raw_json = match.group(1).strip().replace('\n', ' ').replace('\r', ' ')
        payload = json.loads(raw_json)
    except Exception:
        return None
    direction = str(payload.get("direction") or "").strip()
    reason = str(payload.get("reason") or "").strip()
    if not direction:
        return None
    return {"direction": direction, "reason": reason}


def resolve_report_fields(
    result_data: Optional[Dict[str, Any]] = None,
    confidence_override: Optional[int] = None,
    target_price_override: Optional[float] = None,
    stop_loss_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Resolve the final structured fields once for both SSE payloads and DB writes."""
    market_report = sentiment_report = news_report = None
    fundamentals_report = macro_report = smart_money_report = volume_price_report = game_theory_report = None
    investment_plan = trader_investment_plan = None
    final_trade_decision = None

    if result_data:
        market_report = result_data.get("market_report")
        sentiment_report = result_data.get("sentiment_report")
        news_report = result_data.get("news_report")
        fundamentals_report = result_data.get("fundamentals_report")
        macro_report = result_data.get("macro_report")
        smart_money_report = result_data.get("smart_money_report")
        volume_price_report = result_data.get("volume_price_report")
        game_theory_report = result_data.get("game_theory_report")
        investment_plan = result_data.get("investment_plan")
        trader_investment_plan = result_data.get("trader_investment_plan")
        final_trade_decision = result_data.get("final_trade_decision")

    verdict = _extract_verdict(final_trade_decision)
    direction = verdict["direction"] if verdict else None

    if confidence_override is not None:
        confidence = _coerce_confidence_value(confidence_override)
    else:
        confidence = _extract_confidence_regex(final_trade_decision)

    target_price = target_price_override if target_price_override is not None else _extract_price_regex(final_trade_decision, "target")
    if target_price is None:
        target_price = _extract_price_regex(trader_investment_plan, "target")

    stop_loss_price = stop_loss_override if stop_loss_override is not None else _extract_price_regex(final_trade_decision, "stop_loss")
    if stop_loss_price is None:
        stop_loss_price = _extract_price_regex(trader_investment_plan, "stop_loss")

    return {
        "market_report": market_report,
        "sentiment_report": sentiment_report,
        "news_report": news_report,
        "fundamentals_report": fundamentals_report,
        "macro_report": macro_report,
        "smart_money_report": smart_money_report,
        "volume_price_report": volume_price_report,
        "game_theory_report": game_theory_report,
        "investment_plan": investment_plan,
        "trader_investment_plan": trader_investment_plan,
        "final_trade_decision": final_trade_decision,
        "direction": direction,
        "confidence": confidence,
        "target_price": target_price,
        "stop_loss_price": stop_loss_price,
    }


# ─── CRUD ────────────────────────────────────────────────────────────────────

def init_report(
    db: Session,
    report_id: str,
    symbol: str,
    trade_date: str,
    user_id: Optional[str] = None,
) -> ReportDB:
    """Create a pending report record when a job is submitted."""
    now = datetime.now(timezone.utc)
    db_report = ReportDB(
        id=report_id,
        user_id=user_id,
        symbol=symbol,
        trade_date=trade_date,
        status="pending",
        created_at=now,
        updated_at=now,
    )
    db.add(db_report)
    try:
        db.commit()
        db.refresh(db_report)
    except Exception:
        db.rollback()
        raise
    return db_report


def update_report_partial(
    db: Session,
    report_id: str,
    status: Optional[str] = None,
    **fields: Any
) -> Optional[ReportDB]:
    """Update specific fields of an existing report (e.g., partial analyst reports)."""
    db_report = db.query(ReportDB).filter(ReportDB.id == report_id).first()
    if not db_report:
        return None
    
    canonical_fields: Dict[str, Any] = {}
    try:
        for key, value in fields.items():
            if key == "confidence":
                value = _coerce_confidence_value(value)
            elif key == "probability":
                value = _coerce_probability_value(value)
            elif key == "result_data":
                value = canonicalize_report_result_data(value)
            elif key == "risk_items":
                value = _canonicalize_structured_items(value, RiskItemSchema, "risk_items")
            elif key == "key_metrics":
                value = _canonicalize_structured_items(value, KeyMetricSchema, "key_metrics")
            if hasattr(db_report, key):
                canonical_fields[key] = value
    except Exception:
        db.rollback()
        raise

    if status:
        db_report.status = status
    for key, value in canonical_fields.items():
        setattr(db_report, key, value)

    db_report.updated_at = datetime.now(timezone.utc)
    try:
        db.commit()
        db.refresh(db_report)
    except Exception:
        db.rollback()
        raise
    return db_report


def finalize_orphan_report(
    db: Session,
    report: ReportDB,
    *,
    error_message: str = STALE_REPORT_ERROR_MESSAGE,
) -> ReportDB:
    """Mark an orphaned pending/running report as failed."""
    if str(report.status or "") not in ACTIVE_REPORT_STATUSES:
        return report

    report.status = "failed"
    report.error = error_message
    report.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(report)
    return report


def recover_stale_active_reports(
    db: Session,
    *,
    active_job_ids: Optional[Iterable[str]] = None,
    error_message: str = STALE_REPORT_ERROR_MESSAGE,
) -> Dict[str, int]:
    """Recover stale pending/running reports left behind by interrupted jobs."""
    active_job_id_set = {str(job_id) for job_id in (active_job_ids or []) if str(job_id).strip()}
    rows = (
        db.query(ReportDB)
        .filter(ReportDB.status.in_(ACTIVE_REPORT_STATUSES))
        .all()
    )
    if not rows:
        return {"total": 0, "failed": 0}

    failed = 0
    changed = False
    now = datetime.now(timezone.utc)
    for row in rows:
        if str(row.id) in active_job_id_set:
            continue
        row.status = "failed"
        row.error = error_message
        row.updated_at = now
        changed = True
        failed += 1

    if changed:
        db.commit()

    return {
        "total": failed,
        "failed": failed,
    }


def mark_report_failed(
    db: Session,
    report_id: str,
    error_message: str
) -> Optional[ReportDB]:
    """Mark a report as failed with an error message."""
    return update_report_partial(db, report_id, status="failed", error=error_message)


def create_report(
    db: Session,
    symbol: str,
    trade_date: str,
    decision: Optional[str] = None,
    result_data: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    risk_items: Optional[List[dict]] = None,
    key_metrics: Optional[List[dict]] = None,
    probability: Optional[float] = None,
    data_gaps: Optional[List[str]] = None,
    falsification_conditions: Optional[List[str]] = None,
    not_applicable: bool = False,
    analyst_traces: Optional[List[dict]] = None,
    confidence_override: Optional[int] = None,
    target_price_override: Optional[float] = None,
    stop_loss_override: Optional[float] = None,
    report_id: Optional[str] = None,  # If provided, update existing
) -> ReportDB:
    """Create or finalize a report."""
    validated_probability = _coerce_probability_value(probability)
    canonical_result_data = canonicalize_report_result_data(result_data)
    # Bug D: strip AI thinking monologue from report sections before they are
    # persisted, so neither the DB text columns nor result_data carry reasoning
    # filler (e.g. "Let me think", "Hmm, wait,").
    canonical_result_data = clean_report_result_data(canonical_result_data)
    canonical_risk_items = _canonicalize_structured_items(risk_items, RiskItemSchema, "risk_items")
    canonical_key_metrics = _canonicalize_structured_items(key_metrics, KeyMetricSchema, "key_metrics")
    resolved = resolve_report_fields(
        result_data=canonical_result_data,
        confidence_override=confidence_override,
        target_price_override=target_price_override,
        stop_loss_override=stop_loss_override,
    )

    now = datetime.now(timezone.utc)
    
    # Check if we should update an existing record (initialized via init_report)
    db_report = None
    if report_id:
        db_report = db.query(ReportDB).filter(ReportDB.id == report_id).first()

    if db_report:
        # Update existing
        db_report.status = "completed"
        # A report may previously have been marked failed by an older worker
        # or timeout policy.  Successful finalisation is authoritative.
        db_report.error = None
        db_report.decision = decision
        db_report.direction = resolved["direction"]
        db_report.confidence = resolved["confidence"]
        db_report.probability = validated_probability
        db_report.target_price = resolved["target_price"]
        db_report.stop_loss_price = resolved["stop_loss_price"]
        db_report.result_data = canonical_result_data
        db_report.risk_items = canonical_risk_items
        db_report.key_metrics = canonical_key_metrics
        db_report.data_gaps = list(data_gaps or [])
        db_report.falsification_conditions = list(falsification_conditions or [])
        db_report.not_applicable = bool(not_applicable)
        db_report.analyst_traces = analyst_traces
        db_report.market_report = resolved["market_report"]
        db_report.sentiment_report = resolved["sentiment_report"]
        db_report.news_report = resolved["news_report"]
        db_report.fundamentals_report = resolved["fundamentals_report"]
        db_report.macro_report = resolved["macro_report"]
        db_report.smart_money_report = resolved["smart_money_report"]
        db_report.volume_price_report = resolved["volume_price_report"]
        db_report.game_theory_report = resolved["game_theory_report"]
        db_report.investment_plan = resolved["investment_plan"]
        db_report.trader_investment_plan = resolved["trader_investment_plan"]
        db_report.final_trade_decision = resolved["final_trade_decision"]
        db_report.updated_at = now
    else:
        # Create new
        db_report = ReportDB(
            id=report_id or str(uuid4()),
            user_id=user_id,
            symbol=symbol,
            trade_date=trade_date,
            status="completed",
            decision=decision,
            direction=resolved["direction"],
            confidence=resolved["confidence"],
            probability=validated_probability,
            target_price=resolved["target_price"],
            stop_loss_price=resolved["stop_loss_price"],
            result_data=canonical_result_data,
            risk_items=canonical_risk_items,
            key_metrics=canonical_key_metrics,
            data_gaps=list(data_gaps or []),
            falsification_conditions=list(falsification_conditions or []),
            not_applicable=bool(not_applicable),
            analyst_traces=analyst_traces,
            market_report=resolved["market_report"],
            sentiment_report=resolved["sentiment_report"],
            news_report=resolved["news_report"],
            fundamentals_report=resolved["fundamentals_report"],
            macro_report=resolved["macro_report"],
            smart_money_report=resolved["smart_money_report"],
            volume_price_report=resolved["volume_price_report"],
            game_theory_report=resolved["game_theory_report"],
            investment_plan=resolved["investment_plan"],
            trader_investment_plan=resolved["trader_investment_plan"],
            final_trade_decision=resolved["final_trade_decision"],
            created_at=now,
            updated_at=now,
        )
        db.add(db_report)

    try:
        db.commit()
        db.refresh(db_report)
    except Exception:
        db.rollback()
        raise
    return db_report


def get_report(db: Session, report_id: str, user_id: Optional[str] = None) -> Optional[ReportDB]:
    query = db.query(ReportDB).filter(ReportDB.id == report_id)
    if user_id:
        query = query.filter(ReportDB.user_id == user_id)
    return query.first()


def get_reports_by_user(
    db: Session,
    user_id: Optional[str] = None,
    symbol: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
) -> List[ReportDB]:
    query = db.query(ReportDB).options(load_only(*REPORT_SUMMARY_COLUMNS))
    if user_id:
        query = query.filter(ReportDB.user_id == user_id)
    if symbol:
        query = query.filter(ReportDB.symbol == symbol)
    return query.order_by(ReportDB.created_at.desc()).offset(skip).limit(limit).all()


def get_latest_reports_by_symbols(
    db: Session,
    symbols: List[str],
    user_id: Optional[str] = None,
) -> List[ReportDB]:
    normalized_symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    if not normalized_symbols:
        return []

    query = db.query(ReportDB).options(load_only(*REPORT_SUMMARY_COLUMNS))
    if user_id:
        query = query.filter(ReportDB.user_id == user_id)

    rows = (
        query.filter(ReportDB.symbol.in_(normalized_symbols))
        .order_by(ReportDB.symbol.asc(), ReportDB.created_at.desc())
        .all()
    )

    latest_by_symbol: dict[str, ReportDB] = {}
    for row in rows:
        symbol = str(row.symbol or "").upper()
        if symbol and symbol not in latest_by_symbol:
            latest_by_symbol[symbol] = row

    return [latest_by_symbol[symbol] for symbol in normalized_symbols if symbol in latest_by_symbol]


def count_reports(
    db: Session,
    user_id: Optional[str] = None,
    symbol: Optional[str] = None,
) -> int:
    query = db.query(func.count(ReportDB.id))
    if user_id:
        query = query.filter(ReportDB.user_id == user_id)
    if symbol:
        query = query.filter(ReportDB.symbol == symbol)
    return query.scalar() or 0


def delete_report(db: Session, report_id: str, user_id: Optional[str] = None) -> bool:
    query = db.query(ReportDB).filter(ReportDB.id == report_id)
    if user_id:
        query = query.filter(ReportDB.user_id == user_id)
    report = query.first()
    if report:
        db.delete(report)
        db.commit()
        return True
    return False


def batch_delete_reports(db: Session, report_ids: Iterable[str], user_id: Optional[str] = None) -> dict:
    normalized_ids: list[str] = []
    seen: set[str] = set()
    for raw_report_id in report_ids:
        report_id = str(raw_report_id or "").strip()
        if not report_id or report_id in seen:
            continue
        seen.add(report_id)
        normalized_ids.append(report_id)

    if not normalized_ids:
        raise ValueError("请至少选择 1 份报告")

    query = db.query(ReportDB).filter(ReportDB.id.in_(normalized_ids))
    if user_id:
        query = query.filter(ReportDB.user_id == user_id)

    rows = query.all()
    row_by_id = {str(row.id): row for row in rows}
    deleted_ids: list[str] = []
    missing_ids: list[str] = []

    for report_id in normalized_ids:
        row = row_by_id.get(report_id)
        if row is None:
            missing_ids.append(report_id)
            continue
        db.delete(row)
        deleted_ids.append(report_id)

    if deleted_ids:
        db.commit()

    return {
        "deleted_ids": deleted_ids,
        "missing_ids": missing_ids,
    }


_REPORT_GAP_FIELDS = (
    "market_report",
    "sentiment_report",
    "news_report",
    "fundamentals_report",
    "macro_report",
    "smart_money_report",
    "volume_price_report",
    "game_theory_report",
    "investment_plan",
    "trader_investment_plan",
    "final_trade_decision",
)
_DATA_FAILURE_LINE_RE = re.compile(
    r"^\s*(?:(?:[-*•]\s*)|(?:\d+[.)、]\s*))?(【数据获取失败】.*)\s*$"
)


def _normalize_gap_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _explicit_data_failure_lines(text: Any) -> Iterable[str]:
    """Yield only machine-readable failure lines from a report body."""
    if not isinstance(text, str):
        return
    for line in text.splitlines():
        match = _DATA_FAILURE_LINE_RE.match(line)
        if not match:
            continue
        normalized = _normalize_gap_text(match.group(1))
        if normalized:
            yield normalized


def _iter_report_texts(result_data: Any) -> Iterable[str]:
    if not isinstance(result_data, dict):
        return

    for field in _REPORT_GAP_FIELDS:
        value = result_data.get(field)
        if isinstance(value, str):
            yield value

    for nested_key in ("short_term", "medium_term", "horizons", "result_data"):
        nested = result_data.get(nested_key)
        if isinstance(nested, dict):
            if nested_key == "horizons":
                for horizon_result in nested.values():
                    if isinstance(horizon_result, dict):
                        yield from _iter_report_texts(horizon_result)
            else:
                yield from _iter_report_texts(nested)


def merge_data_gaps(
    result_data: Optional[Dict[str, Any]] = None,
    llm_data_gaps: Optional[Iterable[Any]] = None,
) -> List[str]:
    """Merge explicit report failures with model-reported gaps deterministically."""
    merged: List[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        normalized = _normalize_gap_text(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            merged.append(normalized)

    for ledger_entry in _iter_failure_ledger_entries(result_data):
        source = _normalize_gap_text(ledger_entry.get("source"))
        status = _normalize_gap_text(ledger_entry.get("status")).lower()
        reason = _normalize_gap_text(ledger_entry.get("reason"))
        if not source or status not in _FAILURE_LEDGER_STATUSES:
            continue
        explicit_gap = _normalize_gap_text(ledger_entry.get("gap"))
        if explicit_gap.startswith("【数据获取失败】"):
            add(explicit_gap)
        elif reason:
            add(f"【数据获取失败】{source}：{reason}")

    for report_text in _iter_report_texts(result_data):
        for gap in _explicit_data_failure_lines(report_text):
            add(gap)

    if isinstance(llm_data_gaps, str):
        llm_data_gaps = [llm_data_gaps]
    try:
        llm_items = iter(llm_data_gaps or [])
    except TypeError:
        llm_items = iter(())
    for gap in llm_items:
        add(gap)

    return merged


_FAILURE_LEDGER_STATUSES = frozenset(("failed", "timeout", "unavailable", "refused", "error"))


def _iter_failure_ledger_entries(result_data: Any) -> Iterable[Dict[str, Any]]:
    """Yield only ledgers from the known market-data context locations."""
    if not isinstance(result_data, dict):
        return

    market_data_context = result_data.get("market_data_context")
    if isinstance(market_data_context, dict):
        ledger = market_data_context.get("data_failure_ledger")
        if isinstance(ledger, list):
            for entry in ledger:
                if isinstance(entry, dict):
                    yield entry
        # Dual reports keep one context per horizon at this location.
        for nested_context in market_data_context.values():
            if isinstance(nested_context, dict) and nested_context is not market_data_context:
                nested_ledger = nested_context.get("data_failure_ledger")
                if isinstance(nested_ledger, list):
                    for entry in nested_ledger:
                        if isinstance(entry, dict):
                            yield entry

    for nested_key in ("short_term", "medium_term", "horizons", "result_data"):
        nested = result_data.get(nested_key)
        if isinstance(nested, dict):
            if nested_key == "horizons":
                for horizon_result in nested.values():
                    if isinstance(horizon_result, dict):
                        yield from _iter_failure_ledger_entries(horizon_result)
            else:
                yield from _iter_failure_ledger_entries(nested)


def aggregate_horizon_metadata(
    horizon_results: Iterable[tuple[str, Dict[str, Any]]],
    *,
    requested_horizons: Iterable[str],
) -> Dict[str, Any]:
    """Aggregate horizon metadata without collapsing mixed states.

    The flattened list is retained for the legacy ReportDB column, while the
    keyed maps preserve which horizon supplied each value.
    """
    requested = [str(horizon) for horizon in requested_horizons]
    by_horizon = {str(horizon): result for horizon, result in horizon_results}
    not_applicable_by_horizon: Dict[str, Any] = {}
    falsification_by_horizon: Dict[str, List[str]] = {}
    flattened: List[str] = []
    seen_conditions: set[str] = set()

    all_completed = bool(requested)
    all_not_applicable = bool(requested)
    for horizon in requested:
        result = by_horizon.get(horizon) or {}
        completed = result.get("status") == "completed"
        all_completed = all_completed and completed
        value = result.get("not_applicable") if completed else None
        not_applicable_by_horizon[horizon] = value
        all_not_applicable = all_not_applicable and value is True

        conditions = result.get("falsification_conditions") if completed else []
        if not isinstance(conditions, list):
            conditions = []
        canonical_conditions: List[str] = []
        for condition in conditions:
            normalized = _normalize_gap_text(condition)
            if not normalized or normalized in canonical_conditions:
                continue
            canonical_conditions.append(normalized)
            if normalized not in seen_conditions:
                seen_conditions.add(normalized)
                flattened.append(normalized)
        falsification_by_horizon[horizon] = canonical_conditions

    return {
        "falsification_conditions": flattened,
        "falsification_conditions_by_horizon": falsification_by_horizon,
        "not_applicable": bool(all_completed and all_not_applicable),
        "not_applicable_by_horizon": not_applicable_by_horizon,
    }
