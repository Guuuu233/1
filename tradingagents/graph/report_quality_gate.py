"""Report Quality Gate for macro and analyst reports.

Validates core keywords (e.g. 传导 and 联动/外溢/时滞) and prevents
silent rewriting or smoothing of failed/missing global market data.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple


KEYWORD_REQUIRED_CHAIN = "传导"
KEYWORD_LINKAGE_OPTIONS = ("联动", "外溢", "时滞")

ALLOWED_EXPLICIT_MISSING_MARKERS = (
    "【数据缺失】",
    "【数据获取失败】",
    "数据缺失",
)

ALLOWED_INDEX_KEYWORDS = (
    "全球核心指数",
    "标普",
    "恒生",
)

FORBIDDEN_SMOOTH_KEYWORDS = (
    "外围平稳",
    "外围中性",
    "外围市场平稳",
    "外围市场中性",
    "外围表现平稳",
    "外围表现中性",
    "外围整体平稳",
    "外围环境平稳",
    "外盘平稳",
    "外盘中性",
)

GLOBAL_DATA_FAILURE_STATUSES = frozenset(
    ("failed", "partial", "unavailable", "timeout", "error", "refused")
)

_FAILURE_MARKERS = (
    "数据获取失败",
    "【数据获取失败】",
    "数据缺失",
    "【数据缺失】",
    "调用失败",
    "调用异常",
    "拉取失败",
    "抓取失败",
    "获取失败",
    "未获取到",
    "无数据",
    "超时",
    "timeout",
    "failed",
    "unavailable",
    "所有全球指数接口调用失败",
)


def is_global_indices_failed_or_partial(
    market_data_context: Optional[Dict[str, Any]],
) -> bool:
    """Determine whether global_indices is missing, failed, or partial."""
    if not isinstance(market_data_context, dict):
        return False

    # 1. Check data_failure_ledger
    ledger = market_data_context.get("data_failure_ledger")
    if isinstance(ledger, list):
        for entry in ledger:
            if isinstance(entry, dict) and entry.get("source") == "global_indices":
                status = str(entry.get("status", "")).lower().strip()
                if status in GLOBAL_DATA_FAILURE_STATUSES:
                    return True

    # 2. Check source_provenance
    provenance = market_data_context.get("source_provenance")
    if isinstance(provenance, dict):
        g_prov = provenance.get("global_indices")
        if isinstance(g_prov, dict):
            status = str(g_prov.get("status", "")).lower().strip()
            if status in GLOBAL_DATA_FAILURE_STATUSES:
                return True

    # 3. Check direct global_indices value
    global_indices = market_data_context.get("global_indices")
    if isinstance(global_indices, dict):
        status = str(global_indices.get("status", "")).lower().strip()
        completeness = str(global_indices.get("completeness", "")).lower().strip()
        if status in GLOBAL_DATA_FAILURE_STATUSES or completeness in GLOBAL_DATA_FAILURE_STATUSES:
            return True
    elif isinstance(global_indices, str):
        val = global_indices.strip()
        if val.lower() in GLOBAL_DATA_FAILURE_STATUSES:
            return True
        if any(marker in val for marker in _FAILURE_MARKERS):
            return True

    return False


def check_report_keywords(text: str) -> Tuple[bool, List[str]]:
    """Check if report text contains required transmission & linkage keywords."""
    if not text or not isinstance(text, str):
        return False, ["正文为空或无有效文本"]

    reasons: List[str] = []
    if KEYWORD_REQUIRED_CHAIN not in text:
        reasons.append(f"缺少核心关键词：{KEYWORD_REQUIRED_CHAIN}")

    if not any(k in text for k in KEYWORD_LINKAGE_OPTIONS):
        options_str = "/".join(KEYWORD_LINKAGE_OPTIONS)
        reasons.append(f"缺少核心关键词：{options_str}之一")

    passed = len(reasons) == 0
    return passed, reasons


def check_global_indices_compliance(
    text: str,
    market_data_context: Optional[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """Check if report complies with global market data availability rules."""
    if not is_global_indices_failed_or_partial(market_data_context):
        return True, []

    if not text or not isinstance(text, str):
        return False, ["外盘数据缺失/异常但报告正文为空"]

    reasons: List[str] = []

    has_missing_marker = any(m in text for m in ALLOWED_EXPLICIT_MISSING_MARKERS)
    has_index_mention = any(idx in text for idx in ALLOWED_INDEX_KEYWORDS)

    # 必须出现【数据缺失】或 全球核心指数/标普/恒生 之一
    if not (has_missing_marker or has_index_mention):
        reasons.append(
            "外盘数据缺失/异常时，正文必须出现【数据缺失】或全球核心指数/标普/恒生之一"
        )

    # 禁止仅有「外围平稳/外围中性」而无点位或缺失标注
    has_smooth_phrase = any(phrase in text for phrase in FORBIDDEN_SMOOTH_KEYWORDS)
    if has_smooth_phrase and not has_missing_marker:
        # Check if there are explicit point/percentage citations for indices
        # If no explicit points or missing markers, forbid silent smoothing
        has_specific_index_points = bool(
            re.search(r"(?:标普|恒生|道指|纳斯达克|日经|DAX|指数)[^，。！？\n]*?(?:[+-]?\d+(?:\.\d+)?%|\d+点)", text)
        )
        if not has_specific_index_points:
            reasons.append("外盘数据缺失时禁止仅写外围平稳/外围中性而无点位或缺失标注")

    passed = len(reasons) == 0
    return passed, reasons


def check_report_quality(
    macro_report: str = "",
    fundamentals_report: str = "",
    market_data_context: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """Comprehensive quality gate inspection for macro (and fundamentals) reports."""
    target_text = macro_report.strip() if isinstance(macro_report, str) else ""
    if not target_text and isinstance(fundamentals_report, str) and fundamentals_report.strip():
        target_text = fundamentals_report.strip()

    if not target_text:
        # No report text available to validate
        return True, []

    all_reasons: List[str] = []

    # 1. Keyword validation
    kw_passed, kw_reasons = check_report_keywords(target_text)
    if not kw_passed:
        all_reasons.extend(kw_reasons)

    # 2. Global market rewrite compliance
    gi_passed, gi_reasons = check_global_indices_compliance(target_text, market_data_context)
    if not gi_passed:
        all_reasons.extend(gi_reasons)

    return (len(all_reasons) == 0, all_reasons)


def apply_report_quality_gate(
    state_or_result: Dict[str, Any],
    macro_retry_fn: Optional[Callable[..., Any]] = None,
) -> bool:
    """Run quality gate on report state/result and record failures to ledger without blocking."""
    if not isinstance(state_or_result, dict):
        return True

    macro_report = state_or_result.get("macro_report", "")
    fundamentals_report = state_or_result.get("fundamentals_report", "")
    market_data_context = state_or_result.get("market_data_context")
    if not isinstance(market_data_context, dict):
        # Look inside short_term or medium_term
        for sub_key in ("short_term", "medium_term", "result_data"):
            sub_val = state_or_result.get(sub_key)
            if isinstance(sub_val, dict) and isinstance(sub_val.get("market_data_context"), dict):
                market_data_context = sub_val["market_data_context"]
                break

    passed, failure_reasons = check_report_quality(
        macro_report=macro_report,
        fundamentals_report=fundamentals_report,
        market_data_context=market_data_context,
    )

    if not passed and macro_retry_fn is not None:
        try:
            retry_res = macro_retry_fn()
            if isinstance(retry_res, str) and retry_res.strip():
                macro_report = retry_res
                state_or_result["macro_report"] = retry_res
                passed, failure_reasons = check_report_quality(
                    macro_report=macro_report,
                    fundamentals_report=fundamentals_report,
                    market_data_context=market_data_context,
                )
        except Exception:
            pass

    if not passed:
        # Record into data_failure_ledger
        if not isinstance(market_data_context, dict):
            market_data_context = {}
            state_or_result["market_data_context"] = market_data_context

        ledger = market_data_context.get("data_failure_ledger")
        if not isinstance(ledger, list):
            ledger = []
            market_data_context["data_failure_ledger"] = ledger

        for reason in failure_reasons:
            already_recorded = any(
                isinstance(e, dict)
                and e.get("source") == "report_quality_gate"
                and e.get("reason") == reason
                for e in ledger
            )
            if not already_recorded:
                entry = {
                    "source": "report_quality_gate",
                    "status": "failed",
                    "reason": reason,
                    "gap": f"【数据获取失败】report_quality_gate：{reason}",
                }
                ledger.append(entry)

        # Sync data_gaps list if present
        if "data_gaps" in state_or_result and isinstance(state_or_result["data_gaps"], list):
            for entry in ledger:
                if isinstance(entry, dict) and entry.get("source") == "report_quality_gate":
                    gap_str = str(entry.get("gap") or "")
                    if gap_str and gap_str not in state_or_result["data_gaps"]:
                        state_or_result["data_gaps"].append(gap_str)

    return passed
