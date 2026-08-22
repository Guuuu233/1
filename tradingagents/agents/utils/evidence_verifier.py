"""Deterministic factual truth evaluator for debate claims and evidence.

Performs unit normalization, anti-lookahead date checks, deterministic factual
matching against the seven analyst reports and market_data_context, and flags
fatal hallucinations when citations reference failed or unavailable data sources.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

# Status constants
STATUS_VERIFIED = "verified"
STATUS_UNSUPPORTED = "unsupported"
STATUS_CONTRADICTED = "contradicted"
STATUS_SOURCE_UNAVAILABLE = "source_unavailable"

# Common failed status strings
UNAVAILABLE_STATUSES = frozenset(
    {"failed", "unavailable", "empty", "error", "missing", "partial_failure", "not_found", "rejected"}
)

# Standard report keys
SEVEN_REPORT_KEYS = (
    "macro_report",
    "market_report",
    "sentiment_report",
    "news_report",
    "fundamentals_report",
    "smart_money_report",
    "volume_price_report",
)

# Chinese and English quantity/unit patterns
_NUMBER_WITH_UNIT_RE = re.compile(
    r"([+-]?\d+(?:\.\d+)?)\s*(亿元|万元|万|亿|%|％|bp|点|元|港元|美元)?",
    re.IGNORECASE,
)

_ISO_DATE_RE = re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)")
_CN_DATE_RE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日?")


def _parse_date(text: str) -> date | None:
    if not text:
        return None
    iso_m = _ISO_DATE_RE.search(text)
    if iso_m:
        try:
            return date(int(iso_m.group(1)), int(iso_m.group(2)), int(iso_m.group(3)))
        except (ValueError, TypeError):
            pass
    cn_m = _CN_DATE_RE.search(text)
    if cn_m:
        try:
            return date(int(cn_m.group(1)), int(cn_m.group(2)), int(cn_m.group(3)))
        except (ValueError, TypeError):
            pass
    return None


def normalize_numeric_value(val_str: str, unit_str: str = "") -> tuple[float, str] | None:
    """Normalize a value and its unit to a canonical base (e.g. 元, %, or raw number).

    Returns:
        (canonical_number, canonical_unit) or None
    """
    try:
        num = float(val_str)
    except (ValueError, TypeError):
        return None

    unit = (unit_str or "").strip().lower()
    if unit in {"亿元", "亿"}:
        return num * 100_000_000.0, "元"
    elif unit in {"万元", "万"}:
        return num * 10_000.0, "元"
    elif unit in {"元", "港元", "美元"}:
        return num, "元"
    elif unit in {"%", "％"}:
        return num, "%"
    elif unit == "bp":
        return num / 100.0, "%"
    else:
        return num, "raw"


def _extract_numbers_and_units(text: str) -> list[tuple[float, str, str]]:
    """Extract list of (normalized_val, canonical_unit, raw_substr) from text."""
    results = []
    for match in _NUMBER_WITH_UNIT_RE.finditer(text):
        val_str = match.group(1)
        unit_str = match.group(2) or ""
        norm = normalize_numeric_value(val_str, unit_str)
        if norm is not None:
            results.append((norm[0], norm[1], match.group(0)))
    return results


def _extract_metric_keywords(text: str) -> list[str]:
    """Extract financial and market metric keywords from a string."""
    keywords = [
        "pe", "pb", "roe", "eps", "m2", "cpi", "ppi", "gdp", "lpr", "shibor",
        "营收", "收入", "利润", "净利润", "毛利率", "净利率", "负债率", "现金流",
        "主力", "净流入", "净流出", "超大单", "大单", "龙虎榜", "成交量", "成交额", "换手率",
        "降息", "降准", "分红", "估值", "订单", "产能", "利用率", "均线", "突破",
        "油价", "汇率", "关税", "补贴", "长协", "转嫁", "cr3", "价格战", "库存"
    ]
    found = []
    text_lower = text.lower()
    for kw in keywords:
        if kw in text_lower:
            found.append(kw)
    return found


class EvidenceFactualTruthEvaluator:
    """Evaluates evidence statements against 7 analyst reports and market context."""

    def __init__(self, relative_tolerance: float = 0.02, absolute_tolerance: float = 0.05):
        self.rel_tol = relative_tolerance
        self.abs_tol = absolute_tolerance

    def _extract_unavailable_sources(
        self, market_data_context: Mapping[str, Any] | None
    ) -> set[str]:
        unavailable = set()
        if not isinstance(market_data_context, Mapping):
            return unavailable

        # Check data_failure_ledger
        ledger = market_data_context.get("data_failure_ledger")
        if isinstance(ledger, list):
            for entry in ledger:
                if isinstance(entry, dict):
                    src = str(entry.get("source", "")).strip().lower()
                    status = str(entry.get("status", "")).strip().lower()
                    if src and (not status or status in UNAVAILABLE_STATUSES):
                        unavailable.add(src)
                        # Add alias name if available
                        name = str(entry.get("name", "")).strip().lower()
                        if name:
                            unavailable.add(name)

        # Check source_provenance
        prov = market_data_context.get("source_provenance")
        if isinstance(prov, dict):
            for src, info in prov.items():
                if isinstance(info, dict):
                    status = str(info.get("status", "")).strip().lower()
                    if status in UNAVAILABLE_STATUSES:
                        unavailable.add(src.strip().lower())

        # Check data_gaps
        gaps = market_data_context.get("data_gaps")
        if isinstance(gaps, list):
            for g in gaps:
                if isinstance(g, str):
                    unavailable.add(g.strip().lower())

        return unavailable

    def _check_source_unavailable(
        self, raw_evidence: str, unavailable_sources: set[str]
    ) -> tuple[bool, str]:
        if not unavailable_sources or not raw_evidence:
            return False, ""
        evidence_lower = raw_evidence.lower()
        for src in unavailable_sources:
            if src and (src in evidence_lower or evidence_lower in src):
                return True, src
        return False, ""

    def _check_anti_lookahead(
        self, raw_evidence: str, baseline_date_obj: date | None
    ) -> tuple[bool, str]:
        if baseline_date_obj is None or not raw_evidence:
            return True, ""
        ev_date = _parse_date(raw_evidence)
        if ev_date and ev_date > baseline_date_obj:
            # If the text does not indicate forward prediction/target, it violates lookahead
            if not any(w in raw_evidence for w in ("预测", "预期", "目标", "展望", "情景", "未来")):
                return False, f"日期 {ev_date.isoformat()} 晚于基准分析日期 {baseline_date_obj.isoformat()}，存在前视偏差"
        return True, ""

    def evaluate_single_evidence(
        self,
        raw_evidence: str,
        seven_reports: Mapping[str, str],
        market_data_context: Mapping[str, Any] | None = None,
        analysis_baseline_date: str | None = None,
        claim_id: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate a single evidence string for truthfulness, lookahead, and fatal hallucination."""
        raw_text = str(raw_evidence or "").strip()
        if not raw_text:
            return {
                "raw": raw_evidence,
                "claim_id": claim_id,
                "matched_role": None,
                "matched_source": None,
                "status": STATUS_UNSUPPORTED,
                "is_fatal": False,
                "details": "证据文本为空",
            }

        # 1. Check for unavailable data sources (Fatal Hallucination)
        unavailable_sources = self._extract_unavailable_sources(market_data_context)
        is_unavail, failed_src = self._check_source_unavailable(raw_text, unavailable_sources)
        if is_unavail:
            return {
                "raw": raw_text,
                "claim_id": claim_id,
                "matched_role": None,
                "matched_source": failed_src,
                "status": STATUS_SOURCE_UNAVAILABLE,
                "is_fatal": True,
                "details": f"引用了失败账本中不可用或缺失的数据源指标: {failed_src}，属于严重幻觉",
            }

        # 2. Check Anti-lookahead Date
        baseline_date_obj = None
        if analysis_baseline_date:
            baseline_date_obj = _parse_date(analysis_baseline_date)
        elif isinstance(market_data_context, Mapping):
            b_str = market_data_context.get("analysis_baseline_date") or market_data_context.get("data_as_of") or market_data_context.get("trade_date")
            if b_str:
                baseline_date_obj = _parse_date(str(b_str))

        date_ok, date_reason = self._check_anti_lookahead(raw_text, baseline_date_obj)
        if not date_ok:
            return {
                "raw": raw_text,
                "claim_id": claim_id,
                "matched_role": None,
                "matched_source": None,
                "status": STATUS_CONTRADICTED,
                "is_fatal": False,
                "details": date_reason,
            }

        # 3. Deterministic Matching against 7 Reports
        ev_numbers = _extract_numbers_and_units(raw_text)
        ev_keywords = _extract_metric_keywords(raw_text)

        # 3.1 Exact substring match in any report
        for role_key in SEVEN_REPORT_KEYS:
            report_body = str(seven_reports.get(role_key, "") or "")
            if not report_body.strip():
                continue
            if raw_text in report_body or any(len(p) >= 4 and p in report_body for p in raw_text.split("，")):
                return {
                    "raw": raw_text,
                    "claim_id": claim_id,
                    "matched_role": role_key,
                    "matched_source": role_key.replace("_report", ""),
                    "status": STATUS_VERIFIED,
                    "is_fatal": False,
                    "details": f"在 {role_key} 中找到精确匹配事实",
                }

        # 3.2 Numeric and keyword normalized matching in reports
        contradicted_candidate = None

        for role_key in SEVEN_REPORT_KEYS:
            report_body = str(seven_reports.get(role_key, "") or "")
            if not report_body.strip():
                continue

            for line in report_body.splitlines():
                line_text = line.strip()
                if not line_text:
                    continue

                # Check keyword overlap
                line_keywords = _extract_metric_keywords(line_text)
                common_kw = set(ev_keywords).intersection(set(line_keywords))

                line_numbers = _extract_numbers_and_units(line_text)
                if not line_numbers and not ev_numbers:
                    # Pure qualitative text match if keywords strongly match
                    if len(common_kw) >= 2 and any(kw in line_text for kw in ev_keywords):
                        return {
                            "raw": raw_text,
                            "claim_id": claim_id,
                            "matched_role": role_key,
                            "matched_source": role_key.replace("_report", ""),
                            "status": STATUS_VERIFIED,
                            "is_fatal": False,
                            "details": f"在 {role_key} 中找到定性指标匹配: {', '.join(common_kw)}",
                        }
                    continue

                # If numbers exist, check value match
                matched_all_numbers = True
                found_match = False
                for ev_num, ev_unit, ev_raw in ev_numbers:
                    num_found_in_line = False
                    for l_num, l_unit, l_raw in line_numbers:
                        # Check units compatible
                        if ev_unit == l_unit or (ev_unit == "raw" and l_unit == "%") or (ev_unit == "%" and l_unit == "raw"):
                            if math.isclose(ev_num, l_num, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                                num_found_in_line = True
                                found_match = True
                                break
                            elif common_kw:
                                # Numbers are different on the same metric keyword -> potential contradiction
                                diff_pct = abs(ev_num - l_num) / (abs(l_num) + 1e-9)
                                if diff_pct > 0.05:
                                    contradicted_candidate = (
                                        role_key,
                                        f"在 {role_key} 中关键词 '{', '.join(common_kw)}' 数据冲突: 证据声称 {ev_raw}，报告记录为 {l_raw}",
                                    )
                    if not num_found_in_line:
                        matched_all_numbers = False

                if found_match and (matched_all_numbers or len(common_kw) >= 1):
                    return {
                        "raw": raw_text,
                        "claim_id": claim_id,
                        "matched_role": role_key,
                        "matched_source": role_key.replace("_report", ""),
                        "status": STATUS_VERIFIED,
                        "is_fatal": False,
                        "details": f"在 {role_key} 中验证数值与关键词匹配",
                    }

        if contradicted_candidate:
            return {
                "raw": raw_text,
                "claim_id": claim_id,
                "matched_role": contradicted_candidate[0],
                "matched_source": contradicted_candidate[0].replace("_report", ""),
                "status": STATUS_CONTRADICTED,
                "is_fatal": False,
                "details": contradicted_candidate[1],
            }

        # 4. Check market_data_context if provided
        if isinstance(market_data_context, Mapping):
            # Check quotes, indicators, fund_flow_evidence
            md_str = str(market_data_context)
            if raw_text in md_str:
                return {
                    "raw": raw_text,
                    "claim_id": claim_id,
                    "matched_role": "market_data_context",
                    "matched_source": "market_data_context",
                    "status": STATUS_VERIFIED,
                    "is_fatal": False,
                    "details": "在 market_data_context 中找到匹配数据",
                }

        # 5. Unsupported
        return {
            "raw": raw_text,
            "claim_id": claim_id,
            "matched_role": None,
            "matched_source": None,
            "status": STATUS_UNSUPPORTED,
            "is_fatal": False,
            "details": "未在七份分析师报告或市场数据上下文中找到该事实或数据支撑",
        }

    def evaluate_claims(
        self,
        claims: Sequence[Mapping[str, Any]],
        seven_reports: Mapping[str, str],
        market_data_context: Mapping[str, Any] | None = None,
        analysis_baseline_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate all evidence items across tracked debate claims."""
        results: list[dict[str, Any]] = []
        for claim in claims:
            cid = str(claim.get("claim_id", "")).strip() or None
            evidence_list = claim.get("evidence") or []
            if isinstance(evidence_list, str):
                evidence_list = [evidence_list]

            for ev in evidence_list:
                ev_str = str(ev).strip()
                if not ev_str:
                    continue
                ver_res = self.evaluate_single_evidence(
                    raw_evidence=ev_str,
                    seven_reports=seven_reports,
                    market_data_context=market_data_context,
                    analysis_baseline_date=analysis_baseline_date,
                    claim_id=cid,
                )
                results.append(ver_res)

        return results


def normalize_winner(winner_raw: Any, direction_raw: Any = "") -> str:
    """Normalize winner string to one of 'bull', 'bear', 'tie'."""
    w_str = str(winner_raw or "").strip().lower()
    if w_str in {"bull", "bullish", "多头", "多方", "多头胜", "多方胜", "多头全面胜出"}:
        return "bull"
    elif w_str in {"bear", "bearish", "空头", "空方", "空头胜", "空方胜", "空头全面胜出"}:
        return "bear"
    elif w_str in {"tie", "neutral", "平局", "势均力敌", "分歧", "观望", "hold", "中性"}:
        return "tie"

    # Infer from direction if winner is not explicit
    d_str = str(direction_raw or "").strip().lower()
    if d_str in {"看多", "偏多", "buy", "bullish", "lean_bullish", "买入", "增持"}:
        return "bull"
    elif d_str in {"看空", "偏空", "sell", "bearish", "lean_bearish", "卖出", "减持"}:
        return "bear"
    elif d_str in {"中性", "观望", "hold", "neutral", "持有"}:
        return "tie"

    return "tie"


def extract_and_validate_manager_verdict(
    raw_response: str,
    claims_verification: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract structured manager verdict and perform strict consistency check.

    Returns:
        manager_verdict dict with keys:
        - direction
        - winner ("bull" | "bear" | "tie")
        - reason
        - position_pct
        - entry
        - target
        - stop_loss
        - upside
        - downside
        - odds
        - adopted_claim_ids
        - rejected_claim_ids
        - consistency_check_passed
        - failed_checks
    """
    from tradingagents.agents.utils.debate_utils import extract_tagged_json, strip_tagged_json

    # Attempt to extract MANAGER_VERDICT block first, then fallback to VERDICT block
    payload = extract_tagged_json(raw_response, "MANAGER_VERDICT")
    if not payload:
        payload = extract_tagged_json(raw_response, "VERDICT")

    failed_checks: list[str] = []

    if not payload:
        failed_checks.append("未提取到有效的研究总监结构化裁决机读块 (MANAGER_VERDICT 或 VERDICT)")

    direction = str(payload.get("direction", "")).strip() if payload else ""
    winner = normalize_winner(payload.get("winner"), direction)
    reason = str(payload.get("reason", "")).strip() if payload else ""

    position_pct = payload.get("position_pct") if payload else None
    entry = payload.get("entry") if payload else None
    target = payload.get("target") if payload else None
    stop_loss = payload.get("stop_loss") if payload else None
    upside = payload.get("upside") if payload else None
    downside = payload.get("downside") if payload else None
    odds = payload.get("odds") if payload else None

    # Helper for extracting claim id lists
    def _to_str_list(val: Any) -> list[str]:
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        elif isinstance(val, str) and val.strip():
            return [val.strip()]
        return []

    adopted_claim_ids = _to_str_list(payload.get("adopted_claim_ids")) if payload else []
    rejected_claim_ids = _to_str_list(payload.get("rejected_claim_ids")) if payload else []

    # ── Consistency Hard Gate Validation ──────────────────────────────────
    # Check 1: Winner vs Direction
    if winner == "bear":
        if direction.upper() in {"BUY", "BULLISH", "LEAN_BULLISH"} or direction in {"看多", "偏多", "买入", "增持"}:
            failed_checks.append(f"空头胜裁决下方向不得为看多/买入 (当前: {direction})")
    elif winner == "bull":
        if direction.upper() in {"SELL", "BEARISH", "LEAN_BEARISH"} or direction in {"看空", "偏空", "卖出", "减持"}:
            failed_checks.append(f"多头胜裁决下方向不得为看空/卖出 (当前: {direction})")

    # Check 2: Bear position percentage
    if winner == "bear" and position_pct is not None:
        try:
            pos_val = float(str(position_pct).replace("%", "").strip())
            pos_ratio = pos_val / 100.0 if pos_val > 1.0 else pos_val
            if pos_ratio > 0.20:
                failed_checks.append(f"空头胜裁决下建议仓位({pos_val}%)过高，不得高于20%")
        except (ValueError, TypeError):
            pass

    # Check 3: Bull stop loss requirement & validation
    has_manager_block = bool(extract_tagged_json(raw_response, "MANAGER_VERDICT"))
    if winner == "bull":
        if has_manager_block and (not stop_loss or str(stop_loss).strip() in {"无", "null", "None", ""}):
            failed_checks.append("多头胜裁决必须设定明确有效的止损位 (stop_loss)")
        elif stop_loss and str(stop_loss).strip() not in {"无", "null", "None", ""}:
            # If both entry and stop_loss are numbers, stop_loss must be lower than entry
            try:
                e_clean = str(entry).split("-")[0].replace("元", "").strip() if entry else ""
                s_clean = str(stop_loss).replace("元", "").strip()
                if e_clean:
                    e_num = float(e_clean)
                    s_num = float(s_clean)
                    if s_num >= e_num:
                        failed_checks.append(f"多头胜止损位({s_num})必须严格低于入场价({e_num})")
            except (ValueError, TypeError):
                pass

    # Check 4: Tie / Hold validation
    if winner == "tie" or direction.upper() in {"HOLD", "NEUTRAL"} or direction in {"中性", "观望", "持有"}:
        if position_pct is not None:
            try:
                pos_val = float(str(position_pct).replace("%", "").strip())
                pos_ratio = pos_val / 100.0 if pos_val > 1.0 else pos_val
                if pos_ratio > 0.30:
                    failed_checks.append(f"势均力敌/观望裁决下建议仓位({pos_val}%)过高，不得高于30%")
            except (ValueError, TypeError):
                pass

    # Check 5: Contradiction between prose text and verdict winner
    prose = strip_tagged_json(raw_response, "MANAGER_VERDICT")
    prose = strip_tagged_json(prose, "VERDICT")
    if "空头胜" in prose or "空方胜" in prose or "空头全面占优" in prose:
        if winner == "bull":
            failed_checks.append("正文明确判定空头胜，但机读块为多头胜(bull)，正文与机读裁决严重矛盾")
    elif "多头胜" in prose or "多方胜" in prose or "多头全面占优" in prose:
        if winner == "bear":
            failed_checks.append("正文明确判定多头胜，但机读块为空头胜(bear)，正文与机读裁决严重矛盾")

    # Check 6: Fatal hallucination in adopted claim IDs
    if claims_verification:
        fatal_cids = {
            str(item.get("claim_id"))
            for item in claims_verification
            if item.get("is_fatal") or item.get("status") == STATUS_SOURCE_UNAVAILABLE
        }
        for cid in adopted_claim_ids:
            if str(cid) in fatal_cids:
                failed_checks.append(f"裁决采纳了不可用数据源的严重幻觉 claim: {cid}")

    consistency_passed = (len(failed_checks) == 0)

    return {
        "direction": direction,
        "winner": winner,
        "reason": reason,
        "position_pct": position_pct,
        "entry": entry,
        "target": target,
        "stop_loss": stop_loss,
        "upside": upside,
        "downside": downside,
        "odds": odds,
        "adopted_claim_ids": adopted_claim_ids,
        "rejected_claim_ids": rejected_claim_ids,
        "consistency_check_passed": consistency_passed,
        "failed_checks": failed_checks,
    }

