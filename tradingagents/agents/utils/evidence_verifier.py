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

from tradingagents.agents.utils.claim_specs import (
    ClaimApplicability,
    ClaimInvalidationCondition,
    ClaimReviewContract,
    ERR_SPEC_LOOKAHEAD_PIT,
    validate_applicability,
    validate_claim_review_contract,
    validate_invalidation_condition,
)

logger = logging.getLogger(__name__)

# Status constants
STATUS_VERIFIED = "verified"
STATUS_UNSUPPORTED = "unsupported"
STATUS_CONTRADICTED = "contradicted"
STATUS_SOURCE_UNAVAILABLE = "source_unavailable"

# Decision constants
DECISION_ADOPT = "adopt"
DECISION_PARTIAL = "partial"
DECISION_REJECT = "reject"

# Coverage thresholds
MIN_COVERAGE_THRESHOLD = 0.67

# Common failed status strings
UNAVAILABLE_STATUSES = frozenset(
    {
        "failed",
        "unavailable",
        "empty",
        "error",
        "missing",
        "partial_failure",
        "not_found",
        "rejected",
        "available_unverified_as_of",
        "unverified",
        "refused",
        "future",
    }
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

# Mapping from report key to underlying provenance sources
REPORT_TO_PROVENANCE_SOURCES: dict[str, tuple[str, ...]] = {
    "fundamentals_report": ("fundamentals", "balance_sheet", "income_statement", "cashflow"),
    "market_report": ("stock_data",),
    "volume_price_report": ("stock_data",),
    "news_report": ("news",),
    "sentiment_report": ("social_archive", "social_data", "social", "social.xhs", "social.dy"),
}

# Chinese and English quantity/unit patterns
_UNIT_STR = r"(?:万股|亿股|股|亿元|万元|万|亿|%|％|pct|bp|点|元|港元|美元|倍|次|手)"

_RANGE_BOTH_UNIT_PATTERN = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*(" + _UNIT_STR + r")\s*[-~至到]\s*(\d+(?:\.\d+)?)\s*(" + _UNIT_STR + r")(?![\d.])"
)
_RANGE_END_UNIT_PATTERN = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*[-~至到]\s*(\d+(?:\.\d+)?)\s*(" + _UNIT_STR + r")(?![\d.])"
)
_RANGE_NO_UNIT_PATTERN = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*[-~至到]\s*(\d+(?:\.\d+)?)(?![\d.])"
)

_NUMBER_WITH_UNIT_RE = re.compile(
    r"(?<![\d.])([+-]?\d+(?:\.\d+)?)\s*(" + _UNIT_STR + r")?",
    re.IGNORECASE,
)

_DATE_MASK_PATTERN = re.compile(
    r"(?<![\d.])\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?![\d.])|"
    r"(?<![\d.])\d{4}年\d{1,2}月\d{1,2}日?(?![\d.])|"
    r"(?<![\d.])\d{1,2}月\d{1,2}日?(?![\d.])|"
    r"(?<![\d.])(0[1-9]|1[0-2])[-/.](0[1-9]|[12]\d|3[01])(?![\d.])|"
    r"(?<![\d.])\d{4}年?[hHqQ][1-4](?![\d.])|"
    r"(?<![\d.])[hHqQ][1-4](?![\d.])|"
    r"(?<![\d.])\d{4}年?[hH][1-2](?![\d.])|"
    r"(?<![\d.])\d{4}年度?(?![\d.])"
)

_PERIOD_QUARTER_RE = re.compile(r"(\d{4})年?[-_]?[qQ]([1-4])")
_PERIOD_SINGLE_QUARTER_RE = re.compile(r"(?<!\w)[qQ]([1-4])(?!\w)")
_PERIOD_HALF_RE = re.compile(r"(\d{4})年?[-_]?[hH]([1-2])")
_PERIOD_DATE_RE = re.compile(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?")
_PERIOD_MD_RE = re.compile(r"(\d{1,2})月(\d{1,2})日?")
_PERIOD_YEAR_RE = re.compile(r"(\d{4})年(?:度)?")


def normalize_period(text: str) -> str | None:
    """Extract and normalize period string (e.g. 2026Q2, 2026-08-24, 08-20, 2026)."""
    if not text:
        return None
    m = _PERIOD_QUARTER_RE.search(text)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"
    m = _PERIOD_HALF_RE.search(text)
    if m:
        return f"{m.group(1)}H{m.group(2)}"
    m = _PERIOD_DATE_RE.search(text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = _PERIOD_SINGLE_QUARTER_RE.search(text)
    if m:
        return f"Q{m.group(1)}"
    m = _PERIOD_MD_RE.search(text)
    if m:
        return f"{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    m = _PERIOD_YEAR_RE.search(text)
    if m:
        return m.group(1)
    return None

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
    """Normalize a value and its unit to a canonical base (e.g. 元, 股, %, or raw number).

    Returns:
        (canonical_number, canonical_unit) or None
    """
    try:
        num = float(val_str)
    except (ValueError, TypeError):
        return None

    unit = (unit_str or "").strip().lower()
    if unit in {"万股", "亿股", "股"}:
        if unit == "亿股":
            return num * 100_000_000.0, "股"
        elif unit == "万股":
            return num * 10_000.0, "股"
        return num, "股"
    elif unit in {"亿元", "亿"}:
        return num * 100_000_000.0, "元"
    elif unit in {"万元", "万"}:
        return num * 10_000.0, "元"
    elif unit in {"元", "港元", "美元"}:
        return num, "元"
    elif unit in {"%", "％", "pct"}:
        return num, "%"
    elif unit == "bp":
        return num / 100.0, "%"
    else:
        return num, "raw"


def _extract_numbers_and_units(text: str) -> list[tuple[float, str, str]]:
    """Extract list of (normalized_val, canonical_unit, raw_substr) from text."""
    if not text:
        return []
    # 1. Mask dates to prevent temporal anchors from polluting financial metric matching
    cleaned = _DATE_MASK_PATTERN.sub(" ", text)
    # 2. Expand ranges so the first number inherits trailing unit (e.g. 450~470亿元 -> 450亿元 ~ 470亿元)
    cleaned = _RANGE_BOTH_UNIT_PATTERN.sub(r"\1\2 ~ \3\4", cleaned)
    cleaned = _RANGE_END_UNIT_PATTERN.sub(r"\1\3 ~ \2\3", cleaned)
    cleaned = _RANGE_NO_UNIT_PATTERN.sub(r"\1 ~ \2", cleaned)

    results = []
    for match in _NUMBER_WITH_UNIT_RE.finditer(cleaned):
        val_str = match.group(1)
        unit_str = match.group(2) or ""
        norm = normalize_numeric_value(val_str, unit_str)
        if norm is not None:
            results.append((norm[0], norm[1], match.group(0).strip()))
    return results


def _is_num_match(
    ev_num: float,
    ev_unit: str,
    l_num: float,
    l_unit: str,
    rel_tol: float = 0.02,
    abs_tol: float = 0.05,
) -> bool:
    """Check if evidence numeric value matches report numeric value within tolerance."""
    unit_compatible = (
        (ev_unit == l_unit)
        or (ev_unit == "raw" and l_unit == "%")
        or (ev_unit == "%" and l_unit == "raw")
    )
    if not unit_compatible:
        return False
    return (
        math.isclose(ev_num, l_num, rel_tol=rel_tol, abs_tol=abs_tol)
        or math.isclose(abs(ev_num), abs(l_num), rel_tol=rel_tol, abs_tol=abs_tol)
    )


# Expanded domain metric and context keywords
_METRIC_KEYWORDS = [
    # Valuation & Financial metrics
    "pe", "pb", "ps", "roe", "roa", "eps", "m2", "cpi", "ppi", "gdp", "lpr", "shibor",
    "营收", "收入", "利润", "净利润", "净利", "毛利", "毛利率", "净利率", "扣非", "负债率", "资产负债率",
    "现金流", "自由现金流", "fcf", "资本开支", "capex", "研发", "费用", "费用率", "应收账款", "存货",
    "周转率", "商誉", "减值", "利用率", "产能", "cr3", "价格战", "库存", "去库", "补库", "订单",
    "估值", "分红", "股息", "股息率", "回购", "增持", "减持", "重组", "定增", "质押", "现金", "货币资金", "安全垫", "安全边际",
    "底线", "压力测试", "敏感性", "弹性",
    # Capital & Flow metrics
    "主力", "净流入", "净流出", "流出", "流入", "超大单", "大单", "中单", "小单", "全单", "龙虎榜",
    "北向", "北向资金", "机构", "外资", "游资", "散户", "两融", "融资", "融券", "大宗交易", "筹码",
    "吸筹", "出货", "洗盘", "托底", "增仓", "减仓", "持股", "席位",
    # Technical & Volume/Price metrics
    "成交量", "成交额", "换手率", "换手", "量比", "地量", "天量", "放量", "缩量", "量价", "均线",
    "ema", "sma", "vwma", "macd", "rsi", "boll", "布林", "kdj", "atr", "dif", "dea",
    "支撑", "阻力", "突破", "破位", "双底", "筑底", "死叉", "金叉", "超买", "超卖", "多头", "空头",
    "冲高", "回落", "震荡", "趋势", "动量", "k线", "收盘", "开盘", "最高", "最低", "日内", "位置",
    "高点", "低点", "低位", "高位", "实体", "上影", "下影", "涨停", "跌停", "连板", "炸板",
    # Macro & Industry metrics
    "降息", "降准", "加息", "利率", "美债", "汇率", "油价", "原油", "黄金", "铜价", "lme", "大宗商品",
    "关税", "补贴", "以旧换新", "外需", "内需", "财政", "赤字", "信贷", "流动性", "长协", "转嫁",
    "水库", "来水", "发电量", "偏枯", "偏丰", "蓄能", "电量", "纳斯达克", "生物科技", "指数",
    # Pharma & Sector specific
    "管线", "临床", "获批", "授权", "license-out", "医保", "集采", "原料药", "仿制药", "创新药", "adc", "fda",
    # Sentiment & General
    "风险偏好", "进攻", "防守", "避险", "虹吸", "抽水", "情绪", "舆情", "预期差", "公告", "中报", "年报", "季报",
    # Entities frequently referenced
    "大金", "惠而浦", "乌东德", "三峡", "美的", "恒瑞", "礼来"
]


def _extract_metric_keywords(text: str) -> list[str]:
    """Extract financial and market metric keywords from a string."""
    found = []
    text_lower = text.lower()
    for kw in _METRIC_KEYWORDS:
        if kw in text_lower:
            found.append(kw)
    return found


_STRICT_METRICS = {
    "营收", "毛利率", "毛利", "净利率", "净利润", "成本", "应收账款", "存货", "现金流",
    "资产负债率", "roe", "roa", "eps", "pe", "pb", "ps", "股息率", "换手率", "量比",
    "主力", "超大单", "大单", "两融", "概率", "预期收益", "降息", "降准", "关税",
    "lpr", "cpi", "ppi", "m2", "gdp"
}

_METRIC_CANONICAL_MAP: dict[str, str] = {
    # 营收 / 收入
    "营业收入": "营收", "主营业务收入": "营收", "单季营收": "营收", "海外营收": "营收", "营收": "营收", "收入": "营收", "revenue": "营收",
    # 毛利率 / 毛利
    "综合毛利率": "毛利率", "销售毛利率": "毛利率", "毛利率": "毛利率", "毛利": "毛利",
    # 净利率
    "归母净利率": "净利率", "扣非净利率": "净利率", "净利率": "净利率", "净利润率": "净利率",
    # 净利润
    "归母净利润": "净利润", "归母净利": "净利润", "扣非净利润": "净利润", "扣非净利": "净利润", "净利润": "净利润", "净利": "净利润",
    # 成本
    "营业成本": "成本", "生产成本": "成本", "成本": "成本",
    # 应收账款
    "应收账款": "应收账款", "应收款项": "应收账款", "应收账期": "应收账款",
    # 存货
    "存货": "存货", "库存": "存货",
    # 现金流
    "经营活动产生的现金流量净额": "现金流", "经营性现金流": "现金流", "经营现金流": "现金流", "自由现金流": "现金流", "现金流": "现金流", "fcf": "现金流",
    # 资产负债率
    "资产负债率": "资产负债率", "负债率": "资产负债率",
    # 估值 / 收益率
    "加权净资产收益率": "roe", "净资产收益率": "roe", "roe": "roe",
    "总资产收益率": "roa", "roa": "roa",
    "每股收益": "eps", "eps": "eps",
    "市盈率": "pe", "pe": "pe",
    "市净率": "pb", "pb": "pb",
    "市销率": "ps", "ps": "ps",
    "分红率": "股息率", "股息率": "股息率", "分红": "股息率", "股息": "股息率",
    # 交易 / 资金
    "换手率": "换手率", "换手": "换手率",
    "量比": "量比",
    "成交量": "成交量", "成交额": "成交额",
    "主力净流入": "主力", "主力净流出": "主力", "主力": "主力",
    "超大单净流入": "超大单", "超大单净流出": "超大单", "超大单": "超大单",
    "大单净流入": "大单", "大单净流出": "大单", "大单": "大单",
    "散户小单净买入": "散户小单", "散户小单": "散户小单",
    "融资净偿还": "两融", "融资净买入": "两融", "融券净卖出": "两融", "两融": "两融", "融资": "两融", "融券": "两融",
    # 宏观 / 情景
    "情景概率": "概率", "概率": "概率", "情景": "概率",
    "预期收益": "预期收益",
    "降息": "降息", "降准": "降准", "关税": "关税",
    "lpr": "lpr", "cpi": "cpi", "ppi": "ppi", "m2": "m2", "gdp": "gdp",
}


def _canonicalize_metric(raw_metric: str | None, unit: str) -> str | None:
    if not raw_metric:
        return None
    m = _METRIC_CANONICAL_MAP.get(raw_metric, raw_metric)
    if unit == "%":
        if m == "毛利":
            return "毛利率"
        if m in ("净利", "净利润"):
            return "净利率"
    elif unit in ("元", "股"):
        if m == "毛利率":
            return "毛利"
        if m == "净利率":
            return "净利润"
    return m


_SORTED_METRIC_MAP_KEYS = sorted(_METRIC_CANONICAL_MAP.keys(), key=len, reverse=True)


class BoundNumber:
    __slots__ = ("val", "unit", "raw", "metric", "period", "raw_metric")

    def __init__(self, val: float, unit: str, raw: str, metric: str | None, period: str | None, raw_metric: str | None):
        self.val = val
        self.unit = unit
        self.raw = raw
        self.metric = metric
        self.period = period
        self.raw_metric = raw_metric

    def __repr__(self) -> str:
        return f"BoundNumber({self.raw!r}, val={self.val}, unit={self.unit!r}, metric={self.metric!r}, period={self.period!r})"


def extract_bound_numbers(text: str, default_period: str | None = None) -> list[BoundNumber]:
    """Extract numbers from text and bind each number to its closest specific metric and period."""
    if not text:
        return []
    period = normalize_period(text) or default_period
    cleaned = _DATE_MASK_PATTERN.sub(lambda m: " " * len(m.group(0)), text)
    text_lower = text.lower()
    metric_spans = []
    for kw in _SORTED_METRIC_MAP_KEYS:
        start = 0
        while True:
            idx = text_lower.find(kw.lower(), start)
            if idx == -1:
                break
            metric_spans.append((idx, idx + len(kw), kw))
            start = idx + len(kw)
    metric_spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    filtered_spans = []
    for s in metric_spans:
        if not filtered_spans or s[0] >= filtered_spans[-1][1]:
            filtered_spans.append(s)

    matches = list(_NUMBER_WITH_UNIT_RE.finditer(cleaned))
    res = []
    for i, m in enumerate(matches):
        val_str = m.group(1)
        unit_str = m.group(2) or ""
        if not unit_str and i + 1 < len(matches):
            between = cleaned[m.end():matches[i+1].start()]
            if re.fullmatch(r"\s*[-~至到]\s*", between) and matches[i+1].group(2):
                unit_str = matches[i+1].group(2)
        norm = normalize_numeric_value(val_str, unit_str)
        if norm is None:
            continue
        val, unit = norm
        raw = m.group(0).strip()
        n_start, n_end = m.span()

        closest_prec = None
        min_prec_dist = 9999
        for m_start, m_end, m_raw in filtered_spans:
            if m_end <= n_start and (n_start - m_end) < 40:
                intervening = text[m_end:n_start]
                if "；" in intervening or ";" in intervening or "。" in intervening:
                    continue
                dist = n_start - m_end
                if dist < min_prec_dist:
                    min_prec_dist = dist
                    closest_prec = m_raw

        closest_succ = None
        min_succ_dist = 9999
        for m_start, m_end, m_raw in filtered_spans:
            if m_start >= n_end and (m_start - n_end) < 15:
                dist = m_start - n_end
                if dist < min_succ_dist:
                    min_succ_dist = dist
                    closest_succ = m_raw

        raw_metric = closest_prec or closest_succ
        metric = _canonicalize_metric(raw_metric, unit)
        res.append(BoundNumber(val, unit, raw, metric, period, raw_metric))
    return res


_COMPOUND_SPLIT_RE = re.compile(
    r"[;；]|(?<!\d)[,，](?!\d)|(?<=[%\d元股点])\s*(?:且|并且|严重背离|背离|同时)\s*"
)


def split_compound_evidence(text: str) -> list[str]:
    """Split a compound evidence sentence into atomic statements."""
    if not text:
        return []
    parts = [p.strip() for p in _COMPOUND_SPLIT_RE.split(text) if p.strip()]
    return parts if parts else [text.strip()]


def _is_bound_num_match(
    ev_bn: BoundNumber,
    l_bn: BoundNumber,
    rel_tol: float = 0.02,
    abs_tol: float = 0.05,
) -> bool:
    """Check if evidence bound number matches line bound number without metric cross-binding."""
    if not _is_num_match(ev_bn.val, ev_bn.unit, l_bn.val, l_bn.unit, rel_tol, abs_tol):
        return False
    ev_strict = ev_bn.metric if ev_bn.metric in _STRICT_METRICS else None
    l_strict = l_bn.metric if l_bn.metric in _STRICT_METRICS else None
    if ev_strict and l_strict and ev_strict != l_strict:
        return False
    return True


def _is_bound_num_contradicted(
    ev_bn: BoundNumber,
    l_bn: BoundNumber,
) -> bool:
    """Check if evidence bound number contradicts line bound number.

    Requires matching metric name, unit, and period (for percentages) before judging conflict.
    """
    ev_strict = ev_bn.metric if ev_bn.metric in _STRICT_METRICS else None
    l_strict = l_bn.metric if l_bn.metric in _STRICT_METRICS else None
    if not ev_strict or not l_strict or ev_strict != l_strict:
        return False
    if ev_bn.unit != l_bn.unit or ev_bn.unit not in {"%", "元", "股"}:
        return False
    if ev_bn.unit == "%":
        if ev_bn.period != l_bn.period:
            return False
    else:
        if ev_bn.period and l_bn.period and ev_bn.period != l_bn.period:
            return False
    diff_pct = abs(abs(ev_bn.val) - abs(l_bn.val)) / (abs(l_bn.val) + 1e-9)
    return diff_pct > 0.05


class EvidenceFactualTruthEvaluator:
    """Evaluates evidence statements against 7 analyst reports and market context."""

    def __init__(self, relative_tolerance: float = 0.02, absolute_tolerance: float = 0.05):
        self.rel_tol = relative_tolerance
        self.abs_tol = absolute_tolerance

    def _extract_unavailable_sources(
        self,
        market_data_context: Mapping[str, Any] | None,
        social_data_context: Mapping[str, Any] | None = None,
    ) -> set[str]:
        unavailable = set()
        if isinstance(market_data_context, Mapping):
            # Check data_failure_ledger
            ledger = market_data_context.get("data_failure_ledger")
            if isinstance(ledger, list):
                for entry in ledger:
                    if isinstance(entry, dict):
                        src = str(entry.get("source", "")).strip().lower()
                        status = str(entry.get("status", "")).strip().lower()
                        prov_status = str(entry.get("provenance_status", "")).strip().lower()
                        if src and (not status or status in UNAVAILABLE_STATUSES or prov_status in {"unverified", "refused", "future"}):
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
                        prov_status = str(info.get("provenance_status", "")).strip().lower()
                        if status in UNAVAILABLE_STATUSES or prov_status in {"unverified", "refused", "future"}:
                            unavailable.add(src.strip().lower())

            # Check data_gaps
            gaps = market_data_context.get("data_gaps")
            if isinstance(gaps, list):
                for g in gaps:
                    if isinstance(g, str):
                        unavailable.add(g.strip().lower())

        if isinstance(social_data_context, Mapping):
            social_ledger = social_data_context.get("data_failure_ledger")
            if isinstance(social_ledger, list):
                for entry in social_ledger:
                    if isinstance(entry, dict):
                        src = str(entry.get("source", "")).strip().lower()
                        status = str(entry.get("status", "")).strip().lower()
                        prov_status = str(entry.get("provenance_status", "")).strip().lower()
                        if src and (not status or status in UNAVAILABLE_STATUSES or prov_status in {"unverified", "refused", "future"}):
                            unavailable.add(src)
                            name = str(entry.get("name", "")).strip().lower()
                            if name:
                                unavailable.add(name)

            social_prov = social_data_context.get("source_provenance")
            if isinstance(social_prov, dict):
                for src, info in social_prov.items():
                    if isinstance(info, dict):
                        status = str(info.get("status", "")).strip().lower()
                        prov_status = str(info.get("provenance_status", "")).strip().lower()
                        if status in UNAVAILABLE_STATUSES or prov_status in {"unverified", "refused", "future"}:
                            unavailable.add(src.strip().lower())

            social_status = str(social_data_context.get("status", "")).strip().lower()
            if social_status in UNAVAILABLE_STATUSES:
                unavailable.add("social_archive")
                unavailable.add("social_data")
                unavailable.add("social")

        return unavailable

    def _check_source_unavailable(
        self, raw_evidence: str, unavailable_sources: set[str]
    ) -> tuple[bool, str]:
        if not unavailable_sources or not raw_evidence:
            return False, ""
        evidence_lower = raw_evidence.lower()
        sorted_sources = sorted(unavailable_sources, key=len, reverse=True)
        for src in sorted_sources:
            if src and (src in evidence_lower or evidence_lower in src):
                return True, src
        return False, ""

    def _is_report_unavailable(
        self, role_key: str, unavailable_sources: set[str]
    ) -> bool:
        """Check if a report's underlying provenance sources are marked unavailable/unverified."""
        if not unavailable_sources:
            return False
        mapped_sources = (
            REPORT_TO_PROVENANCE_SOURCES.get(role_key)
            or REPORT_TO_PROVENANCE_SOURCES.get(f"{role_key}_report", ())
        )
        return any(src in unavailable_sources for src in mapped_sources)

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
        social_data_context: Mapping[str, Any] | None = None,
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
        unavailable_sources = self._extract_unavailable_sources(market_data_context, social_data_context)
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

        # 1.1 Check if evidence asserts a directional social sentiment score when direction is disallowed or social is insufficient/empty
        if isinstance(social_data_context, Mapping):
            social_status = str(social_data_context.get("status", "")).strip().lower()
            social_mode = str(social_data_context.get("mode", "")).strip().lower()
            dir_allowed = bool(social_data_context.get("direction_allowed", False))
            bundle = social_data_context.get("bundle") if isinstance(social_data_context.get("bundle"), dict) else {}
            sent_label = bundle.get("social_sentiment", {}).get("label") if isinstance(bundle.get("social_sentiment"), dict) else None

            if (
                not dir_allowed
                or social_mode in ("disabled", "shadow")
                or social_status in ("insufficient", "empty", "not_applicable", "failed", "timeout", "refused")
                or sent_label == "insufficient"
            ):
                raw_lower = raw_text.lower()
                is_social_mention = any(kw in raw_lower for kw in ("社交", "舆情", "散户", "小红书", "抖音", "social"))
                has_score_or_dir = any(kw in raw_lower for kw in ("得分", "分数", "score", "看多", "看空", "多头", "空头", "0.", "极度", "狂热", "高涨", "情绪分"))
                if is_social_mention and has_score_or_dir:
                    return {
                        "raw": raw_text,
                        "claim_id": claim_id,
                        "matched_role": None,
                        "matched_source": None,
                        "status": STATUS_UNSUPPORTED,
                        "is_fatal": False,
                        "details": "社交数据处于不可用/不足态或 direction_allowed=false，禁止将社交情绪分数或多空方向作为已验证事实",
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
        ev_keywords = _extract_metric_keywords(raw_text)

        # 3.1 Exact substring match in any report
        for role_key in SEVEN_REPORT_KEYS:
            if self._is_report_unavailable(role_key, unavailable_sources):
                continue
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

        parent_period = normalize_period(raw_text)
        atomic_clauses = split_compound_evidence(raw_text)
        all_ev_bns: list[BoundNumber] = []
        for clause in atomic_clauses:
            clause_period = normalize_period(clause) or parent_period
            c_bns = extract_bound_numbers(clause, default_period=clause_period)
            all_ev_bns.extend(c_bns)

        # 3.2 Single-line full match in reports
        for role_key in SEVEN_REPORT_KEYS:
            if self._is_report_unavailable(role_key, unavailable_sources):
                continue
            report_body = str(seven_reports.get(role_key, "") or "")
            if not report_body.strip():
                continue

            for line in report_body.splitlines():
                line_text = line.strip()
                if not line_text:
                    continue

                line_keywords = _extract_metric_keywords(line_text)
                ev_canon = {_METRIC_CANONICAL_MAP.get(k, k) for k in ev_keywords}
                line_canon = {_METRIC_CANONICAL_MAP.get(k, k) for k in line_keywords}
                common_kw = ev_canon.intersection(line_canon)
                line_bns = extract_bound_numbers(line_text)

                if not line_bns and not all_ev_bns:
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

                # If numbers exist, check single-line full value match with metric binding
                if all_ev_bns:
                    matched_all_numbers = True
                    found_match = False
                    for ev_bn in all_ev_bns:
                        num_found_in_line = False
                        for l_bn in line_bns:
                            if _is_bound_num_match(ev_bn, l_bn, self.rel_tol, self.abs_tol):
                                num_found_in_line = True
                                found_match = True
                                break
                        if not num_found_in_line:
                            matched_all_numbers = False
                            break

                    if found_match and matched_all_numbers:
                        if not ev_keywords or len(common_kw) >= 1:
                            return {
                                "raw": raw_text,
                                "claim_id": claim_id,
                                "matched_role": role_key,
                                "matched_source": role_key.replace("_report", ""),
                                "status": STATUS_VERIFIED,
                                "is_fatal": False,
                                "details": f"在 {role_key} 中验证数值与关键词匹配",
                            }

        # 3.3 Multi-line / atomic aggregation mode (when single line did not match all numbers)
        if all_ev_bns:
            num_hits_by_report: dict[str, set[int]] = {}
            all_hit_num_indices: set[int] = set()

            for num_idx, ev_bn in enumerate(all_ev_bns):
                for role_key in SEVEN_REPORT_KEYS:
                    if self._is_report_unavailable(role_key, unavailable_sources):
                        continue
                    report_body = str(seven_reports.get(role_key, "") or "")
                    if not report_body.strip():
                        continue
                    matched_in_report = False
                    for line in report_body.splitlines():
                        line_text = line.strip()
                        if not line_text:
                            continue
                        line_keywords = _extract_metric_keywords(line_text)
                        ev_canon = {_METRIC_CANONICAL_MAP.get(k, k) for k in ev_keywords}
                        line_canon = {_METRIC_CANONICAL_MAP.get(k, k) for k in line_keywords}
                        common_kw = ev_canon.intersection(line_canon)
                        # Each hit line must share >= 1 canonical keyword with the evidence sentence
                        if ev_keywords and not common_kw:
                            continue
                        line_bns = extract_bound_numbers(line_text)
                        for l_bn in line_bns:
                            if _is_bound_num_match(ev_bn, l_bn, self.rel_tol, self.abs_tol):
                                num_hits_by_report.setdefault(role_key, set()).add(num_idx)
                                all_hit_num_indices.add(num_idx)
                                matched_in_report = True
                                break
                        if matched_in_report:
                            break

            total_nums = len(all_ev_bns)
            # Check single-report multi-line aggregation first
            for role_key in SEVEN_REPORT_KEYS:
                if self._is_report_unavailable(role_key, unavailable_sources):
                    continue
                hit_set = num_hits_by_report.get(role_key, set())
                if len(hit_set) == total_nums:
                    return {
                        "raw": raw_text,
                        "claim_id": claim_id,
                        "matched_role": role_key,
                        "matched_source": role_key.replace("_report", ""),
                        "status": STATUS_VERIFIED,
                        "is_fatal": False,
                        "details": f"在 {role_key} 中通过多行聚合验证数值与关键词匹配 (multi_line_match)",
                    }

            # Check cross-report multi-line aggregation
            if len(all_hit_num_indices) == total_nums:
                matched_roles = [
                    r for r in SEVEN_REPORT_KEYS
                    if r in num_hits_by_report and num_hits_by_report[r]
                    and not self._is_report_unavailable(r, unavailable_sources)
                ]
                if matched_roles:
                    return {
                        "raw": raw_text,
                        "claim_id": claim_id,
                        "matched_role": ",".join(matched_roles) if len(matched_roles) > 1 else matched_roles[0],
                        "matched_source": ",".join(r.replace("_report", "") for r in matched_roles),
                        "status": STATUS_VERIFIED,
                        "is_fatal": False,
                        "details": f"在 {','.join(matched_roles)} 中跨报告多行聚合验证数值与关键词匹配 (multi_line_match)",
                    }

        # 3.4 Contradiction check across reports when evidence is not verified
        contradicted_candidate = None
        if all_ev_bns:
            for role_key in SEVEN_REPORT_KEYS:
                if self._is_report_unavailable(role_key, unavailable_sources):
                    continue
                report_body = str(seven_reports.get(role_key, "") or "")
                if not report_body.strip():
                    continue
                for line in report_body.splitlines():
                    line_text = line.strip()
                    if not line_text:
                        continue
                    line_bns = extract_bound_numbers(line_text)
                    for num_idx, ev_bn in enumerate(all_ev_bns):
                        if num_idx in all_hit_num_indices:
                            continue
                        if not ev_bn.metric:
                            continue
                        for l_bn in line_bns:
                            if not l_bn.metric:
                                continue
                            if _is_bound_num_contradicted(ev_bn, l_bn):
                                contradicted_candidate = (
                                    role_key,
                                    f"在 {role_key} 中指标 '{ev_bn.metric}' 数据冲突: 证据声称 {ev_bn.raw}，报告记录为 {l_bn.raw}",
                                )
                                break
                        if contradicted_candidate:
                            break
                    if contradicted_candidate:
                        break
                if contradicted_candidate:
                    break

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
            # Check quotes, indicators, fund_flow_evidence (excluding metadata and unavailable sources)
            for k, v in market_data_context.items():
                if k in {
                    "source_provenance",
                    "data_failure_ledger",
                    "data_gaps",
                    "analysis_baseline_date",
                    "trade_date",
                    "data_as_of",
                }:
                    continue
                if str(k).strip().lower() in unavailable_sources:
                    continue
                v_str = str(v or "")
                if raw_text in v_str:
                    return {
                        "raw": raw_text,
                        "claim_id": claim_id,
                        "matched_role": "market_data_context",
                        "matched_source": str(k),
                        "status": STATUS_VERIFIED,
                        "is_fatal": False,
                        "details": f"在 market_data_context[{k}] 中找到匹配数据",
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
        social_data_context: Mapping[str, Any] | None = None,
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
                    social_data_context=social_data_context,
                )
                results.append(ver_res)

        return results

    def evaluate_challenges(
        self,
        challenges: Sequence[Mapping[str, Any]],
        seven_reports: Mapping[str, str],
        market_data_context: Mapping[str, Any] | None = None,
        analysis_baseline_date: str | None = None,
        social_data_context: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate all evidence items across challenges and compute challenge-level evidence status."""
        results: list[dict[str, Any]] = []
        for ch in challenges:
            chid = str(ch.get("challenge_id", "")).strip() or None
            target_id = str(ch.get("target_claim_id", "")).strip()
            speaker = str(ch.get("speaker") or ch.get("speaker_key") or "")
            speaker_key = str(ch.get("speaker_key") or ch.get("speaker") or "")
            severity = str(ch.get("severity", "major")).strip().lower()
            ev_list = ch.get("evidence") or []
            if isinstance(ev_list, str):
                ev_list = [ev_list]

            ver_items: list[dict[str, Any]] = []
            for ev in ev_list:
                ev_str = str(ev).strip()
                if not ev_str:
                    continue
                ver_res = self.evaluate_single_evidence(
                    raw_evidence=ev_str,
                    seven_reports=seven_reports,
                    market_data_context=market_data_context,
                    analysis_baseline_date=analysis_baseline_date,
                    claim_id=chid,
                    social_data_context=social_data_context,
                )
                ver_items.append(ver_res)

            verified_items = [v for v in ver_items if v.get("status") == STATUS_VERIFIED]
            contradicted_items = [v for v in ver_items if v.get("status") == STATUS_CONTRADICTED]
            unavail_items = [
                v for v in ver_items
                if v.get("status") == STATUS_SOURCE_UNAVAILABLE or v.get("is_fatal")
            ]
            unsupported_items = [v for v in ver_items if v.get("status") == STATUS_UNSUPPORTED]

            total_count = len(ver_items)
            verified_count = len(verified_items)
            contradicted_count = len(contradicted_items)
            unavail_count = len(unavail_items)
            unsupported_count = len(unsupported_items)

            if contradicted_count > 0 or unavail_count > 0:
                evidence_status = "contradicted"
            elif verified_count == total_count and total_count > 0:
                evidence_status = "verified"
            else:
                evidence_status = "unsupported"

            ch_res = {
                "challenge_id": chid,
                "target_claim_id": target_id,
                "speaker": speaker,
                "speaker_key": speaker_key,
                "severity": severity,
                "evidence_status": evidence_status,
                "counts": {
                    "total": total_count,
                    "verified": verified_count,
                    "unsupported": unsupported_count,
                    "contradicted": contradicted_count,
                    "source_unavailable": unavail_count,
                },
                "verification_items": ver_items,
            }
            results.append(ch_res)

        return results

    def aggregate_claim_evidence(
        self,
        claims: Sequence[Mapping[str, Any]] | None = None,
        claims_verification: Sequence[Mapping[str, Any]] | None = None,
        *,
        analysis_baseline_date: str | None = None,
        expected_symbol: str | None = None,
        market_data_context: Mapping[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Aggregate evidence verification by claim and compute coverage ratio and adoption decisions."""
        return aggregate_claim_evidence(
            claims=claims,
            claims_verification=claims_verification,
            analysis_baseline_date=analysis_baseline_date,
            expected_symbol=expected_symbol,
            market_data_context=market_data_context,
        )


_OBSERVATION_HYPOTHESIS_PATTERN = re.compile(
    r"(?:^|[\[\(【（])\s*(?:观察|假设|假说|推测|情景假设|hypothesis|observation)\s*(?:[\]\)】）]|$|:|\s)",
    re.IGNORECASE,
)
_OBSERVATION_HYPOTHESIS_TYPES = frozenset(
    {
        "observation",
        "hypothesis",
        "hypothetical",
        "assumption",
        "observational",
        "conjecture",
        "scenario",
    }
)


def is_observation_or_hypothesis_claim(claim: Mapping[str, Any] | Any) -> bool:
    """Check if a claim is an observation or hypothesis rather than a verified fact assertion (E-03c)."""
    if not isinstance(claim, Mapping):
        if isinstance(claim, str):
            return bool(_OBSERVATION_HYPOTHESIS_PATTERN.search(claim))
        return False
    if bool(claim.get("is_observation")) or bool(claim.get("is_hypothesis")):
        return True
    for k in ("claim_type", "type", "epistemic_status", "claim_nature", "nature"):
        val = str(claim.get(k) or "").strip().lower()
        if val in _OBSERVATION_HYPOTHESIS_TYPES:
            return True
    claim_text = str(claim.get("claim") or "").strip()
    if _OBSERVATION_HYPOTHESIS_PATTERN.search(claim_text):
        return True
    if claim_text.startswith(("观察：", "假设：", "推测：", "情景：", "Observation:", "Hypothesis:")):
        return True
    status_val = str(claim.get("status") or "").strip().lower()
    if status_val in {"observation", "hypothesis"}:
        return True
    return False


def _check_invalidation_condition_triggered(
    cond: Mapping[str, Any],
    market_data_context: Mapping[str, Any],
) -> tuple[bool, str]:
    """Check whether a machine-checkable invalidation condition has triggered (E-03c)."""
    metric = str(cond.get("metric") or "").strip().lower()
    op = str(cond.get("operator") or "").strip()
    threshold = cond.get("threshold")
    if threshold is None or not op:
        return False, ""
    try:
        th_val = float(threshold)
    except (ValueError, TypeError):
        return False, ""

    actual_val = None
    for k in (metric, metric.replace(" ", "_"), f"current_{metric}", f"last_{metric}"):
        if k in market_data_context:
            v = market_data_context[k]
            if v is not None and not isinstance(v, bool):
                try:
                    actual_val = float(v)
                    break
                except (ValueError, TypeError):
                    pass
    if actual_val is None:
        for sub_key in ("indicators", "quotes", "daily", "realtime"):
            sub = market_data_context.get(sub_key)
            if isinstance(sub, Mapping) and metric in sub:
                v = sub[metric]
                if v is not None and not isinstance(v, bool):
                    try:
                        actual_val = float(v)
                        break
                    except (ValueError, TypeError):
                        pass

    if actual_val is None:
        return False, ""

    triggered = False
    if op == "<":
        triggered = (actual_val < th_val)
    elif op == "<=":
        triggered = (actual_val <= th_val)
    elif op == ">":
        triggered = (actual_val > th_val)
    elif op == ">=":
        triggered = (actual_val >= th_val)
    elif op == "==":
        triggered = math.isclose(actual_val, th_val, abs_tol=1e-5)
    elif op == "!=":
        triggered = not math.isclose(actual_val, th_val, abs_tol=1e-5)

    if triggered:
        cid = cond.get("condition_id", "cond")
        return True, f"失效条件 [{cid}] 已触发: 实际指标 {metric}={actual_val} 满足 {op} 阈值 {th_val}"
    return False, ""


def aggregate_claim_evidence(
    claims: Sequence[Mapping[str, Any]] | None = None,
    claims_verification: Sequence[Mapping[str, Any]] | None = None,
    *,
    analysis_baseline_date: str | None = None,
    expected_symbol: str | None = None,
    market_data_context: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Aggregate evidence verification results by claim_id and evaluate deterministic decisions.

    Returns:
        A dict mapping claim_id -> {
            "claim_id": str,
            "speaker": str,
            "speaker_key": str,
            "stance": str,
            "claim": str,
            "counts": {
                "total": int,
                "verified": int,
                "unsupported": int,
                "contradicted": int,
                "source_unavailable": int,
            },
            "coverage": float,
            "decision": "adopt" | "partial" | "reject",
            "reason": str,
            "verified_evidence": list[str],
            "unsupported_evidence": list[str],
            "contradicted_evidence": list[str],
            "source_unavailable_evidence": list[str],
            "excluded_evidence": list[str],
            "is_observation_or_hypothesis": bool,
            "applicability": dict | None,
            "invalidation_conditions": list[dict],
            "pit_failed": bool,
        }
    """
    claims_list = list(claims or [])
    ver_list = list(claims_verification or [])

    effective_baseline_date = analysis_baseline_date
    if not effective_baseline_date and isinstance(market_data_context, Mapping):
        effective_baseline_date = str(
            market_data_context.get("analysis_baseline_date")
            or market_data_context.get("data_as_of")
            or market_data_context.get("trade_date")
            or ""
        ).strip() or None

    effective_expected_symbol = expected_symbol
    if not effective_expected_symbol and isinstance(market_data_context, Mapping):
        effective_expected_symbol = str(
            market_data_context.get("symbol")
            or market_data_context.get("ticker")
            or ""
        ).strip() or None

    # Map verification items by claim_id
    ver_by_cid: dict[str, list[Mapping[str, Any]]] = {}
    for item in ver_list:
        cid = str(item.get("claim_id", "") or "").strip()
        if cid:
            ver_by_cid.setdefault(cid, []).append(item)

    # All known claim objects
    known_claims: dict[str, Mapping[str, Any]] = {}
    for c in claims_list:
        cid = str(c.get("claim_id", "") or "").strip()
        if cid:
            known_claims[cid] = c

    # Union of all CIDs preserving order
    all_cids = list(known_claims.keys())
    for cid in ver_by_cid:
        if cid not in known_claims:
            all_cids.append(cid)

    summary_map: dict[str, dict[str, Any]] = {}
    for cid in all_cids:
        claim_obj = known_claims.get(cid, {})
        claim_ver_items = ver_by_cid.get(cid, [])

        is_obs_hypo = is_observation_or_hypothesis_claim(claim_obj)

        # 1. Applicability validation
        norm_applicability: dict[str, Any] | None = None
        applicability_pit_failed = False
        applicability_failed = False
        app_error_msg = ""
        if "applicability" in claim_obj and claim_obj["applicability"] is not None:
            app_val = claim_obj["applicability"]
            if isinstance(app_val, ClaimApplicability):
                app_val = app_val.to_dict()
            ok_app, err_app, norm_app = validate_applicability(
                app_val,
                expected_symbol=effective_expected_symbol,
                current_trade_date=effective_baseline_date,
            )
            if not ok_app:
                if err_app == ERR_SPEC_LOOKAHEAD_PIT:
                    applicability_pit_failed = True
                    app_error_msg = f"适用性规格存在前视偏差/PIT失败 (pit_date={app_val.get('pit_date')} > baseline={effective_baseline_date})"
                else:
                    applicability_failed = True
                    app_error_msg = f"适用性规格校验失败: {err_app}"
            else:
                norm_applicability = norm_app

        # 2. Invalidation conditions validation
        norm_conditions: list[dict[str, Any]] = []
        conditions_pit_failed = False
        conditions_failed = False
        conditions_triggered = False
        cond_error_msgs: list[str] = []
        if "invalidation_conditions" in claim_obj and claim_obj["invalidation_conditions"] is not None:
            raw_conds = claim_obj["invalidation_conditions"]
            if isinstance(raw_conds, (list, tuple, Sequence)) and not isinstance(raw_conds, (str, bytes, Mapping)):
                cond_baseline = effective_baseline_date or (norm_applicability.get("pit_date") if norm_applicability else None)
                for idx, cond in enumerate(raw_conds, start=1):
                    if isinstance(cond, ClaimInvalidationCondition):
                        cond = cond.to_dict()
                    ok_c, err_c, norm_c = validate_invalidation_condition(
                        cond,
                        index=idx,
                        current_trade_date=cond_baseline,
                    )
                    if not ok_c:
                        if err_c == ERR_SPEC_LOOKAHEAD_PIT:
                            conditions_pit_failed = True
                            cond_error_msgs.append(f"失效条件 [{cond.get('condition_id', idx)}] 存在前视偏差/PIT失败 ({err_c})")
                        else:
                            conditions_failed = True
                            cond_error_msgs.append(f"失效条件 [{cond.get('condition_id', idx)}] 规格校验失败 ({err_c})")
                    else:
                        norm_conditions.append(norm_c)
                        if market_data_context and isinstance(market_data_context, Mapping):
                            is_trig, trig_msg = _check_invalidation_condition_triggered(norm_c, market_data_context)
                            if is_trig:
                                conditions_triggered = True
                                cond_error_msgs.append(trig_msg)

        pit_failed = applicability_pit_failed or conditions_pit_failed

        verified_items = [v for v in claim_ver_items if v.get("status") == STATUS_VERIFIED]
        unsupported_items = [v for v in claim_ver_items if v.get("status") == STATUS_UNSUPPORTED]
        contradicted_items = [v for v in claim_ver_items if v.get("status") == STATUS_CONTRADICTED]
        source_unavail_items = [
            v for v in claim_ver_items
            if v.get("status") == STATUS_SOURCE_UNAVAILABLE or v.get("is_fatal")
        ]

        total_count = len(claim_ver_items)
        if total_count == 0:
            ev_field = claim_obj.get("evidence") or []
            if isinstance(ev_field, str):
                ev_field = [ev_field]
            total_count = len([e for e in ev_field if str(e).strip()])

        verified_count = len(verified_items)
        unsupported_count = len(unsupported_items)
        contradicted_count = len(contradicted_items)
        source_unavail_count = len(source_unavail_items)

        if pit_failed:
            contradicted_count = max(contradicted_count, 1)

        counts = {
            "total": total_count,
            "verified": verified_count,
            "unsupported": unsupported_count,
            "contradicted": contradicted_count,
            "source_unavailable": source_unavail_count,
        }

        coverage = (verified_count / total_count) if total_count > 0 else 0.0

        verified_ev = [str(v.get("raw", "")).strip() for v in verified_items if str(v.get("raw", "")).strip()]
        unsupported_ev = [str(v.get("raw", "")).strip() for v in unsupported_items if str(v.get("raw", "")).strip()]
        contradicted_ev = [str(v.get("raw", "")).strip() for v in contradicted_items if str(v.get("raw", "")).strip()]
        source_unavail_ev = [str(v.get("raw", "")).strip() for v in source_unavail_items if str(v.get("raw", "")).strip()]
        excluded_ev = [
            str(v.get("raw", "")).strip()
            for v in claim_ver_items
            if v.get("status") != STATUS_VERIFIED and str(v.get("raw", "")).strip()
        ]

        if pit_failed:
            decision = DECISION_REJECT
            all_pit_reasons = []
            if applicability_pit_failed:
                all_pit_reasons.append(app_error_msg)
            if conditions_pit_failed:
                all_pit_reasons.extend(cond_error_msgs)
            reason = f"命题规格契约存在前视偏差/PIT失败 (contradicted): {'; '.join(all_pit_reasons)}"
        elif applicability_failed or conditions_failed:
            decision = DECISION_REJECT
            all_spec_reasons = []
            if applicability_failed:
                all_spec_reasons.append(app_error_msg)
            if conditions_failed:
                all_spec_reasons.extend(cond_error_msgs)
            reason = f"命题规格契约校验失败: {'; '.join(all_spec_reasons)}"
        elif conditions_triggered:
            decision = DECISION_REJECT
            reason = f"命题失效条件已触发证伪 (contradicted): {'; '.join(cond_error_msgs)}"
            contradicted_count = max(contradicted_count, 1)
            counts["contradicted"] = contradicted_count
        elif contradicted_count > 0:
            decision = DECISION_REJECT
            reason = f"存在 {contradicted_count} 条与报告事实冲突/前视偏差证据 (contradicted)"
        elif source_unavail_count > 0:
            decision = DECISION_REJECT
            reason = f"存在 {source_unavail_count} 条引用不可用数据源的严重幻觉证据 (source_unavailable)"
        elif total_count == 0 or verified_count == 0:
            decision = DECISION_REJECT
            reason = "未提供有效证据或全部证据未获验证 (unsupported)"
        elif verified_count == total_count:
            if is_obs_hypo:
                decision = DECISION_PARTIAL
                reason = f"观察/假设类命题 (observation/hypothesis) 核验通过但须留在可审计的观察状态 (verified {verified_count}/{total_count})，不升级为已验证事实 (adopt)"
            else:
                decision = DECISION_ADOPT
                reason = f"全部证据核验通过 (verified {verified_count}/{total_count}, coverage=100.0%)"
        elif coverage >= MIN_COVERAGE_THRESHOLD or round(coverage, 2) >= MIN_COVERAGE_THRESHOLD or math.isclose(coverage, 2 / 3, abs_tol=1e-3):
            decision = DECISION_PARTIAL
            reason = f"混合证据部分通过核验 (verified {verified_count}/{total_count}, coverage={coverage:.1%})，仅可采纳 verified 子结论并剔除未验证项"
        else:
            decision = DECISION_REJECT
            reason = f"证据覆盖率不足 (verified {verified_count}/{total_count}, coverage={coverage:.1%} < {MIN_COVERAGE_THRESHOLD:.0%})，予以驳回/降权"

        summary_map[cid] = {
            "claim_id": cid,
            "speaker": str(claim_obj.get("speaker", "") or ""),
            "speaker_key": str(claim_obj.get("speaker_key", "") or ""),
            "stance": str(claim_obj.get("stance", "") or ""),
            "claim": str(claim_obj.get("claim", "") or ""),
            "counts": counts,
            "coverage": coverage,
            "decision": decision,
            "reason": reason,
            "verified_evidence": verified_ev,
            "unsupported_evidence": unsupported_ev,
            "contradicted_evidence": contradicted_ev,
            "source_unavailable_evidence": source_unavail_ev,
            "excluded_evidence": excluded_ev,
            "is_observation_or_hypothesis": is_obs_hypo,
            "applicability": norm_applicability,
            "invalidation_conditions": norm_conditions,
            "pit_failed": pit_failed,
        }

    return summary_map


def format_claims_with_verification_for_prompt(
    claims: Sequence[Mapping[str, Any]] | None,
    claims_verification: Sequence[Mapping[str, Any]] | None = None,
    claim_evidence_summary: Mapping[str, Mapping[str, Any]] | None = None,
    focus_claim_ids: Sequence[str] | None = None,
    empty_message: str = "当前没有已登记 claim。",
) -> str:
    """Format claim overview with deterministic factual verification details for research manager prompt."""
    claim_list = list(claims or [])
    if not claim_list and not claims_verification:
        return empty_message

    if claim_evidence_summary is None:
        summary_map = aggregate_claim_evidence(claim_list, claims_verification or [])
    else:
        summary_map = dict(claim_evidence_summary)

    focus_set = {str(item) for item in (focus_claim_ids or []) if str(item).strip()}
    lines: list[str] = []

    badge_map = {
        DECISION_ADOPT: "证据充分 / 全Verified",
        DECISION_PARTIAL: "部分支持 / 混合证据(仅采纳Verified子结论)",
        DECISION_REJECT: "证据薄弱/不支持/矛盾(驳回)",
    }

    for claim in claim_list:
        cid = str(claim.get("claim_id", "")).strip()
        status = str(claim.get("status", "open")).strip() or "open"
        speaker = str(claim.get("speaker", "")).strip() or "Unknown"
        stance = str(claim.get("stance", "")).strip() or ""
        summary_text = str(claim.get("claim", "")).strip() or "未提供 claim 文本"

        sum_info = summary_map.get(cid)
        prefix = "* " if cid in focus_set else "- "
        stance_str = f" ({stance})" if stance else ""

        if not sum_info:
            evidence = claim.get("evidence") or []
            if isinstance(evidence, str):
                evidence = [evidence]
            ev_text = "；".join(str(e).strip() for e in evidence if str(e).strip()) or "无明确证据"
            lines.append(f"{prefix}{cid} [{status}] {speaker}{stance_str}: {summary_text} | 证据: {ev_text}")
            continue

        decision = sum_info.get("decision", DECISION_REJECT)
        is_obs = sum_info.get("is_observation_or_hypothesis", False)
        badge = "观察/假设类命题 (可审计未验证/部分采纳)" if is_obs else badge_map.get(decision, "待核验")
        cov = sum_info.get("coverage", 0.0)
        counts = sum_info.get("counts", {})
        total = counts.get("total", 0)
        verified = counts.get("verified", 0)
        reason = sum_info.get("reason", "")

        lines.append(f"{prefix}{cid} [{status}] {speaker}{stance_str}: {summary_text}")
        lines.append(f"  * 核验评级: 【{badge}】 覆盖率={cov:.1%} ({verified}/{total} verified) | 规则判定: {decision}")
        lines.append(f"  * 判定说明: {reason}")
        if sum_info.get("applicability"):
            app = sum_info["applicability"]
            lines.append(f"  * 适用档规格: 标的={app.get('symbol')}, 观察窗={app.get('horizon')}, 计量基准={app.get('metric_basis')}, PIT截止日={app.get('pit_date')}")
        if sum_info.get("invalidation_conditions"):
            for cond in sum_info["invalidation_conditions"]:
                lines.append(f"  * 证伪/失效条件 [{cond.get('condition_id')}]: {cond.get('metric')} {cond.get('operator')} {cond.get('threshold')} {cond.get('unit')} (周期: {cond.get('period')}, 来源: {cond.get('source')}, PIT={cond.get('pit_date')})")

        ver_ev = sum_info.get("verified_evidence", [])
        unsupp_ev = sum_info.get("unsupported_evidence", [])
        contra_ev = sum_info.get("contradicted_evidence", [])
        unavail_ev = sum_info.get("source_unavailable_evidence", [])

        if ver_ev:
            for e in ver_ev:
                lines.append(f"    - [VERIFIED / 真实核验] {e}")
        if unsupp_ev:
            for e in unsupp_ev:
                lines.append(f"    - [UNSUPPORTED / 未获支撑] {e} (严禁作为采纳依据，必须剔除)")
        if contra_ev:
            for e in contra_ev:
                lines.append(f"    - [CONTRADICTED / 事实冲突] {e} (严禁采纳，必须驳回)")
        if unavail_ev:
            for e in unavail_ev:
                lines.append(f"    - [UNAVAILABLE / 严重幻觉] {e} (数据源不可用，严禁采纳)")

    return "\n".join(lines)


def format_challenges_for_prompt(
    challenges: Sequence[Mapping[str, Any]] | None,
    challenge_verification: Sequence[Mapping[str, Any]] | None = None,
    empty_message: str = "当前没有提出交叉盘问 (challenges)。",
) -> str:
    """Format challenges with verification status for research manager prompt."""
    ch_list = list(challenges or [])
    if not ch_list:
        return empty_message

    ver_map: dict[str, Mapping[str, Any]] = {}
    if challenge_verification:
        for v in challenge_verification:
            chid = str(v.get("challenge_id", "")).strip()
            if chid:
                ver_map[chid] = v

    lines: list[str] = []
    for ch in ch_list:
        chid = str(ch.get("challenge_id", "")).strip()
        speaker = str(ch.get("speaker") or ch.get("speaker_key") or "Unknown").strip()
        target_id = str(ch.get("target_claim_id", "")).strip()
        weakest = str(ch.get("weakest_point", "")).strip()
        sev = str(ch.get("severity", "major")).strip().lower()
        ev_list = [str(e).strip() for e in (ch.get("evidence") or []) if str(e).strip()]
        ev_str = "；".join(ev_list) if ev_list else "无"
        status = str(ch.get("status", "open")).strip()

        ver_info = ver_map.get(chid)
        ev_status = ver_info.get("evidence_status") if ver_info else ch.get("evidence_status", "unverified")
        badge = f"【证据核验: {ev_status}】" if ev_status else ""

        line = f"- {chid} [{status}] {speaker} 攻击对手 {target_id} (严厉度: {sev}) {badge}: 弱点={weakest} | 证据: {ev_str}"
        lines.append(line)

    return "\n".join(lines)


def format_challenge_verification_summary(
    challenges: Sequence[Mapping[str, Any]] | None,
    challenge_verification: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Format summary of challenge evidence verification status for research manager prompt."""
    ch_list = list(challenges or [])
    if not ch_list:
        return "暂无交叉盘问核验数据。"

    ver_list = list(challenge_verification or [])
    if not ver_list:
        return "交叉盘问证据尚未核验。"

    verified_count = sum(1 for v in ver_list if v.get("evidence_status") == "verified")
    unsupported_count = sum(1 for v in ver_list if v.get("evidence_status") == "unsupported")
    contradicted_count = sum(1 for v in ver_list if v.get("evidence_status") == "contradicted")

    lines = [
        f"交叉盘问核验汇总 (共 {len(ver_list)} 项): Verified={verified_count}, Unsupported={unsupported_count}, Contradicted={contradicted_count}",
    ]
    for v in ver_list:
        chid = v.get("challenge_id", "")
        target_id = v.get("target_claim_id", "")
        sev = v.get("severity", "")
        ev_st = v.get("evidence_status", "")
        lines.append(f"  * {chid} (针对 {target_id}, 严厉度: {sev}): 证据状态={ev_st}")

    return "\n".join(lines)


def format_battlefield_coverage(claims: Sequence[Mapping[str, Any]] | None) -> str:
    """Format summary of covered battlefields by camp for research manager prompt."""
    claim_list = list(claims or [])
    if not claim_list:
        return "暂无战场覆盖数据。"

    bull_bfs = set()
    bear_bfs = set()
    for c in claim_list:
        bf = str(c.get("battlefield", "")).strip()
        if not bf:
            continue
        sp = str(c.get("speaker_key") or c.get("speaker") or "")
        st = str(c.get("stance") or "").lower()
        if "bull" in sp.lower() or "bull" in st:
            bull_bfs.add(bf)
        elif "bear" in sp.lower() or "bear" in st:
            bear_bfs.add(bf)

    b_str = ", ".join(sorted(bull_bfs)) if bull_bfs else "未指定"
    be_str = ", ".join(sorted(bear_bfs)) if bear_bfs else "未指定"
    return f"多头覆盖战场 ({len(bull_bfs)}/5): {b_str} | 空头覆盖战场 ({len(bear_bfs)}/5): {be_str}"


def normalize_winner(winner_raw: Any, direction_raw: Any = "") -> str:
    """Normalize winner string to one of 'bull', 'bear', 'tie'."""
    w_str = str(winner_raw or "").strip().lower()
    if w_str in {"bull", "bullish", "多头", "多方", "多头胜", "多方胜", "多头全面胜出"}:
        return "bull"
    elif w_str in {"bear", "bearish", "空头", "空方", "空头胜", "空方胜", "空头全面胜出"}:
        return "bear"
    elif w_str in {"tie", "neutral", "平局", "势均力敌", "分歧", "观望", "hold", "中性", "unresolved"}:
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



def is_daily_ohlcv_unavailable(market_data_context: Any) -> bool:
    """True when daily OHLCV is missing or failed in a *provided* market_data_context.

    Compatibility: ``None`` (caller did not pass context) returns False so legacy
    call sites keep prior behavior. Any explicitly provided Mapping — including
    ``{}``, missing ``stock_data`` provenance, or no usable daily as_of — is
    treated as unavailable (fail-closed).
    """
    if market_data_context is None:
        return False
    if not isinstance(market_data_context, Mapping):
        return True

    provenance = market_data_context.get("source_provenance")
    if isinstance(provenance, Mapping):
        stock = provenance.get("stock_data")
        if isinstance(stock, Mapping):
            status = str(stock.get("status") or "").strip().lower()
            prov_status = str(stock.get("provenance_status") or "").strip().lower()
            if status in {"unavailable", "failed", "timeout", "refused", "error", "future", "available_unverified_as_of"}:
                return True
            if prov_status in {"refused", "future", "unverified"}:
                return True
            gap = str(stock.get("gap") or "")
            if "无有效完整日线" in gap or "【数据获取失败】stock_data" in gap:
                return True
            if status == "available" and stock.get("as_of") and prov_status == "verified":
                return False
            if status == "available" and stock.get("as_of") and not prov_status:
                return False
            # Provenance present but not a usable available+as_of bar.
            return True
        # Context provided with provenance map but no stock_data entry.
        if "stock_data" not in provenance:
            # Fall through to daily / ledger checks before concluding.
            pass
        else:
            return True

    ledger = market_data_context.get("data_failure_ledger")
    if isinstance(ledger, Sequence) and not isinstance(ledger, (str, bytes)):
        for entry in ledger:
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("source") or "").strip() != "stock_data":
                continue
            status = str(entry.get("status") or "").strip().lower()
            prov_status = str(entry.get("provenance_status") or "").strip().lower()
            gap = str(entry.get("gap") or "")
            if status in {"unavailable", "failed", "timeout", "refused", "error", "future", "available_unverified_as_of"}:
                return True
            if prov_status in {"refused", "future", "unverified"}:
                return True
            if "无有效完整日线" in gap or "【数据获取失败】stock_data" in gap:
                return True

    daily = market_data_context.get("daily")
    if isinstance(daily, Mapping):
        daily_status = str(daily.get("status") or "").strip().lower()
        if daily_status in {"unavailable", "failed", "timeout", "refused", "error", "future", "available_unverified_as_of"}:
            return True
        completeness = str(daily.get("completeness") or "").strip().lower()
        if completeness == "completed" and daily.get("as_of"):
            return False
        if daily.get("as_of") and completeness not in {"unavailable", ""}:
            return False
        # Explicit daily block without usable as_of.
        if "as_of" in daily or "completeness" in daily or "status" in daily:
            return True

    # Provided context but no usable OHLCV evidence anywhere.
    return True



_FUND_FLOW_IN_MARKERS = ("流入", "净流入", "吸筹", "inflow", "accumulation")
_FUND_FLOW_OUT_MARKERS = ("流出", "净流出", "派发", "出货", "outflow", "distribution")
_FUND_FLOW_ABSORPTION_MARKERS = (
    "流出=吸筹",
    "流出等于吸筹",
    "流出即吸筹",
    "边打边吸",
    "边出边吸",
    "流出当作吸筹",
)


def is_conflicting_fund_flow_dispute(
    data_point: str = "",
    bull_interpretation: str = "",
    bear_interpretation: str = "",
    evidence_decision: str = "",
) -> bool:
    """True when a dispute row mixes inflow/outflow prints or uses outflow=absorption."""
    blob = " ".join(
        [
            str(data_point or ""),
            str(bull_interpretation or ""),
            str(bear_interpretation or ""),
            str(evidence_decision or ""),
        ]
    )
    if not blob.strip():
        return False
    if any(marker in blob for marker in _FUND_FLOW_ABSORPTION_MARKERS):
        return True
    has_in = any(marker in blob for marker in _FUND_FLOW_IN_MARKERS)
    has_out = any(marker in blob for marker in _FUND_FLOW_OUT_MARKERS)
    return has_in and has_out


def extract_and_validate_manager_verdict(
    raw_response: str,
    claims_verification: Sequence[Mapping[str, Any]] | None = None,
    claims: Sequence[Mapping[str, Any]] | None = None,
    challenges: Sequence[Mapping[str, Any]] | None = None,
    challenges_verification: Sequence[Mapping[str, Any]] | None = None,
    market_data_context: Mapping[str, Any] | None = None,
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
        - partially_adopted_claims
        - rejected_claim_ids
        - excluded_evidence
        - claim_evidence_summary
        - dispute_map
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
    ohlcv_gate_applied = False

    # A2: missing daily OHLCV → fail-closed; never allow bull/bear winner.
    if is_daily_ohlcv_unavailable(market_data_context) and winner in {"bull", "bear"}:
        winner = "tie"
        direction = "中性"
        gate_note = "日线 OHLCV 不可用，禁止方向性裁决"
        reason = f"{reason}；{gate_note}" if reason else gate_note
        ohlcv_gate_applied = True

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
    partially_adopted_claims = _to_str_list(payload.get("partially_adopted_claims")) if payload else []
    rejected_claim_ids = _to_str_list(payload.get("rejected_claim_ids")) if payload else []
    excluded_evidence = _to_str_list(payload.get("excluded_evidence")) if payload else []

    # ── Extract Dispute Map ───────────────────────────────────────────────
    raw_dispute_map = payload.get("dispute_map") or []
    dispute_map: list[dict[str, Any]] = []
    fund_flow_dispute_gate_applied = False
    if isinstance(raw_dispute_map, list):
        for item in raw_dispute_map:
            if isinstance(item, Mapping):
                dp = str(item.get("data_point") or "").strip()
                b_interp = str(item.get("bull_interpretation") or "").strip()
                be_interp = str(item.get("bear_interpretation") or "").strip()
                ev_dec = str(item.get("evidence_decision") or "").strip()
                w_raw = str(item.get("winner") or "").strip()
                row_winner = normalize_winner(w_raw)
                if is_conflicting_fund_flow_dispute(dp, b_interp, be_interp, ev_dec):
                    row_winner = "tie"
                    ev_dec = "分单对打/冲突资金流不得单独支撑方向（禁止流出=吸筹）"
                    fund_flow_dispute_gate_applied = True
                dispute_map.append({
                    "data_point": dp,
                    "bull_interpretation": b_interp,
                    "bear_interpretation": be_interp,
                    "evidence_decision": ev_dec,
                    "winner": row_winner,
                })

    # A4: if every directional dispute was fund-flow conflict, do not keep bull/bear.
    if fund_flow_dispute_gate_applied and winner in {"bull", "bear"}:
        directional_rows = [r for r in dispute_map if r.get("winner") in {"bull", "bear"}]
        if not directional_rows:
            winner = "tie"
            direction = "中性"
            gate_note = "资金流分单冲突，禁止据此给出方向性裁决"
            reason = f"{reason}；{gate_note}" if reason else gate_note

    # ── Deterministic Claim Evidence Summary Computation ──────────────────
    claim_evidence_summary: dict[str, dict[str, Any]] = {}
    if claims is not None or claims_verification is not None:
        b_date = None
        exp_sym = None
        if isinstance(market_data_context, Mapping):
            b_date = str(
                market_data_context.get("analysis_baseline_date")
                or market_data_context.get("data_as_of")
                or market_data_context.get("trade_date")
                or ""
            ).strip() or None
            exp_sym = str(
                market_data_context.get("symbol")
                or market_data_context.get("ticker")
                or ""
            ).strip() or None
        claim_evidence_summary = aggregate_claim_evidence(
            claims=claims,
            claims_verification=claims_verification,
            analysis_baseline_date=b_date,
            expected_symbol=exp_sym,
            market_data_context=market_data_context,
        )
    elif payload and isinstance(payload.get("claim_evidence_summary"), dict):
        claim_evidence_summary = payload["claim_evidence_summary"]

    deterministic_excluded: list[str] = []
    for cid, s in claim_evidence_summary.items():
        if cid in partially_adopted_claims or cid in rejected_claim_ids or s.get("decision") in {DECISION_PARTIAL, DECISION_REJECT}:
            deterministic_excluded.extend(s.get("excluded_evidence", []))
    combined_excluded = list(dict.fromkeys(excluded_evidence + deterministic_excluded))

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

    # Check 6: Claim ledger subset and existence validation
    if claims is not None:
        known_cids = {
            str(c.get("claim_id", "")).strip()
            for c in claims
            if str(c.get("claim_id", "")).strip()
        }
        for cid in adopted_claim_ids:
            if cid not in known_cids:
                failed_checks.append(f"裁决采纳了不存在的 claim ID: {cid} (当前账本: {sorted(known_cids)})")
        for cid in partially_adopted_claims:
            if cid not in known_cids:
                failed_checks.append(f"裁决部分采纳了不存在的 claim ID: {cid} (当前账本: {sorted(known_cids)})")
        for cid in rejected_claim_ids:
            if cid not in known_cids:
                failed_checks.append(f"裁决拒绝了不存在的 claim ID: {cid} (当前账本: {sorted(known_cids)})")

    # Check 7: Claim Evidence Coverage & Consistency Hard Gate
    if claim_evidence_summary:
        for cid in adopted_claim_ids:
            if cid in claim_evidence_summary:
                s = claim_evidence_summary[cid]
                cnt = s.get("counts", {})
                cov = s.get("coverage", 0.0)
                dec = s.get("decision")
                is_obs = s.get("is_observation_or_hypothesis", False)
                if s.get("pit_failed") or cnt.get("contradicted", 0) > 0:
                    failed_checks.append(f"裁决采纳了存在事实冲突/前视偏差的矛盾 claim: {cid}")
                elif cnt.get("source_unavailable", 0) > 0:
                    failed_checks.append(f"裁决采纳了不可用数据源的严重幻觉 claim: {cid}")
                elif is_obs:
                    failed_checks.append(f"裁决全额采纳了观察/假设类 claim: {cid}，观察/假设类命题不得升级为已验证事实 (adopt)")
                elif cnt.get("verified", 0) == 0 or cnt.get("total", 0) == 0:
                    failed_checks.append(f"裁决采纳了全部证据未获验证 (unsupported) 的 claim: {cid}")
                elif cov < MIN_COVERAGE_THRESHOLD and not math.isclose(cov, 2 / 3, abs_tol=1e-3):
                    failed_checks.append(f"裁决采纳了证据覆盖率不足 ({cov:.1%} < 67%) 的 claim: {cid}")
                elif dec == DECISION_PARTIAL or (0.67 <= cov < 1.0 and not math.isclose(cov, 1.0)):
                    failed_checks.append(
                        f"裁决全额采纳了含未核实混合证据的 claim: {cid} (coverage={cov:.1%})，混合证据仅允许记录于 partially_adopted_claims 并剔除未验证项"
                    )

        for cid in partially_adopted_claims:
            if cid in claim_evidence_summary:
                s = claim_evidence_summary[cid]
                cnt = s.get("counts", {})
                cov = s.get("coverage", 0.0)
                is_obs = s.get("is_observation_or_hypothesis", False)
                if s.get("pit_failed") or cnt.get("contradicted", 0) > 0:
                    failed_checks.append(f"部分采纳列表中包含了存在事实冲突/前视偏差的矛盾 claim: {cid}")
                elif cnt.get("source_unavailable", 0) > 0:
                    failed_checks.append(f"部分采纳列表中包含了不可用数据源的严重幻觉 claim: {cid}")
                elif not is_obs and (cnt.get("verified", 0) == 0 or cnt.get("total", 0) == 0):
                    failed_checks.append(f"部分采纳列表中包含了全部证据未获验证 (unsupported) 的 claim: {cid}")
                elif not is_obs and (cov < MIN_COVERAGE_THRESHOLD and not math.isclose(cov, 2 / 3, abs_tol=1e-3)):
                    failed_checks.append(f"部分采纳列表中包含了证据覆盖率不足 ({cov:.1%} < 67%) 的 claim: {cid}")

        # Check prose consistency against claim verification
        for cid, s in claim_evidence_summary.items():
            cov = s.get("coverage", 0.0)
            dec = s.get("decision")
            if dec != DECISION_ADOPT or cov < 1.0:
                pattern = re.compile(
                    rf"{re.escape(cid)}[^\n。；]*?证据充分|证据充分[^\n。；]*?{re.escape(cid)}",
                    re.IGNORECASE,
                )
                for line in prose.splitlines():
                    if pattern.search(line):
                        if not re.search(r"非[^\n]*?证据充分|不[^\n]*?证据充分|未[^\n]*?证据充分|不能[^\n]*?证据充分", line):
                            failed_checks.append(
                                f"裁决正文将未完全核实的 claim {cid} (coverage={cov:.1%}, decision={dec}) 标注为'证据充分'，正文与证据核验严重冲突"
                            )
                            break
    elif claims_verification:
        # Fallback fatal check if only raw verification list was provided without claim summary
        fatal_cids = {
            str(item.get("claim_id"))
            for item in claims_verification
            if item.get("is_fatal") or item.get("status") == STATUS_SOURCE_UNAVAILABLE
        }
        for cid in adopted_claim_ids:
            if str(cid) in fatal_cids:
                failed_checks.append(f"裁决采纳了不可用数据源的严重幻觉 claim: {cid}")

    # ── Check 8: Fatal Challenge Consistency Hard Gate ──────────────────
    ch_map: dict[str, Mapping[str, Any]] = {}
    if challenges:
        for ch in challenges:
            chid = str(ch.get("challenge_id", "")).strip()
            if chid:
                ch_map[chid] = ch

    ch_ver_map: dict[str, Mapping[str, Any]] = {}
    if challenges_verification:
        for cv in challenges_verification:
            chid = str(cv.get("challenge_id", "")).strip()
            if chid:
                ch_ver_map[chid] = cv

    # Rule 8.1: Unverified fatal challenge cannot reject 100% verified claim
    for chid, ch in ch_map.items():
        sev = str(ch.get("severity", "major")).strip().lower()
        if sev == "fatal":
            target_id = str(ch.get("target_claim_id", "")).strip()
            cv = ch_ver_map.get(chid, {})
            ev_status = cv.get("evidence_status") or ch.get("evidence_status", "unverified")

            # If fatal challenge evidence is unsupported or contradicted
            if ev_status in ("unsupported", "contradicted", "unverified"):
                if target_id in rejected_claim_ids and claim_evidence_summary:
                    target_summary = claim_evidence_summary.get(target_id, {})
                    target_cov = target_summary.get("coverage", 0.0)
                    target_dec = target_summary.get("decision")
                    if target_dec == DECISION_ADOPT and (target_cov >= 1.0 or math.isclose(target_cov, 1.0)):
                        failed_checks.append(
                            f"未经验证的 fatal challenge ({chid}, status={ev_status}) 不得作为否决高质量已验证 claim {target_id} 的依据"
                        )

            # Rule 8.2: Contradicted fatal challenge must be rejected
            if ev_status == "contradicted":
                ch_status = str(ch.get("status", "")).strip().lower()
                adopted_challenges = payload.get("adopted_challenge_ids") or []
                if ch_status == "adopted" or chid in adopted_challenges:
                    failed_checks.append(
                        f"存在事实冲突的 fatal challenge ({chid}) 必须被驳回，不得采纳"
                    )

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
        "partially_adopted_claims": partially_adopted_claims,
        "rejected_claim_ids": rejected_claim_ids,
        "excluded_evidence": combined_excluded,
        "claim_evidence_summary": claim_evidence_summary,
        "dispute_map": dispute_map,
        "consistency_check_passed": consistency_passed,
        "failed_checks": failed_checks,
        "ohlcv_gate_applied": ohlcv_gate_applied,
        "fund_flow_dispute_gate_applied": fund_flow_dispute_gate_applied,
    }

