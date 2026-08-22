"""Report Quality Gate for macro and analyst reports.

Validates core keywords (e.g. 传导 and 联动/外溢/时滞), prevents
silent rewriting or smoothing of failed/missing global market data,
and provides role-aware deterministic depth scoring for analysts:
macro, fundamentals, news, and volume_price.
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
    "数据获取失败",
    "暂无数据",
    "未获取到",
    "无数据",
    "暂无相关数据",
    "相关数据缺失",
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


def is_industry_linkage_present(
    market_data_context: Optional[Dict[str, Any]],
) -> bool:
    """Determine whether industry_linkage exists in market_data_context."""
    if not isinstance(market_data_context, dict):
        return False

    linkage = market_data_context.get("industry_linkage")
    if isinstance(linkage, dict):
        if (
            linkage.get("industry_name")
            or linkage.get("upstream_cost")
            or linkage.get("downstream_demand")
            or linkage.get("international_benchmark")
        ):
            return True

    provenance = market_data_context.get("source_provenance")
    if isinstance(provenance, dict):
        ind_prov = provenance.get("industry_linkage")
        if isinstance(ind_prov, dict) and ind_prov.get("status") in ("available", "partial"):
            return True

    return False


def check_industry_linkage_compliance(
    macro_report: str = "",
    fundamentals_report: str = "",
    market_data_context: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """Check if report contains industry linkage data section or explicit indicators when context has industry_linkage."""
    if not is_industry_linkage_present(market_data_context):
        return True, []

    linkage = (
        market_data_context.get("industry_linkage")
        if isinstance(market_data_context, dict)
        else None
    )

    indicator_names: List[str] = []
    if isinstance(linkage, dict):
        for section in ("upstream_cost", "downstream_demand", "international_benchmark"):
            items = linkage.get(section)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        name = item.get("name")
                        if name and isinstance(name, str) and len(name.strip()) >= 2:
                            indicator_names.append(name.strip())
                        symbol = item.get("symbol")
                        if symbol and isinstance(symbol, str) and len(symbol.strip()) >= 2:
                            indicator_names.append(symbol.strip())

    def _satisfies_linkage(text: str) -> bool:
        if not text or not isinstance(text, str):
            return False
        if "【产业链联想数据】" in text:
            return True
        if "产业链" in text:
            if any(marker in text for marker in ALLOWED_EXPLICIT_MISSING_MARKERS):
                return True
            if indicator_names and any(name in text for name in indicator_names):
                return True
            common_indicators = (
                "铜价", "LME", "碳酸锂", "原油", "三星", "台积电",
                "出货量", "渗透率", "交付量", "指数", "价格", "对标"
            )
            if any(ind in text for ind in common_indicators):
                return True
        return False

    if _satisfies_linkage(macro_report) or _satisfies_linkage(fundamentals_report):
        return True, []

    return (
        False,
        ["context包含产业链数据但宏观与基本面报告均缺少【产业链联想数据】或明确产业链指标/数据缺失标注"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 角色感知确定性深度评分器 (Role-Aware Deterministic Depth Scorers)
# ─────────────────────────────────────────────────────────────────────────────


def _has_explicit_missing(text: str) -> bool:
    """Check whether text contains explicit data missing markers."""
    if not text or not isinstance(text, str):
        return False
    return any(m in text for m in ALLOWED_EXPLICIT_MISSING_MARKERS)


def evaluate_macro_depth(
    text: str,
    market_data_context: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, float, List[str], str]:
    """Evaluate depth of macro report.

    Dimensions:
    1. entities_or_metrics: valid entity or numbers (points or percentage) or explicit missing marker.
    2. causal_chain: at least one causal/arrow-style transmission chain (not an empty slogan).
    3. direction_or_magnitude: clear directional or magnitude assessment.
    4. lag_or_missing: time lag / latency / validation horizon or explicit missing marker.
    5. anti_smoothing: forbid forbidden smoothing when global indices failed.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return False, 0.0, ["empty_report"], "macro报告正文为空"

    t = text.strip()
    failed_dims: List[str] = []
    dim_count = 4

    # 1. entities_or_metrics
    has_missing = _has_explicit_missing(t)
    has_numbers = bool(re.search(r"(?:[+-]?\d+(?:\.\d+)?%|[+-]?\d+(?:\.\d+)?(?:点|bp|BP|个基点|美元|元|亿元|万|手))", t))
    has_macro_entity = bool(
        re.search(
            r"(?:标普|道琼斯|纳斯达克|恒生|上证|沪深300|中证500|中证1000|美联储|央行|降息|加息|MLF|LPR|CPI|PPI|PMI|GDP|美债|汇率|美元指数|人民币|国债|北向资金|大宗商品|原油|铜价|黄金|LME|十年期国债)",
            t,
        )
    )
    if not (has_missing or (has_macro_entity and has_numbers) or (has_macro_entity and ("指数" in t or "点位" in t or "%" in t)) or (has_numbers and ("指数" in t or "点" in t or "%" in t))):
        failed_dims.append("entities_or_metrics")

    # 2. causal_chain (arrow or structured causal transmission, rejects empty slogans like "传导与联动值得关注")
    has_arrow = bool(re.search(r"(?:[→\->=>]|-->|==>)", t))
    has_causal_structure = bool(
        re.search(
            r"(?:通过|由于|因为|伴随|随着|在.+驱动下).{1,35}(?:向|对|导致|引发|驱动|推升|拉动|压制|促使|形成|带来|传导|溢出|反馈)",
            t,
        )
        or re.search(
            r"(?:向.+传导|传导至|传导路径|传导机制|溢出至|溢出效应|反馈至|映射到|直接导致|间接导致|引发|驱动|推升|拉动|压制|提振|带动|吞噬|侵蚀).{1,35}(?:A股|国内|市场|板块|行业|资产|估值|流动性|情绪|成长|核心资产|汇率|利率|价格|需求|成本)",
            t,
        )
        or re.search(
            r"(?:导致|促使|引发|使得|驱动|推升|拉动|压制).{1,25}(?:上涨|下跌|走强|走弱|承压|分化|反弹|回落|收紧|宽松|修复)",
            t,
        )
    )
    if not (has_missing or has_arrow or has_causal_structure):
        failed_dims.append("causal_chain")

    # 3. direction_or_magnitude
    has_direction = bool(
        re.search(
            r"(?:上涨|下跌|走强|走弱|承压|下行|上行|收窄|扩张|放缓|攀升|下滑|回落|反弹|震荡|修复|提振|压制|增厚|侵蚀|紧缩|宽松|升值|贬值|分化|企稳|回升|恶化|改善)",
            t,
        )
    )
    has_magnitude = bool(
        re.search(
            r"(?:大幅|温和|显著|微幅|快速|小幅|剧烈|超预期|有限|平稳|[+-]?\d+(?:\.\d+)?%|[+-]?\d+(?:\.\d+)?点)",
            t,
        )
    )
    if not (has_missing or has_direction or has_magnitude):
        failed_dims.append("direction_or_magnitude")

    # 4. lag_or_missing
    has_lag = bool(
        re.search(
            r"(?:时滞|滞后|传导周期|反应时滞|短期内|短期|中期|长期|T\+\d|季度|月度|见效时间|时差|窗口期|兑现周期|传导时间|延迟|逐步显现|时间节点|时间窗口|验证期|周内|年内|数月|数周|阶段性)",
            t,
        )
    )
    if not (has_missing or has_lag):
        failed_dims.append("lag_or_missing")

    # 5. Anti-smoothing check if global indices failed
    if is_global_indices_failed_or_partial(market_data_context):
        dim_count += 1
        gi_passed, _ = check_global_indices_compliance(t, market_data_context)
        if not gi_passed:
            failed_dims.append("anti_smoothing")

    passed_count = dim_count - len(failed_dims)
    score = round(max(0.0, passed_count / dim_count), 2)
    passed = len(failed_dims) == 0

    reasons: List[str] = []
    if "entities_or_metrics" in failed_dims:
        reasons.append("缺少有效实体/宏观数字(点位或百分比)或数据缺失标记")
    if "causal_chain" in failed_dims:
        reasons.append("缺少因果/箭头式传导链(空话或纯数字无因果)")
    if "direction_or_magnitude" in failed_dims:
        reasons.append("缺少明确方向或量级判断")
    if "lag_or_missing" in failed_dims:
        reasons.append("缺少时滞/时间窗口或明确数据缺失标记")
    if "anti_smoothing" in failed_dims:
        reasons.append("外盘数据缺失时存在违规平滑或未标注缺失")

    reason_str = f"macro深度不足：{'；'.join(reasons)}" if reasons else ""
    return passed, score, failed_dims, reason_str


def evaluate_fundamentals_depth(
    text: str,
    market_data_context: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, float, List[str], str]:
    """Evaluate depth of fundamentals report.

    Dimensions:
    1. financial_metrics: financial numbers / metrics or explicit missing marker.
    2. industry_chain_or_pricing_power: industry chain position or pricing/bargaining power.
    3. sensitivity_relationship: at least one sensitivity relationship (cost/sales -> margin/profit) or explicit missing marker.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return False, 0.0, ["empty_report"], "fundamentals报告正文为空"

    t = text.strip()
    failed_dims: List[str] = []
    dim_count = 3
    has_missing = _has_explicit_missing(t)

    # 1. financial_metrics
    has_fin_keywords = bool(
        re.search(
            r"(?:营收|营业收入|收入|净利润|扣非|毛利|毛利率|净利率|PE|PB|ROE|ROIC|EPS|资产负债率|现金流|经营性现金流|同比|环比|归母|估值|市盈率|市净率|分红|股息率|业绩)",
            t,
        )
    )
    has_numbers = bool(
        re.search(
            r"(?:\d+(?:\.\d+)?%|\d+(?:\.\d+)?(?:亿|万|元|倍)|[+-]?\d+(?:\.\d+)?)",
            t,
        )
    )
    if not (has_missing or (has_fin_keywords and has_numbers)):
        failed_dims.append("financial_metrics")

    # 2. industry_chain_or_pricing_power
    has_chain_power = bool(
        re.search(
            r"(?:产业链|供应链|上游|下游|中游|议价权|定价权|话语权|护城河|壁垒|市场份额|市占率|竞争格局|龙头|集中度|供应商|大客户|核心客户|代工|自研|成本转嫁|卡脖子|国产替代|终端需求|垂直一体化|行业地位)",
            t,
        )
    )
    if not (has_missing or has_chain_power):
        failed_dims.append("industry_chain_or_pricing_power")

    # 3. sensitivity_relationship (cost/sales change -> margin/profit impact)
    has_arrow_sensitivity = bool(
        re.search(
            r"(?:成本|原材料|锂价|铜价|单价|运费|价格|费用|销量|出货量|需求|产能|开工率|汇率|产销量|客单价).{0,30}(?:[+-]?\d+%|[+-]?\d+|上涨|下跌|增长|下降|增加|减少|波动|变动|上升|下滑).{0,15}[→\->=>].{0,30}(?:毛利率|毛利|净利润|净利|利润|业绩|盈利能力|EPS|ROE|收益|营收)",
            t,
        )
    )
    has_verbal_sensitivity = bool(
        re.search(
            r"(?:成本|原材料|锂价|铜价|单价|运费|价格|费用|销量|出货量|需求|产能|开工率|汇率|产销量|客单价).{0,30}(?:[+-]?\d+%|[+-]?\d+|上涨|下跌|增长|下降|增加|减少|波动|变动|上升|下滑).{0,30}(?:导致|引发|拉动|吞噬|提振|压制|增厚|侵蚀|影响|带动|压缩|挤压|改善|恶化|削减|使|造成|测算|承压|下降|提升|放缓).{0,30}(?:毛利率|毛利|净利润|净利|利润|业绩|盈利|EPS|ROE|收益|综合毛利)",
            t,
        )
    )
    has_explicit_sensitivity = bool(
        re.search(
            r"(?:敏感性|敏感度|利润弹性|业绩弹性|若.+变动.+则.+利润|每变动.+影响.+利润|每上涨.+影响.+毛利|每下降.+影响.+毛利|每上升.+影响.+净利)",
            t,
        )
    )
    if not (has_missing or has_arrow_sensitivity or has_verbal_sensitivity or has_explicit_sensitivity):
        failed_dims.append("sensitivity_relationship")

    passed_count = dim_count - len(failed_dims)
    score = round(max(0.0, passed_count / dim_count), 2)
    passed = len(failed_dims) == 0

    reasons: List[str] = []
    if "financial_metrics" in failed_dims:
        reasons.append("缺少财务量化数字或数据缺失标记")
    if "industry_chain_or_pricing_power" in failed_dims:
        reasons.append("缺少产业链位置或议价权/定价权分析")
    if "sensitivity_relationship" in failed_dims:
        reasons.append("缺少敏感性关系(成本/销量变动→毛利/利润影响)或明确数据缺失标记")

    reason_str = f"fundamentals深度不足：{'；'.join(reasons)}" if reasons else ""
    return passed, score, failed_dims, reason_str


def evaluate_news_depth(
    text: str,
    market_data_context: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, float, List[str], str]:
    """Evaluate depth of news report.

    Dimensions:
    1. event_facts_or_source: event facts / source or explicit missing marker.
    2. direct_impact: direct impact analysis on company/asset.
    3. indirect_transmission: indirect upstream/downstream/peer/international transmission.
    4. lag_or_verification: time lag or verification node.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return False, 0.0, ["empty_report"], "news报告正文为空"

    t = text.strip()
    failed_dims: List[str] = []
    dim_count = 4
    has_missing = _has_explicit_missing(t) or "暂无重大新闻" in t or "无近期重大新闻" in t or "未检索到" in t

    # 1. event_facts_or_source
    has_source_fact = bool(
        re.search(
            r"(?:公告|新闻|政策|发布|财联社|新华社|证券时报|据.+报道|披露|监管|会议|签约|中标|获批|立案|处罚|重组|定增|减持|增持|回购|事件|消息|动态|通知|指引|发改委|证监会|商务部|工信部|国务院|工信|央行|指导意见|管理办法|调研|通报|报道|声明)",
            t,
        )
    )
    if not (has_missing or has_source_fact):
        failed_dims.append("event_facts_or_source")

    # 2. direct_impact
    has_direct = bool(
        re.search(
            r"(?:直接影响|直接利好|直接利空|直接冲击|直接催化|直接带动|直接提振|直接压制|对公司|对标的|对主营业务|对核心业务|短期影响|短期来看|短期情绪|业绩直接|直接反应|直接作用|直接驱动|直接拉动|直接受损|直接受益|直接改善)",
            t,
        )
    )
    if not (has_missing or has_direct):
        failed_dims.append("direct_impact")

    # 3. indirect_transmission
    has_indirect = bool(
        re.search(
            r"(?:间接|上下游|产业链|同行|竞对|竞争对手|外溢|传导|扩散|联动|海外|国际|行业层面|板块联动|供应链|外延|溢出效应|次生影响|板块效应|带动同业|行业格局|向.+传导)",
            t,
        )
    )
    if not (has_missing or has_indirect):
        failed_dims.append("indirect_transmission")

    # 4. lag_or_verification
    has_lag_verify = bool(
        re.search(
            r"(?:时滞|滞后|验证节点|观察点|时间窗口|短期|中期|长期|落地时间|生效日期|业绩兑现|后续关注|跟踪节点|催化时点|落地节奏|兑现周期|观察期|关键节点|时间表|预计在.+体现|验证周期|财报兑现|窗口期|落地进度|兑现节点)",
            t,
        )
    )
    if not (has_missing or has_lag_verify):
        failed_dims.append("lag_or_verification")

    passed_count = dim_count - len(failed_dims)
    score = round(max(0.0, passed_count / dim_count), 2)
    passed = len(failed_dims) == 0

    reasons: List[str] = []
    if "event_facts_or_source" in failed_dims:
        reasons.append("缺少事件事实/来源或数据缺失标记")
    if "direct_impact" in failed_dims:
        reasons.append("缺少直接影响分析")
    if "indirect_transmission" in failed_dims:
        reasons.append("缺少间接上下游/同行/国际传导链")
    if "lag_or_verification" in failed_dims:
        reasons.append("缺少时滞或验证节点标注")

    reason_str = f"news深度不足：{'；'.join(reasons)}" if reasons else ""
    return passed, score, failed_dims, reason_str


def evaluate_volume_price_depth(
    text: str,
    market_data_context: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, float, List[str], str]:
    """Evaluate depth of volume_price report.

    Dimensions:
    1. metrics_or_dates: volume/price numbers or dates or explicit missing marker.
    2. confirmation_or_anomaly: volume-price confirmation or anomaly / divergence.
    3. supply_demand_or_stage: supply/demand or market stage judgment.
    4. verification_conditions: subsequent verification conditions / triggers.
    5. cross_dimension_reference: utilizes Phase 1 context (macro/market/sentiment) or explicitly notes missing context.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return False, 0.0, ["empty_report"], "volume_price报告正文为空"

    t = text.strip()
    failed_dims: List[str] = []
    dim_count = 5
    has_missing = _has_explicit_missing(t)

    # 1. metrics_or_dates
    has_numbers_metrics = bool(
        re.search(
            r"(?:[+-]?\d+(?:\.\d+)?%|[+-]?\d+(?:\.\d+)?(?:元|手|股|亿|万|点|日均|MA|EMA|KDJ|MACD|RSI|量比|换手率|成交量|成交额|收盘价|均线|均价))",
            t,
        )
    )
    has_dates = bool(
        re.search(
            r"(?:\d{4}-\d{2}-\d{2}|\d+月\d+日|昨日|今日|T-\d|前一交易日|近\d+日|近\d+天)",
            t,
        )
    )
    if not (has_missing or has_numbers_metrics or has_dates):
        failed_dims.append("metrics_or_dates")

    # 2. confirmation_or_anomaly
    has_confirm_anomaly = bool(
        re.search(
            r"(?:量价配合|量价齐升|量价背离|放量滞涨|缩量回调|放量突破|缩量企稳|放量下挫|缩量上涨|放量杀跌|缩量整理|缩量震荡|异常放量|天量|地量|确认|异动|背离|突破|假突破|洗盘|吸筹|出货|价升量增|价跌量缩|价升量缩|价跌量增|多头排列|空头排列|金叉|死叉|放量|缩量|底背离|顶背离|量能放大|量能萎缩|量能配合)",
            t,
        )
    )
    if not (has_missing or has_confirm_anomaly):
        failed_dims.append("confirmation_or_anomaly")

    # 3. supply_demand_or_stage
    has_supply_stage = bool(
        re.search(
            r"(?:供需|买盘|卖盘|多头|空头|获利盘|套牢盘|主力|筹码|底部|顶部|通道|震荡|整理|蓄势|盘整|筑底|主升浪|回调|派发|支撑|阻力|压力位|支撑位|筹码集中度|筹码结构|浮筹|阶段|多空博弈|吸筹阶段|洗盘阶段|拉升阶段|出货阶段)",
            t,
        )
    )
    if not (has_missing or has_supply_stage):
        failed_dims.append("supply_demand_or_stage")

    # 4. verification_conditions
    has_verification = bool(
        re.search(
            r"(?:后续验证|观察条件|若突破|若跌破|站稳|放量上攻|需关注|关键点位|触发条件|确认信号|止损|止盈|验证标准|验证条件|下行风险|一旦有效突破|如果放量|密切留意|以.+为防守位|防守位|跌破.+则|突破.+则|观察后续|观察.+日|防守线|防守点)",
            t,
        )
    )
    if not (has_missing or has_verification):
        failed_dims.append("verification_conditions")

    # 5. cross_dimension_reference (Phase 1 context usage or honest missing)
    has_cross_ref = bool(
        re.search(
            r"(?:宏观|大盘|市场面|板块|情绪|舆情|阶段一|外盘|指数|上证|沪深|海外市场|行业板块|资金面|跨市场|北向)",
            t,
        )
    )
    has_context_missing = bool(
        re.search(
            r"(?:无可用上下文|阶段一报告缺失|【数据缺失】|上下文缺失|未获取到阶段一|阶段一数据缺失|缺乏宏观/情绪上下文|无前序分析输入|前序报告缺失|无阶段一上下文|无可用前序)",
            t,
        )
    )
    if not (has_missing or has_cross_ref or has_context_missing):
        failed_dims.append("cross_dimension_reference")

    passed_count = dim_count - len(failed_dims)
    score = round(max(0.0, passed_count / dim_count), 2)
    passed = len(failed_dims) == 0

    reasons: List[str] = []
    if "metrics_or_dates" in failed_dims:
        reasons.append("缺少量价数字或交易日期")
    if "confirmation_or_anomaly" in failed_dims:
        reasons.append("缺少量价确认/异动/背离形态判断")
    if "supply_demand_or_stage" in failed_dims:
        reasons.append("缺少供需关系或行情阶段判断")
    if "verification_conditions" in failed_dims:
        reasons.append("缺少后续验证触发条件或防守点位")
    if "cross_dimension_reference" in failed_dims:
        reasons.append("缺少阶段一跨维度上下文引用或明确无上下文标记")

    reason_str = f"volume_price深度不足：{'；'.join(reasons)}" if reasons else ""
    return passed, score, failed_dims, reason_str


def evaluate_role_depth(
    role: str,
    text: str,
    market_data_context: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, float, List[str], str]:
    """Dispatch evaluation to appropriate role depth scorer."""
    r = (role or "").strip().lower()
    if r == "macro":
        return evaluate_macro_depth(text, market_data_context)
    elif r == "fundamentals":
        return evaluate_fundamentals_depth(text, market_data_context)
    elif r == "news":
        return evaluate_news_depth(text, market_data_context)
    elif r in ("volume_price", "volumeprice", "volume_price_analyst"):
        return evaluate_volume_price_depth(text, market_data_context, state)
    else:
        return True, 1.0, [], ""


def check_analyst_depth_quality(
    reports: Dict[str, str],
    market_data_context: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """Batch evaluate depth quality for all present analyst reports.

    Args:
        reports: dict mapping role name (e.g. 'macro', 'fundamentals', 'news', 'volume_price') to report text.
        market_data_context: market data context dict.
        state: graph state dict.

    Returns:
        (overall_passed, evaluation_results_by_role)
    """
    results: Dict[str, Dict[str, Any]] = {}
    all_passed = True

    for role, text in reports.items():
        if not text or not isinstance(text, str) or not text.strip():
            continue
        passed, score, failed_dims, reason = evaluate_role_depth(
            role=role,
            text=text,
            market_data_context=market_data_context,
            state=state,
        )
        results[role] = {
            "passed": passed,
            "score": score,
            "failed_dimensions": failed_dims,
            "reason": reason,
        }
        if not passed:
            all_passed = False

    return all_passed, results


def check_report_quality(
    macro_report: str = "",
    fundamentals_report: str = "",
    market_data_context: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """Comprehensive quality gate inspection for macro (and fundamentals) reports."""
    target_text = macro_report.strip() if isinstance(macro_report, str) else ""
    if not target_text and isinstance(fundamentals_report, str) and fundamentals_report.strip():
        target_text = fundamentals_report.strip()

    all_reasons: List[str] = []

    # 1. Keyword validation
    if target_text:
        kw_passed, kw_reasons = check_report_keywords(target_text)
        if not kw_passed:
            all_reasons.extend(kw_reasons)

        # 2. Global market rewrite compliance
        gi_passed, gi_reasons = check_global_indices_compliance(target_text, market_data_context)
        if not gi_passed:
            all_reasons.extend(gi_reasons)
    elif not is_industry_linkage_present(market_data_context):
        return True, []

    # 3. Industry linkage compliance
    il_passed, il_reasons = check_industry_linkage_compliance(
        macro_report=macro_report,
        fundamentals_report=fundamentals_report,
        market_data_context=market_data_context,
    )
    if not il_passed:
        all_reasons.extend(il_reasons)

    return (len(all_reasons) == 0, all_reasons)


def apply_report_quality_gate(
    state_or_result: Dict[str, Any],
    macro_retry_fn: Optional[Callable[..., Any]] = None,
) -> bool:
    """Run quality gate on report state/result and record structured failures to ledger without blocking."""
    if not isinstance(state_or_result, dict):
        return True

    macro_report = state_or_result.get("macro_report", "")
    fundamentals_report = state_or_result.get("fundamentals_report", "")
    news_report = state_or_result.get("news_report", "")
    volume_price_report = state_or_result.get("volume_price_report", "")

    market_data_context = state_or_result.get("market_data_context")
    if not isinstance(market_data_context, dict):
        for sub_key in ("short_term", "medium_term", "result_data"):
            sub_val = state_or_result.get(sub_key)
            if isinstance(sub_val, dict) and isinstance(sub_val.get("market_data_context"), dict):
                market_data_context = sub_val["market_data_context"]
                break

    # Look inside sub-keys for reports if missing at top level
    for sub_key in ("short_term", "medium_term", "result_data"):
        sub_val = state_or_result.get(sub_key)
        if isinstance(sub_val, dict):
            if not macro_report and sub_val.get("macro_report"):
                macro_report = sub_val["macro_report"]
            if not fundamentals_report and sub_val.get("fundamentals_report"):
                fundamentals_report = sub_val["fundamentals_report"]
            if not news_report and sub_val.get("news_report"):
                news_report = sub_val["news_report"]
            if not volume_price_report and sub_val.get("volume_price_report"):
                volume_price_report = sub_val["volume_price_report"]

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

    # Run detailed per-role depth evaluation
    reports_map = {
        "macro": macro_report,
        "fundamentals": fundamentals_report,
        "news": news_report,
        "volume_price": volume_price_report,
    }
    depth_all_passed, depth_results = check_analyst_depth_quality(
        reports=reports_map,
        market_data_context=market_data_context,
        state=state_or_result,
    )

    overall_passed = passed and depth_all_passed

    if not overall_passed:
        # Record into data_failure_ledger
        if not isinstance(market_data_context, dict):
            market_data_context = {}
            state_or_result["market_data_context"] = market_data_context

        ledger = market_data_context.get("data_failure_ledger")
        if not isinstance(ledger, list):
            ledger = []
            market_data_context["data_failure_ledger"] = ledger

        # 1. Handle general / legacy failure reasons first to preserve ordering
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

        # 2. Handle structured per-role depth failures
        for role, res in depth_results.items():
            if not res["passed"]:
                role_entry = {
                    "source": "report_quality_gate",
                    "role": role,
                    "status": "failed",
                    "score": res["score"],
                    "failed_dimensions": res["failed_dimensions"],
                    "reason": res["reason"],
                    "gap": f"【数据获取失败】report_quality_gate：{res['reason']}",
                }
                found = False
                for existing in ledger:
                    if (
                        isinstance(existing, dict)
                        and existing.get("source") == "report_quality_gate"
                        and existing.get("role") == role
                    ):
                        existing.update(role_entry)
                        found = True
                        break
                if not found:
                    ledger.append(role_entry)

        # Sync data_gaps list if present
        if "data_gaps" in state_or_result and isinstance(state_or_result["data_gaps"], list):
            for entry in ledger:
                if isinstance(entry, dict) and entry.get("source") == "report_quality_gate":
                    gap_str = str(entry.get("gap") or "")
                    if gap_str and gap_str not in state_or_result["data_gaps"]:
                        state_or_result["data_gaps"].append(gap_str)

    return overall_passed
