"""产业链数据层指标采集器实现 (Industry Linkage Provider).

本模块实现产业链数据层 MVP (DAV-196 / DAV-201 M2) 所需的数据采集器 `IndustryLinkageProvider`：
1. `get_industry_linkage(industry, as_of=None, use_cache=True)`:
   - 依据行业名称获取配置映射；
   - 支持内存 TTL 缓存（默认 1 小时），降低高频重复请求压力；
   - 依次采集上游成本、下游需求、国际对标等指标数据；
2. `_fetch_indicator(config, as_of=None)`:
   - LME铜价：对接 akshare 国际期货日行情（CAD / 伦敦铜），计算最新值、月环比、季度环比与趋势；
   - 三星电子股价：对接 yfinance 行情（005930.KS），计算最新值、月环比、季度环比与趋势；
   - 碳酸锂价格等未接入指标：返回结构化缺失状态 `{"trend": "数据缺失", "confidence": "低（待接入API）"}`；
   - 手动录入指标：返回结构化标注状态 `{"trend": "数据缺失", "confidence": "低（待手动录入）"}`；
3. 容错与防前视纪律：
   - 所有外部调用异常全面捕获，返回结构化错误说明，绝不抛出异常中断上层分析；
   - 支持 `as_of` 参数过滤，严格遵循防前视纪律；
   - 依据 AGENTS.md 规范按列名解析、显式排序，杜绝位置切片与伪造默认值。
"""

import copy
from datetime import datetime, timezone
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from tradingagents.dataflows.industry_linkage import (
    IndustryLinkage,
    IndustryLinkageIndicator,
    get_industry_linkage_config,
)

logger = logging.getLogger(__name__)

# 尝试导入 AKSHARE_CALL_LOCK 细粒度并发锁
try:
    from tradingagents.dataflows.providers.cn_akshare_provider import AKSHARE_CALL_LOCK
except ImportError:
    AKSHARE_CALL_LOCK = threading.Lock()

# 默认缓存时间（1小时）
DEFAULT_CACHE_TTL_SECONDS = 3600

# 环比计算参考交易日步长
APPROX_TRADING_DAYS_PER_MONTH = 22
APPROX_TRADING_DAYS_PER_QUARTER = 63

# 趋势判定阈值 (%)
TREND_UPWARD_THRESHOLD_PCT = 1.0
TREND_DOWNWARD_THRESHOLD_PCT = -1.0

# ── Tushare API 常量与错误码分类 ──
_TUSHARE_DEFAULT_URL = "https://api.tushare.pro"
_TUSHARE_TIMEOUT = 10
_TUSHARE_AUTH_CODES = {2001, 2002, 40101, 40102, 40103}
_TUSHARE_RATE_LIMIT_CODES = {2003, 40203, 40204, 40205, 40206}
_FUTURES_EXCHANGE_SUFFIXES = (".GFE", ".SHF", ".DCE", ".CZC", ".INE", ".CFX", ".ZCE")


def _get_tushare_url() -> str:
    """解析 Tushare API 请求 URL（优先环境变量 TUSHARE_API_URL / TUSHARE_BASE_URL）。"""
    return (
        os.getenv("TUSHARE_API_URL", "").strip()
        or os.getenv("TUSHARE_BASE_URL", "").strip()
        or _TUSHARE_DEFAULT_URL
    )


def _get_tushare_token() -> str:
    """安全读取 Tushare Token，严禁硬编码或打印到日志/错误信息中。"""
    return os.getenv("TUSHARE_TOKEN", "").strip()


def _resolve_tushare_api_name(
    symbol: str, metadata: Optional[Dict[str, Any]] = None
) -> str:
    """根据标的代码或元数据判定 Tushare API 接口类型。

    - 期货标的 (以交易所后缀结尾，如 .GFE, .SHF 等): 调用 fut_daily
    - 全球指数标的 (如 SPX, IXIC 等): 调用 index_global
    """
    if metadata and metadata.get("api_name"):
        return str(metadata["api_name"]).strip()
    sym_upper = symbol.strip().upper()
    if sym_upper.endswith(_FUTURES_EXCHANGE_SUFFIXES):
        return "fut_daily"
    return "index_global"


def _normalize_as_of_date(as_of: Optional[str]) -> Optional[str]:
    """标准化 as_of 日期为 YYYY-MM-DD 字符串，若无法解析则返回 None。"""
    if not as_of:
        return None
    clean = str(as_of).strip()
    if not clean:
        return None
    try:
        dt = pd.to_datetime(clean, format="mixed")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def _query_tushare_api(
    api_name: str,
    ts_code: str,
    as_of: Optional[str] = None,
    fields: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """向 Tushare 发起行情请求并进行响应校验与错误分类。

    Args:
        api_name: 接口名称 (fut_daily / index_global 等)
        ts_code: 证券代码 (如 LC.GFE / SPX)
        as_of: 截止基准日期 (YYYY-MM-DD 或 YYYYMMDD)
        fields: 请求字段列表

    Returns:
        (DataFrame, error_category, error_note): 成功时 DataFrame 非空，失败时返回错误分类与说明
    """
    token = _get_tushare_token()
    if not token:
        return None, "token", "Tushare Token 未配置 (TUSHARE_TOKEN missing)"

    url = _get_tushare_url()
    req_fields = fields or "ts_code,trade_date,open,high,low,close,vol"
    params: Dict[str, Any] = {"ts_code": ts_code}
    if as_of:
        clean_as_of = str(as_of).replace("-", "").strip()
        if len(clean_as_of) == 8 and clean_as_of.isdigit():
            params["end_date"] = clean_as_of

    payload = {
        "api_name": api_name,
        "token": token,
        "params": params,
        "fields": req_fields,
    }

    try:
        resp = requests.post(url, json=payload, timeout=_TUSHARE_TIMEOUT)
    except requests.Timeout as e:
        return None, "timeout", f"Tushare 请求超时: {e}"
    except requests.RequestException as e:
        return None, "network_error", f"Tushare 请求异常: {e}"
    except Exception as e:
        return None, "network_error", f"Tushare 调用未知异常: {e}"

    if resp.status_code == 403:
        return None, "403", "Tushare HTTP 403 权限不足"
    if resp.status_code == 429:
        return None, "rate_limited", "Tushare HTTP 429 请求超限"
    if resp.status_code != 200:
        return None, "http_error", f"Tushare HTTP 错误 (status={resp.status_code})"

    try:
        res_json = resp.json()
    except Exception as e:
        return None, "parse_error", f"Tushare 响应 JSON 解析失败: {e}"

    if not isinstance(res_json, dict):
        return None, "parse_error", "Tushare 响应格式非法 (非 dict)"

    code = res_json.get("code")
    msg = str(res_json.get("msg") or "")

    try:
        code_val = int(code) if code is not None else -1
    except (TypeError, ValueError):
        code_val = -1

    if code_val != 0:
        msg_lower = msg.lower()
        if code_val in _TUSHARE_AUTH_CODES or any(
            k in msg_lower for k in ("权限", "permission", "403", "unauthor")
        ):
            return None, "403", f"Tushare API 权限不足 (code={code}): {msg}"
        if code_val in _TUSHARE_RATE_LIMIT_CODES or any(
            k in msg_lower for k in ("频率", "rate", "limit")
        ):
            return None, "rate_limited", f"Tushare API 触发限频 (code={code}): {msg}"
        return None, "api_error", f"Tushare API 错误 (code={code}): {msg}"

    data = res_json.get("data")
    if not data or not isinstance(data, dict):
        return None, "empty_rows", "Tushare 未返回有效数据（data 字段为空）"

    resp_fields = data.get("fields")
    items = data.get("items")
    if not items or not isinstance(items, list) or len(items) == 0:
        return None, "empty_rows", "Tushare 未返回有效行情记录（空行）"

    try:
        df = pd.DataFrame(items, columns=resp_fields)
        return df, None, None
    except Exception as e:
        return None, "parse_error", f"Tushare 数据构造 DataFrame 失败: {e}"


class IndustryLinkageProvider:
    """产业链数据层指标采集器，负责行业上下游联动数据拉取与计算。"""

    def __init__(self, cache_ttl: int = DEFAULT_CACHE_TTL_SECONDS):
        """初始化采集器与内存缓存。

        Args:
            cache_ttl: 内存缓存有效时长（秒），默认 3600 秒 (1 小时)
        """
        self._cache_ttl = cache_ttl
        self._cache: Dict[tuple[str, Optional[str]], Dict[str, Any]] = {}
        self._cache_timestamps: Dict[tuple[str, Optional[str]], float] = {}
        self._lock = threading.Lock()

    def clear_cache(self) -> None:
        """清空内存缓存。"""
        with self._lock:
            self._cache.clear()
            self._cache_timestamps.clear()

    def get_industry_linkage(
        self,
        industry: str,
        as_of: Optional[str] = None,
        use_cache: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """获取指定行业的产业链联动数据。

        Args:
            industry: 行业名称或行业关键词 (如 "消费电子", "新能源车")
            as_of: 分析基准日期 (YYYY-MM-DD 或 YYYYMMDD)，防前视截止日期
            use_cache: 是否使用内存缓存，默认 True

        Returns:
            结构化的行业产业链数据字典，若未找到行业配置则返回 None
        """
        if not industry or not isinstance(industry, str):
            logger.warning("IndustryLinkageProvider: 无效的行业参数 %s", industry)
            return None

        config: Optional[IndustryLinkage] = get_industry_linkage_config(industry)
        if config is None:
            logger.info("IndustryLinkageProvider: 未找到行业 '%s' 的产业链配置映射", industry)
            return None

        cache_key = (config.industry_name, as_of)
        now = time.time()

        if use_cache:
            with self._lock:
                if cache_key in self._cache:
                    cached_time = self._cache_timestamps.get(cache_key, 0.0)
                    if (now - cached_time) < self._cache_ttl:
                        logger.debug("IndustryLinkageProvider: 命中缓存 %s", cache_key)
                        return copy.deepcopy(self._cache[cache_key])

        # 依次采集各维度指标数据
        upstream_results = [
            self._fetch_indicator(ind, as_of=as_of) for ind in config.upstream_cost
        ]
        downstream_results = [
            self._fetch_indicator(ind, as_of=as_of) for ind in config.downstream_demand
        ]
        benchmark_results = [
            self._fetch_indicator(ind, as_of=as_of) for ind in config.international_benchmark
        ]

        finished_time = time.time()
        result_payload: Dict[str, Any] = {
            "industry_name": config.industry_name,
            "upstream_cost": upstream_results,
            "downstream_demand": downstream_results,
            "international_benchmark": benchmark_results,
            "policy_catalysts": list(config.policy_catalysts),
            "description": config.description,
            "as_of": as_of,
            "cached_at": finished_time,
        }

        with self._lock:
            self._cache[cache_key] = copy.deepcopy(result_payload)
            self._cache_timestamps[cache_key] = finished_time

        return result_payload

    def _fetch_indicator(
        self,
        config: IndustryLinkageIndicator,
        as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """根据指标配置拉取单项指标数据并计算环比趋势。

        Args:
            config: 指标配置定义对象
            as_of: 基准日期

        Returns:
            包含采集数值与趋势分析的结构化指标字典
        """
        # 基础数据字典备份
        base_dict = config.model_dump()

        try:
            # 1. 待接入 API 指标 (显式标记 pending_api)
            if config.status == "pending_api" or config.source == "pending_api" or config.note == "待接入API":
                base_dict.update({
                    "current_value": None,
                    "trend": "数据缺失",
                    "confidence": "低（待接入API）",
                    "note": config.note or "待接入API",
                })
                return base_dict

            # 2. 手动录入/标注指标 (显式标记 manual)
            if config.status == "manual" or config.source == "manual" or config.note == "手动":
                base_dict.update({
                    "current_value": None,
                    "trend": "数据缺失",
                    "confidence": "低（待手动录入）",
                    "note": config.note or "手动",
                })
                return base_dict

            # 3. LME铜价 / akshare 期货历史行情 (支持 Tushare 沪铜 CU.SHF 备源)
            if config.name == "LME铜价" or (config.source == "akshare" and config.symbol in ("铜", "CAD")):
                return self._fetch_lme_copper(config, as_of=as_of)

            # 4. Tushare 付费数据源标的 (碳酸锂 LC.GFE、多晶硅 PS.GFE、全球指数 SPX 等)
            if config.source == "tushare":
                return self._fetch_tushare_indicator(config, as_of=as_of)

            # 5. yfinance 标的 (三星电子股价、费城半导体指数、台积电、布伦特原油、埃克森美孚、摩根大通、标普500金融指数等)
            if config.source == "yfinance" and config.symbol:
                return self._fetch_yfinance_indicator(config, as_of=as_of)

            # 6. 其余默认未实现指标
            base_dict.update({
                "current_value": None,
                "trend": "数据缺失",
                "confidence": "低（待实现）",
                "note": config.note or "未接入",
            })
            return base_dict

        except Exception as e:
            logger.warning("IndustryLinkageProvider: 采集指标 '%s' 发生异常: %s", config.name, e)
            base_dict.update({
                "current_value": None,
                "trend": "数据缺失",
                "confidence": "低（接口异常）",
                "note": f"数据获取失败: {e}",
            })
            if config.source == "tushare":
                base_dict.update({
                    "requested_as_of": as_of,
                    "actual_as_of": None,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "transport_provider": "tushare",
                    "api_name": _resolve_tushare_api_name(config.symbol or "", config.metadata),
                    "category": "api_error",
                })
            return base_dict

    def _fetch_tushare_indicator(
        self,
        config: IndustryLinkageIndicator,
        as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """采集 Tushare 标的历史行情并计算最新值、月环比、季度环比与趋势，附带完整 Provenance 证据链。"""
        result = config.model_dump()
        retrieved_at = datetime.now(timezone.utc).isoformat()
        symbol = config.symbol
        api_name = _resolve_tushare_api_name(symbol or "", config.metadata)

        if not symbol:
            result.update({
                "current_value": None,
                "trend": "数据缺失",
                "confidence": "低（代码缺失）",
                "note": "未配置有效的 Tushare 证券代码",
                "requested_as_of": as_of,
                "actual_as_of": None,
                "retrieved_at": retrieved_at,
                "transport_provider": "tushare",
                "api_name": api_name,
                "category": "symbol_missing",
            })
            return result

        df, err_cat, err_note = _query_tushare_api(
            api_name=api_name,
            ts_code=symbol,
            as_of=as_of,
        )

        if df is None:
            conf_map = {
                "token": "低（Token缺失）",
                "403": "低（无权限403）",
                "rate_limited": "低（频率限制）",
                "empty_rows": "低（数据源为空）",
                "timeout": "低（网络超时）",
                "network_error": "低（接口异常）",
                "http_error": "低（接口异常）",
                "api_error": "低（接口异常）",
                "parse_error": "低（接口异常）",
            }
            confidence = conf_map.get(err_cat or "", "低（接口异常）")
            result.update({
                "current_value": None,
                "trend": "数据缺失",
                "confidence": confidence,
                "note": err_note or "Tushare 数据获取失败",
                "requested_as_of": as_of,
                "actual_as_of": None,
                "retrieved_at": retrieved_at,
                "transport_provider": "tushare",
                "api_name": api_name,
                "category": err_cat or "api_error",
            })
            return result

        metrics = self._calculate_series_metrics(
            df, as_of=as_of, price_col="close", date_col="trade_date"
        )

        if not metrics:
            result.update({
                "current_value": None,
                "trend": "数据缺失",
                "confidence": "低（有效数据不足）",
                "note": "无符合截止日期的有效价格序列",
                "requested_as_of": as_of,
                "actual_as_of": None,
                "retrieved_at": retrieved_at,
                "transport_provider": "tushare",
                "api_name": api_name,
                "category": "empty_rows",
            })
            return result

        actual_as_of = metrics.get("actual_as_of")
        normalized_req = _normalize_as_of_date(as_of)

        # 严格防前视纪律校验：actual_as_of <= requested_as_of，不满足时 fail-closed
        if normalized_req is not None and actual_as_of is not None and actual_as_of > normalized_req:
            logger.warning(
                "IndustryLinkageProvider: 防前视校验失败，actual_as_of (%s) > requested_as_of (%s)",
                actual_as_of,
                as_of,
            )
            result.update({
                "current_value": None,
                "mom_change": None,
                "qoq_change": None,
                "trend": "数据缺失",
                "confidence": "低（前视偏差异常）",
                "note": f"防前视校验失败: 实际数据日期 ({actual_as_of}) 晚于请求基准日期 ({as_of})",
                "requested_as_of": as_of,
                "actual_as_of": None,
                "retrieved_at": retrieved_at,
                "transport_provider": "tushare",
                "api_name": api_name,
                "category": "lookahead_violation",
            })
            return result

        result.update({
            "current_value": metrics["current_value"],
            "mom_change": metrics["mom_change"],
            "qoq_change": metrics["qoq_change"],
            "trend": metrics["trend"],
            "confidence": "高",
            "status": "active",
            "note": f"数据源: tushare (接口: {api_name}, 代码: {symbol})",
            "requested_as_of": as_of,
            "actual_as_of": actual_as_of,
            "retrieved_at": retrieved_at,
            "transport_provider": "tushare",
            "api_name": api_name,
        })
        return result

    def _fetch_lme_copper(
        self,
        config: IndustryLinkageIndicator,
        as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """采集 LME 铜价历史行情（优先 akshare 伦敦铜，失败时回退 Tushare 沪铜 CU.SHF 备源）。"""
        result = config.model_dump()
        symbol = config.symbol or "CAD"
        # akshare foreign hist symbol 映射: 铜 -> CAD
        if symbol == "铜":
            symbol = "CAD"

        akshare_err_note = None
        akshare_conf = "低（接口异常）"
        try:
            import akshare as ak

            with AKSHARE_CALL_LOCK:
                # 调用外盘期货历史行情
                df = ak.futures_foreign_hist(symbol=symbol)

            if df is not None and not df.empty:
                metrics = self._calculate_series_metrics(
                    df, as_of=as_of, price_col="close", date_col="date"
                )
                if metrics:
                    result.update({
                        "current_value": metrics["current_value"],
                        "mom_change": metrics["mom_change"],
                        "qoq_change": metrics["qoq_change"],
                        "trend": metrics["trend"],
                        "confidence": "高",
                        "status": "active",
                        "note": f"数据源: akshare (代码: {symbol})",
                    })
                    return result
                else:
                    akshare_err_note = "无符合截止日期的有效价格序列"
                    akshare_conf = "低（有效数据不足）"
            else:
                akshare_err_note = "akshare 未返回有效行情记录"
                akshare_conf = "低（数据源为空）"

        except Exception as e:
            logger.warning("IndustryLinkageProvider: 获取 LME铜价 (akshare) 失败: %s", e)
            akshare_err_note = str(e)
            akshare_conf = "低（接口异常）"

        # akshare 不可用时，尝试 Tushare fut_daily CU.SHF 作为备用数据源
        retrieved_at_ts = datetime.now(timezone.utc).isoformat()
        df_ts, ts_err_cat, ts_err_note = _query_tushare_api(
            api_name="fut_daily",
            ts_code="CU.SHF",
            as_of=as_of,
        )
        if df_ts is not None and not df_ts.empty:
            metrics_ts = self._calculate_series_metrics(
                df_ts, as_of=as_of, price_col="close", date_col="trade_date"
            )
            if metrics_ts:
                actual_as_of_ts = metrics_ts.get("actual_as_of")
                normalized_req = _normalize_as_of_date(as_of)
                if not (normalized_req is not None and actual_as_of_ts is not None and actual_as_of_ts > normalized_req):
                    result.update({
                        "current_value": metrics_ts["current_value"],
                        "mom_change": metrics_ts["mom_change"],
                        "qoq_change": metrics_ts["qoq_change"],
                        "trend": metrics_ts["trend"],
                        "confidence": "高",
                        "status": "active",
                        "note": "数据源: tushare (备源接口: fut_daily, 代码: CU.SHF)",
                        "requested_as_of": as_of,
                        "actual_as_of": actual_as_of_ts,
                        "retrieved_at": retrieved_at_ts,
                        "transport_provider": "tushare",
                        "api_name": "fut_daily",
                    })
                    return result

        result.update({
            "current_value": None,
            "trend": "数据缺失",
            "confidence": akshare_conf,
            "note": (
                f"akshare 失败 ({akshare_err_note})；Tushare 备源失败 ({ts_err_note or '未知'})"
                if akshare_conf == "低（接口异常）"
                else (akshare_err_note or "未获取到有效数据")
            ),
        })
        return result

    def _fetch_yfinance_indicator(
        self,
        config: IndustryLinkageIndicator,
        as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """采集 yfinance 标的历史行情并计算最新值、月环比、季度环比与趋势。"""
        result = config.model_dump()
        symbol = config.symbol
        if not symbol:
            result.update({
                "current_value": None,
                "trend": "数据缺失",
                "confidence": "低（代码缺失）",
                "note": "未配置有效的 yfinance 证券代码",
            })
            return result

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            if as_of:
                try:
                    as_of_dt = pd.to_datetime(as_of)
                    start_dt = as_of_dt - pd.Timedelta(days=120)
                    end_dt = as_of_dt + pd.Timedelta(days=1)
                    data = ticker.history(
                        start=start_dt.strftime("%Y-%m-%d"),
                        end=end_dt.strftime("%Y-%m-%d"),
                    )
                except Exception:
                    data = ticker.history(period="3mo")
            else:
                data = ticker.history(period="3mo")

            if data is None or data.empty:
                result.update({
                    "current_value": None,
                    "trend": "数据缺失",
                    "confidence": "低（数据源为空）",
                    "note": "yfinance 未返回有效行情记录",
                })
                return result

            # 清理时区与索引
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            df = data.reset_index()

            metrics = self._calculate_series_metrics(
                df, as_of=as_of, price_col="Close", date_col="Date"
            )

            if not metrics:
                result.update({
                    "current_value": None,
                    "trend": "数据缺失",
                    "confidence": "低（有效数据不足）",
                    "note": "无符合截止日期的有效价格序列",
                })
                return result

            result.update({
                "current_value": metrics["current_value"],
                "mom_change": metrics["mom_change"],
                "qoq_change": metrics["qoq_change"],
                "trend": metrics["trend"],
                "confidence": "高",
                "status": "active",
                "note": f"数据源: yfinance (代码: {symbol})",
            })
            return result

        except Exception as e:
            logger.warning("IndustryLinkageProvider: 获取 %s (%s) 失败: %s", config.name, symbol, e)
            result.update({
                "current_value": None,
                "trend": "数据缺失",
                "confidence": "低（接口异常）",
                "note": f"数据获取失败: {e}",
            })
            return result

    def _fetch_samsung_stock(
        self,
        config: IndustryLinkageIndicator,
        as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """采集三星电子股价行情（委托给 _fetch_yfinance_indicator 统一处理）。"""
        return self._fetch_yfinance_indicator(config, as_of=as_of)

    def _calculate_series_metrics(
        self,
        df: pd.DataFrame,
        as_of: Optional[str] = None,
        price_col: str = "close",
        date_col: str = "date",
    ) -> Optional[Dict[str, Any]]:
        """根据历史时序数据计算最新价、月环比、季度环比与趋势状态。

        Args:
            df: 原始行情 DataFrame
            as_of: 截止基准日期
            price_col: 收盘价列名候选
            date_col: 日期列名候选

        Returns:
            计算后的度量字典，数据不满足时返回 None
        """
        if df is None or df.empty:
            return None

        df_work = df.copy()

        # 按列名不区分大小写匹配
        col_lower_map = {str(c).lower(): c for c in df_work.columns}

        # 匹配日期列
        real_date_col = None
        for cand in (date_col.lower(), "date", "日期", "trade_date", "datetime", "index"):
            if cand in col_lower_map:
                real_date_col = col_lower_map[cand]
                break
        if real_date_col is None:
            return None

        # 匹配价格列
        real_price_col = None
        for cand in (price_col.lower(), "close", "收盘", "收盘价", "adj close", "value"):
            if cand in col_lower_map:
                real_price_col = col_lower_map[cand]
                break
        if real_price_col is None:
            return None

        # 日期安全解析（兼容 int YYYYMMDD, str YYYYMMDD, ISO 字符串与 datetime 对象）
        date_series = df_work[real_date_col]
        if pd.api.types.is_datetime64_any_dtype(date_series):
            df_work["_std_date"] = pd.to_datetime(date_series, errors="coerce")
        else:
            df_work["_std_date"] = pd.to_datetime(
                date_series.astype(str), format="mixed", errors="coerce"
            )
        df_work["_std_price"] = pd.to_numeric(df_work[real_price_col], errors="coerce")
        df_work = df_work.dropna(subset=["_std_date", "_std_price"])

        if df_work.empty:
            return None

        # 严格按日期升序排序
        df_work = df_work.sort_values("_std_date", ascending=True).reset_index(drop=True)

        # 防前视纪律过滤
        if as_of:
            try:
                as_of_str = str(as_of).strip()
                as_of_dt = pd.to_datetime(as_of_str, format="mixed")
                df_work = df_work[df_work["_std_date"] <= as_of_dt]
            except Exception as e:
                logger.warning("IndustryLinkageProvider: 解析 as_of 日期 '%s' 失败: %s", as_of, e)

        if df_work.empty:
            return None

        total_rows = len(df_work)
        latest_row = df_work.iloc[-1]
        latest_price = float(latest_row["_std_price"])
        actual_as_of = latest_row["_std_date"].strftime("%Y-%m-%d")

        # 计算月环比 (MoM)
        mom_change: Optional[float] = None
        if total_rows > APPROX_TRADING_DAYS_PER_MONTH:
            base_idx = total_rows - 1 - APPROX_TRADING_DAYS_PER_MONTH
            base_price = float(df_work.iloc[base_idx]["_std_price"])
            if base_price > 0:
                mom_change = round(((latest_price - base_price) / base_price) * 100.0, 2)
        elif total_rows >= 2:
            # 数据样本较少时用首个样本作为月度参考
            base_price = float(df_work.iloc[0]["_std_price"])
            if base_price > 0:
                mom_change = round(((latest_price - base_price) / base_price) * 100.0, 2)

        # 计算季度环比 (QoQ)
        qoq_change: Optional[float] = None
        if total_rows > APPROX_TRADING_DAYS_PER_QUARTER:
            base_idx = total_rows - 1 - APPROX_TRADING_DAYS_PER_QUARTER
            base_price = float(df_work.iloc[base_idx]["_std_price"])
            if base_price > 0:
                qoq_change = round(((latest_price - base_price) / base_price) * 100.0, 2)

        # 判定趋势
        if mom_change is not None:
            if mom_change >= TREND_UPWARD_THRESHOLD_PCT:
                trend = "上升"
            elif mom_change <= TREND_DOWNWARD_THRESHOLD_PCT:
                trend = "下降"
            else:
                trend = "平稳"
        else:
            trend = "数据缺失"

        return {
            "current_value": round(latest_price, 2),
            "mom_change": mom_change,
            "qoq_change": qoq_change,
            "trend": trend,
            "actual_as_of": actual_as_of,
        }
