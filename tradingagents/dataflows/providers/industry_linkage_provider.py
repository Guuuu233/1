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
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import pandas as pd

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
            # 1. LME铜价 (akshare CAD/铜 历史行情)
            if config.name == "LME铜价" or (config.source == "akshare" and config.symbol in ("铜", "CAD")):
                return self._fetch_lme_copper(config, as_of=as_of)

            # 2. 三星电子股价 (yfinance 005930.KS 历史行情)
            if config.name == "三星电子股价" or (config.source == "yfinance" and config.symbol == "005930.KS"):
                return self._fetch_samsung_stock(config, as_of=as_of)

            # 3. 碳酸锂价格 (待接入 API)
            if config.name == "碳酸锂价格" or config.status == "pending_api" or config.note == "待接入API":
                base_dict.update({
                    "current_value": None,
                    "trend": "数据缺失",
                    "confidence": "低（待接入API）",
                    "note": config.note or "待接入API",
                })
                return base_dict

            # 4. 手动录入/标注指标
            if config.status == "manual" or config.source == "manual" or config.note == "手动":
                base_dict.update({
                    "current_value": None,
                    "trend": "数据缺失",
                    "confidence": "低（待手动录入）",
                    "note": config.note or "手动",
                })
                return base_dict

            # 5. 其余默认未实现指标
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
            return base_dict

    def _fetch_lme_copper(
        self,
        config: IndustryLinkageIndicator,
        as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """采集 LME 铜价历史行情并计算最新值、月环比、季度环比与趋势。"""
        result = config.model_dump()
        symbol = config.symbol or "CAD"
        # akshare foreign hist symbol 映射: 铜 -> CAD
        if symbol == "铜":
            symbol = "CAD"

        try:
            import akshare as ak

            with AKSHARE_CALL_LOCK:
                # 调用外盘期货历史行情
                df = ak.futures_foreign_hist(symbol=symbol)

            if df is None or df.empty:
                result.update({
                    "current_value": None,
                    "trend": "数据缺失",
                    "confidence": "低（数据源为空）",
                    "note": "akshare 未返回有效行情记录",
                })
                return result

            metrics = self._calculate_series_metrics(
                df, as_of=as_of, price_col="close", date_col="date"
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
                "note": f"数据源: akshare (代码: {symbol})",
            })
            return result

        except Exception as e:
            logger.warning("IndustryLinkageProvider: 获取 LME铜价 失败: %s", e)
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
        """采集三星电子股价行情并计算最新值、月环比、季度环比与趋势。"""
        result = config.model_dump()
        symbol = config.symbol or "005930.KS"

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
            logger.warning("IndustryLinkageProvider: 获取 三星电子股价 失败: %s", e)
            result.update({
                "current_value": None,
                "trend": "数据缺失",
                "confidence": "低（接口异常）",
                "note": f"数据获取失败: {e}",
            })
            return result

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

        df_work["_std_date"] = pd.to_datetime(df_work[real_date_col], errors="coerce")
        df_work["_std_price"] = pd.to_numeric(df_work[real_price_col], errors="coerce")
        df_work = df_work.dropna(subset=["_std_date", "_std_price"])

        if df_work.empty:
            return None

        # 严格按日期升序排序
        df_work = df_work.sort_values("_std_date", ascending=True).reset_index(drop=True)

        # 防前视纪律过滤
        if as_of:
            try:
                as_of_dt = pd.to_datetime(as_of)
                df_work = df_work[df_work["_std_date"] <= as_of_dt]
            except Exception as e:
                logger.warning("IndustryLinkageProvider: 解析 as_of 日期 '%s' 失败: %s", as_of, e)

        if df_work.empty:
            return None

        total_rows = len(df_work)
        latest_price = float(df_work.iloc[-1]["_std_price"])

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
        }
