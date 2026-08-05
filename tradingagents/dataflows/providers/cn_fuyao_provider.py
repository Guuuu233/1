"""A-share market data via 同花顺 (THS) 金融数据 API (fuyao.aicubes.cn).

覆盖：行情快照（批量）、历史日 K（前复权）、三大报表、财务指标（五类能力）、
涨跌停池 / 连板天梯、龙虎榜、交易日历。统一 ``ApiResponse`` 信封，错误码按
``code`` 映射到现有 vendor 链语义（见 ``tradingagents/dataflows/vendor_result.py``）：

- ``0`` 成功
- ``1001~1004`` 参数错误 —— 客户端 bug，显式 ``ValueError``
- ``2001/2003`` Key 无效 / 无权限 —— 显式 ``NotImplementedError``
- ``3001`` 标的不存在、``3002`` 数据未就绪 —— ``VendorEmpty``（确认无数据）
- ``4001`` 频率超限 —— 退避重试后仍失败走 ``VendorFail``（切换备用源）
- ``5001~5003`` 服务端错误 —— ``VendorFail``

API Key 读取顺序：配置 ``fuyao_api_key``，其次环境变量 ``FUYAO_API_KEY``。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from .base import BaseMarketDataProvider
from ..config import get_config
from ..trade_calendar import (
    CN_TZ,
    DateDataUnavailable,
    dedupe_daily_bars,
    drop_incomplete_today_bar,
    fetch_with_date_fallback,
    snapshot_historical_refusal,
)
from ..utils import format_hist_csv, safe_float, shrink_table, slice_hist_df
from ..vendor_result import VendorEmpty, VendorFail

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://fuyao.aicubes.cn"

_SNAPSHOT_PATH = "/api/a-share/prices/snapshot"
_HISTORICAL_PATH = "/api/a-share/prices/historical"
_FINANCIALS_BASE = "/api/a-share/financials"
_INDICATORS_PATH = "/api/a-share/financials/indicators"
_LIMIT_UP_POOL_PATH = "/api/a-share/special-data/limit-up-pool"
_LIMIT_UP_LADDER_PATH = "/api/a-share/special-data/limit-up-ladder"
_DRAGON_TIGER_LIST_PATH = "/api/a-share/special-data/dragon-tiger-list"
_TRADING_DAYS_PATH = "/api/a-share/calendar/trading-days"

_REQUEST_TIMEOUT_SECONDS = 20.0
_RATE_LIMIT_RETRIES = 2
_RATE_LIMIT_BACKOFF_SECONDS = 1.0
# 批量行情快照单次请求的 thscode 数量上限（避免 URL 超长），超出则分块。
_SNAPSHOT_BATCH_SIZE = 100
_ZT_POOL_PAGE_SIZE = 200
_ZT_POOL_MAX_PAGES = 20
_ONE_YEAR_DAYS = 365

_FINANCIAL_ENDPOINTS = {
    "income": "income-statements",
    "balance": "balance-sheets",
    "cashflow": "cash-flow-statements",
}

_ABILITY_LABELS = {
    "growth": "成长能力",
    "profitability": "盈利能力",
    "solvency": "偿债能力",
    "operation": "营运能力",
    "cash-flow": "现金流",
}


class FuyaoApiError(Exception):
    """业务错误：HTTP 恒为 200，错误经信封 ``code`` 字段表达。"""

    def __init__(self, code: int, message: str):
        super().__init__(f"code={code} message={message}")
        self.code = int(code)
        self.message = str(message or "")


class CnFuyaoProvider(BaseMarketDataProvider):
    """同花顺金融数据 API：行情、日 K、财报、财务指标、涨跌停池、龙虎榜、交易日历。"""

    _RATE_LIMIT_RETRIES = _RATE_LIMIT_RETRIES
    _RATE_LIMIT_BACKOFF_SECONDS = _RATE_LIMIT_BACKOFF_SECONDS

    @property
    def name(self) -> str:
        return "cn_fuyao"

    # ── 配置 ──────────────────────────────────────────────────────────

    def _resolve_api_key(self) -> str:
        config = get_config()
        return (
            str(config.get("fuyao_api_key", "")).strip()
            or os.getenv("FUYAO_API_KEY", "").strip()
        )

    def _require_api_key(self) -> str:
        key = self._resolve_api_key()
        if not key:
            raise NotImplementedError(
                "cn_fuyao 需要 API Key。请在配置中设置 fuyao_api_key "
                "或环境变量 FUYAO_API_KEY。"
            )
        return key

    def _resolve_base_url(self) -> str:
        config = get_config()
        return (
            str(config.get("fuyao_base_url", "")).strip()
            or os.getenv("FUYAO_BASE_URL", "").strip()
            or _DEFAULT_BASE_URL
        ).rstrip("/")

    # ── 工具方法 ──────────────────────────────────────────────────────

    @staticmethod
    def _normalize_thscode(symbol: str) -> str | None:
        """把任意写法（600519 / 600519.SH / 600519.SS / SH600519）归一到 thscode。

        A 股后缀规则：60/68 开头 → ``.SH``；00/30 开头 → ``.SZ``；8/4/92 开头 → ``.BJ``。
        无法识别返回 None。
        """
        s = str(symbol or "").strip().upper()
        if not s:
            return None
        m = re.search(r"(\d{6})", s)
        if not m:
            return None
        code = m.group(1)
        if ".SH" in s or ".SS" in s:
            exchange = "SH"
        elif ".SZ" in s:
            exchange = "SZ"
        elif ".BJ" in s:
            exchange = "BJ"
        elif code.startswith(("60", "68")):
            exchange = "SH"
        elif code.startswith(("00", "30")):
            exchange = "SZ"
        elif code.startswith(("8", "4", "92")):
            exchange = "BJ"
        else:
            return None
        return f"{code}.{exchange}"

    @staticmethod
    def _ms_to_date_str(ms: Any) -> str | None:
        """毫秒 Unix 时间戳（Asia/Shanghai）→ ``YYYY-MM-DD``。"""
        try:
            f = float(ms)
        except (TypeError, ValueError):
            return None
        return datetime.fromtimestamp(f / 1000.0, tz=CN_TZ).strftime("%Y-%m-%d")

    @staticmethod
    def _date_to_ms(date_str: str) -> int:
        """``YYYY-MM-DD``（Asia/Shanghai 零点）→ 毫秒 Unix 时间戳。"""
        d = datetime.strptime(str(date_str).strip(), "%Y-%m-%d").replace(tzinfo=CN_TZ)
        return int(d.timestamp() * 1000)

    @staticmethod
    def _is_quarterly_freq(freq: str) -> bool:
        f = (freq or "").strip().lower()
        return f in ("quarterly", "quarter", "q")

    def _request_fuyao(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """发起 GET 并校验信封；业务错误抛 :class:`FuyaoApiError`。

        网络异常直接抛 ``requests.RequestException`` 子类，由调用方转 ``VendorFail``。
        4001 频率超限在函数内退避重试（``_RATE_LIMIT_RETRIES`` 次），仍失败抛 4001。
        """
        api_key = self._require_api_key()
        base_url = self._resolve_base_url()
        url = f"{base_url}{path}"
        headers = {"X-api-key": api_key}
        clean_params = {k: v for k, v in params.items() if v is not None}

        attempts = self._RATE_LIMIT_RETRIES + 1
        for attempt in range(attempts):
            resp = requests.get(
                url,
                params=clean_params,
                headers=headers,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict):
                raise FuyaoApiError(-1, "响应信封格式异常（顶层非 dict）")
            code = payload.get("code")
            if code == 0:
                return payload
            if code == 4001 and attempt < attempts - 1:
                time.sleep(self._RATE_LIMIT_BACKOFF_SECONDS * (attempt + 1))
                continue
            raise FuyaoApiError(
                int(code) if isinstance(code, int) else -1,
                str(payload.get("message") or "未知错误"),
            )

        raise FuyaoApiError(4001, "频率超限（重试后仍失败）")  # pragma: no cover

    def _map_api_error(self, exc: FuyaoApiError) -> Any:
        """按错误码把 :class:`FuyaoApiError` 转成 vendor 链语义（返回类型或抛异常）。"""
        if exc.code in (1001, 1002, 1003, 1004):
            raise ValueError(f"[cn_fuyao] 参数错误 code={exc.code}: {exc.message}")
        if exc.code in (2001, 2003):
            raise NotImplementedError(
                f"[cn_fuyao] API Key 无效或无权限 code={exc.code}: {exc.message}"
            )
        if exc.code in (3001, 3002, 3004):
            return VendorEmpty(f"[cn_fuyao] {exc.message}（code={exc.code}）")
        if exc.code == 4001:
            return VendorFail(f"[cn_fuyao] 频率超限 code=4001: {exc.message}")
        if exc.code in (5001, 5002, 5003):
            return VendorFail(f"[cn_fuyao] 服务端错误 code={exc.code}: {exc.message}")
        return VendorFail(f"[cn_fuyao] 未知错误码 code={exc.code}: {exc.message}")

    def _request_or_map(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """调用接口；业务错误码映射为 vendor 语义结果，调用方用返回值判定。"""
        try:
            return self._request_fuyao(path, params)
        except FuyaoApiError as exc:
            outcome = self._map_api_error(exc)
            if isinstance(outcome, (VendorEmpty, VendorFail)):
                raise _MappedVendorOutcome(outcome) from exc
            raise  # pragma: no cover —— _map_api_error 对未知码返回 VendorFail，不会走到这里

    @staticmethod
    def _shrink_table(
        df: pd.DataFrame,
        max_rows: int = 12,
        max_cols: int = 16,
        *,
        table_kind: str | None = "generic",
        require_core_fields: bool = False,
        max_prompt_chars: int | None = None,
    ) -> str:
        """按名称选择列的 LLM 注入表格渲染（对齐现有 provider）。"""
        kwargs = {
            "max_rows": max_rows,
            "table_kind": table_kind,
            "require_core_fields": require_core_fields,
        }
        if max_prompt_chars is not None:
            kwargs["max_prompt_chars"] = max_prompt_chars
        _ = max_cols  # positional column cuts are forbidden
        return shrink_table(df, **kwargs)

    # ── 行情快照 / 历史 K 线 ──────────────────────────────────────────

    def get_realtime_quotes(self, symbols: list[str], curr_date: str = None) -> str:
        """批量行情快照：``GET /api/a-share/prices/snapshot``（thscodes 逗号分隔）。"""
        refusal = snapshot_historical_refusal(
            curr_date, source_label="实时行情（同花顺 fuyao 快照）"
        )
        if refusal:
            return refusal

        original_by_thscode: dict[str, str] = {}
        for s in symbols:
            if not s or not str(s).strip():
                continue
            thscode = self._normalize_thscode(str(s))
            if thscode and thscode not in original_by_thscode:
                original_by_thscode[thscode] = str(s).strip().upper()

        if not original_by_thscode:
            return json.dumps({})

        result: dict[str, dict[str, Any]] = {}
        thscodes = list(original_by_thscode.keys())
        try:
            for i in range(0, len(thscodes), _SNAPSHOT_BATCH_SIZE):
                chunk = thscodes[i : i + _SNAPSHOT_BATCH_SIZE]
                payload = self._request_or_map(
                    _SNAPSHOT_PATH, {"thscodes": ",".join(chunk)}
                )
                data = payload.get("data") or {}
                snapshot_ts = data.get("timestamp")
                items = data.get("item") or []
                for row in items:
                    if not isinstance(row, dict):
                        continue
                    ths = row.get("thscode")
                    if ths not in original_by_thscode:
                        continue
                    result[original_by_thscode[ths]] = self._map_snapshot_row(
                        row, snapshot_ts
                    )
        except _MappedVendorOutcome as exc:
            raise NotImplementedError(
                f"cn_fuyao 实时行情请求失败：{exc.outcome.to_prompt()}"
            ) from exc

        if not result:
            raise NotImplementedError(
                "cn_fuyao 未获取到任何实时行情（请检查 thscode 与 API Key）。"
            )
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _map_snapshot_row(row: dict[str, Any], snapshot_ts: Any) -> dict[str, Any]:
        price = safe_float(row.get("last_price"))
        prev = safe_float(row.get("prev_price"))
        change = None
        if price is not None and prev is not None:
            change = round(price - prev, 4)
        change_pct = safe_float(row.get("price_change_ratio_pct"))
        return {
            "price": price,
            "open": safe_float(row.get("open_price")),
            "high": safe_float(row.get("high_price")),
            "low": safe_float(row.get("low_price")),
            "previous_close": prev,
            "change": change,
            "change_pct": change_pct,
            "volume": safe_float(row.get("volume")),
            "amount": safe_float(row.get("turnover")),
            "quote_time": CnFuyaoProvider._ms_to_date_str(snapshot_ts),
            "source": "fuyao",
        }

    def _fetch_historical_df(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """历史日 K（前复权）：``GET /api/a-share/prices/historical``。"""
        thscode = self._normalize_thscode(symbol)
        if not thscode:
            raise ValueError(f"[cn_fuyao] 无法解析证券代码: {symbol}")

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        if end_dt - start_dt > timedelta(days=365 * 10):
            raise ValueError(
                f"[cn_fuyao] 历史K线窗口超过 10 年上限: {start_date} ~ {end_date}"
            )

        params = {
            "thscode": thscode,
            "interval": "1d",
            "start": self._date_to_ms(start_date),
            "end": self._date_to_ms(end_date),
            "adjust": "forward",
        }
        payload = self._request_or_map(_HISTORICAL_PATH, params)
        items = ((payload.get("data") or {}).get("item")) or []
        recs: list[dict[str, Any]] = []
        for r in items:
            if not isinstance(r, dict):
                continue
            date_str = self._ms_to_date_str(r.get("date_ms"))
            if not date_str:
                continue
            recs.append(
                {
                    "Date": date_str,
                    "Open": safe_float(r.get("open_price")),
                    "High": safe_float(r.get("high_price")),
                    "Low": safe_float(r.get("low_price")),
                    "Close": safe_float(r.get("close_price")),
                    "Volume": safe_float(r.get("volume")),
                }
            )
        df = pd.DataFrame(recs)
        if df.empty:
            return df
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for c in ("Open", "High", "Low", "Close", "Volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
        df["Volume"] = df["Volume"].astype(float)
        return dedupe_daily_bars(
            df, "Date", ["Open", "High", "Low", "Close", "Volume"]
        )

    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> str:
        """前复权日 K 线，输出与 AkShare 一致的 CSV 头。"""
        try:
            df = self._fetch_historical_df(symbol, start_date, end_date)
        except _MappedVendorOutcome as exc:
            return exc.outcome
        df = slice_hist_df(df, start_date, end_date)
        df = drop_incomplete_today_bar(df, "Date", end_date)
        if df is None or df.empty:
            return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
        return format_hist_csv(df, symbol, start_date, end_date)

    # ── 三大报表 / 财务指标 ───────────────────────────────────────────

    def _financial_report_markdown(
        self, kind: str, title_cn: str, ticker: str, freq: str, curr_date: str | None
    ) -> Any:
        """三大报表：``period`` 按 freq 选择，``start``/``end`` 时间区间模式。

        ``curr_date`` 必填（内部层不得默认今天）；窗口以 curr_date 为终点，
        避免把未来报告期注入历史分析。
        """
        if not curr_date:
            return (
                f"【数据获取失败】{title_cn} 缺少 curr_date，"
                "内部层不得默认今天，本项不可用。"
            )
        thscode = self._normalize_thscode(ticker)
        if not thscode:
            raise ValueError(f"[cn_fuyao] 无法解析证券代码: {ticker}")

        period = "quarterly" if self._is_quarterly_freq(freq) else "annual"
        window_years = 3 if period == "quarterly" else 8
        end_ms = self._date_to_ms(curr_date)
        start_ms = self._date_to_ms(
            (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=365 * window_years))
            .strftime("%Y-%m-%d")
        )
        path = f"{_FINANCIALS_BASE}/{_FINANCIAL_ENDPOINTS[kind]}"
        params = {
            "thscode": thscode,
            "period": period,
            "start": start_ms,
            "end": end_ms,
        }
        try:
            payload = self._request_fuyao(path, params)
        except FuyaoApiError as exc:
            return self._map_api_error(exc)

        items = ((payload.get("data") or {}).get("item")) or []
        if not items:
            return (
                f"## {title_cn} ({ticker})\n\n"
                f"未获取到报表数据（同花顺 fuyao {path}，截至 {curr_date}）。"
            )
        df = pd.DataFrame(
            [{k: v for k, v in r.items()} for r in items if isinstance(r, dict)]
        )
        table = self._shrink_table(df, max_rows=12, max_cols=18, table_kind="generic")
        freq_note = "单季度" if period == "quarterly" else "报告期口径以接口为准"
        return (
            f"## {title_cn} ({ticker}) — 同花顺 fuyao {path}"
            f"（{freq_note}，截至 {curr_date}）\n\n{table}"
        )

    def get_balance_sheet(
        self, ticker: str, freq: str = "quarterly", curr_date: str = None
    ) -> Any:
        return self._financial_report_markdown(
            "balance", "资产负债表", ticker, freq, curr_date
        )

    def get_cashflow(
        self, ticker: str, freq: str = "quarterly", curr_date: str = None
    ) -> Any:
        return self._financial_report_markdown(
            "cashflow", "现金流量表", ticker, freq, curr_date
        )

    def get_income_statement(
        self, ticker: str, freq: str = "quarterly", curr_date: str = None
    ) -> Any:
        return self._financial_report_markdown(
            "income", "利润表", ticker, freq, curr_date
        )

    @staticmethod
    def _latest_report_period(curr_date: str | None) -> str:
        """分析日 → 最接近且通常已披露的报告期（``yyyy-N``）。

        披露截止（约）：一季报 4/30、中报 8/31、三季报 10/31、年报次年 4/30。
        """
        if curr_date:
            d = datetime.strptime(str(curr_date).strip(), "%Y-%m-%d")
        else:
            d = datetime.now(CN_TZ)
        md = (d.month, d.day)
        if md >= (10, 31):
            return f"{d.year}-3"
        if md >= (8, 31):
            return f"{d.year}-2"
        if md >= (4, 30):
            return f"{d.year}-1"
        return f"{d.year - 1}-4"

    def get_fundamentals(self, ticker: str, curr_date: str = None) -> Any:
        """财务指标：``GET /api/a-share/financials/indicators``（五类能力）。"""
        thscode = self._normalize_thscode(ticker)
        if not thscode:
            raise ValueError(f"[cn_fuyao] 无法解析证券代码: {ticker}")
        report = self._latest_report_period(curr_date)
        params = {"thscode": thscode, "report": report}
        try:
            payload = self._request_fuyao(_INDICATORS_PATH, params)
        except FuyaoApiError as exc:
            return self._map_api_error(exc)

        data = payload.get("data") or {}
        abilities = data.get("abilities") or []
        lines: list[str] = []
        for block in abilities:
            if not isinstance(block, dict):
                continue
            ability = str(block.get("ability") or "")
            label = _ABILITY_LABELS.get(ability, ability)
            indicators = block.get("indicators") or []
            items = []
            for ind in indicators:
                if not isinstance(ind, dict):
                    continue
                idx = str(ind.get("index_id") or "unknown")
                val = ind.get("value")
                items.append(f"{idx}={val}" if val is not None else f"{idx}=缺失")
            if items:
                lines.append(f"- **{label}**：{'; '.join(items)}")
        if not lines:
            return VendorEmpty(
                f"[cn_fuyao] {ticker} 在 {report} 报告期暂无财务指标数据"
                "（code=0 但 abilities 为空）。"
            )
        header = f"## Fundamentals for {ticker}（同花顺 fuyao 财务指标，report={report}）"
        return header + "\n\n" + "\n".join(lines)

    # ── 涨跌停池 / 连板天梯 / 龙虎榜 ──────────────────────────────────

    @staticmethod
    def _format_zt_pool(data: dict[str, Any], day: str) -> str:
        pagination = data.get("pagination") or {}
        total = pagination.get("total") or 0
        items = data.get("item") or []
        lines = [f"涨停池（{day}，同花顺 fuyao）：共 {total} 只"]
        if items:
            cnt_dist = Counter(
                int(i.get("continue_day_cnt") or 0) for i in items if isinstance(i, dict)
            )
            if cnt_dist:
                dist_parts = []
                for board_cnt in sorted(cnt_dist):
                    label = f"{board_cnt}连板" if board_cnt >= 2 else "首板"
                    dist_parts.append(f"{label} {cnt_dist[board_cnt]}只")
                lines.append("连板分布：" + "；".join(dist_parts))
            lines.append("")
            for item in items[:15]:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or item.get("thscode") or "?"
                ths = item.get("thscode") or ""
                ratio = item.get("price_change_ratio_pct")
                ratio_txt = f"{ratio}%" if ratio is not None else "-"
                lines.append(
                    f"- {name}（{ths}）{item.get('continue_day_text') or ''} "
                    f"{ratio_txt} 涨停时间 {item.get('limit_up_time') or '-'} "
                    f"原因：{item.get('limit_up_reason') or '-'}"
                )
        return "\n".join(lines)

    def _fetch_zt_pool_for_day(self, day: str) -> str:
        """拉取指定交易日全部涨跌停池条目（分页合并）并格式化。"""
        page = 1
        items: list[dict[str, Any]] = []
        pagination: dict[str, Any] = {}
        while page <= _ZT_POOL_MAX_PAGES:
            payload = self._request_fuyao(
                _LIMIT_UP_POOL_PATH,
                {
                    "date_ms": self._date_to_ms(day),
                    "page": page,
                    "size": _ZT_POOL_PAGE_SIZE,
                    "sort_field": "continue_day_cnt",
                    "sort_dir": "desc",
                },
            )
            data = payload.get("data") or {}
            page_items = data.get("item") or []
            if not isinstance(page_items, list):
                break
            for it in page_items:
                if isinstance(it, dict):
                    items.append(it)
            pagination = data.get("pagination") or {}
            total_pages = pagination.get("pages") or 0
            if page >= total_pages:
                break
            page += 1
        if not items:
            raise DateDataUnavailable(f"{day} 涨停池无数据")
        data = {"pagination": pagination, "item": items}
        return self._format_zt_pool(data, day)

    def get_zt_pool(self, date: str) -> Any:
        """涨跌停池（东财主源失败时的备用源）。

        先试请求日；数据未就绪（3001/3002/空）则回退最近交易日
        （max_back=3）以覆盖发布延迟；历史日期仍按该日取数。
        """
        if not date:
            return (
                "【数据获取失败】涨停板情绪池缺少 date/curr_date，"
                "内部层不得默认今天，本项不可用。"
            )

        def _fetch_one(day: str):
            try:
                return self._fetch_zt_pool_for_day(day)
            except FuyaoApiError as exc:
                if exc.code in (3001, 3002):
                    raise DateDataUnavailable(f"{day} 涨停池无数据") from exc
                raise

        try:
            return self._fetch_zt_pool_for_day(date)
        except FuyaoApiError as exc:
            if exc.code not in (3001, 3002):
                return self._map_api_error(exc)
        except DateDataUnavailable:
            pass

        result = fetch_with_date_fallback(_fetch_one, date, max_back=3)
        if not result.ok:
            return VendorFail(f"涨停板情绪池数据获取失败（同花顺 fuyao）：{result.error}")
        return result.data

    def get_lhb_detail(self, symbol: str, date: str) -> Any:
        """龙虎榜：``GET /api/a-share/special-data/dragon-tiger-list``（board_type=all）。

        返回全市场榜单后按 symbol 过滤；该票未上榜视为「非异动日」正常空结果。
        """
        if not date:
            return (
                "【数据获取失败】龙虎榜缺少 date/curr_date，"
                "内部层不得默认今天，本项不可用。"
            )
        code = self._normalize_thscode(symbol)
        if not code:
            raise ValueError(f"[cn_fuyao] 无法解析证券代码: {symbol}")
        ticker6 = code.split(".")[0]

        request_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=CN_TZ)
        if request_dt < datetime.now(CN_TZ) - timedelta(days=_ONE_YEAR_DAYS):
            raise ValueError(
                f"[cn_fuyao] 龙虎榜 date 仅支持一年内数据: {date}"
            )

        def _fetch_one(day: str):
            try:
                payload = self._request_fuyao(
                    _DRAGON_TIGER_LIST_PATH, {"board_type": "all", "date": day}
                )
            except FuyaoApiError as exc:
                if exc.code in (3001, 3002):
                    raise DateDataUnavailable(f"{day} 龙虎榜无数据") from exc
                raise
            data = payload.get("data") or {}
            stock_items = data.get("stock_items") or []
            matched = [
                it
                for it in stock_items
                if isinstance(it, dict)
                and (
                    str(it.get("thscode") or "").split(".")[0] == ticker6
                    or str(it.get("ticker") or "").zfill(6) == ticker6
                )
            ]
            if not matched:
                return f"{symbol} 在 {day} 无龙虎榜数据（非异动日属正常）。"
            rows = [
                {
                    "名称": it.get("name"),
                    "代码": it.get("thscode"),
                    "涨跌幅": it.get("change"),
                    "净买入": it.get("net_value"),
                    "净占比": it.get("net_rate"),
                    "买方": it.get("buy_value"),
                    "卖方": it.get("sell_value"),
                    "上榜天数": it.get("range_days"),
                    "原因": it.get("limit_reason"),
                }
                for it in matched
            ]
            table = pd.DataFrame(rows).to_string(index=False)
            return f"{symbol} 龙虎榜明细（{day}，同花顺 fuyao）：\n{table}"

        try:
            result = fetch_with_date_fallback(_fetch_one, date, max_back=3)
        except FuyaoApiError as exc:
            return self._map_api_error(exc)

        if not result.ok:
            return VendorFail(f"龙虎榜数据获取失败（同花顺 fuyao）：{result.error}")
        return result.data

    # ── 交易日历 ──────────────────────────────────────────────────────

    def get_trading_days(self, curr_date: str = None) -> Any:
        """近一年交易日序列：``GET /api/a-share/calendar/trading-days``。"""
        try:
            payload = self._request_fuyao(_TRADING_DAYS_PATH, {})
        except FuyaoApiError as exc:
            return self._map_api_error(exc)
        items = ((payload.get("data") or {}).get("item")) or []
        dates = [
            str(it.get("date"))
            for it in items
            if isinstance(it, dict) and it.get("date")
        ]
        if not dates:
            return VendorEmpty("[cn_fuyao] 交易日历无数据（code=0 但 item 为空）。")
        first, last = dates[0], dates[-1]
        return (
            f"同花顺交易日历（{first} ~ {last}，共 {len(dates)} 个交易日）：\n"
            + ",".join(dates)
        )

    # ── 不支持的抽象方法：显式 NotImplementedError，交给 vendor 链回退 ──

    def get_indicators(
        self, symbol: str, indicator: str, curr_date: str, look_back_days: int
    ) -> str:
        raise NotImplementedError("cn_fuyao 不支持技术指标（get_indicators）。")

    def get_news(self, ticker: str, start_date: str, end_date: str) -> str:
        raise NotImplementedError("cn_fuyao 不支持个股新闻（get_news）。")

    def get_global_news(
        self, curr_date: str, look_back_days: int = 7, limit: int = 50
    ) -> str:
        raise NotImplementedError("cn_fuyao 不支持全市场新闻（get_global_news）。")

    def get_insider_transactions(self, symbol: str, curr_date: str = None) -> str:
        raise NotImplementedError(
            "cn_fuyao 不支持高管持股变动（get_insider_transactions）。"
        )


class _MappedVendorOutcome(Exception):
    """内部透传：错误码已映射为 VendorResult 结果，调用方把它当作返回值。"""

    def __init__(self, outcome: Any):
        super().__init__(str(outcome))
        self.outcome = outcome


def fetch_trading_days_ths(api_key: str, base_url: str | None = None) -> list[str]:
    """低层交易日历抓取（供 ``trade_calendar`` 作 akshare 失败后的在线对照/备用）。

    :returns: ``yyyyMMdd`` 字符串列表（升序）。失败抛异常由调用方决定兜底。
    """
    base = (base_url or _DEFAULT_BASE_URL).rstrip("/")
    resp = requests.get(
        f"{base}{_TRADING_DAYS_PATH}",
        headers={"X-api-key": api_key},
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict) or payload.get("code") != 0:
        raise RuntimeError(
            f"fuyao 交易日历返回异常 code="
            f"{payload.get('code') if isinstance(payload, dict) else 'N/A'}"
        )
    items = ((payload.get("data") or {}).get("item")) or []
    return [
        str(it.get("date"))
        for it in items
        if isinstance(it, dict) and it.get("date")
    ]
