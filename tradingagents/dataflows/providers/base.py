from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ProviderResourcePolicy:
    """每个 provider 独立的外部调用超时/重试/并发上限。"""

    timeout_seconds: float
    max_retries: int
    max_concurrency: int

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")


DEFAULT_PROVIDER_RESOURCE_POLICY = ProviderResourcePolicy(
    timeout_seconds=60.0,
    max_retries=1,
    max_concurrency=4,
)


@dataclass
class DataResult:
    """所有数据接口的统一返回类型。"""
    ok: bool
    data: Any
    as_of: Optional[str] = None
    source: str = ""
    title: str = ""
    error: Optional[str] = None
    stale: bool = False

    def to_prompt(self) -> str:
        """转换为注入 LLM 的文本。失败或为空时必须返回显式的处理说明。"""
        item_title = self.title or self.source or "相关数据"
        if not self.ok:
            error_msg = self.error or "接口超时或返回异常"
            return (
                f"【数据获取失败】{item_title} — 原因：{error_msg} (来源: {self.source})\n"
                f"该项分析不可用，请在报告中标注\"{item_title}未排查/获取失败\"，不要基于记忆推测。\n"
            )
        
        if self.data is None or (isinstance(self.data, str) and not self.data.strip()):
            return (
                f"【数据排查结果】{item_title} — 暂无相关事件/无触发记录 (来源: {self.source})\n"
                f"该项排查结果：已排查，无异常或未触发风险记录。\n"
            )

        if isinstance(self.data, str):
            return self.data
        return str(self.data)


class BaseMarketDataProvider(ABC):
    """Abstract interface for market data providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier used by config routing."""
        raise NotImplementedError

    @abstractmethod
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_indicators(
        self, symbol: str, indicator: str, curr_date: str, look_back_days: int
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_fundamentals(self, ticker: str, curr_date: str = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_balance_sheet(
        self, ticker: str, freq: str = "quarterly", curr_date: str = None
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_cashflow(
        self, ticker: str, freq: str = "quarterly", curr_date: str = None
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_income_statement(
        self, ticker: str, freq: str = "quarterly", curr_date: str = None
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_news(self, ticker: str, start_date: str, end_date: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_global_news(
        self, curr_date: str, look_back_days: int = 7, limit: int = 50
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_insider_transactions(self, symbol: str, curr_date: str = None) -> str:
        raise NotImplementedError

    def get_realtime_quotes(self, symbols: list[str], curr_date: str = None) -> str:
        """Return real-time quotes for a list of symbols as a JSON string.

        ``curr_date`` is required by time-sensitive providers so historical
        analysis can refuse snapshot quotes; base default is None for
        providers that only support live bars.
        """
        raise NotImplementedError

    def get_restricted_release(self, symbol: str, curr_date: str = None) -> str:
        """Return restricted share release events."""
        raise NotImplementedError

    def get_share_pledge(self, symbol: str, curr_date: str = None) -> str:
        """Return share pledge ratio and risks."""
        raise NotImplementedError

    def get_earnings_forecast(self, symbol: str, curr_date: str = None) -> str:
        """Return earnings forecast and quick reports."""
        raise NotImplementedError

    def get_shareholder_count(self, symbol: str, curr_date: str = None) -> str:
        """Return shareholder count and chip concentration."""
        raise NotImplementedError

    def get_margin_trading(self, symbol: str, curr_date: str = None) -> str:
        """Return margin trading and short selling details."""
        raise NotImplementedError

    def get_northbound_flow(self, symbol: str, curr_date: str = None) -> str:
        """Return northbound / HK connect holding details."""
        raise NotImplementedError

    def get_global_indices(
        self, curr_date: str = None, look_back_days: int = 30
    ) -> str:
        """Return historical and latest daily series for global core indices."""
        raise NotImplementedError

    def get_major_assets(
        self, curr_date: str = None, look_back_days: int = 30
    ) -> str:
        """Return historical and latest daily series for major macro assets."""
        raise NotImplementedError

    def get_cn_indices(
        self, curr_date: str = None, look_back_days: int = 30
    ) -> str:
        """Return historical and latest daily series for China core market indices."""
        raise NotImplementedError
