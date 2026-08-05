from typing import Dict

from .base import (
    DEFAULT_PROVIDER_RESOURCE_POLICY,
    BaseMarketDataProvider,
    ProviderResourcePolicy,
)
from .yfinance_provider import YFinanceProvider
from .alpha_vantage_provider import AlphaVantageProvider
from .china_equity_provider import CnStubProvider
from .cn_akshare_provider import CnAkshareProvider
from .cn_baostock_provider import CnBaoStockProvider
from .cn_fuyao_provider import CnFuyaoProvider
from .cn_investoday_provider import CnInvestodayProvider


DEFAULT_PROVIDER_RESOURCE_POLICIES: Dict[str, ProviderResourcePolicy] = {
    "cn_akshare": ProviderResourcePolicy(
        timeout_seconds=90.0,
        max_retries=1,
        max_concurrency=5,
    ),
    "cn_baostock": ProviderResourcePolicy(
        timeout_seconds=45.0,
        max_retries=1,
        max_concurrency=2,
    ),
    "cn_investoday": ProviderResourcePolicy(
        timeout_seconds=30.0,
        max_retries=1,
        max_concurrency=4,
    ),
    "cn_fuyao": ProviderResourcePolicy(
        timeout_seconds=30.0,
        max_retries=1,
        max_concurrency=4,
    ),
    "yfinance": ProviderResourcePolicy(
        timeout_seconds=30.0,
        max_retries=1,
        max_concurrency=2,
    ),
    "alpha_vantage": ProviderResourcePolicy(
        timeout_seconds=20.0,
        max_retries=1,
        max_concurrency=3,
    ),
    "cn_stub": ProviderResourcePolicy(
        timeout_seconds=5.0,
        max_retries=0,
        max_concurrency=4,
    ),
}


class DataProviderRegistry:
    """Simple in-memory provider registry."""

    def __init__(self):
        self._providers: Dict[str, BaseMarketDataProvider] = {}
        self._resource_policies: Dict[str, ProviderResourcePolicy] = {}

    def register(
        self,
        provider: BaseMarketDataProvider,
        resource_policy: ProviderResourcePolicy | None = None,
    ) -> None:
        self._providers[provider.name] = provider
        self._resource_policies[provider.name] = (
            resource_policy
            or DEFAULT_PROVIDER_RESOURCE_POLICIES.get(
                provider.name, DEFAULT_PROVIDER_RESOURCE_POLICY
            )
        )

    def get(self, provider_name: str) -> BaseMarketDataProvider | None:
        return self._providers.get(provider_name)

    def list_names(self) -> list[str]:
        return list(self._providers.keys())

    def resource_policy(self, provider_name: str) -> ProviderResourcePolicy:
        return self._resource_policies.get(
            provider_name, DEFAULT_PROVIDER_RESOURCE_POLICY
        )

    def list_resource_policies(self) -> Dict[str, ProviderResourcePolicy]:
        return dict(self._resource_policies)


def build_default_registry() -> DataProviderRegistry:
    registry = DataProviderRegistry()
    registry.register(CnAkshareProvider())
    registry.register(CnBaoStockProvider())
    registry.register(CnInvestodayProvider())
    registry.register(CnFuyaoProvider())
    registry.register(YFinanceProvider())
    registry.register(AlphaVantageProvider())
    registry.register(CnStubProvider())
    return registry
