"""LLM base_url proxy reachability startup guard (DAV-777).

Enforces fail-closed proxy check for LLM base_url targets:
When an LLM base_url points to a literal IP address (e.g., self-hosted CLIProxyAPI),
httpx must not route it to an outbound forward proxy (e.g. 127.0.0.1:7897),
which causes 502 Bad Gateway failures.

Uses httpx's internal transport routing determination (trust_env=True) rather
than re-implementing no_proxy parsing.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import urllib.parse
from typing import Any, Optional

import httpx

_logger = logging.getLogger(__name__)

# In-memory cache to ensure self-check runs once per unique (base_url, env) combination.
# Avoids repeated inspection and ensures zero network roundtrips during LLM calls.
_VERIFIED_CACHE: set[tuple[str, str, str, str, str, str, str]] = set()


class LLMProxyRoutingError(RuntimeError):
    """Raised when an LLM base_url is incorrectly routed through a proxy."""

    def __init__(self, base_url: str, proxy_url: str, repair_command: str):
        self.base_url = base_url
        self.proxy_url = proxy_url
        self.repair_command = repair_command
        super().__init__(
            f"LLM base_url '{base_url}' 代理可达性自检失败（fail-closed）："
            f"httpx 判定该 base_url 会经代理 '{proxy_url}' 转发，"
            f"导致无法直连上游（如产生 502 Bad Gateway）。\n"
            f"修复命令（需包含小写 no_proxy 与字面量 IP）：\n"
            f"{repair_command}"
        )


def _cache_key(base_url: str) -> tuple[str, str, str, str, str, str, str]:
    return (
        base_url,
        os.environ.get("no_proxy", ""),
        os.environ.get("NO_PROXY", ""),
        os.environ.get("http_proxy", ""),
        os.environ.get("HTTP_PROXY", ""),
        os.environ.get("https_proxy", ""),
        os.environ.get("HTTPS_PROXY", ""),
    )


def clear_proxy_guard_cache() -> None:
    """Clear in-memory cache (primarily used for unit testing)."""
    _VERIFIED_CACHE.clear()


def _is_loopback_or_localhost(host: str) -> bool:
    """Check if host is localhost or a loopback address (Contract 5)."""
    if not host:
        return False
    host_lower = host.lower()
    if host_lower in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_unspecified
    except ValueError:
        return False


def _is_literal_ip(host: str) -> bool:
    """Check if host is an IPv4 or IPv6 address literal."""
    if not host:
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _format_proxy_url(proxy_url: Any) -> str:
    """Extract a human-readable proxy address from httpx / httpcore URL object."""
    if proxy_url is None:
        return ""
    if hasattr(proxy_url, "origin"):
        return str(proxy_url.origin)
    if hasattr(proxy_url, "scheme") and hasattr(proxy_url, "host"):
        scheme = (
            proxy_url.scheme.decode()
            if isinstance(proxy_url.scheme, bytes)
            else str(proxy_url.scheme)
        )
        host = (
            proxy_url.host.decode()
            if isinstance(proxy_url.host, bytes)
            else str(proxy_url.host)
        )
        port = getattr(proxy_url, "port", None)
        port_str = f":{port}" if port else ""
        return f"{scheme}://{host}{port_str}"
    return str(proxy_url)


def check_llm_proxy_guard(base_url: Optional[str]) -> None:
    """Check whether httpx will route the given LLM base_url through a proxy.

    Contracts:
    1. Uses httpx's internal transport routing determination (Client(trust_env=True)._transport_for_url()).
    2. Fail-closed on proxy detection for literal IP base_urls with detailed repair command.
    3. Overseas services (domain names) are allowed to go through proxy (preserves legitimate proxy needs).
    4. Evaluated in-memory with caching; does not perform network roundtrips.
    5. base_url on localhost / 127.0.0.1 skips check.
    6. Does not degrade to another provider on failure.
    """
    if not base_url or not isinstance(base_url, str):
        return

    raw_url = base_url.strip()
    if not raw_url:
        return

    cache_key = _cache_key(raw_url)
    if cache_key in _VERIFIED_CACHE:
        return

    normalized_url = raw_url if "://" in raw_url else f"http://{raw_url}"
    try:
        parsed = urllib.parse.urlparse(normalized_url)
        host = parsed.hostname
    except Exception:
        return

    if not host:
        return

    # Contract 5: base_url 为 localhost/127.0.0.1 时跳过自检
    if _is_loopback_or_localhost(host):
        _VERIFIED_CACHE.add(cache_key)
        return

    # Contract 3: Domain names are allowed to go through proxy (preserves legitimate overseas proxy needs).
    # Only literal IP base_urls (self-hosted / internal gateways) must fail-closed if routed to a proxy.
    if not _is_literal_ip(host):
        _VERIFIED_CACHE.add(cache_key)
        return

    # Contract 1: Use httpx's own routing logic
    with httpx.Client(trust_env=True) as client:
        tr = client._transport_for_url(httpx.URL(normalized_url))
        pool = getattr(tr, "_pool", None)
        proxy_url = getattr(pool, "_proxy_url", None)

    if proxy_url is not None:
        proxy_addr = _format_proxy_url(proxy_url)
        repair_command = (
            f'export no_proxy="${{no_proxy}},{host}"\n'
            f'export NO_PROXY="${{NO_PROXY}},{host}"'
        )
        _logger.error(
            "[ProxyGuard] Fail-closed: base_url '%s' is routed through proxy '%s'. Repair command: export no_proxy=\"${no_proxy},%s\"",
            raw_url,
            proxy_addr,
            host,
        )
        raise LLMProxyRoutingError(
            base_url=raw_url,
            proxy_url=proxy_addr,
            repair_command=repair_command,
        )

    _VERIFIED_CACHE.add(cache_key)
