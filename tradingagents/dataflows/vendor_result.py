"""Typed vendor outcomes for the dataflow router (KNOWN_ISSUES #1).

A provider method called through ``route_to_vendor`` may return a plain value
(a string / DataFrame / dict), which is treated as a successful hit — backward
compatible — or one of the typed results below to explicitly signal a non-OK
outcome.  ``route_to_vendor`` uses the outcome to drive the vendor chain:

- ``VendorRefuse`` — this source cannot serve the request (snapshot-only under a
  historical analysis date, missing capability, date-blind semantics).  The chain
  must NOT silently fall through to a weaker/date-blind vendor.
- ``VendorEmpty``  — the query succeeded and there is genuinely no data for the
  requested scope.  Stop and report "confirmed none".
- ``VendorFail``   — network / timeout / parse error / temporary breakage.  Try
  the next vendor.
- ``VendorOk``     — an explicit successful hit (payload).

Before this type was introduced, a refusal / failure / confirmed-empty returned
as an ordinary **string** all looked like a successful hit and stopped the chain,
so a temporary akshare failure could mask a real yfinance fallback, and a
near-window empty could block a historical-capable source.
"""

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class VendorResult:
    """Base marker for all typed vendor outcomes."""

    def to_prompt(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class VendorOk(VendorResult):
    """Explicit success with a payload."""

    payload: Any

    def to_prompt(self) -> str:
        return self.payload if isinstance(self.payload, str) else str(self.payload)


@dataclass(frozen=True)
class VendorRefuse(VendorResult):
    """This source cannot serve the request; do not fall through.

    ``allow_peers`` optionally lists same-semantics providers the router may
    continue through (e.g. other historical-capable sources), instead of the
    whole fallback chain.
    """

    reason: str
    allow_peers: Tuple[str, ...] = ()

    def to_prompt(self) -> str:
        return self.reason


@dataclass(frozen=True)
class VendorEmpty(VendorResult):
    """The query succeeded and there is genuinely no data (confirmed none)."""

    message: str

    def to_prompt(self) -> str:
        return self.message


@dataclass(frozen=True)
class VendorFail(VendorResult):
    """Transient failure; the router should try the next vendor."""

    error: str

    def to_prompt(self) -> str:
        return self.error


def result_to_prompt(result: Any) -> str:
    """Normalize a provider return value into a prompt string.

    Plain strings / other values pass through unchanged (backward compatible);
    typed ``VendorResult`` objects are unwrapped via ``to_prompt``.
    """
    if isinstance(result, VendorResult):
        return result.to_prompt()
    if isinstance(result, str):
        return result
    return str(result)
