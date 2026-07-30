"""Shared utility for custom-prompt injection into agent prompts.

Phase C scope: build_injection_slots() is the single owner of slot-filling logic.
Three agent factories (bull_researcher, bear_researcher, research_manager) call it;
they must not duplicate placement conditionals or separator logic.

Placement semantics (anchored in zh.py template markers):
- "before_data"  : slot fires between role/priority block and the first data field.
- "after_data"   : slot fires between last data field and built-in output requirements.

When custom_prompt is empty (switch off, or role has no text) both slots return "",
which means the zh.py template expands to the same bytes as before this feature
existed — T2 (byte-identity when switch off) is guaranteed here, not in the callers.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Literal

logger = logging.getLogger(__name__)

Placement = Literal["before_data", "after_data"]
_VALID_PLACEMENTS: tuple[str, ...] = ("before_data", "after_data")

# Single authoritative default — all callers must import and use this constant
# rather than hard-coding a string, so a future placement change is one-line.
DEFAULT_PLACEMENT: Placement = "after_data"


def _render_slot(custom_prompt: str) -> str:
    """Return '' for empty text; 'text\\n\\n' otherwise.

    The trailing double-newline is the only separator this module emits.
    Callers must not add their own separators.
    """
    if not custom_prompt:
        return ""
    return f"{custom_prompt}\n\n"


def build_injection_slots(
    custom_prompt: str,
    placement: Placement,
    role_key: str = "",
) -> dict[str, str]:
    """Return the two zh.py slot values for a given prompt + placement.

    Always returns both keys so callers can unconditionally unpack into
    .format(). The slot that is NOT active always gets "".

    Args:
        custom_prompt: Resolved text for this role (empty string = no injection).
        placement: "before_data" or "after_data".
        role_key: Agent role identifier used only for logging (no effect on slots).

    Returns:
        {"custom_prompt_before_data": str, "custom_prompt_after_data": str}

    Raises:
        ValueError: If placement is not one of the two valid values.
    """
    if placement not in _VALID_PLACEMENTS:
        raise ValueError(
            f"[prompt_injection] Unknown placement {placement!r}; "
            f"must be one of {_VALID_PLACEMENTS}"
        )

    rendered = _render_slot(custom_prompt)

    if placement == "before_data":
        slots = {
            "custom_prompt_before_data": rendered,
            "custom_prompt_after_data": "",
        }
    else:  # after_data
        slots = {
            "custom_prompt_before_data": "",
            "custom_prompt_after_data": rendered,
        }

    # Log every call so each debate round is traceable; suppress prompt body.
    logger.info(
        "[prompt_injection] role=%s placement=%s injected=%s length=%d hash=%s",
        role_key or "unknown",
        placement,
        bool(custom_prompt),
        len(custom_prompt),
        hashlib.sha256(custom_prompt.encode("utf-8")).hexdigest()[:12] if custom_prompt else "none",
    )

    return slots
