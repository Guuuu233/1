"""Service module for user custom analysis prompts (global + per-role/group overrides).

Phase B scope only: persistence, validation, and a read-only "resolved" preview.
No agent-construction / injection logic lives here — that is Phase C's job, and it
must import `resolve_role_prompt` / `resolve_all_roles_prompts` from this module
rather than re-deriving prompt text on its own.

Role-key source of truth: this module imports ALL_ROLES / ROLE_GROUPS / ROLE_TO_GROUP
from role_routing_service instead of redefining a role list. The codebase already has
a second, unrelated "role key" convention used only for SSE progress display
(api/main.py ANALYST_AGENT_NAMES, e.g. "aggressive" / "portfolio_manager") — that one
must never be used here; role_routing_service.ALL_ROLES is the convention that matches
how agents are actually constructed (tradingagents/graph/setup.py's role_llms keys).
"""

from __future__ import annotations

import hashlib
import logging
from uuid import uuid4
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from sqlalchemy.orm import Session

from api.database import UserCustomPromptDB, UserDB
from api.services.role_routing_service import ALL_ROLES, ROLE_GROUPS, ROLE_TO_GROUP

logger = logging.getLogger(__name__)

GLOBAL_TARGET_KEY = ""
VALID_TARGET_TYPES = {"global", "role", "group"}
VALID_GROUP_KEYS = set(ROLE_GROUPS.keys())

# Field-level caps (see plan doc section 3 / 9 for the reasoning behind these numbers).
GLOBAL_PROMPT_MAX_CHARS = 4000
ROLE_PROMPT_MAX_CHARS = 2000
GROUP_PROMPT_MAX_CHARS = 2000
# Cap on the *resolved* (global + "\n\n" + override) text actually injected per role.
# Note: two fields both at their own max (4000 + 2000) plus the 2-char separator total
# 6002, which is over this cap by 2 — that is intentional, not a bug. See plan doc.
RESOLVED_PROMPT_MAX_CHARS = 6000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _validate_item(item: Dict[str, Any]) -> None:
    target_type = item.get("target_type")
    target_key = item.get("target_key") or ""
    prompt_text = item.get("prompt_text")

    if target_type not in VALID_TARGET_TYPES:
        raise ValueError(f"未知的 target_type: {target_type!r}，必须是 global/role/group 之一")

    if not prompt_text or not prompt_text.strip():
        raise ValueError(f"target_type={target_type} target_key={target_key!r} 的 prompt_text 不能为空")

    if target_type == "global":
        if target_key not in ("", None):
            raise ValueError("target_type=global 时 target_key 必须为空")
        if len(prompt_text) > GLOBAL_PROMPT_MAX_CHARS:
            raise ValueError(
                f"全局提示超过 {GLOBAL_PROMPT_MAX_CHARS} 字符上限（当前 {len(prompt_text)} 字符）"
            )
    elif target_type == "role":
        if target_key not in ALL_ROLES:
            raise ValueError(f"未知的 role_key: {target_key!r}，必须是 role_routing_service.ALL_ROLES 之一")
        if len(prompt_text) > ROLE_PROMPT_MAX_CHARS:
            raise ValueError(
                f"角色 {target_key} 的覆盖提示超过 {ROLE_PROMPT_MAX_CHARS} 字符上限（当前 {len(prompt_text)} 字符）"
            )
    elif target_type == "group":
        if target_key not in VALID_GROUP_KEYS:
            raise ValueError(f"未知的 group_key: {target_key!r}，必须是 role_routing_service.ROLE_GROUPS 之一")
        if len(prompt_text) > GROUP_PROMPT_MAX_CHARS:
            raise ValueError(
                f"分组 {target_key} 的覆盖提示超过 {GROUP_PROMPT_MAX_CHARS} 字符上限（当前 {len(prompt_text)} 字符）"
            )


def _row_to_dict(row: UserCustomPromptDB) -> Dict[str, Any]:
    return {
        "id": row.id,
        "target_type": row.target_type,
        "target_key": row.target_key,
        "prompt_text": row.prompt_text,
        "prompt_hash": row.prompt_hash,
        "char_count": len(row.prompt_text or ""),
        "enabled": row.enabled,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def _lite_from_row(row: Optional[UserCustomPromptDB]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {"prompt_text": row.prompt_text, "enabled": row.enabled}


def _lite_from_item(item: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if item is None:
        return None
    return {"prompt_text": item["prompt_text"], "enabled": item.get("enabled", True)}


def _resolve_for_role(
    role_key: str,
    global_lite: Optional[Dict[str, Any]],
    role_lites_by_key: Dict[str, Dict[str, Any]],
    group_lites_by_key: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Priority: role override > group override > global only. Resolved = global + \\n\\n + override."""
    global_text = global_lite["prompt_text"] if global_lite and global_lite["enabled"] else ""

    override_text = ""
    override_source: Optional[str] = None

    role_lite = role_lites_by_key.get(role_key)
    if role_lite and role_lite["enabled"] and role_lite["prompt_text"]:
        override_text = role_lite["prompt_text"]
        override_source = "role"
    else:
        group_key = ROLE_TO_GROUP.get(role_key)
        group_lite = group_lites_by_key.get(group_key) if group_key else None
        if group_lite and group_lite["enabled"] and group_lite["prompt_text"]:
            override_text = group_lite["prompt_text"]
            override_source = "group"

    if global_text and override_text:
        resolved_text = f"{global_text}\n\n{override_text}"
    else:
        resolved_text = global_text or override_text

    return {
        "role_key": role_key,
        "global_text": global_text,
        "override_text": override_text,
        "override_source": override_source,
        "resolved_text": resolved_text,
        "resolved_length": len(resolved_text),
        "resolved_hash": _hash_text(resolved_text) if resolved_text else None,
    }


def list_custom_prompts(db: Session, user_id: str) -> List[Dict[str, Any]]:
    rows = db.query(UserCustomPromptDB).filter(UserCustomPromptDB.user_id == user_id).all()
    return [_row_to_dict(r) for r in rows]


def resolve_role_prompt(db: Session, user_id: str, role_key: str) -> Dict[str, Any]:
    if role_key not in ALL_ROLES:
        raise ValueError(f"未知的 role_key: {role_key!r}")

    global_row = (
        db.query(UserCustomPromptDB)
        .filter(UserCustomPromptDB.user_id == user_id, UserCustomPromptDB.target_type == "global")
        .first()
    )
    role_row = (
        db.query(UserCustomPromptDB)
        .filter(
            UserCustomPromptDB.user_id == user_id,
            UserCustomPromptDB.target_type == "role",
            UserCustomPromptDB.target_key == role_key,
        )
        .first()
    )
    group_key = ROLE_TO_GROUP.get(role_key)
    group_row = None
    if group_key:
        group_row = (
            db.query(UserCustomPromptDB)
            .filter(
                UserCustomPromptDB.user_id == user_id,
                UserCustomPromptDB.target_type == "group",
                UserCustomPromptDB.target_key == group_key,
            )
            .first()
        )

    return _resolve_for_role(
        role_key,
        _lite_from_row(global_row),
        {role_key: _lite_from_row(role_row)} if role_row else {},
        {group_key: _lite_from_row(group_row)} if group_row else {},
    )


def resolve_all_roles_prompts(db: Session, user_id: str) -> List[Dict[str, Any]]:
    """Resolve the preview for all 15 agent roles in one query round-trip."""
    rows = db.query(UserCustomPromptDB).filter(UserCustomPromptDB.user_id == user_id).all()
    global_row = next((r for r in rows if r.target_type == "global"), None)
    role_lites_by_key = {r.target_key: _lite_from_row(r) for r in rows if r.target_type == "role"}
    group_lites_by_key = {r.target_key: _lite_from_row(r) for r in rows if r.target_type == "group"}
    global_lite = _lite_from_row(global_row)

    return [
        _resolve_for_role(role_key, global_lite, role_lites_by_key, group_lites_by_key)
        for role_key in ALL_ROLES
    ]


def replace_custom_prompts(db: Session, user_id: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Whole-set replace, mirroring role_routing_service.update_role_bindings's contract:
    the submitted list becomes the user's complete set of custom-prompt rows.

    Validates every item's own field cap, then validates the *resolved* length for all
    15 roles against the candidate final state (derived purely from `items`, since this
    is a full replace) before touching the database. Delete + insert happen inside one
    transaction; any failure rolls back and leaves existing rows untouched.
    """
    for item in items:
        _validate_item(item)

    candidate_global = next((i for i in items if i["target_type"] == "global"), None)
    candidate_roles = {i["target_key"]: i for i in items if i["target_type"] == "role"}
    candidate_groups = {i["target_key"]: i for i in items if i["target_type"] == "group"}

    candidate_global_lite = _lite_from_item(candidate_global)
    candidate_role_lites = {k: _lite_from_item(v) for k, v in candidate_roles.items()}
    candidate_group_lites = {k: _lite_from_item(v) for k, v in candidate_groups.items()}

    for role_key in ALL_ROLES:
        resolved = _resolve_for_role(role_key, candidate_global_lite, candidate_role_lites, candidate_group_lites)
        if resolved["resolved_length"] > RESOLVED_PROMPT_MAX_CHARS:
            raise ValueError(
                f"角色 {role_key} 合并后的提示词长度为 {resolved['resolved_length']} 字符，"
                f"超过 {RESOLVED_PROMPT_MAX_CHARS} 字符上限"
            )

    now = _utcnow()
    try:
        db.query(UserCustomPromptDB).filter(UserCustomPromptDB.user_id == user_id).delete(synchronize_session=False)
        for item in items:
            target_key = item["target_key"] if item["target_type"] != "global" else GLOBAL_TARGET_KEY
            db.add(
                UserCustomPromptDB(
                    id=uuid4().hex,
                    user_id=user_id,
                    target_type=item["target_type"],
                    target_key=target_key,
                    prompt_text=item["prompt_text"],
                    prompt_hash=_hash_text(item["prompt_text"]),
                    enabled=item.get("enabled", True),
                    created_at=now,
                    updated_at=now,
                )
            )
        db.commit()
    except Exception:
        db.rollback()
        raise

    return list_custom_prompts(db, user_id)


def migrate_legacy_prompt(db: Session, user_id: str, legacy_text: str) -> List[Dict[str, Any]]:
    """One-time upload of the frontend's localStorage global prompt.

    Idempotent by design: only inserts if the user has no 'global' row yet. A second
    call (retry, another device, page reload racing the first migration) is a no-op
    that returns the current rows unchanged — it must never overwrite a value the user
    has since edited on the backend.
    """
    legacy_text = (legacy_text or "").strip()
    if not legacy_text:
        return list_custom_prompts(db, user_id)

    existing_global = (
        db.query(UserCustomPromptDB)
        .filter(UserCustomPromptDB.user_id == user_id, UserCustomPromptDB.target_type == "global")
        .first()
    )
    if existing_global:
        logger.info("[custom_prompts] Skipping migration for user %s: global row already exists.", user_id)
        return list_custom_prompts(db, user_id)

    if len(legacy_text) > GLOBAL_PROMPT_MAX_CHARS:
        raise ValueError(
            f"待迁移的提示词为 {len(legacy_text)} 字符，超过 {GLOBAL_PROMPT_MAX_CHARS} 字符上限，"
            "请先在前端缩短后再迁移"
        )

    now = _utcnow()
    db.add(
        UserCustomPromptDB(
            id=uuid4().hex,
            user_id=user_id,
            target_type="global",
            target_key=GLOBAL_TARGET_KEY,
            prompt_text=legacy_text,
            prompt_hash=_hash_text(legacy_text),
            enabled=True,
            created_at=now,
            updated_at=now,
        )
    )
    db.commit()
    return list_custom_prompts(db, user_id)


def get_prompt_injection_enabled(db: Session, user_id: str) -> bool:
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not user:
        return False
    return bool(user.prompt_injection_enabled)


def set_prompt_injection_enabled(db: Session, user_id: str, enabled: bool) -> bool:
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not user:
        raise ValueError("User not found")
    user.prompt_injection_enabled = enabled
    db.commit()
    return bool(user.prompt_injection_enabled)
