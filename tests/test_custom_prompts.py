"""Tests for api/services/custom_prompt_service.py (Phase B: prompt persistence).

Covers: global+role/group concatenation semantics ("resolved" = global + "\\n\\n" +
override, override priority role > group), field-level length caps, the resolved-length
cap catching combinations that pass field-level validation individually, whole-set
replace transaction safety (rollback leaves old data intact), migration idempotency,
the default-off master switch, per-user isolation, and that the 15 role_key whitelist
is the same list role_routing_service.ALL_ROLES exposes (not a locally redefined copy).
"""

from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base, UserCustomPromptDB, UserDB
from api.services import custom_prompt_service as svc
from api.services.role_routing_service import ALL_ROLES


def _make_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return TestingSessionLocal()


def _make_user(db) -> str:
    user_id = uuid4().hex
    db.add(UserDB(id=user_id, email=f"{user_id}@test.local"))
    db.commit()
    return user_id


# --- resolve concatenation / priority semantics ---

def test_resolve_concatenates_global_and_role_override_with_blank_line():
    db = _make_session()
    try:
        user_id = _make_user(db)
        svc.replace_custom_prompts(
            db,
            user_id,
            [
                {"target_type": "global", "target_key": "", "prompt_text": "全局约束A"},
                {"target_type": "role", "target_key": "bull_researcher", "prompt_text": "多头补充B"},
            ],
        )

        resolved = svc.resolve_role_prompt(db, user_id, "bull_researcher")
        assert resolved["resolved_text"] == "全局约束A\n\n多头补充B"
        assert resolved["override_source"] == "role"

        # A role with no override falls back to global text only.
        other = svc.resolve_role_prompt(db, user_id, "market")
        assert other["resolved_text"] == "全局约束A"
        assert other["override_source"] is None
    finally:
        db.close()


def test_resolve_falls_back_to_group_when_no_role_override():
    db = _make_session()
    try:
        user_id = _make_user(db)
        svc.replace_custom_prompts(
            db,
            user_id,
            [{"target_type": "group", "target_key": "analysts", "prompt_text": "分析师组补充"}],
        )

        resolved = svc.resolve_role_prompt(db, user_id, "market")  # 'market' is in the 'analysts' group
        assert resolved["override_source"] == "group"
        assert resolved["resolved_text"] == "分析师组补充"

        not_in_group = svc.resolve_role_prompt(db, user_id, "trader")  # 'trader' group has no row
        assert not_in_group["override_source"] is None
        assert not_in_group["resolved_text"] == ""
    finally:
        db.close()


def test_role_override_takes_precedence_over_group_override():
    db = _make_session()
    try:
        user_id = _make_user(db)
        svc.replace_custom_prompts(
            db,
            user_id,
            [
                {"target_type": "group", "target_key": "analysts", "prompt_text": "组级"},
                {"target_type": "role", "target_key": "market", "prompt_text": "角色级"},
            ],
        )
        resolved = svc.resolve_role_prompt(db, user_id, "market")
        assert resolved["override_source"] == "role"
        assert resolved["override_text"] == "角色级"
    finally:
        db.close()


def test_disabled_row_is_excluded_from_resolution():
    db = _make_session()
    try:
        user_id = _make_user(db)
        svc.replace_custom_prompts(
            db,
            user_id,
            [
                {"target_type": "global", "target_key": "", "prompt_text": "全局"},
                {"target_type": "role", "target_key": "trader", "prompt_text": "被停用", "enabled": False},
            ],
        )
        resolved = svc.resolve_role_prompt(db, user_id, "trader")
        assert resolved["override_source"] is None
        assert resolved["resolved_text"] == "全局"
    finally:
        db.close()


# --- length validation ---

def test_global_prompt_over_cap_rejected():
    db = _make_session()
    try:
        user_id = _make_user(db)
        with pytest.raises(ValueError, match="4000"):
            svc.replace_custom_prompts(
                db, user_id, [{"target_type": "global", "target_key": "", "prompt_text": "字" * 4001}]
            )
    finally:
        db.close()


def test_role_prompt_over_cap_rejected():
    db = _make_session()
    try:
        user_id = _make_user(db)
        with pytest.raises(ValueError, match="2000"):
            svc.replace_custom_prompts(
                db, user_id, [{"target_type": "role", "target_key": "trader", "prompt_text": "字" * 2001}]
            )
    finally:
        db.close()


def test_resolved_length_cap_catches_combination_that_passes_field_level_caps():
    """Both fields are individually within their own caps (4000 / 2000), but
    concatenated with the "\\n\\n" separator they total 6002 chars — 2 over the
    6000 resolved-length cap. Field-level validation alone would miss this."""
    db = _make_session()
    try:
        user_id = _make_user(db)
        with pytest.raises(ValueError, match="research_manager"):
            svc.replace_custom_prompts(
                db,
                user_id,
                [
                    {"target_type": "global", "target_key": "", "prompt_text": "字" * 4000},
                    {"target_type": "role", "target_key": "research_manager", "prompt_text": "字" * 2000},
                ],
            )
    finally:
        db.close()


def test_unknown_role_key_rejected():
    db = _make_session()
    try:
        user_id = _make_user(db)
        with pytest.raises(ValueError):
            svc.replace_custom_prompts(
                db, user_id, [{"target_type": "role", "target_key": "not_a_real_role", "prompt_text": "x"}]
            )
    finally:
        db.close()


def test_unknown_target_type_rejected():
    db = _make_session()
    try:
        user_id = _make_user(db)
        with pytest.raises(ValueError):
            svc.replace_custom_prompts(
                db, user_id, [{"target_type": "bogus", "target_key": "", "prompt_text": "x"}]
            )
    finally:
        db.close()


def test_empty_prompt_text_rejected():
    db = _make_session()
    try:
        user_id = _make_user(db)
        with pytest.raises(ValueError):
            svc.replace_custom_prompts(
                db, user_id, [{"target_type": "global", "target_key": "", "prompt_text": "   "}]
            )
    finally:
        db.close()


def test_all_15_role_keys_from_role_routing_service_are_accepted():
    """Anchors the whitelist to role_routing_service.ALL_ROLES so the two role-key
    registries (this table's validation vs. graph/setup.py's role_llms keys) can't
    silently drift apart."""
    db = _make_session()
    try:
        user_id = _make_user(db)
        assert len(ALL_ROLES) == 15
        items = [{"target_type": "role", "target_key": role_key, "prompt_text": "ok"} for role_key in ALL_ROLES]
        result = svc.replace_custom_prompts(db, user_id, items)
        assert {r["target_key"] for r in result if r["target_type"] == "role"} == set(ALL_ROLES)
    finally:
        db.close()


# --- transaction atomicity ---

def test_replace_rolls_back_and_keeps_old_data_when_commit_fails(monkeypatch):
    db = _make_session()
    try:
        user_id = _make_user(db)
        svc.replace_custom_prompts(
            db, user_id, [{"target_type": "global", "target_key": "", "prompt_text": "原始版本"}]
        )

        def _boom():
            raise RuntimeError("simulated commit failure")

        monkeypatch.setattr(db, "commit", _boom)

        with pytest.raises(RuntimeError):
            svc.replace_custom_prompts(
                db, user_id, [{"target_type": "global", "target_key": "", "prompt_text": "新版本"}]
            )

        # Restore the real commit before reading back to confirm nothing was lost.
        monkeypatch.undo()
        rows = db.query(UserCustomPromptDB).filter(UserCustomPromptDB.user_id == user_id).all()
        assert len(rows) == 1
        assert rows[0].prompt_text == "原始版本"
    finally:
        db.close()


# --- migration idempotency ---

def test_migrate_inserts_when_no_global_row_exists():
    db = _make_session()
    try:
        user_id = _make_user(db)
        result = svc.migrate_legacy_prompt(db, user_id, "从 localStorage 迁移的旧提示词")
        assert len(result) == 1
        assert result[0]["prompt_text"] == "从 localStorage 迁移的旧提示词"
        assert result[0]["target_type"] == "global"
    finally:
        db.close()


def test_migrate_is_noop_when_global_row_already_exists():
    db = _make_session()
    try:
        user_id = _make_user(db)
        svc.replace_custom_prompts(
            db,
            user_id,
            [{"target_type": "global", "target_key": "", "prompt_text": "用户已经在后端设置过的新内容"}],
        )
        result = svc.migrate_legacy_prompt(db, user_id, "旧的 localStorage 内容，不应该覆盖")
        assert len(result) == 1
        assert result[0]["prompt_text"] == "用户已经在后端设置过的新内容"
    finally:
        db.close()


def test_migrate_rejects_legacy_text_over_cap_without_losing_anything():
    db = _make_session()
    try:
        user_id = _make_user(db)
        with pytest.raises(ValueError, match="4000"):
            svc.migrate_legacy_prompt(db, user_id, "字" * 4001)
        # Nothing partially written.
        assert svc.list_custom_prompts(db, user_id) == []
    finally:
        db.close()


def test_migrate_with_blank_text_is_noop():
    db = _make_session()
    try:
        user_id = _make_user(db)
        result = svc.migrate_legacy_prompt(db, user_id, "   ")
        assert result == []
    finally:
        db.close()


# --- master switch ---

def test_prompt_injection_switch_defaults_to_false_and_can_be_toggled():
    db = _make_session()
    try:
        user_id = _make_user(db)
        assert svc.get_prompt_injection_enabled(db, user_id) is False

        svc.set_prompt_injection_enabled(db, user_id, True)
        assert svc.get_prompt_injection_enabled(db, user_id) is True

        svc.set_prompt_injection_enabled(db, user_id, False)
        assert svc.get_prompt_injection_enabled(db, user_id) is False
    finally:
        db.close()


# --- per-user isolation ---

def test_custom_prompts_are_scoped_per_user():
    db = _make_session()
    try:
        user_a = _make_user(db)
        user_b = _make_user(db)
        svc.replace_custom_prompts(
            db, user_a, [{"target_type": "global", "target_key": "", "prompt_text": "user A 的提示词"}]
        )
        assert svc.list_custom_prompts(db, user_b) == []
        assert len(svc.list_custom_prompts(db, user_a)) == 1
        assert svc.get_prompt_injection_enabled(db, user_b) is False
    finally:
        db.close()


# --- hash column ---

def test_prompt_hash_changes_when_content_changes():
    db = _make_session()
    try:
        user_id = _make_user(db)
        first = svc.replace_custom_prompts(
            db, user_id, [{"target_type": "global", "target_key": "", "prompt_text": "版本一"}]
        )
        second = svc.replace_custom_prompts(
            db, user_id, [{"target_type": "global", "target_key": "", "prompt_text": "版本二"}]
        )
        assert first[0]["prompt_hash"] != second[0]["prompt_hash"]
        assert len(second[0]["prompt_hash"]) == 12
    finally:
        db.close()
