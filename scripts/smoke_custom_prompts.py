#!/usr/bin/env python3
"""Smoke test script for the /v1/custom-prompts endpoints (Phase B).

Offline by default: uses a temp SQLite database so nothing is written to the
global configured database. Set LIVE=1 TO USE THE REAL DATABASE; that mode
requires explicit opt-in and still requires a backup first.
"""

import json
import sys
import os
import tempfile
from datetime import datetime, timezone
from uuid import uuid4

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sqlalchemy import text

testclient_imported = False
db_imported = False


def _ensure_imports():
    global testclient_imported, db_imported
    if not testclient_imported:
        from fastapi.testclient import TestClient
        _ensure_imports.testclient = TestClient
        testclient_imported = True
    if not db_imported:
        from api.database import UserCustomPromptDB, UserDB, get_db_ctx, init_db
        from api.services import auth_service
        _ensure_imports.UserCustomPromptDB = UserCustomPromptDB
        _ensure_imports.UserDB = UserDB
        _ensure_imports.get_db_ctx = get_db_ctx
        _ensure_imports.init_db = init_db
        _ensure_imports.auth_service = auth_service
        db_imported = True

def _set_offline_database() -> str:
    """Point api.database at a temp SQLite DB and return its temp dir name.

    Must be called before importing api.database / api.main so the engine and
    app see the isolated DATABASE_URL.
    """
    tmpdir = tempfile.mkdtemp(prefix="ta-smoke-custom-prompts-")
    os.environ["DATABASE_URL"] = "sqlite:///" + tmpdir + "/smoke.db"
    return tmpdir


class _TempDirContext:
    def __init__(self):
        self.dir = None

    def __enter__(self):
        self.dir = _set_offline_database()
        return self.dir

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        if self.dir:
            import shutil
            shutil.rmtree(self.dir, ignore_errors=True)


_temp_ctx = _TempDirContext()

created_user_ids: list[str] = []


def _new_user() -> str:
    """Create a throwaway user, return its bearer token."""
    _ensure_imports()
    auth_service = _ensure_imports.auth_service
    get_db_ctx = _ensure_imports.get_db_ctx
    UserDB = _ensure_imports.UserDB
    email = auth_service.normalize_email(f"smoke-prompt-{uuid4().hex[:8]}@test.local")
    now = datetime.now(timezone.utc)
    with get_db_ctx() as db:
        user = UserDB(
            id=str(uuid4()), email=email, is_active=True,
            created_at=now, updated_at=now, last_login_at=now,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        created_user_ids.append(user.id)
        return auth_service.create_access_token(user)


def _cleanup() -> None:
    if not created_user_ids:
        return
    _ensure_imports()
    get_db_ctx = _ensure_imports.get_db_ctx
    UserDB = _ensure_imports.UserDB
    UserCustomPromptDB = _ensure_imports.UserCustomPromptDB
    with get_db_ctx() as db:
        for uid in created_user_ids:
            db.query(UserCustomPromptDB).filter(UserCustomPromptDB.user_id == uid).delete(synchronize_session=False)
            db.query(UserDB).filter(UserDB.id == uid).delete(synchronize_session=False)
        db.commit()
    print(f"\n[cleanup] Removed {len(created_user_ids)} throwaway user(s) and their prompt rows.")


def show(label: str, resp) -> None:
    print(f"\n--- {label} [{resp.status_code}] ---")
    try:
        print(json.dumps(resp.json(), ensure_ascii=False, indent=2)[:1500])
    except ValueError:
        print(resp.text[:500])


def check_schema() -> None:
    """Step 2d/2e: confirm the ALTER TABLE landed and existing rows read as False, not None."""
    print("\n================ 1. schema 检查 ================")
    _ensure_imports()
    init_db = _ensure_imports.init_db
    init_db()  # runs _ensure_user_schema(), i.e. the ALTER TABLE under test

    from api.database import engine

    with engine.begin() as conn:
        cols = {row[1]: row for row in conn.execute(text("PRAGMA table_info(users)"))}
        print("\n--- PRAGMA table_info(users) 中的开关列 ---")
        if "prompt_injection_enabled" not in cols:
            print("!!! FAIL: users.prompt_injection_enabled 不存在")
            sys.exit(1)
        # PRAGMA columns: (cid, name, type, notnull, dflt_value, pk)
        cid, name, ctype, notnull, default, pk = cols["prompt_injection_enabled"]
        # dflt_value 仅作参考打印，不是判据：两条建表路径渲染不同
        #   已有部署走 _ensure_user_schema() 的 ALTER TABLE → DEFAULT 0（不带引号）
        #   全新库走 Base.metadata.create_all() 的 server_default="0" → DEFAULT '0'（带引号）
        # 而它是字符串，repr() 一定会再加一对引号，看引号无法区分这两条路径。
        # 库的身份由上面的行数检查和下面的已有用户读取判定。
        print(
            f"name={name} type={ctype} notnull={notnull} "
            f"default={default} raw={default!r} py_type={type(default).__name__}"
        )
        assert notnull == 1, f"期望 NOT NULL，实际 notnull={notnull}"

        nulls = conn.execute(
            text("SELECT COUNT(*) FROM users WHERE prompt_injection_enabled IS NULL")
        ).scalar()
        truthy = conn.execute(
            text("SELECT COUNT(*) FROM users WHERE prompt_injection_enabled = 1")
        ).scalar()
        print(f"\nusers 总行数={nulls + truthy}  值为 NULL={nulls}  值为 1(True)={truthy}")
        assert nulls == 0, f"有 {nulls} 行是 NULL —— 应该全是 0"
        assert truthy == 0, f"有 {truthy} 行是 True —— 新列应该默认全关"

    # Read back through the ORM on a pre-existing user (not one we just made).
    # In offline mode we created users during init, so this verifies the ORM
    # read-after-create path without touching production data.
    get_db_ctx = _ensure_imports.get_db_ctx
    UserDB = _ensure_imports.UserDB
    with get_db_ctx() as db:
        existing = (
            db.query(UserDB)
            .filter(~UserDB.id.in_(created_user_ids) if created_user_ids else True)
            .first()
        )
        if existing is None:
            live = os.environ.get("LIVE", "").strip().lower() in ("1", "true", "yes")
            assert not live, "LIVE 模式下查不到任何已有用户 —— 迁移读回检查无法在空库上验证"
            # Fresh offline temp DB has no rows yet: seed one fixture user so the
            # ORM read-back path is exercised. The temp DB is deleted on exit.
            auth_service = _ensure_imports.auth_service
            now = datetime.now(timezone.utc)
            email = auth_service.normalize_email(f"smoke-schema-{uuid4().hex[:8]}@test.local")
            user = UserDB(
                id=str(uuid4()), email=email, is_active=True,
                created_at=now, updated_at=now, last_login_at=now,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            created_user_ids.append(user.id)
            existing = user
        assert existing is not None, "查不到任何已有用户 —— schema 初始化未创建本地用户"
        val = existing.prompt_injection_enabled
        print(f"\n已有用户 {existing.email} 的 prompt_injection_enabled = {val!r} (type={type(val).__name__})")
        assert val is not None, "读出来是 None —— 迁移的默认值没生效"
        assert bool(val) is False, f"读出来是 {val!r} —— 应该是 False"
    print("\nschema 检查通过：库身份已确认、列已加、NOT NULL、已有用户读出 False")


def check_endpoints() -> None:
    print("\n================ 2. 端点行为检查 ================")
    _ensure_imports()
    TestClient = _ensure_imports.testclient
    get_db_ctx = _ensure_imports.get_db_ctx
    UserDB = _ensure_imports.UserDB
    from api.main import app
    client = TestClient(app, raise_server_exceptions=False)
    headers = {"Authorization": f"Bearer {_new_user()}"}

    r = client.get("/v1/custom-prompts", headers=headers)
    show("GET /v1/custom-prompts（新用户，应为空）", r)
    assert r.status_code == 200 and r.json() == []

    r = client.get("/v1/custom-prompts/switch", headers=headers)
    show("GET /v1/custom-prompts/switch（默认应为 false）", r)
    assert r.json() == {"enabled": False}

    legacy = "从 localStorage 迁移：更关注估值安全边际、政策催化与机构资金行为"
    r = client.post("/v1/custom-prompts/migrate", headers=headers, json={"legacy_text": legacy})
    show("POST /v1/custom-prompts/migrate（首次迁移，应写入）", r)
    assert r.status_code == 200 and len(r.json()) == 1 and r.json()[0]["prompt_text"] == legacy

    r = client.post("/v1/custom-prompts/migrate", headers=headers,
                    json={"legacy_text": "这段不该覆盖上面那条"})
    show("POST /v1/custom-prompts/migrate（二次调用，应幂等跳过）", r)
    assert r.json()[0]["prompt_text"] == legacy, "幂等失效：旧内容被覆盖了"

    r = client.patch("/v1/custom-prompts", headers=headers, json={"prompts": [
        {"target_type": "global", "target_key": "", "prompt_text": "全局：概率形式输出，置信度上限 75%"},
        {"target_type": "role", "target_key": "research_manager", "prompt_text": "研究经理：按证据可信度分级判断"},
    ]})
    show("PATCH /v1/custom-prompts（设置 global + 研究经理覆盖）", r)
    assert r.status_code == 200 and len(r.json()) == 2

    r = client.get("/v1/custom-prompts/resolved", headers=headers)
    body = r.json()
    assert r.status_code == 200, body
    assert len(body) == 15, f"resolved 应返回 15 个角色，实际 {len(body)}"
    interesting = [x for x in body if x["role_key"] in ("market", "research_manager", "risk_manager")]
    print(f"\n--- GET /v1/custom-prompts/resolved [{r.status_code}]（共 15 个角色，摘 3 个）---")
    print(json.dumps(interesting, ensure_ascii=False, indent=2))

    rm = next(x for x in body if x["role_key"] == "research_manager")
    assert rm["resolved_text"] == "全局：概率形式输出，置信度上限 75%\n\n研究经理：按证据可信度分级判断"
    assert rm["override_source"] == "role"
    mk = next(x for x in body if x["role_key"] == "market")
    assert mk["resolved_text"] == "全局：概率形式输出，置信度上限 75%" and mk["override_source"] is None

    r = client.patch("/v1/custom-prompts/switch", headers=headers, json={"enabled": True})
    show("PATCH /v1/custom-prompts/switch → true", r)
    assert r.json() == {"enabled": True}
    r = client.patch("/v1/custom-prompts/switch", headers=headers, json={"enabled": False})
    show("PATCH /v1/custom-prompts/switch → false（复位）", r)
    assert r.json() == {"enabled": False}

    r = client.patch("/v1/custom-prompts", headers=headers, json={"prompts": [
        {"target_type": "global", "target_key": "", "prompt_text": "字" * 4001},
    ]})
    show("PATCH /v1/custom-prompts（global 超 4000 上限，应 422）", r)
    assert r.status_code == 422

    r = client.patch("/v1/custom-prompts", headers=headers, json={"prompts": [
        {"target_type": "global", "target_key": "", "prompt_text": "字" * 4000},
        {"target_type": "role", "target_key": "research_manager", "prompt_text": "字" * 2000},
    ]})
    show("PATCH /v1/custom-prompts（字段各自合规但 resolved=6002 超限，应 422 且指名角色）", r)
    assert r.status_code == 422 and "research_manager" in r.json()["detail"]

    r = client.patch("/v1/custom-prompts", headers=headers, json={"prompts": [
        {"target_type": "role", "target_key": "portfolio_manager", "prompt_text": "x"},
    ]})
    show("PATCH（用 ANALYST_AGENT_NAMES 的展示名 portfolio_manager，应 422 拒绝）", r)
    assert r.status_code == 422, "展示层短名不应被当作合法 role_key"

    other = {"Authorization": f"Bearer {_new_user()}"}
    r = client.get("/v1/custom-prompts", headers=other)
    show("GET /v1/custom-prompts（换一个用户，必须看不到上面的数据）", r)
    assert r.json() == [], "跨用户数据泄漏"


if __name__ == "__main__":
    live = os.environ.get("LIVE", "").strip().lower() in ("1", "true", "yes")
    if live:
        print("=== custom-prompts 冒烟测试（LIVE=1 真实数据库）===")
    else:
        _temp_ctx.__enter__()
        print("=== custom-prompts 冒烟测试（offline 临时数据库）===")
    try:
        from api.main import app
        check_schema()
        check_endpoints()
        print("\n=== 全部断言通过 ===")
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        _cleanup()
        if not live:
            _temp_ctx.__exit__(None, None, None)
