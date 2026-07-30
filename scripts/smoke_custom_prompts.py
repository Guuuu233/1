#!/usr/bin/env python3
"""Smoke test script for the /v1/custom-prompts endpoints (Phase B).

Runs against the REAL configured database (no DATABASE_URL override) so it also
verifies that `_ensure_user_schema()`'s ALTER TABLE for `users.prompt_injection_enabled`
applied correctly to an existing deployment.

`tests/test_custom_prompts.py` already covers the service layer against in-memory
sqlite; this script covers what unit tests can't: real routing, pydantic validation,
the auth dependency, HTTPException→422 mapping, and the live schema migration.

Run inside the container:
    /app/.venv/bin/python scripts/smoke_custom_prompts.py

BACK UP THE DATABASE FIRST. This writes to the real db. It creates two throwaway
users and deletes them (plus their prompt rows) in the finally block, but a crash
between create and cleanup can leave test users behind.
"""

import json
import sys
from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import text

from api.database import UserCustomPromptDB, UserDB, engine, get_db_ctx, init_db
from api.main import app
from api.services import auth_service

created_user_ids: list[str] = []


def _new_user() -> str:
    """Create a throwaway user, return its bearer token."""
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
    print("\n================ 1. 真实 db 的 schema 检查 ================")
    init_db()  # runs _ensure_user_schema(), i.e. the ALTER TABLE under test

    with engine.begin() as conn:
        # 库身份检查必须在 schema 断言之前：连到一个新建的空库时，下面所有 schema
        # 断言在空库上一样会通过（列是 create_all() 建的、0 行也满足 nulls==0），
        # 反而掩盖"根本没连到生产库"这个真问题。用数据存在性判定库的身份，
        # 不依赖 DDL 渲染细节。
        total = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
        report_total = conn.execute(text("SELECT COUNT(*) FROM reports")).scalar()
        print(f"\n--- 库身份检查 ---\nusers 行数={total}  reports 行数={report_total}")
        assert total > 0, f"users 表 {total} 行 —— 疑似连到新建库而非生产库，立即停止"
        assert report_total > 0, f"reports 表 {report_total} 行 —— 疑似连到新建库而非生产库，立即停止"

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
        print(f"\nusers 总行数={total}  值为 NULL={nulls}  值为 1(True)={truthy}")
        assert nulls == 0, f"有 {nulls} 行是 NULL —— 应该全是 0"
        assert truthy == 0, f"有 {truthy} 行是 True —— 新列应该默认全关"

    # Read back through the ORM on a pre-existing user (not one we just made).
    # 这也是库身份的第二道判据：新建库里没有已有用户，查询会返回 None。
    with get_db_ctx() as db:
        existing = (
            db.query(UserDB)
            .filter(~UserDB.id.in_(created_user_ids) if created_user_ids else True)
            .first()
        )
        assert existing is not None, "查不到任何已有用户 —— 疑似连到新建库而非生产库，立即停止"
        val = existing.prompt_injection_enabled
        print(f"\n已有用户 {existing.email} 的 prompt_injection_enabled = {val!r} (type={type(val).__name__})")
        assert val is not None, "读出来是 None —— 迁移的默认值没生效"
        assert bool(val) is False, f"读出来是 {val!r} —— 应该是 False"
    print("\nschema 检查通过：库身份已确认、列已加、NOT NULL、已有用户读出 False")


def check_endpoints() -> None:
    print("\n================ 2. 端点行为检查 ================")
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
    print("=== custom-prompts 冒烟测试（真实数据库）===")
    try:
        check_schema()
        check_endpoints()
        print("\n=== 全部断言通过 ===")
    finally:
        _cleanup()
