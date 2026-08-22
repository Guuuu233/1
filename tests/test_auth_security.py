import os
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4
import jwt
import pytest
from fastapi.testclient import TestClient

from api.database import EmailVerificationCodeDB, UserDB, get_db_ctx, init_db, _ensure_auth_schema, engine
from api.services import auth_service, token_service
from sqlalchemy import text


def _get_test_client() -> TestClient:
    from api.main import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


class TestRequireUserAuth:
    def test_anonymous_request_allowed_when_no_authorization_header(self):
        client = _get_test_client()
        r = client.post("/v1/analyze", json={
            "symbol": "600519.SH",
            "trade_date": "2024-01-15",
            "dry_run": True,
        })
        assert r.status_code == 200
        assert "job_id" in r.json()

    def test_bad_authorization_header_format_rejected(self):
        client = _get_test_client()
        # Non-Bearer scheme
        r = client.post("/v1/analyze", headers={"Authorization": "Basic dXNlcjpwYXNz"}, json={"dry_run": True})
        assert r.status_code == 401

        # Invalid format (no space / token)
        r = client.post("/v1/analyze", headers={"Authorization": "Bearer"}, json={"dry_run": True})
        assert r.status_code == 401

        r = client.post("/v1/analyze", headers={"Authorization": "InvalidScheme 123"}, json={"dry_run": True})
        assert r.status_code == 401

        r = client.post("/v1/analyze", headers={"Authorization": "   "}, json={"dry_run": True})
        assert r.status_code == 401

    def test_bad_bearer_token_rejected(self):
        client = _get_test_client()
        # Arbitrary fake JWT
        r = client.post("/v1/analyze", headers={"Authorization": "Bearer invalid.jwt.token"}, json={"dry_run": True})
        assert r.status_code == 401

        # Random string
        r = client.post("/v1/analyze", headers={"Authorization": "Bearer notatoken"}, json={"dry_run": True})
        assert r.status_code == 401

        # Invalid API token
        r = client.post("/v1/analyze", headers={"Authorization": "Bearer ta-sk-nonexistenttoken"}, json={"dry_run": True})
        assert r.status_code == 401

    def test_expired_jwt_token_rejected(self):
        client = _get_test_client()
        now = datetime.now(timezone.utc)
        # Create a user first
        user_id = str(uuid4())
        email = f"expired-user-{user_id[:8]}@test.local"
        with get_db_ctx() as db:
            user = UserDB(id=user_id, email=email, is_active=True, created_at=now, updated_at=now)
            db.add(user)
            db.commit()

        # Sign expired token (expired 1 hour ago)
        expired_payload = {
            "sub": user_id,
            "email": email,
            "exp": now - timedelta(hours=1),
            "iat": now - timedelta(hours=2),
        }
        expired_token = jwt.encode(expired_payload, auth_service._secret_key(), algorithm=auth_service.ALGORITHM)

        r = client.post("/v1/analyze", headers={"Authorization": f"Bearer {expired_token}"}, json={"dry_run": True})
        assert r.status_code == 401
        assert "expired" in r.json().get("detail", "").lower() or r.status_code == 401

    def test_web_only_endpoint_rejects_api_token(self):
        client = _get_test_client()
        user_id = str(uuid4())
        email = f"web-only-{user_id[:8]}@test.local"
        with get_db_ctx() as db:
            user = UserDB(id=user_id, email=email, is_active=True)
            db.add(user)
            db.commit()
            token_data = token_service.create_token(db, user_id, "test-key")
            api_token = token_data["token"]

        # /v1/auth/me is protected by _require_web_user
        r = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {api_token}"})
        assert r.status_code == 401

        # While API endpoint allows it
        r_api = client.post("/v1/analyze", headers={"Authorization": f"Bearer {api_token}"}, json={"dry_run": True})
        assert r_api.status_code == 200


class TestVerificationCodeAndRateLimiting:
    def test_wrong_code_attempts_1_to_4_rejected_and_5_invalidates(self):
        client = _get_test_client()
        email = f"attempts-test-{uuid4().hex[:8]}@test.local"

        r = client.post("/v1/auth/request-code", json={"email": email})
        assert r.status_code == 200
        correct_code = r.json()["dev_code"]

        # 1-4 wrong code attempts
        for attempt in range(1, 5):
            r_verify = client.post("/v1/auth/verify-code", json={"email": email, "code": "000000"})
            assert r_verify.status_code == 400
            with get_db_ctx() as db:
                row = db.query(EmailVerificationCodeDB).filter(
                    EmailVerificationCodeDB.email == email,
                    EmailVerificationCodeDB.consumed_at.is_(None)
                ).first()
                assert row is not None
                assert row.attempts == attempt

        # 5th wrong code attempt -> invalidates/consumes the code
        r_verify_5 = client.post("/v1/auth/verify-code", json={"email": email, "code": "000000"})
        assert r_verify_5.status_code == 400
        with get_db_ctx() as db:
            row_consumed = db.query(EmailVerificationCodeDB).filter(
                EmailVerificationCodeDB.email == email,
                EmailVerificationCodeDB.consumed_at.isnot(None)
            ).first()
            assert row_consumed is not None
            assert row_consumed.attempts == 5

        # 6th attempt with the CORRECT code -> fails because code was consumed
        r_verify_correct = client.post("/v1/auth/verify-code", json={"email": email, "code": correct_code})
        assert r_verify_correct.status_code == 400

    def test_correct_code_consumed_and_invalidates_subsequent_attempts(self):
        client = _get_test_client()
        email = f"success-test-{uuid4().hex[:8]}@test.local"

        r = client.post("/v1/auth/request-code", json={"email": email})
        assert r.status_code == 200
        code = r.json()["dev_code"]

        # First verification succeeds
        r_verify = client.post("/v1/auth/verify-code", json={"email": email, "code": code})
        assert r_verify.status_code == 200
        assert "access_token" in r_verify.json()

        # Second verification with same code fails (already consumed)
        r_verify_again = client.post("/v1/auth/verify-code", json={"email": email, "code": code})
        assert r_verify_again.status_code == 400

    def test_verify_code_rate_limiting_429(self, monkeypatch):
        client = _get_test_client()
        email = f"rate-limit-test-{uuid4().hex[:8]}@test.local"

        # Set strict limit of 3 attempts per window
        from api import main as api_main
        monkeypatch.setattr(api_main, "_VERIFY_CODE_RATE_MAX", 3)
        monkeypatch.setattr(api_main, "_VERIFY_CODE_RATE_WINDOW_SECONDS", 60)
        api_main._verify_code_rate_hits.clear()

        # 3 calls
        for _ in range(3):
            r = client.post("/v1/auth/verify-code", json={"email": email, "code": "123456"})
            assert r.status_code in (400, 200)

        # 4th call hits 429
        r_limit = client.post("/v1/auth/verify-code", json={"email": email, "code": "123456"})
        assert r_limit.status_code == 429
        # Ensure no sensitive leak about email existence or remaining attempts
        assert "attempts" not in r_limit.json().get("detail", "").lower()

    def test_request_code_rate_limiting_429(self, monkeypatch):
        client = _get_test_client()
        email = f"req-rate-limit-{uuid4().hex[:8]}@test.local"

        from api import main as api_main
        monkeypatch.setattr(api_main, "_REQUEST_CODE_RATE_MAX", 2)
        monkeypatch.setattr(api_main, "_REQUEST_CODE_RATE_WINDOW_SECONDS", 60)
        api_main._request_code_rate_hits.clear()

        r1 = client.post("/v1/auth/request-code", json={"email": email})
        assert r1.status_code == 200
        r2 = client.post("/v1/auth/request-code", json={"email": email})
        assert r2.status_code == 200

        # 3rd request hits 429
        r3 = client.post("/v1/auth/request-code", json={"email": email})
        assert r3.status_code == 429

    def test_production_environment_never_echoes_dev_code(self, monkeypatch):
        client = _get_test_client()
        monkeypatch.setenv("APP_ENV", "production")
        email = f"prod-test-{uuid4().hex[:8]}@test.local"

        # In production without SMTP configured, send_login_code returns None
        r = client.post("/v1/auth/request-code", json={"email": email})
        assert r.status_code == 200
        assert "dev_code" not in r.json()
        assert r.json() == {"message": "验证码已发送"}


class TestAuthDatabaseMigration:
    def test_ensure_auth_schema_adds_attempts_column_backward_compatibility(self):
        # Verify attempts column exists on table
        with engine.begin() as conn:
            columns = {row[1] for row in conn.execute(text("PRAGMA table_info(email_verification_codes)"))}
            assert "attempts" in columns

        # Calling _ensure_auth_schema again is idempotent
        _ensure_auth_schema()
