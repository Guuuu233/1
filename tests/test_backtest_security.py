"""Offline tests for backtest auth, owner scoping, and resource limits."""
from __future__ import annotations

import queue
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api.services import backtest_service as bt


def _backtest_kwargs(**overrides):
    kwargs = {
        "symbol": "600519.SH",
        "start_date": "2024-01-02",
        "end_date": "2024-01-31",
        "selected_analysts": ["market"],
        "hold_days": 5,
        "sample_interval": 1,
        "config": {},
    }
    kwargs.update(overrides)
    return kwargs


def _create(job_id, user_id="u1", **overrides):
    fields = {
        "job_id": job_id,
        "user_id": user_id,
        "status": "completed",
        "created_at": "2024-01-01T00:00:00+00:00",
        "records": [],
        "stats": None,
    }
    fields.update(overrides)
    bt._create_job(**fields)


def _user_token():
    from api.database import UserDB, get_db_ctx, init_db
    from api.services import auth_service

    init_db()
    now = datetime.now(timezone.utc)
    user_id = str(uuid4())
    email = f"backtest-{uuid4().hex[:12]}@test.com"
    with get_db_ctx() as db:
        user = UserDB(
            id=user_id,
            email=email,
            is_active=True,
            created_at=now,
            updated_at=now,
            last_login_at=now,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return user_id, auth_service.create_access_token(user)


@pytest.fixture
def client():
    from api.main import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def reset_backtest_state():
    with bt._lock:
        bt._backtest_jobs.clear()
        bt._job_queue.queue.clear()
        # Reset the worker registry too so one test cannot hand a real
        # daemon thread to another test.
        bt._workers_started = False
        bt._workers = []
    yield


class TestSampleIntervalValidation:
    @pytest.mark.parametrize("bad", [0, -1, 366, True])
    def test_submit_rejects_bad_intervals(self, bad):
        with pytest.raises(ValueError, match="sample_interval"):
            bt.submit(user_id="u1", **_backtest_kwargs(sample_interval=bad))
        assert bt._backtest_jobs == {}

    def test_upper_bound_is_allowed(self):
        assert bt.validate_sample_interval(365) == 365


class TestOwnerScoping:
    def test_get_list_delete_only_touch_owner_jobs(self):
        _create("job-a", user_id="alice", status="completed")
        _create("job-b", user_id="bob", status="running")

        assert [job["job_id"] for job in bt.list_jobs("alice")] == ["job-a"]
        assert bt.get_job("job-b", "alice") is None
        assert bt.delete_job("job-b", "alice") is False
        assert bt.get_job("job-b", "bob") is not None
        assert bt.delete_job("job-b", "bob") is True
        assert bt.get_job("job-b", "bob") is None


class TestResourceLimits:
    def test_prune_does_not_delete_when_store_is_under_cap(self, monkeypatch):
        monkeypatch.setattr(bt, "MAX_RETAINED_BACKTEST_JOBS", 100)
        for index in range(98):
            _create(
                f"term-{index}",
                user_id="u1",
                status="completed",
                created_at=f"2024-01-{index + 1:02d}T00:00:00+00:00",
            )
        _create("running-1", user_id="u1", status="running", created_at="2024-03-01T00:00:00+00:00")

        bt._prune_old_jobs()

        assert len(bt._backtest_jobs) == 99

    def test_submit_rejects_when_queue_is_full(self, monkeypatch):
        monkeypatch.setattr(bt, "_job_queue", queue.Queue(maxsize=1))
        monkeypatch.setattr(bt, "_ensure_workers", lambda: None)

        first = bt.submit(user_id="u1", **_backtest_kwargs())
        with pytest.raises(bt.BacktestQueueFullError, match="queue is full"):
            bt.submit(user_id="u2", **_backtest_kwargs())

        assert set(bt._backtest_jobs) == {first}

    def test_worker_pool_size_is_bounded(self, monkeypatch):
        class FakeThread:
            def __init__(self, *, target=None, name="", daemon=None):
                self.target = target
                self.name = name
                self.daemon = daemon
                self.started = False

            def start(self):
                self.started = True

        monkeypatch.setattr(bt.threading, "Thread", FakeThread)
        monkeypatch.setattr(bt, "MAX_BACKTEST_WORKERS", 3)
        monkeypatch.setattr(bt, "_workers_started", False)
        monkeypatch.setattr(bt, "_workers", [])

        bt._ensure_workers()

        assert len(bt._workers) == 3
        assert all(worker.daemon for worker in bt._workers)
        assert all(worker.started for worker in bt._workers)
        assert bt._workers_started is True

    def test_terminal_history_is_pruned(self, monkeypatch):
        monkeypatch.setattr(bt, "MAX_RETAINED_BACKTEST_JOBS", 2)
        _create("old", user_id="u1", status="completed", created_at="2024-01-01T00:00:00+00:00")
        _create("new", user_id="u1", status="failed", created_at="2024-01-02T00:00:00+00:00")
        _create("running", user_id="u1", status="running", created_at="2024-01-03T00:00:00+00:00")

        bt._prune_old_jobs()

        assert set(bt._backtest_jobs) == {"new", "running"}

    def test_backtest_failure_marks_job_failed(self, monkeypatch):
        _create("fail", user_id="u1", status="pending")
        with patch.object(bt, "_get_trading_dates", side_effect=ValueError("bad date range")):
            bt._run_backtest("fail", **_backtest_kwargs())

        job = bt.get_job("fail", "u1")
        assert job["status"] == "failed"
        assert "bad date range" in job["error"]
        assert job["records"] == []

    def test_deleted_job_is_not_resurrected(self, monkeypatch):
        _create("gone", user_id="u1", status="pending")
        bt.delete_job("gone", "u1")
        with (
            patch.object(bt, "_get_trading_dates", return_value=["2024-01-02"]),
            patch.object(
                bt,
                "_run_single_analysis",
                return_value={"decision": "HOLD", "final_trade_decision": "HOLD"},
            ),
        ):
            bt._run_backtest("gone", **_backtest_kwargs())

        assert "gone" not in bt._backtest_jobs


class TestApiWiring:
    def test_backtest_read_endpoints_require_api_user(self):
        from api.main import _require_api_user, app

        for path in ("/v1/backtest", "/v1/backtest/{job_id}"):
            route = next(r for r in app.routes if getattr(r, "path", None) == path)
            calls = [dependency.call for dependency in route.dependant.dependencies]
            assert _require_api_user in calls

    def test_backtest_get_list_delete_are_owner_scoped(self, client):
        uid_a, token_a = _user_token()
        uid_b, token_b = _user_token()
        _create("job-a", user_id=uid_a, status="completed")
        _create("job-b", user_id=uid_b, status="running")
        headers_a = {"Authorization": f"Bearer {token_a}"}
        headers_b = {"Authorization": f"Bearer {token_b}"}

        list_a = client.get("/v1/backtest", headers=headers_a)
        list_b = client.get("/v1/backtest", headers=headers_b)
        assert list_a.status_code == 200
        assert [job["job_id"] for job in list_a.json()["jobs"]] == ["job-a"]
        assert list_a.json()["total"] == 1
        assert [job["job_id"] for job in list_b.json()["jobs"]] == ["job-b"]

        assert client.get("/v1/backtest/job-a", headers=headers_b).status_code == 404
        assert client.get("/v1/backtest/job-b", headers=headers_a).status_code == 404
        assert client.delete("/v1/backtest/job-a", headers=headers_b).status_code == 404
        assert client.delete("/v1/backtest/job-a", headers=headers_a).status_code == 200
        assert client.get("/v1/backtest/job-a", headers=headers_a).status_code == 404

    def test_submit_records_api_user_as_owner(self, client):
        user_id, token = _user_token()
        headers = {"Authorization": f"Bearer {token}"}
        with (
            patch.object(bt, "_run_backtest", lambda *args, **kwargs: None),
            patch.object(bt, "_ensure_workers", lambda: None),
        ):
            response = client.post(
                "/v1/backtest",
                headers=headers,
                json={
                    "symbol": "600519.SH",
                    "start_date": "2024-01-02",
                    "end_date": "2024-01-03",
                    "sample_interval": 1,
                },
            )

        assert response.status_code == 200
        job_id = response.json()["job_id"]
        job = bt.get_job(job_id, user_id)
        assert job is not None
        assert job["user_id"] == user_id
        assert job["status"] == "pending"

    @pytest.mark.parametrize("bad", [0, -1, 366, True, "5", 1.5])
    def test_submit_rejects_invalid_sample_interval_via_api(self, client, bad):
        _, token = _user_token()
        headers = {"Authorization": f"Bearer {token}"}
        response = client.post(
            "/v1/backtest",
            headers=headers,
            json={
                "symbol": "600519.SH",
                "start_date": "2024-01-02",
                "end_date": "2024-01-03",
                "sample_interval": bad,
            },
        )

        assert response.status_code == 400
        assert "sample_interval" in response.json()["detail"]

    def test_submit_returns_429_when_queue_is_full(self, client, monkeypatch):
        monkeypatch.setattr(bt, "_job_queue", queue.Queue(maxsize=1))
        monkeypatch.setattr(bt, "_ensure_workers", lambda: None)
        _, token = _user_token()
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "symbol": "600519.SH",
            "start_date": "2024-01-02",
            "end_date": "2024-01-03",
            "sample_interval": 1,
        }

        assert client.post("/v1/backtest", headers=headers, json=payload).status_code == 200
        assert client.post("/v1/backtest", headers=headers, json=payload).status_code == 429
