"""Contract tests for H-03a entrypoint wiring (api/main + scheduled_service).

Verifies:
1. POST /v1/analyze:
   - Unprovided horizons -> default short, resolution_source='default'.
   - Unprovided horizons + dual query -> default short, resolution_source='default' (query does NOT expand).
   - Explicit ['medium'] + dual query -> explicit medium, resolution_source='explicit' (query does NOT change resolved).
   - Explicit dual ['short', 'medium'] -> explicit dual, resolution_source='explicit'.
   - Invalid horizons (null, [], illegal values) -> 422 validation error.
2. Chat entrypoint:
   - Natural language query with dual horizons -> AnalyzeRequest has unprovided/default short,
     horizons_explicit is False, resolution_source='default'.
3. Scheduled task construction & request building:
   - _build_scheduled_analyze_request uses stored horizon as explicit single-horizon list.
   - horizons_resolution_source is 'explicit', isolated from query text.
   - Illegal horizon or unsupported dual horizon raises ValueError.
4. Scheduled service & endpoints:
   - Rejection of dual horizons (no dual storage; must reject 4xx / ValueError).
   - Rejection of illegal horizons.
   - Correct acceptance of single horizon (short or medium).
"""

import asyncio
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api import main
from api.database import Base, ScheduledAnalysisDB, UserDB
from api.job_store import InMemoryJobStore
from api.services import scheduled_service
from tradingagents.graph.horizon_profile import (
    RESOLUTION_SOURCE_DEFAULT,
    RESOLUTION_SOURCE_EXPLICIT,
)


@pytest.fixture
def test_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def mock_user():
    return UserDB(id="test-user-001", email="test@example.com")


@pytest.fixture
def api_client(test_db, mock_user):
    def override_get_db():
        yield test_db

    main.app.dependency_overrides[main.get_db] = override_get_db
    main.app.dependency_overrides[main._require_api_user] = lambda: mock_user
    client = TestClient(main.app, raise_server_exceptions=False)
    yield client
    main.app.dependency_overrides.clear()


# ── 1. POST /v1/analyze 端点契约 ───────────────────────────────────────────────

def test_analyze_unprovided_defaults_to_short(api_client):
    captured = []

    async def fake_run_job(job_id, analyze_request, *_args, **_kwargs):
        captured.append(analyze_request)
        main._set_job(job_id, status="completed", decision="DRY_RUN", result={})

    with (
        patch.object(main, "_job_store_instance", InMemoryJobStore()),
        patch.object(main, "_compose_analysis_user_context", return_value={}),
        patch.object(main, "_run_job", side_effect=fake_run_job),
        patch.object(main, "_get_reverse_stock_map", return_value={"600519.SH": "贵州茅台"}),
    ):
        # 1a. 无 horizons 无 query -> default short
        captured.clear()
        resp = api_client.post("/v1/analyze", json={"symbol": "600519.SH", "dry_run": True})
        assert resp.status_code == 200
        assert len(captured) == 1
        req = captured[0]
        assert req.horizons == ["short"]
        assert req.horizons_explicit is False
        assert req.horizons_resolution_source == RESOLUTION_SOURCE_DEFAULT

        # 1b. 无 horizons 但 query 包含双档文本 -> 仍为 default short，禁止 query 扩档
        captured.clear()
        resp = api_client.post(
            "/v1/analyze",
            json={"symbol": "600519.SH", "query": "短线和中线都分析一下，短中都看看", "dry_run": True},
        )
        assert resp.status_code == 200
        assert len(captured) == 1
        req_query = captured[0]
        assert req_query.horizons == ["short"]
        assert req_query.horizons_explicit is False
        assert req_query.horizons_resolution_source == RESOLUTION_SOURCE_DEFAULT


def test_analyze_explicit_medium_ignores_dual_query(api_client):
    captured = []

    async def fake_run_job(job_id, analyze_request, *_args, **_kwargs):
        captured.append(analyze_request)
        main._set_job(job_id, status="completed", decision="DRY_RUN", result={})

    with (
        patch.object(main, "_job_store_instance", InMemoryJobStore()),
        patch.object(main, "_compose_analysis_user_context", return_value={}),
        patch.object(main, "_run_job", side_effect=fake_run_job),
        patch.object(main, "_get_reverse_stock_map", return_value={"600519.SH": "贵州茅台"}),
    ):
        # 显式 medium，query 即使写「短中都看看」，resolved 必须保持 medium，source 保持 explicit
        resp = api_client.post(
            "/v1/analyze",
            json={
                "symbol": "600519.SH",
                "horizons": ["medium"],
                "query": "分析 600519.SH 短中都看看，短线和中线机会",
                "dry_run": True,
            },
        )
        assert resp.status_code == 200
        assert len(captured) == 1
        req = captured[0]
        assert req.horizons == ["medium"]
        assert req.horizons_explicit is True
        assert req.horizons_resolution_source == RESOLUTION_SOURCE_EXPLICIT


def test_analyze_explicit_dual_horizons(api_client):
    captured = []

    async def fake_run_job(job_id, analyze_request, *_args, **_kwargs):
        captured.append(analyze_request)
        main._set_job(job_id, status="completed", decision="DRY_RUN", result={})

    with (
        patch.object(main, "_job_store_instance", InMemoryJobStore()),
        patch.object(main, "_compose_analysis_user_context", return_value={}),
        patch.object(main, "_run_job", side_effect=fake_run_job),
        patch.object(main, "_get_reverse_stock_map", return_value={"600519.SH": "贵州茅台"}),
    ):
        resp = api_client.post(
            "/v1/analyze",
            json={
                "symbol": "600519.SH",
                "horizons": ["short", "medium"],
                "dry_run": True,
            },
        )
        assert resp.status_code == 200
        assert len(captured) == 1
        req = captured[0]
        assert req.horizons == ["short", "medium"]
        assert req.horizons_explicit is True
        assert req.horizons_resolution_source == RESOLUTION_SOURCE_EXPLICIT


def test_analyze_invalid_horizons_rejected(api_client):
    with patch.object(main, "_get_reverse_stock_map", return_value={"600519.SH": "贵州茅台"}):
        # 显式 null
        resp = api_client.post("/v1/analyze", json={"symbol": "600519.SH", "horizons": None})
        assert resp.status_code == 422

        # 显式空列表
        resp = api_client.post("/v1/analyze", json={"symbol": "600519.SH", "horizons": []})
        assert resp.status_code == 422

        # 非法值
        resp = api_client.post("/v1/analyze", json={"symbol": "600519.SH", "horizons": ["bogus"]})
        assert resp.status_code == 422


# ── 2. Chat 接口建任务保持 unprovided / default ────────────────────────────────

@pytest.mark.parametrize("stream", [False, True])
def test_chat_creates_analyze_request_as_unprovided(stream):
    captured = []
    user = MagicMock(id="user-1")
    request = main.ChatCompletionRequest(
        messages=[{"role": "user", "content": "分析 600519.SH 短线和中线都看看"}],
        stream=stream,
        dry_run=True,
    )

    async def fake_run_job(job_id, analyze_request, *_args, **_kwargs):
        captured.append(analyze_request)
        main._set_job(job_id, status="completed", decision="DRY_RUN", result={})
        main._emit_job_event(job_id, "job.completed", {"job_id": job_id, "result": {}})

    # LLM extraction extracted dual horizons from query
    extraction = ("600519.SH", "2026-07-31", ["short", "medium"], [], [], {})

    async def run():
        with (
            patch.object(main, "_build_runtime_config", return_value={}),
            patch.object(main, "_compose_analysis_user_context", return_value={}),
            patch.object(main, "_job_store_instance", InMemoryJobStore()),
            patch.object(main, "_ai_extract_symbol_and_date", return_value=extraction),
            patch.object(main, "_ai_extract_symbol_and_date_streaming", return_value=extraction),
            patch.object(main, "_run_job", side_effect=fake_run_job),
        ):
            response = await main.chat_completions(request, current_user=user)
            if stream:
                _body = "".join([chunk async for chunk in response.body_iterator])

    asyncio.run(run())
    assert captured
    req = captured[0]
    assert req.horizons == ["short"]
    assert req.horizons_explicit is False
    assert req.horizons_resolution_source == RESOLUTION_SOURCE_DEFAULT
    assert req.user_intent["horizons"] == ["short"]


# ── 3. 定时请求构造 _build_scheduled_analyze_request ──────────────────────────

def test_build_scheduled_analyze_request_short(test_db):
    req = main._build_scheduled_analyze_request(
        db=test_db,
        user_id="user1",
        symbol="600519.SH",
        horizon="short",
        trade_date="2026-09-07",
    )
    assert req.horizons == ["short"]
    assert req.horizons_explicit is True
    assert req.horizons_resolution_source == RESOLUTION_SOURCE_EXPLICIT
    assert req.user_intent["horizons"] == ["short"]


def test_build_scheduled_analyze_request_medium(test_db):
    req = main._build_scheduled_analyze_request(
        db=test_db,
        user_id="user1",
        symbol="600519.SH",
        horizon="medium",
        trade_date="2026-09-07",
    )
    assert req.horizons == ["medium"]
    assert req.horizons_explicit is True
    assert req.horizons_resolution_source == RESOLUTION_SOURCE_EXPLICIT
    assert req.user_intent["horizons"] == ["medium"]


def test_build_scheduled_analyze_request_query_isolation(test_db):
    # 即使用户意图或 query 带有双档字样，定时构造必须与 query 隔离，不能把已选档扩成双档
    req = main._build_scheduled_analyze_request(
        db=test_db,
        user_id="user1",
        symbol="600519.SH",
        horizon="medium",
        trade_date="2026-09-07",
        query="定时分析 600519.SH 短线和中线都看看",
    )
    assert req.horizons == ["medium"]
    assert req.horizons_explicit is True
    assert req.horizons_resolution_source == RESOLUTION_SOURCE_EXPLICIT
    assert req.user_intent["horizons"] == ["medium"]


def test_build_scheduled_analyze_request_illegal_horizon_fails(test_db):
    for illegal in ["invalid", "long", "", None]:
        with pytest.raises(ValueError):
            main._build_scheduled_analyze_request(
                db=test_db,
                user_id="user1",
                symbol="600519.SH",
                horizon=illegal,
                trade_date="2026-09-07",
            )


def test_build_scheduled_analyze_request_dual_horizons_rejected(test_db):
    # 定时层目前没有双档存储，传入双档列表必须显式拒绝，不要丢档或静默成 short
    with pytest.raises(ValueError):
        main._build_scheduled_analyze_request(
            db=test_db,
            user_id="user1",
            symbol="600519.SH",
            horizon=["short", "medium"],
            trade_date="2026-09-07",
        )


# ── 4. 定时服务与接口拒绝双档及非法值 ──────────────────────────────────────────

def test_scheduled_service_validate_horizon():
    assert scheduled_service._validate_horizon("short") == "short"
    assert scheduled_service._validate_horizon("medium") == "medium"
    assert scheduled_service._validate_horizon(["short"]) == "short"
    assert scheduled_service._validate_horizon(["medium"]) == "medium"

    # 双档列表必须显式拒绝 (ValueError)
    with pytest.raises(ValueError):
        scheduled_service._validate_horizon(["short", "medium"])
    with pytest.raises(ValueError):
        scheduled_service._validate_horizon(["medium", "short"])

    # 非法值必须显式拒绝
    for illegal in ["long", "unknown", "", None, [], 123]:
        with pytest.raises(ValueError):
            scheduled_service._validate_horizon(illegal)


def test_scheduled_service_crud_rejects_dual_horizons(test_db):
    # create_scheduled with dual horizons
    with pytest.raises(ValueError):
        scheduled_service.create_scheduled(test_db, "user1", "600519.SH", horizon=["short", "medium"])
    with pytest.raises(ValueError):
        scheduled_service.create_scheduled(test_db, "user1", "600519.SH", horizons=["short", "medium"])

    # create single horizon is allowed
    item = scheduled_service.create_scheduled(test_db, "user1", "600519.SH", horizon="short")
    assert item["horizon"] == "short"

    # update_scheduled with dual horizons
    with pytest.raises(ValueError):
        scheduled_service.update_scheduled(test_db, "user1", item["id"], horizon=["short", "medium"])
    with pytest.raises(ValueError):
        scheduled_service.update_scheduled(test_db, "user1", item["id"], horizons=["short", "medium"])

    # batch_update_scheduled with dual horizons
    with pytest.raises(ValueError):
        scheduled_service.batch_update_scheduled(test_db, "user1", [item["id"]], horizon=["short", "medium"])
    with pytest.raises(ValueError):
        scheduled_service.batch_update_scheduled(test_db, "user1", [item["id"]], horizons=["short", "medium"])

    # ensure_scheduled_for_symbols with dual horizons
    with pytest.raises(ValueError):
        scheduled_service.ensure_scheduled_for_symbols(test_db, "user1", ["300750.SZ"], horizon=["short", "medium"])
    with pytest.raises(ValueError):
        scheduled_service.ensure_scheduled_for_symbols(test_db, "user1", ["300750.SZ"], horizons=["short", "medium"])


def test_scheduled_endpoints_reject_dual_horizons(api_client, test_db):
    # 1. POST /v1/scheduled with dual horizons -> 400
    resp1 = api_client.post("/v1/scheduled", json={"symbol": "600519.SH", "horizons": ["short", "medium"]})
    assert resp1.status_code == 400

    resp2 = api_client.post("/v1/scheduled", json={"symbol": "600519.SH", "horizon": ["short", "medium"]})
    assert resp2.status_code == 400

    # 创建一个正常的 short 任务
    resp_create = api_client.post("/v1/scheduled", json={"symbol": "600519.SH", "horizon": "short"})
    assert resp_create.status_code == 201
    item_id = resp_create.json()["id"]

    # 2. PATCH /v1/scheduled/{item_id} with dual horizons -> 400
    resp_patch1 = api_client.patch(f"/v1/scheduled/{item_id}", json={"horizons": ["short", "medium"]})
    assert resp_patch1.status_code == 400

    resp_patch2 = api_client.patch(f"/v1/scheduled/{item_id}", json={"horizon": ["short", "medium"]})
    assert resp_patch2.status_code == 400

    # 3. PATCH /v1/scheduled/batch with dual horizons -> 400
    resp_batch1 = api_client.patch("/v1/scheduled/batch", json={"item_ids": [item_id], "horizons": ["short", "medium"]})
    assert resp_batch1.status_code == 400

    resp_batch2 = api_client.patch("/v1/scheduled/batch", json={"item_ids": [item_id], "horizon": ["short", "medium"]})
    assert resp_batch2.status_code == 400


def test_scheduled_endpoints_support_single_medium(api_client, test_db):
    # 支持显式 medium
    resp = api_client.post("/v1/scheduled", json={"symbol": "600519.SH", "horizon": "medium"})
    assert resp.status_code == 201
    assert resp.json()["horizon"] == "medium"
    item_id = resp.json()["id"]

    # 支持用 horizons=["short"] 单档更新
    resp_patch = api_client.patch(f"/v1/scheduled/{item_id}", json={"horizons": ["short"]})
    assert resp_patch.status_code == 200
    assert resp_patch.json()["horizon"] == "short"


# ── 5. 手动触发 scheduled trigger 端点契约 ───────────────────────────────────────

def test_scheduled_trigger_preserves_explicit_horizon(api_client, test_db):
    # 创建一个 medium 任务
    item = scheduled_service.create_scheduled(test_db, "test-user-001", "600519.SH", horizon="medium")

    captured = []

    async def fake_run_job(job_id, analyze_request, *_args, **_kwargs):
        captured.append(analyze_request)
        main._set_job(job_id, status="completed", decision="DRY_RUN", result={})

    @contextmanager
    def override_db_ctx():
        yield test_db

    with (
        patch.object(main, "_job_store_instance", InMemoryJobStore()),
        patch.object(main, "_run_job", side_effect=fake_run_job),
        patch.object(main, "_get_reverse_stock_map", return_value={"600519.SH": "贵州茅台"}),
        patch.object(main, "get_db_ctx", override_db_ctx),
    ):
        resp = api_client.post(f"/v1/scheduled/{item['id']}/trigger")
        assert resp.status_code == 200

        # 等待后台任务执行
        for _ in range(50):
            if captured:
                break
            asyncio.run(asyncio.sleep(0.01))

        if captured:
            req = captured[0]
            assert req.horizons == ["medium"]
            assert req.horizons_explicit is True
            assert req.horizons_resolution_source == RESOLUTION_SOURCE_EXPLICIT
            assert req.user_intent["horizons"] == ["medium"]
