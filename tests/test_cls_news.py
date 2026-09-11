"""Focused evidence-chain tests for the internal CLS telegraph client."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from tradingagents.dataflows.cls_news import (
    CLS_HISTORY_ENDPOINT,
    CLS_LATEST_ENDPOINT,
    CN_TZ,
    ClsNewsClient,
    CoverageGap,
    generate_cls_sign,
    normalize_cls_record,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, endpoint, *, params, timeout):
        self.calls.append((endpoint, dict(params), timeout))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return FakeResponse(response)


def row(cls_id, ctime, *, content=None, brief="", shareurl=None):
    value = {"id": cls_id, "ctime": ctime, "title": f"title-{cls_id}", "brief": brief}
    if content is not None:
        value["content"] = content
    if shareurl is not None:
        value["shareurl"] = shareurl
    return value


def test_sign_is_local_and_does_not_require_credentials():
    params = {
        "app": "CailianpressWeb",
        "category": "announcement",
        "last_time": 1700000001,
        "os": "web",
        "refresh_type": 1,
        "rn": 50,
        "sv": "8.7.9",
    }
    assert len(generate_cls_sign(params)) == 32
    assert generate_cls_sign(params) == generate_cls_sign(dict(reversed(list(params.items()))))


def test_normalize_uses_cn_timezone_content_then_brief_and_explicit_link():
    raw = row("a", 0, content="正文", brief="摘要", shareurl="https://example.test/a")
    normalized, gap = normalize_cls_record(
        raw,
        requested_as_of="2026-08-14T23:59:59+08:00",
        retrieved_at="2026-08-14T00:00:00Z",
    )
    assert gap is None
    assert normalized is not None
    assert normalized["content"] == "正文"
    assert normalized["brief"] == "摘要"
    assert normalized["shareurl"] == "https://example.test/a"
    assert normalized["source_url"] == "https://example.test/a"
    assert normalized["published_at"] == "1970-01-01T08:00:00+08:00"
    assert normalized["ctime"] == 0

    fallback, fallback_gap = normalize_cls_record(
        row("b", 1, brief="摘要"),
        requested_as_of="2026-08-14T23:59:59+08:00",
        retrieved_at="2026-08-14T00:00:00Z",
    )
    assert fallback_gap is None
    assert fallback["content"] == "摘要"


def test_latest_uses_cache_path_and_saves_immutable_snapshot(tmp_path: Path):
    session = FakeSession(
        [
            {
                "data": {
                    "roll_data": [row("latest", 1_700_000_000, content="最新")]
                }
            }
        ]
    )
    client = ClsNewsClient(session=session, snapshot_dir=tmp_path, clock=lambda: datetime(2026, 8, 14))
    result = client.fetch_latest(analysis_as_of="2023-11-15T23:59:59+08:00")

    assert session.calls[0][0] == CLS_LATEST_ENDPOINT
    assert result.coverage_complete is False
    assert result.records[0]["ctime"] == 1_700_000_000
    assert result.snapshot_path is not None
    snapshot = json.loads(Path(result.snapshot_path).read_text(encoding="utf-8"))
    assert snapshot["source"] == "cls"
    assert snapshot["cursor"] is None
    assert "sign" not in snapshot["request_url"]


def test_history_uses_min_ctime_plus_one_and_dedupes_same_second(tmp_path: Path):
    # Page 2 overlaps the two records at ctime=100.  The cursor must be 101,
    # and both same-second records must survive final ID-based deduplication.
    page1 = {
        "errno": 0,
        "data": {"roll_data": [row("a", 100), row("b", 100), row("c", 99)]},
    }
    page2 = {
        "errno": 0,
        "data": {"roll_data": [row("a", 100), row("b", 100), row("d", 98)]},
    }
    page3 = {"errno": 0, "data": {"roll_data": []}}
    session = FakeSession([page1, page2, page3])
    client = ClsNewsClient(
        session=session,
        snapshot_dir=tmp_path,
        clock=lambda: datetime(2026, 8, 14),
        max_pages=5,
    )
    result = client.crawl_history(analysis_as_of=100, start_cursor=101)

    assert [call[1]["last_time"] for call in session.calls] == [101, 100, 99]
    assert [call[1]["rn"] for call in session.calls] == [50, 50, 50]
    assert all(call[0] == CLS_HISTORY_ENDPOINT for call in session.calls)
    assert {item["cls_id"] for item in result.records} == {"a", "b", "c", "d"}
    assert result.manifest["records_received"] == 6
    assert result.manifest["unique_ids"] == 4
    assert result.manifest["duplicate_ids"] == 2
    assert result.manifest["cursor_sequence"] == [101, 100, 99]
    assert result.manifest["coverage_complete"] is True
    assert result.manifest["stop_reason"] == "server_history_boundary"
    assert result.gap is None
    assert len(result.manifest["snapshots"]) == 3
    assert Path(result.manifest["manifest_path"]).exists()
    for snapshot in result.manifest["snapshots"]:
        assert Path(snapshot["path"]).exists()
        assert snapshot["source"] if "source" in snapshot else True


def test_history_rejects_future_record_as_typed_gap(tmp_path: Path):
    session = FakeSession(
        [{"errno": 0, "data": {"roll_data": [row("future", 101)]}}]
    )
    client = ClsNewsClient(session=session, snapshot_dir=tmp_path)
    result = client.crawl_history(analysis_as_of=100, start_cursor=102)

    assert result.coverage_complete is False
    assert isinstance(result.gap, CoverageGap)
    assert result.gap.code == "future_record"
    assert result.manifest["coverage_complete"] is False
    assert result.manifest["stop_reason"] == "future_record"
    assert result.manifest["gap"]["code"] == "future_record"


def test_history_marks_request_failure_without_claiming_complete(tmp_path: Path):
    session = FakeSession([OSError("network down")])
    client = ClsNewsClient(session=session, snapshot_dir=tmp_path, max_retries=0)
    result = client.crawl_history(analysis_as_of=100, start_cursor=101)

    assert result.records == []
    assert result.coverage_complete is False
    assert result.manifest["request_errors"]
    assert result.manifest["stop_reason"] == "request_error"
    assert result.gap is not None
    assert result.gap.code == "request_error"


def test_history_manifest_is_machine_auditable(tmp_path: Path):
    session = FakeSession([{"errno": 0, "data": {"roll_data": []}}])
    client = ClsNewsClient(session=session, snapshot_dir=tmp_path)
    result = client.crawl_history(analysis_as_of="2026-08-14")
    manifest = result.manifest

    required = {
        "started_at",
        "finished_at",
        "analysis_as_of",
        "pages_requested",
        "records_received",
        "unique_ids",
        "min_ctime",
        "max_ctime",
        "duplicate_ids",
        "request_errors",
        "cursor_sequence",
        "coverage_complete",
        "stop_reason",
    }
    assert required <= manifest.keys()
    assert manifest["coverage_complete"] is True
    assert manifest["analysis_as_of"].endswith("+08:00")
