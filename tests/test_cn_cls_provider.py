"""Focused tests for the CLS provider evidence chain."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from tradingagents.dataflows.providers.cn_cls_provider import CnClsProvider


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class Session:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def get(self, endpoint, *, params, timeout):
        self.calls.append((endpoint, dict(params), timeout))
        payload = self.payloads.pop(0)
        if isinstance(payload, BaseException):
            raise payload
        return Response(payload)


def item(item_id, ctime, *, content=None, brief="", shareurl=None):
    row = {"id": item_id, "ctime": ctime, "title": f"title-{item_id}", "brief": brief}
    if content is not None:
        row["content"] = content
    if shareurl is not None:
        row["shareurl"] = shareurl
    return row


def test_latest_cache_path_and_normalization(tmp_path: Path):
    session = Session([{"errno": 0, "data": {"roll_data": [item("latest", 1700000000, content="正文", brief="摘要", shareurl="https://example.test/a")]}}])
    provider = CnClsProvider(session=session, snapshot_dir=tmp_path)
    today = datetime.now().astimezone().date().isoformat()
    out = provider.get_global_news(today, limit=1)

    assert session.calls[0][0].endswith("/api/cache")
    assert session.calls[0][1] == {"app": "CailianpressWeb", "name": "telegraph", "os": "web", "sv": "8.7.9"}
    assert "正文" in out
    assert "摘要" not in out
    assert "https://example.test/a" in out
    assert "coverage_complete=false" in out


def test_history_cursor_overlap_dedup_manifest_and_snapshots(tmp_path: Path):
    pages = [
        {"errno": 0, "data": {"roll_data": [item("a", 100), item("b", 100), item("c", 99)]}},
        {"errno": 0, "data": {"roll_data": [item("a", 100), item("b", 100), item("d", 98)]}},
        {"errno": 0, "data": {"roll_data": []}},
    ]
    session = Session(pages)
    provider = CnClsProvider(session=session, snapshot_dir=tmp_path, max_pages=5)
    out = provider.get_global_news("1970-01-01T00:01:40+00:00", limit=20)

    assert [call[1]["last_time"] for call in session.calls] == [101, 100, 99]
    assert all(call[1]["rn"] == 50 for call in session.calls)
    assert all("sign" in call[1] and len(call[1]["sign"]) == 32 for call in session.calls)
    assert "coverage_complete=true" in out
    manifest_paths = list(tmp_path.glob("*_manifest.json"))
    assert len(manifest_paths) == 1
    manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    assert manifest["cursor_sequence"] == [101, 100, 99]
    assert manifest["records_received"] == 6
    assert manifest["unique_ids"] == 4
    assert manifest["duplicate_ids"] == 2
    assert manifest["stop_reason"] == "server_history_boundary"
    assert {"started_at", "finished_at", "analysis_as_of", "pages_requested", "records_received", "unique_ids", "min_ctime", "max_ctime", "duplicate_ids", "request_errors", "cursor_sequence", "coverage_complete", "stop_reason"} <= manifest.keys()
    snapshots = list(tmp_path.glob("*_page_*.json"))
    assert len(snapshots) == 3
    snapshot = json.loads(snapshots[0].read_text(encoding="utf-8"))
    assert snapshot["source"] == "cls"
    assert "sign=%5Bredacted%5D" in snapshot["request_url"]


def test_history_future_record_is_typed_gap(tmp_path: Path):
    session = Session([{"errno": 0, "data": {"roll_data": [item("future", 101)]}}])
    provider = CnClsProvider(session=session, snapshot_dir=tmp_path)
    out = provider.get_global_news("1970-01-01T00:01:40+00:00")
    assert "coverage_complete=false" in out
    assert "future_record" in out


def test_history_request_error_is_typed_gap(tmp_path: Path):
    session = Session([OSError("network down")])
    provider = CnClsProvider(session=session, snapshot_dir=tmp_path, max_retries=0)
    out = provider.get_global_news("1970-01-01T00:01:40+00:00")
    assert "coverage_complete=false" in out
    assert "request_error" in out


def test_earliest_boundary_is_incomplete_typed_gap(tmp_path: Path):
    session = Session([
        {"errno": 0, "data": {"roll_data": [item("old", 100), item("new", 101)]}},
    ])
    provider = CnClsProvider(session=session, snapshot_dir=tmp_path, earliest_ctime=100)
    out = provider.get_global_news("1970-01-01T00:01:41+00:00")
    assert "coverage_complete=false" in out
    assert "earliest_boundary" in out


def test_invalid_record_is_not_silently_skipped(tmp_path: Path):
    session = Session([
        {"errno": 0, "data": {"roll_data": [item("valid", 100), {"id": "missing-time"}]}},
    ])
    provider = CnClsProvider(session=session, snapshot_dir=tmp_path)
    out = provider.get_global_news("1970-01-01T00:01:40+00:00")
    assert "coverage_complete=false" in out
    assert "missing_ctime" in out
    manifest = json.loads(next(tmp_path.glob("*_manifest.json")).read_text(encoding="utf-8"))
    assert manifest["records_received"] == 0


def test_get_news_explicitly_rejects_unsupported_ticker_semantics(tmp_path: Path):
    provider = CnClsProvider(snapshot_dir=tmp_path)
    out = provider.get_news("600519", "1970-01-01", "1970-01-02")
    assert "unsupported" in out
    assert "全局" in out


def test_normalized_timestamp_uses_shanghai_and_no_detail_url(tmp_path: Path):
    provider = CnClsProvider(snapshot_dir=tmp_path)
    normalized = provider._normalize_item(item("a", 0), requested_as_of="x", retrieved_at="y")
    assert normalized["published_at"] == "1970-01-01T08:00:00+08:00"
    assert normalized["source_url"] == ""
    assert "detail" not in json.dumps(normalized)
