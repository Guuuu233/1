"""Internal CLS (财联社) telegraph provider with historical evidence tracking."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import requests

from .base import BaseMarketDataProvider
from ..config import get_config
from ..vendor_result import VendorFail, VendorRefuse

CLS_LATEST_ENDPOINT = "https://www.cls.cn/api/cache"
CLS_HISTORY_ENDPOINT = "https://www.cls.cn/v1/roll/get_roll_list"
CLS_SOURCE = "cls"
CLS_TZ = ZoneInfo("Asia/Shanghai")
CLS_HISTORY_RN = 50


class CnClsProvider(BaseMarketDataProvider):
    """CLS website telegraph source for latest and auditable historical news."""

    def __init__(
        self,
        *,
        session: requests.Session | Any | None = None,
        snapshot_dir: str | os.PathLike[str] | None = None,
        timeout_seconds: float = 15.0,
        max_retries: int = 1,
        max_pages: int = 80,
        earliest_ctime: int | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if max_pages < 1:
            raise ValueError("max_pages must be positive")
        if earliest_ctime is not None and earliest_ctime < 0:
            raise ValueError("earliest_ctime must be non-negative")
        self.session = session or requests.Session()
        configured_results_dir = get_config().get("results_dir", "./results")
        self.snapshot_dir = (
            Path(snapshot_dir)
            if snapshot_dir is not None
            else Path(configured_results_dir) / "cls_snapshots"
        )
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_pages = max_pages
        self.earliest_ctime = earliest_ctime

    @property
    def name(self) -> str:
        return "cn_cls"

    @staticmethod
    def _as_of(value: datetime | date | str | int | float) -> tuple[int, str]:
        if isinstance(value, bool):
            raise ValueError("analysis_as_of must not be boolean")
        if isinstance(value, (int, float)):
            ctime = int(value)
            if ctime < 0:
                raise ValueError("analysis_as_of must be non-negative")
            return ctime, datetime.fromtimestamp(ctime, tz=CLS_TZ).isoformat()
        if isinstance(value, datetime):
            parsed = value if value.tzinfo else value.replace(tzinfo=CLS_TZ)
            parsed = parsed.astimezone(CLS_TZ)
            return int(parsed.timestamp()), parsed.isoformat()
        if isinstance(value, date):
            parsed = datetime.combine(value, time.max, tzinfo=CLS_TZ)
            return int(parsed.timestamp()), parsed.isoformat()
        text = str(value).strip()
        if not text:
            raise ValueError("analysis_as_of must not be empty")
        if text.isdigit():
            return CnClsProvider._as_of(int(text))
        if len(text) == 10:
            return CnClsProvider._as_of(date.fromisoformat(text))
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
        return CnClsProvider._as_of(parsed)

    @staticmethod
    def _text(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return "" if text.lower() in {"none", "nan", "null"} else text

    @classmethod
    def _item_id(cls, item: Mapping[str, Any]) -> str:
        return next((cls._text(item.get(k)) for k in ("cls_id", "id") if cls._text(item.get(k))), "")

    @classmethod
    def _item_ctime(cls, item: Mapping[str, Any]) -> int | None:
        value = item.get("ctime", item.get("create_time", item.get("timestamp")))
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(parsed) or parsed < 0 or parsed > 253402300799:
            return None
        try:
            return int(parsed)
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _sign(params: Mapping[str, Any]) -> str:
        raw = "&".join(f"{key}={params[key]}" for key in sorted(params))
        return hashlib.md5(hashlib.sha1(raw.encode("utf-8")).hexdigest().encode("ascii")).hexdigest()

    @staticmethod
    def _redacted_url(endpoint: str, params: Mapping[str, Any]) -> str:
        safe = {str(k): "[redacted]" if str(k).lower() == "sign" else v for k, v in params.items()}
        parts = urlsplit(endpoint)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(safe), ""))

    @staticmethod
    def _roll_data(payload: Any) -> list[Any] | None:
        if not isinstance(payload, Mapping):
            return None
        for candidate in (payload, payload.get("data")):
            if isinstance(candidate, Mapping) and isinstance(candidate.get("roll_data"), list):
                return candidate["roll_data"]
            if isinstance(candidate, list):
                return candidate
        return None

    def _fetch_json(self, endpoint: str, params: Mapping[str, Any]) -> tuple[Any | None, str | None]:
        error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(endpoint, params=dict(params), timeout=self.timeout_seconds)
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, Mapping) and payload.get("errno") not in (None, 0, "0"):
                    return None, f"CLS errno={payload.get('errno')}"
                return payload, None
            except (requests.RequestException, OSError, ValueError, RuntimeError) as exc:
                error = f"attempt={attempt + 1}: {type(exc).__name__}: {exc}"
        return None, error or "CLS request failed"

    def _fetch_latest(self, *, analysis_ctime: int | None = None, requested_as_of: str = "") -> tuple[list[dict[str, Any]], Any, str | None, str | None]:
        params = {"app": "CailianpressWeb", "name": "telegraph", "os": "web", "sv": "8.7.9"}
        payload, error = self._fetch_json(CLS_LATEST_ENDPOINT, params)
        if error:
            return [], None, self._redacted_url(CLS_LATEST_ENDPOINT, params), error
        rows = self._roll_data(payload)
        if rows is None:
            return [], payload, self._redacted_url(CLS_LATEST_ENDPOINT, params), "latest payload has no roll_data"
        retrieved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        out = []
        for row in rows:
            normalized = self._normalize_item(row, requested_as_of=requested_as_of, retrieved_at=retrieved_at)
            if normalized and (analysis_ctime is None or normalized["ctime"] <= analysis_ctime):
                out.append(normalized)
        return out, payload, self._redacted_url(CLS_LATEST_ENDPOINT, params), None

    def _fetch_history_page(self, cursor: int, *, category: str = "announcement") -> tuple[list[Any] | None, Any, dict[str, Any], str | None]:
        params = {"app": "CailianpressWeb", "category": category, "last_time": int(cursor), "os": "web", "refresh_type": 1, "rn": CLS_HISTORY_RN, "sv": "8.7.9"}
        params["sign"] = self._sign(params)
        payload, error = self._fetch_json(CLS_HISTORY_ENDPOINT, params)
        return self._roll_data(payload) if payload is not None else None, payload, params, error

    def _normalize_item(self, item: Mapping[str, Any], *, requested_as_of: str, retrieved_at: str) -> dict[str, Any] | None:
        if not isinstance(item, Mapping):
            return None
        cls_id = self._item_id(item)
        ctime = self._item_ctime(item)
        if not cls_id or ctime is None:
            return None
        try:
            published_at = datetime.fromtimestamp(ctime, tz=ZoneInfo("Asia/Shanghai")).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
        content = self._text(item.get("content"))
        brief = self._text(item.get("brief"))
        source_url = self._text(item.get("source_url")) or self._text(item.get("shareurl"))
        return {
            "cls_id": cls_id,
            "ctime": ctime,
            "published_at": published_at,
            "title": self._text(item.get("title")),
            "brief": brief,
            "content": content or brief,
            "level": item.get("level"),
            "subjects": item.get("subjects", item.get("subject")),
            "stock_list": item.get("stock_list", item.get("stockList")),
            "source_url": source_url,
            "shareurl": self._text(item.get("shareurl")),
            "source": CLS_SOURCE,
            "requested_as_of": requested_as_of,
            "retrieved_at": retrieved_at,
        }

    @staticmethod
    def _dedupe_items(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
        seen: dict[str, dict[str, Any]] = {}
        duplicates: set[str] = set()
        for item in items:
            item_id = str(item["cls_id"])
            if item_id in seen:
                duplicates.add(item_id)
            else:
                seen[item_id] = item
        return list(seen.values()), sorted(duplicates)

    def _write_page_snapshot(self, *, run_id: str, page: int, cursor: int | None, payload: Any, request_url: str, retrieved_at: str) -> str:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.snapshot_dir / f"cls_{run_id}_page_{page:04d}.json"
        if path.exists():
            path = self.snapshot_dir / f"cls_{run_id}_page_{page:04d}_{uuid.uuid4().hex}.json"
        body = {"source": CLS_SOURCE, "request_url": request_url, "retrieved_at": retrieved_at, "page": page, "cursor": cursor, "payload": payload}
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(body, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        return str(path)

    def _write_manifest(self, *, run_id: str, manifest: dict[str, Any]) -> str:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.snapshot_dir / f"cls_{run_id}_manifest.json"
        if path.exists():
            path = self.snapshot_dir / f"cls_{run_id}_manifest_{uuid.uuid4().hex}.json"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        return str(path)

    @staticmethod
    def _build_typed_gap(code: str, message: str, **details: Any) -> dict[str, Any]:
        return {"type": "coverage_gap", "code": code, "message": message, "details": details}

    def get_global_news(self, curr_date: str, look_back_days: int = 7, limit: int = 50) -> str | VendorFail:
        """Return CLS latest for today, or auditable historical telegraphs."""
        analysis_ctime, analysis_iso = self._as_of(curr_date)
        is_today = datetime.fromtimestamp(analysis_ctime, tz=CLS_TZ).date() == datetime.now(CLS_TZ).date()
        if is_today:
            latest, payload, request_url, error = self._fetch_latest(
                analysis_ctime=analysis_ctime,
                requested_as_of=analysis_iso,
            )
            retrieved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            if error:
                return VendorFail(f"财联社最新路径失败：{error}")
            snapshot = self._write_page_snapshot(
                run_id=uuid.uuid4().hex,
                page=1,
                cursor=None,
                payload=payload,
                request_url=request_url or CLS_LATEST_ENDPOINT,
                retrieved_at=retrieved_at,
            )
            return self._format(latest[:limit], analysis_iso, False, None, snapshot)

        cursor = analysis_ctime + 1
        lower_bound = analysis_ctime - max(0, int(look_back_days)) * 86400
        started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        run_id = uuid.uuid4().hex
        pages: list[dict[str, Any]] = []
        all_items: list[dict[str, Any]] = []
        errors: list[str] = []
        cursors: list[int] = []
        duplicate_ids: list[str] = []
        accepted_count = 0
        invalid_count = 0
        valid_count = 0
        records_received = 0
        received_ctimes: list[int] = []
        previous_min: int | None = None
        coverage_complete = False
        stop_reason = "not_started"
        gap: dict[str, Any] | None = None
        for page in range(1, self.max_pages + 1):
            cursors.append(cursor)
            rows, raw_payload, params, page_error = self._fetch_history_page(cursor)
            request_url = self._redacted_url(CLS_HISTORY_ENDPOINT, params)
            retrieved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            if page_error:
                errors.append(page_error)
                gap = self._build_typed_gap("request_error", "CLS historical request failed", page=page, cursor=cursor)
                stop_reason = "request_error"
                break
            snapshot_path = self._write_page_snapshot(run_id=run_id, page=page, cursor=cursor, payload=raw_payload, request_url=request_url, retrieved_at=retrieved_at)
            pages.append({"page": page, "cursor": cursor, "path": snapshot_path, "source": CLS_SOURCE, "request_url": request_url, "retrieved_at": retrieved_at})
            if rows is None:
                gap = self._build_typed_gap("invalid_payload", "CLS historical payload has no roll_data", page=page)
                stop_reason = "invalid_payload"
                break
            records_received += len(rows)
            if not rows:
                coverage_complete = True
                stop_reason = "server_history_boundary"
                break
            page_items: list[dict[str, Any]] = []
            raw_page_ctimes: list[int] = []
            first_gap: dict[str, Any] | None = None
            for row_index, row in enumerate(rows):
                item = self._normalize_item(row, requested_as_of=analysis_iso, retrieved_at=retrieved_at)
                if item is None:
                    invalid_count += 1
                    if not isinstance(row, Mapping):
                        gap_code = "invalid_record"
                        gap_message = "historical page contains a non-object record"
                    elif not self._item_id(row):
                        gap_code = "missing_id"
                        gap_message = "historical record has neither cls_id nor id"
                    else:
                        gap_code = "missing_ctime"
                        gap_message = "historical record has no valid ctime"
                    if first_gap is None:
                        first_gap = self._build_typed_gap(
                            gap_code,
                            gap_message,
                            page=page,
                            row_index=row_index,
                        )
                    continue
                valid_count += 1
                received_ctimes.append(item["ctime"])
                raw_page_ctimes.append(item["ctime"])
                if item["ctime"] > analysis_ctime:
                    if first_gap is None:
                        first_gap = self._build_typed_gap(
                            "future_record",
                            "record exceeds analysis_as_of",
                            page=page,
                            cls_id=item["cls_id"],
                            ctime=item["ctime"],
                        )
                    continue
                if item["ctime"] < lower_bound:
                    continue
                if self.earliest_ctime is not None and item["ctime"] < self.earliest_ctime:
                    continue
                page_items.append(item)
                accepted_count += 1
            all_items.extend(page_items)
            if first_gap is not None:
                gap = first_gap
                stop_reason = str(gap["code"])
                break
            if not raw_page_ctimes:
                gap = self._build_typed_gap("missing_ctime", "page has no valid ctime")
                stop_reason = "missing_ctime"
                break
            min_ctime = min(raw_page_ctimes)
            if previous_min is not None and min_ctime >= previous_min:
                gap = self._build_typed_gap("cursor_not_decreasing", "historical cursor did not move older", previous_min=previous_min, min_ctime=min_ctime)
                stop_reason = "cursor_not_decreasing"
                break
            if min_ctime >= cursor:
                gap = self._build_typed_gap("cursor_not_decreasing", "page minimum ctime is not older than cursor", cursor=cursor, min_ctime=min_ctime)
                stop_reason = "cursor_not_decreasing"
                break
            if self.earliest_ctime is not None and min_ctime <= self.earliest_ctime:
                gap = self._build_typed_gap(
                    "earliest_boundary",
                    "configured earliest_ctime reached; older history is not covered",
                    earliest_ctime=self.earliest_ctime,
                    min_ctime=min_ctime,
                )
                stop_reason = "earliest_boundary"
                break
            previous_min = min_ctime
            cursor = min_ctime + 1
        else:
            gap = self._build_typed_gap("max_pages", "max_pages reached before complete coverage", max_pages=self.max_pages)
            stop_reason = "max_pages"

        unique_items, duplicate_ids = self._dedupe_items(all_items)
        if not coverage_complete and gap is None:
            gap = self._build_typed_gap("incomplete_coverage", "historical coverage was not proven")
            stop_reason = "incomplete_coverage"
        all_ctimes = [int(item["ctime"]) for item in all_items]
        manifest = {
            "source": CLS_SOURCE,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "analysis_as_of": analysis_iso,
            "pages_requested": len(cursors),
            "records_received": records_received,
            "accepted_records": accepted_count,
            "invalid_records": invalid_count,
            "unique_ids": len(unique_items),
            "min_ctime": min(all_ctimes) if all_ctimes else None,
            "max_ctime": max(all_ctimes) if all_ctimes else None,
            "duplicate_ids": len(duplicate_ids),
            "duplicate_id_values": duplicate_ids,
            "request_errors": errors,
            "cursor_sequence": cursors,
            "coverage_complete": coverage_complete,
            "stop_reason": stop_reason,
            "snapshots": pages,
            "gap": gap,
        }
        manifest_path = self._write_manifest(run_id=run_id, manifest=manifest)
        if not coverage_complete:
            return VendorFail(
                f"CLS 历史覆盖不完整（{stop_reason}）；manifest={manifest_path}; "
                "不得将部分集合视为完整历史。"
            )
        return self._format(unique_items[:limit], analysis_iso, coverage_complete, gap, manifest_path)

    @staticmethod
    def _format(items: list[dict[str, Any]], analysis_as_of: str, coverage_complete: bool, gap: dict[str, Any] | None, artifact_path: str | None) -> str:
        output = [f"## 财联社电报（来源：财联社；analysis_as_of={analysis_as_of}；coverage_complete={str(coverage_complete).lower()}）"]
        if gap:
            output.append(f"数据缺口（{gap['code']}）：{gap['message']}")
        if artifact_path:
            output.append(f"evidence_manifest: {artifact_path}")
        for item in items:
            output.append(f"### {item['title'] or '无标题'} [发布时间：{item['published_at']}]")
            output.append(item["content"] or "正文缺口：接口未返回 content 或 brief。")
            url = item.get("source_url") or item.get("shareurl")
            if url:
                output.append(f"source_url: {url}")
        return "\n\n".join(output) + "\n"

    # Non-news methods remain deliberately unsupported; the registry only routes
    # get_global_news to this provider.
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> str:
        raise NotImplementedError("cn_cls only provides news")

    def get_indicators(self, symbol: str, indicator: str, curr_date: str, look_back_days: int) -> str:
        raise NotImplementedError("cn_cls only provides news")

    def get_fundamentals(self, ticker: str, curr_date: str = None) -> str:
        raise NotImplementedError("cn_cls only provides news")

    def get_balance_sheet(self, ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
        raise NotImplementedError("cn_cls only provides news")

    def get_cashflow(self, ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
        raise NotImplementedError("cn_cls only provides news")

    def get_income_statement(self, ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
        raise NotImplementedError("cn_cls only provides news")

    def get_news(self, ticker: str, start_date: str, end_date: str) -> VendorRefuse:
        return VendorRefuse(
            "cn_cls 仅提供全局财联社电报，不支持按 ticker 查询个股新闻（unsupported）。",
            allow_peers=("cn_akshare", "cn_investoday"),
        )

    def get_insider_transactions(self, symbol: str, curr_date: str = None) -> str:
        raise NotImplementedError("cn_cls only provides news")


__all__ = ["CnClsProvider", "CLS_HISTORY_ENDPOINT", "CLS_LATEST_ENDPOINT"]
