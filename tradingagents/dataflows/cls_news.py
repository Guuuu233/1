"""Internal CLS (财联社) telegraph client with historical evidence tracking.

The CLS endpoints are website endpoints intended for internal research use.  This
module deliberately keeps the source client independent from user/provider
configuration so historical crawls can be audited without changing the trading
provider chain.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import requests

CLS_LATEST_ENDPOINT = "https://www.cls.cn/api/cache"
CLS_HISTORY_ENDPOINT = "https://www.cls.cn/v1/roll/get_roll_list"
CLS_SOURCE = "cls"
CLS_APP = "CailianpressWeb"
CLS_OS = "web"
CLS_VERSION = "8.7.9"
CLS_HISTORY_RN = 50
CN_TZ = ZoneInfo("Asia/Shanghai")
UTC = timezone.utc


class ClsNewsError(RuntimeError):
    """Base error for a CLS request or evidence-chain failure."""


@dataclass(frozen=True)
class CoverageGap:
    """Typed explanation for why a historical crawl is incomplete."""

    code: str
    message: str
    details: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.details:
            result["details"] = dict(self.details)
        return result


@dataclass
class ClsLatestBatch:
    """Latest cache response; it is not a claim of complete history."""

    records: list[dict[str, Any]]
    retrieved_at: str
    request_url: str
    snapshot_path: str | None = None
    gap: CoverageGap | None = None

    @property
    def coverage_complete(self) -> bool:
        return False


@dataclass
class ClsHistoryResult:
    """Normalized historical records plus the saved audit manifest."""

    records: list[dict[str, Any]]
    manifest: dict[str, Any]
    gap: CoverageGap | None = None

    @property
    def coverage_complete(self) -> bool:
        return bool(self.manifest.get("coverage_complete", False))

    def to_prompt(self) -> str:
        """Render records without inventing links or hiding an incomplete crawl."""
        return format_cls_records(
            self.records,
            analysis_as_of=self.manifest.get("analysis_as_of"),
            coverage_complete=self.coverage_complete,
            gap=self.gap,
        )


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _cn_iso(value: datetime) -> str:
    return value.astimezone(CN_TZ).isoformat()


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _parse_analysis_as_of(value: datetime | date | str | int | float) -> tuple[int, str]:
    """Parse an as-of value without consulting the host machine timezone."""
    if isinstance(value, bool):
        raise ValueError("analysis_as_of must not be boolean")
    if isinstance(value, (int, float)):
        ctime = int(value)
        if ctime < 0:
            raise ValueError("analysis_as_of Unix time must be non-negative")
        return ctime, _cn_iso(datetime.fromtimestamp(ctime, tz=CN_TZ))
    if isinstance(value, datetime):
        parsed = value if value.tzinfo is not None else value.replace(tzinfo=CN_TZ)
        parsed = parsed.astimezone(CN_TZ)
        return int(parsed.timestamp()), parsed.isoformat()
    if isinstance(value, date):
        parsed = datetime.combine(value, time.max, tzinfo=CN_TZ)
        return int(parsed.timestamp()), parsed.isoformat()
    text = str(value).strip()
    if not text:
        raise ValueError("analysis_as_of must not be empty")
    if text.isdigit():
        return _parse_analysis_as_of(int(text))
    if len(text) == 10:
        parsed_date = date.fromisoformat(text)
        return _parse_analysis_as_of(parsed_date)
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    parsed_dt = datetime.fromisoformat(normalized)
    return _parse_analysis_as_of(parsed_dt)


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "nan", "null"} else text


def _first_text(item: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = _text(item.get(key))
        if value:
            return value
    return ""


def _record_id(item: Mapping[str, Any]) -> str:
    for key in ("cls_id", "id"):
        value = item.get(key)
        if value is not None and _text(value):
            return _text(value)
    return ""


def _record_ctime(item: Mapping[str, Any]) -> int | None:
    value = item.get("ctime")
    if value is None:
        value = item.get("create_time", item.get("timestamp"))
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return int(parsed)


def normalize_cls_record(
    raw: Mapping[str, Any],
    *,
    requested_as_of: str,
    retrieved_at: str,
    source: str = CLS_SOURCE,
) -> tuple[dict[str, Any] | None, CoverageGap | None]:
    """Normalize one raw CLS row and preserve explicit source fields only."""
    if not isinstance(raw, Mapping):
        return None, CoverageGap("invalid_record", "CLS row is not an object")

    cls_id = _record_id(raw)
    ctime = _record_ctime(raw)
    if not cls_id:
        return None, CoverageGap("missing_id", "CLS row has neither cls_id nor id")
    if ctime is None:
        return None, CoverageGap("missing_ctime", f"CLS row {cls_id} has no valid ctime")

    raw_content = _first_text(raw, ("content",))
    brief = _first_text(raw, ("brief",))
    body = raw_content or brief
    source_url = _first_text(raw, ("source_url", "shareurl", "detail_url"))
    shareurl = _text(raw.get("shareurl"))
    title = _first_text(raw, ("title", "brief_title"))

    record = {
        "cls_id": cls_id,
        "ctime": ctime,
        "published_at": _cn_iso(datetime.fromtimestamp(ctime, tz=CN_TZ)),
        "title": title,
        "brief": brief,
        # ``content`` is the selected body for consumers: content first, brief
        # second.  ``body`` makes that precedence explicit without losing brief.
        "content": body,
        "body": body,
        "level": raw.get("level"),
        "subjects": raw.get("subjects", raw.get("subject")),
        "stock_list": raw.get("stock_list", raw.get("stockList")),
        "source_url": source_url,
        "shareurl": shareurl,
        "source": source,
        "requested_as_of": requested_as_of,
        "retrieved_at": retrieved_at,
        "content_gap": not bool(body),
    }
    return record, None


def generate_cls_sign(params: Mapping[str, Any]) -> str:
    """Generate the website endpoint signature without any credential."""
    raw = "&".join(f"{key}={params[key]}" for key in sorted(params))
    sha1_hex = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return hashlib.md5(sha1_hex.encode("ascii")).hexdigest()


def _extract_roll_data(payload: Any) -> list[Any] | None:
    if not isinstance(payload, Mapping):
        return None
    candidates: list[Any] = [payload]
    data = payload.get("data")
    candidates.append(data)
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            rows = candidate.get("roll_data")
            if isinstance(rows, list):
                return rows
        elif isinstance(candidate, list):
            return candidate
    return None


def _redacted_url(endpoint: str, params: Mapping[str, Any]) -> str:
    safe_params = {
        str(key): "[redacted]" if str(key).lower() == "sign" else value
        for key, value in params.items()
    }
    parts = urlsplit(endpoint)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(safe_params), "")
    )


def format_cls_records(
    records: Iterable[Mapping[str, Any]],
    *,
    analysis_as_of: str | None = None,
    coverage_complete: bool = True,
    gap: CoverageGap | None = None,
) -> str:
    """Render normalized records for prompts while exposing coverage status."""
    rows = list(records)
    latest = max((str(row.get("published_at", "")) for row in rows), default="未知")
    header = (
        "## 财联社电报（来源：财联社；"
        f"最新发布时间：{latest}；coverage_complete={str(coverage_complete).lower()}"
    )
    if analysis_as_of:
        header += f"；analysis_as_of={analysis_as_of}"
    header += "）\n"
    if gap:
        header += f"数据缺口（{gap.code}）：{gap.message}\n"
    if not rows:
        return header + "暂无可验证电报。\n"

    output = [header]
    for row in rows:
        title = _text(row.get("title")) or "无标题"
        published = _text(row.get("published_at")) or "未知"
        output.append(f"### {title} [发布时间：{published}]")
        body = _text(row.get("content")) or _text(row.get("brief"))
        output.append(body or "正文缺口：接口未返回 content 或 brief。")
        source_url = _text(row.get("source_url")) or _text(row.get("shareurl"))
        if source_url:
            output.append(f"source_url: {source_url}")
        output.append("")
    return "\n".join(output).rstrip() + "\n"


class ClsNewsClient:
    """CLS latest/history client with immutable raw snapshots and manifests."""

    def __init__(
        self,
        *,
        session: requests.Session | Any | None = None,
        snapshot_dir: str | os.PathLike[str] = "results/cls_snapshots",
        timeout_seconds: float = 15.0,
        max_retries: int = 1,
        max_pages: int = 80,
        earliest_ctime: int | None = None,
        clock: Callable[[], datetime] = _now_utc,
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
        self.snapshot_dir = Path(snapshot_dir)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_pages = max_pages
        self.earliest_ctime = earliest_ctime
        self.clock = clock

    def _retrieved_at(self) -> str:
        return _utc_iso(self.clock())

    def _unique_path(self, prefix: str, run_id: str, suffix: str = ".json") -> Path:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.snapshot_dir / f"{prefix}_{run_id}{suffix}"
        if not path.exists():
            return path
        return self.snapshot_dir / f"{prefix}_{run_id}_{uuid.uuid4().hex}{suffix}"

    def _write_immutable_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(path, flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.write("\n")
        except BaseException:
            try:
                path.unlink()
            except OSError:
                pass
            raise

    def _get_json(
        self, endpoint: str, params: Mapping[str, Any]
    ) -> tuple[Any | None, str | None]:
        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    endpoint,
                    params=dict(params),
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, Mapping):
                    errno = payload.get("errno")
                    if errno not in (None, 0, "0"):
                        return None, f"CLS endpoint errno={errno}"
                return payload, None
            except (requests.RequestException, OSError, ValueError) as exc:
                last_error = f"attempt={attempt + 1}: {type(exc).__name__}: {exc}"
        return None, last_error or "unknown CLS request error"

    @staticmethod
    def _latest_params() -> dict[str, Any]:
        return {
            "app": CLS_APP,
            "name": "telegraph",
            "os": CLS_OS,
            "sv": CLS_VERSION,
        }

    @staticmethod
    def _history_params(cursor: int, *, category: str = "announcement") -> dict[str, Any]:
        params: dict[str, Any] = {
            "app": CLS_APP,
            "category": category,
            "last_time": int(cursor),
            "os": CLS_OS,
            "refresh_type": 1,
            "rn": CLS_HISTORY_RN,
            "sv": CLS_VERSION,
        }
        params["sign"] = generate_cls_sign(params)
        return params

    def fetch_latest(
        self,
        *,
        analysis_as_of: datetime | date | str | int | float | None = None,
        save_snapshot: bool = True,
    ) -> ClsLatestBatch:
        """Fetch the fast cache path as a first/latest batch only."""
        retrieved_at = self._retrieved_at()
        params = self._latest_params()
        request_url = _redacted_url(CLS_LATEST_ENDPOINT, params)
        payload, error = self._get_json(CLS_LATEST_ENDPOINT, params)
        if error:
            return ClsLatestBatch(
                records=[],
                retrieved_at=retrieved_at,
                request_url=request_url,
                gap=CoverageGap("request_error", error),
            )
        rows = _extract_roll_data(payload)
        if rows is None:
            return ClsLatestBatch(
                records=[],
                retrieved_at=retrieved_at,
                request_url=request_url,
                gap=CoverageGap("invalid_payload", "CLS latest response has no roll_data"),
            )

        requested_as_of = ""
        analysis_ctime: int | None = None
        if analysis_as_of is not None:
            analysis_ctime, requested_as_of = _parse_analysis_as_of(analysis_as_of)
        records: list[dict[str, Any]] = []
        for raw in rows:
            record, gap = normalize_cls_record(
                raw,
                requested_as_of=requested_as_of,
                retrieved_at=retrieved_at,
            )
            if record is None:
                continue
            if analysis_ctime is not None and record["ctime"] > analysis_ctime:
                continue
            records.append(record)

        snapshot_path: str | None = None
        if save_snapshot:
            run_id = uuid.uuid4().hex
            path = self._unique_path("cls_latest", run_id)
            self._write_immutable_json(
                path,
                {
                    "source": CLS_SOURCE,
                    "request_url": request_url,
                    "retrieved_at": retrieved_at,
                    "cursor": None,
                    "page": 1,
                    "payload": payload,
                },
            )
            snapshot_path = str(path)
        return ClsLatestBatch(
            records=records,
            retrieved_at=retrieved_at,
            request_url=request_url,
            snapshot_path=snapshot_path,
        )

    def crawl_history(
        self,
        analysis_as_of: datetime | date | str | int | float,
        *,
        start_cursor: int | None = None,
        category: str = "announcement",
    ) -> ClsHistoryResult:
        """Fetch history with ``min_ctime + 1`` overlap and an audit manifest."""
        analysis_ctime, analysis_iso = _parse_analysis_as_of(analysis_as_of)
        started_at = _utc_iso(self.clock())
        run_id = uuid.uuid4().hex
        manifest_path = self._unique_path("cls_manifest", run_id)
        cursor = int(start_cursor) if start_cursor is not None else analysis_ctime + 1
        if cursor < 0:
            raise ValueError("start_cursor must be non-negative")

        records_by_id: dict[str, dict[str, Any]] = {}
        cursor_sequence: list[int] = []
        snapshots: list[dict[str, Any]] = []
        request_errors: list[str] = []
        duplicate_id_values: set[str] = set()
        all_ctimes: list[int] = []
        content_gaps = 0
        pages_requested = 0
        records_received = 0
        previous_min_ctime: int | None = None
        coverage_complete = False
        stop_reason = "not_started"
        gap: CoverageGap | None = None

        while pages_requested < self.max_pages:
            pages_requested += 1
            page = pages_requested
            cursor_sequence.append(cursor)
            params = self._history_params(cursor, category=category)
            request_url = _redacted_url(CLS_HISTORY_ENDPOINT, params)
            retrieved_at = self._retrieved_at()
            payload, error = self._get_json(CLS_HISTORY_ENDPOINT, params)
            if error:
                request_errors.append(error)
                gap = CoverageGap(
                    "request_error",
                    "CLS historical page request failed",
                    {"page": page, "cursor": cursor},
                )
                stop_reason = "request_error"
                break

            snapshot_path = self._unique_path("cls_page", f"{run_id}_{page:04d}")
            self._write_immutable_json(
                snapshot_path,
                {
                    "source": CLS_SOURCE,
                    "request_url": request_url,
                    "retrieved_at": retrieved_at,
                    "cursor": cursor,
                    "page": page,
                    "payload": payload,
                },
            )
            snapshots.append(
                {
                    "page": page,
                    "cursor": cursor,
                    "path": str(snapshot_path),
                    "request_url": request_url,
                    "retrieved_at": retrieved_at,
                }
            )

            rows = _extract_roll_data(payload)
            if rows is None:
                gap = CoverageGap(
                    "invalid_payload",
                    "CLS historical response has no roll_data",
                    {"page": page, "cursor": cursor},
                )
                stop_reason = "invalid_payload"
                break
            records_received += len(rows)
            if not rows:
                coverage_complete = True
                stop_reason = "server_history_boundary"
                break

            page_ctimes: list[int] = []
            for raw in rows:
                record, row_gap = normalize_cls_record(
                    raw,
                    requested_as_of=analysis_iso,
                    retrieved_at=retrieved_at,
                )
                if record is None:
                    gap = row_gap or CoverageGap("invalid_record", "CLS row invalid")
                    stop_reason = gap.code
                    break
                ctime = int(record["ctime"])
                page_ctimes.append(ctime)
                all_ctimes.append(ctime)
                if ctime > analysis_ctime:
                    gap = CoverageGap(
                        "future_record",
                        "CLS returned a record newer than analysis_as_of",
                        {"page": page, "cls_id": record["cls_id"], "ctime": ctime},
                    )
                    stop_reason = "future_record"
                    break
                if self.earliest_ctime is not None and ctime < self.earliest_ctime:
                    continue
                if record.get("content_gap"):
                    content_gaps += 1
                cls_id = str(record["cls_id"])
                if cls_id in records_by_id:
                    duplicate_id_values.add(cls_id)
                else:
                    records_by_id[cls_id] = record
            if gap is not None:
                break

            if not page_ctimes:
                gap = CoverageGap(
                    "no_valid_ctime",
                    "CLS page contains no valid ctime rows",
                    {"page": page, "cursor": cursor},
                )
                stop_reason = "no_valid_ctime"
                break
            min_ctime = min(page_ctimes)
            if previous_min_ctime is not None and min_ctime >= previous_min_ctime:
                gap = CoverageGap(
                    "cursor_not_decreasing",
                    "CLS historical cursor did not move toward older records",
                    {
                        "page": page,
                        "previous_min_ctime": previous_min_ctime,
                        "min_ctime": min_ctime,
                    },
                )
                stop_reason = "cursor_not_decreasing"
                break
            if min_ctime >= cursor:
                gap = CoverageGap(
                    "cursor_not_decreasing",
                    "CLS page minimum ctime is not older than requested cursor",
                    {"page": page, "cursor": cursor, "min_ctime": min_ctime},
                )
                stop_reason = "cursor_not_decreasing"
                break

            if self.earliest_ctime is not None and min_ctime <= self.earliest_ctime:
                coverage_complete = True
                stop_reason = "earliest_ctime_reached"
                break

            previous_min_ctime = min_ctime
            cursor = min_ctime + 1

        else:
            gap = CoverageGap(
                "max_pages",
                "CLS historical crawl reached max_pages before coverage was proven",
                {"max_pages": self.max_pages},
            )
            stop_reason = "max_pages"

        if not coverage_complete and gap is None:
            gap = CoverageGap(
                "incomplete_coverage",
                "CLS historical coverage could not be proven",
            )
            stop_reason = stop_reason if stop_reason != "not_started" else "incomplete_coverage"

        finished_at = _utc_iso(self.clock())
        manifest: dict[str, Any] = {
            "source": CLS_SOURCE,
            "started_at": started_at,
            "finished_at": finished_at,
            "analysis_as_of": analysis_iso,
            "pages_requested": pages_requested,
            "records_received": records_received,
            "unique_ids": len(records_by_id),
            "min_ctime": min(all_ctimes) if all_ctimes else None,
            "max_ctime": max(all_ctimes) if all_ctimes else None,
            "duplicate_ids": len(duplicate_id_values),
            "duplicate_id_values": sorted(duplicate_id_values),
            "request_errors": request_errors,
            "cursor_sequence": cursor_sequence,
            "coverage_complete": coverage_complete,
            "stop_reason": stop_reason,
            "content_gaps": content_gaps,
            "snapshots": snapshots,
            "manifest_path": str(manifest_path),
            "gap": gap.to_dict() if gap else None,
        }
        self._write_immutable_json(manifest_path, manifest)
        records = sorted(
            records_by_id.values(),
            key=lambda row: (int(row["ctime"]), str(row["cls_id"])),
            reverse=True,
        )
        return ClsHistoryResult(records=records, manifest=manifest, gap=gap)


def fetch_cls_latest(**kwargs: Any) -> ClsLatestBatch:
    """Convenience wrapper around :class:`ClsNewsClient.fetch_latest`."""
    client = kwargs.pop("client", None) or ClsNewsClient()
    return client.fetch_latest(**kwargs)


def crawl_cls_history(
    analysis_as_of: datetime | date | str | int | float,
    **kwargs: Any,
) -> ClsHistoryResult:
    """Convenience wrapper around :class:`ClsNewsClient.crawl_history`."""
    client = kwargs.pop("client", None) or ClsNewsClient()
    return client.crawl_history(analysis_as_of, **kwargs)


__all__ = [
    "CLS_HISTORY_ENDPOINT",
    "CLS_LATEST_ENDPOINT",
    "CLS_HISTORY_RN",
    "CLS_SOURCE",
    "CN_TZ",
    "CoverageGap",
    "ClsHistoryResult",
    "ClsLatestBatch",
    "ClsNewsClient",
    "ClsNewsError",
    "crawl_cls_history",
    "fetch_cls_latest",
    "format_cls_records",
    "generate_cls_sign",
    "normalize_cls_record",
]
