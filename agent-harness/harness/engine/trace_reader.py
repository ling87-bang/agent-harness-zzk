"""Read and inspect JSONL trace files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def default_trace_dir() -> Path:
    return Path.home() / ".zzk" / "traces"


@dataclass(frozen=True, slots=True)
class TraceRunEntry:
    """Summary row for trace list."""

    run_id: str
    path: Path
    timestamp: str
    query: str
    final_status: str
    step_count: int


@dataclass(frozen=True, slots=True)
class TraceLoadResult:
    """Loaded trace file or error code."""

    run_id: str
    path: Path
    records: tuple[dict[str, object], ...]
    error_code: str | None = None


def list_trace_runs(
    *,
    trace_dir: Path | None = None,
    limit: int = 20,
) -> tuple[list[TraceRunEntry], str | None]:
    directory = trace_dir or default_trace_dir()
    if not directory.is_dir():
        return ([], None)

    entries: list[TraceRunEntry] = []
    files = sorted(directory.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in files[:limit]:
        summary = _read_run_summary(path)
        if summary is None:
            continue
        entries.append(summary)
    return (entries, None)


def load_trace_run(
    run_id: str,
    *,
    trace_dir: Path | None = None,
) -> TraceLoadResult:
    directory = trace_dir or default_trace_dir()
    if not directory.is_dir():
        return TraceLoadResult(run_id=run_id, path=directory, records=(), error_code="trace_dir_missing")

    matches = _resolve_trace_paths(run_id, directory)
    if not matches:
        return TraceLoadResult(run_id=run_id, path=directory, records=(), error_code="trace_not_found")
    if len(matches) > 1:
        return TraceLoadResult(run_id=run_id, path=directory, records=(), error_code="trace_ambiguous")

    path = matches[0]
    try:
        records = tuple(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, json.JSONDecodeError):
        return TraceLoadResult(run_id=run_id, path=path, records=(), error_code="trace_read_failed")

    return TraceLoadResult(run_id=run_id, path=path, records=records)


def _resolve_trace_paths(run_id: str, directory: Path) -> list[Path]:
    exact = directory / f"{run_id}.jsonl"
    if exact.is_file():
        return [exact]
    prefix_matches = sorted(directory.glob(f"{run_id}*.jsonl"))
    if len(prefix_matches) == 1:
        return prefix_matches
    if len(prefix_matches) > 1:
        return prefix_matches
    return []


def _read_run_summary(path: Path) -> TraceRunEntry | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    if not lines:
        return None

    header: dict[str, object] = {}
    final_status = "unknown"
    step_count = 0
    try:
        first = json.loads(lines[0])
        if isinstance(first, dict) and first.get("record_type") == "run":
            header = first
        for line in lines[1:]:
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            if row.get("record_type") == "step":
                step_count += 1
            if row.get("record_type") == "run_summary":
                status_value = row.get("final_status")
                if isinstance(status_value, str):
                    final_status = status_value
    except json.JSONDecodeError:
        return None

    run_id_value = header.get("run_id")
    if not isinstance(run_id_value, str):
        run_id_value = path.stem

    timestamp_value = header.get("timestamp")
    query_value = header.get("query")
    return TraceRunEntry(
        run_id=run_id_value,
        path=path,
        timestamp=str(timestamp_value) if isinstance(timestamp_value, str) else "",
        query=str(query_value) if isinstance(query_value, str) else "",
        final_status=final_status,
        step_count=step_count,
    )
