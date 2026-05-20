"""CLI commands for trace inspection."""

from __future__ import annotations

from pathlib import Path

import typer

from harness.engine.trace_reader import (
    default_trace_dir,
    list_trace_runs,
    load_trace_run,
)


def run_trace_list(*, limit: int = 20, trace_dir: Path | None = None) -> int:
    entries, _ = list_trace_runs(trace_dir=trace_dir, limit=limit)
    directory = trace_dir or default_trace_dir()
    if not directory.is_dir():
        typer.echo(f"[trace] no trace directory at {directory}")
        return 0
    if not entries:
        typer.echo(f"[trace] no runs under {directory}")
        return 0

    typer.echo(f"[trace] directory={directory}")
    for entry in entries:
        query_preview = entry.query.replace("\n", " ")[:60]
        typer.echo(
            f"{entry.run_id}  status={entry.final_status}  steps={entry.step_count}  "
            f"time={entry.timestamp}  query={query_preview}"
        )
    return 0


def run_trace_show(*, run_id: str, trace_dir: Path | None = None) -> int:
    loaded = load_trace_run(run_id, trace_dir=trace_dir)
    if loaded.error_code is not None:
        if loaded.error_code == "trace_ambiguous":
            typer.echo(f"[trace:error] multiple runs match prefix '{run_id}'")
        elif loaded.error_code == "trace_not_found":
            typer.echo(f"[trace:error] run not found: {run_id}")
        else:
            typer.echo(f"[trace:error:{loaded.error_code}] failed to load trace for {run_id}")
        return 1

    typer.echo(f"[trace] run_id={loaded.run_id} path={loaded.path}")
    for record in loaded.records:
        record_type = record.get("record_type")
        if record_type == "run":
            typer.echo(
                f"[run] query={record.get('query')} provider={record.get('llm_provider')} "
                f"model={record.get('llm_model')} conversation_id={record.get('conversation_id')}"
            )
            continue
        if record_type == "run_summary":
            typer.echo(
                f"[summary] final_status={record.get('final_status')} "
                f"total_steps={record.get('total_steps')}"
            )
            continue
        if record_type == "step":
            parts = [
                f"[step {record.get('step')}]",
                f"type={record.get('step_type')}",
                f"status={record.get('status')}",
            ]
            skill = record.get("skill")
            if isinstance(skill, str) and skill:
                parts.append(f"skill={skill}")
            error_code = record.get("error_code")
            if isinstance(error_code, str) and error_code:
                parts.append(f"error_code={error_code}")
            latency = record.get("latency_ms")
            if isinstance(latency, (int, float)) and latency:
                parts.append(f"latency_ms={latency}")
            error = record.get("error")
            if isinstance(error, str) and error:
                parts.append(f"error={error}")
            typer.echo(" ".join(parts))
    return 0
