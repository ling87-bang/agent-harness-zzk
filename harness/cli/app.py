"""Typer app entrypoint for zzk."""

from __future__ import annotations

from pathlib import Path

import typer

from harness.cli.chain_commands import run_chain_list, run_chain_sequential
from harness.cli.commands import run_chat, run_eval, run_query
from harness.cli.trace_commands import run_trace_list, run_trace_show

app = typer.Typer(
    help="zzk agent harness CLI",
    add_completion=False,
    invoke_without_command=True,
)

trace_app = typer.Typer(help="Inspect JSONL traces under ~/.zzk/traces")
app.add_typer(trace_app, name="trace")

chain_app = typer.Typer(help="Run and manage deterministic chains")
app.add_typer(chain_app, name="chain")


def _parse_user_skill_dirs(values: list[str]) -> tuple[Path, ...]:
    return tuple(Path(value) for value in values)


@app.command("run")
def run_command(
    query: str = typer.Argument(..., help="Query text"),
    prompt_version: str | None = typer.Option(
        None,
        "--prompt-version",
        help="System prompt version (v1 or v2)",
    ),
    enable_user_skills: bool = typer.Option(
        False,
        "--enable-user-skills",
        help="Load skills from ~/.zzk/skills and --user-skills-dir",
    ),
    user_skills_dir: list[str] = typer.Option(
        [],
        "--user-skills-dir",
        help="Extra directory to scan for user skills (repeatable)",
    ),
) -> None:
    """Run one single-turn query."""

    raise typer.Exit(
        code=run_query(
            query,
            prompt_version=prompt_version,
            enable_user_skills=enable_user_skills,
            user_skill_dirs=_parse_user_skill_dirs(user_skills_dir),
        )
    )


@app.command("chat")
def chat_command(
    conversation_id: str | None = typer.Option(None, "--conversation-id", help="Existing conversation id"),
    prompt_version: str | None = typer.Option(
        None,
        "--prompt-version",
        help="System prompt version (v1 or v2)",
    ),
    enable_user_skills: bool = typer.Option(
        False,
        "--enable-user-skills",
        help="Load skills from ~/.zzk/skills and --user-skills-dir",
    ),
    user_skills_dir: list[str] = typer.Option(
        [],
        "--user-skills-dir",
        help="Extra directory to scan for user skills (repeatable)",
    ),
) -> None:
    """Run interactive chat mode."""

    raise typer.Exit(
        code=run_chat(
            conversation_id=conversation_id,
            prompt_version=prompt_version,
            enable_user_skills=enable_user_skills,
            user_skill_dirs=_parse_user_skill_dirs(user_skills_dir),
        )
    )


@app.command("eval")
def eval_command(
    cases: str = typer.Option(
        ...,
        "--cases",
        "--cases-file",
        help="Path to eval cases JSON file",
    ),
    report_out: str | None = typer.Option(
        None,
        "--report-out",
        "--out",
        help="Write machine-readable eval report JSON",
    ),
    prompt_version: str | None = typer.Option(
        None,
        "--prompt-version",
        help="System prompt version for a single eval run (v1 or v2)",
    ),
    compare_prompts: str | None = typer.Option(
        None,
        "--compare-prompts",
        help="Compare prompt versions, e.g. v1,v2 (writes prompt_comparison report)",
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        min=1,
        help="Parallel case workers for eval",
    ),
    enable_user_skills: bool = typer.Option(
        False,
        "--enable-user-skills",
        help="Load skills from ~/.zzk/skills and --user-skills-dir",
    ),
    user_skills_dir: list[str] = typer.Option(
        [],
        "--user-skills-dir",
        help="Extra directory to scan for user skills (repeatable)",
    ),
) -> None:
    """Run golden case evaluation."""

    compare_versions: tuple[str, ...] | None = None
    if compare_prompts:
        compare_versions = tuple(
            part.strip() for part in compare_prompts.split(",") if part.strip()
        )

    output_path = Path(report_out) if report_out is not None else None
    raise typer.Exit(
        code=run_eval(
            cases_file=Path(cases),
            output_path=output_path,
            prompt_version=prompt_version,
            compare_prompt_versions=compare_versions,
            workers=workers,
            enable_user_skills=enable_user_skills,
            user_skill_dirs=_parse_user_skill_dirs(user_skills_dir),
        )
    )


@trace_app.command("list")
def trace_list_command(
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum runs to list"),
    trace_dir: str | None = typer.Option(None, "--trace-dir", help="Override trace directory"),
) -> None:
    """List recent trace runs."""

    directory = Path(trace_dir) if trace_dir is not None else None
    raise typer.Exit(code=run_trace_list(limit=limit, trace_dir=directory))


@chain_app.command("list")
def chain_list_command() -> None:
    """List registered chains and supported step tokens."""

    raise typer.Exit(code=run_chain_list())


@chain_app.command("run")
def chain_run_command(
    chain_type: str = typer.Argument(..., help="Chain type (sequential)"),
    input_text: str = typer.Argument(..., help="Chain input text"),
    steps: str = typer.Option(..., "--steps", help="Comma-separated steps, e.g. llm,skill:file_reader,llm"),
) -> None:
    """Run a chain pipeline."""

    if chain_type != "sequential":
        typer.echo(f"Unsupported chain type: {chain_type}")
        raise typer.Exit(code=1)
    raise typer.Exit(code=run_chain_sequential(steps=steps, input_text=input_text))


@trace_app.command("show")
def trace_show_command(
    run_id: str = typer.Argument(..., help="Run id or unique prefix"),
    trace_dir: str | None = typer.Option(None, "--trace-dir", help="Override trace directory"),
) -> None:
    """Show steps for one trace run."""

    directory = Path(trace_dir) if trace_dir is not None else None
    raise typer.Exit(code=run_trace_show(run_id=run_id, trace_dir=directory))


@app.callback()
def default(ctx: typer.Context) -> None:
    """Show usage when no command is provided."""

    if ctx.invoked_subcommand:
        return
    typer.echo(
        "Usage: zzk run <query> | zzk chat [--conversation-id <id>] "
        "| zzk eval --cases-file <file> [--report-out <report.json>] "
        "[--prompt-version v1|v2] [--compare-prompts v1,v2] [--workers N] "
        "[--enable-user-skills] [--user-skills-dir <dir>] "
        "| zzk trace list | zzk trace show <run_id> "
        "| zzk chain list | zzk chain run sequential --steps <steps> <input>"
    )
    raise typer.Exit(code=1)
