"""Typer app entrypoint for zzk."""

from __future__ import annotations

import typer

from harness.cli.commands import run_chat, run_query

app = typer.Typer(
    help="zzk agent harness CLI",
    add_completion=False,
    invoke_without_command=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)


@app.callback()
def default(
    ctx: typer.Context,
    query: str | None = typer.Argument(default=None, help="Query text"),
) -> None:
    """Support `zzk "hello"` entry style."""

    if ctx.invoked_subcommand:
        return
    if not query:
        typer.echo("Provide a query, e.g. zzk \"hello\"")
        raise typer.Exit(code=1)
    if query == "run":
        if not ctx.args:
            typer.echo("Usage: zzk run <query>")
            raise typer.Exit(code=1)
        raise typer.Exit(code=run_query(" ".join(ctx.args)))
    if query == "chat":
        conversation_id = _parse_conversation_id(ctx.args)
        raise typer.Exit(code=run_chat(conversation_id=conversation_id))
    raise typer.Exit(code=run_query(query))

def _parse_conversation_id(args: list[str]) -> str | None:
    if len(args) >= 2 and args[0] == "--conversation-id":
        return args[1]
    return None
