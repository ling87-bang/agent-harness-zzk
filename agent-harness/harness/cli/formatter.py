"""Terminal stream formatting helpers."""

from __future__ import annotations

import sys

import typer

from harness.state import StreamEvent


def render_stream_event(event: StreamEvent) -> None:
    """Render a single stream event to terminal."""

    if event.event_type == "token":
        sys.stdout.write(event.content)
        sys.stdout.flush()
        return
    if event.event_type == "error":
        typer.echo(f"\n[error:{event.error_code}] {event.content}")
        return
    if event.event_type == "done":
        if event.error_code:
            typer.echo(f"\n[降级:{event.error_code}]")
        else:
            typer.echo("")
