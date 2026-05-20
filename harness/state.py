"""Core immutable state models."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class Message:
    """A single chat message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """Stream event emitted by LLM and engine."""

    event_type: Literal["token", "done", "error"]
    content: str = ""
    error_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TraceStep:
    """A single trace step record."""

    record_type: str = "step"
    step: int = 0
    step_type: str = ""
    run_id: str = ""
    request_id: str = ""
    conversation_id: str = ""
    parent_step: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str | None = None
    skill: str | None = None
    status: str = "succeeded"
    error: str | None = None
    error_code: str | None = None
