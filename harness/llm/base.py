"""LLM protocol and response parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

from harness.errors import ERROR_PARSE_FAILED
from harness.state import Message, StreamEvent

_TOOL_OBSERVATION_PREFIX = "[tool_observation]"


def messages_to_chat_api_payload(messages: list[Message]) -> list[dict[str, str]]:
    """Map internal roles to roles accepted by OpenAI-compatible chat APIs.

    Many providers (including DeepSeek) reject ``role=tool`` unless native function
    calling is enabled. Harness encodes tool results as user follow-ups instead.
    """

    payload: list[dict[str, str]] = []
    for item in messages:
        if item.role == "tool":
            payload.append(
                {
                    "role": "user",
                    "content": f"{_TOOL_OBSERVATION_PREFIX}\n{item.content}",
                }
            )
            continue
        if item.role not in {"system", "user", "assistant"}:
            payload.append({"role": "user", "content": item.content})
            continue
        payload.append({"role": item.role, "content": item.content})
    return payload


@dataclass(frozen=True, slots=True)
class ParsedLLMResponse:
    """Normalized LLM output after schema parsing."""

    action: str
    content: str = ""
    name: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    error_code: str | None = None


class LLMProvider(Protocol):
    """Provider protocol for chat completion and streaming."""

    @property
    def name(self) -> str:
        """Provider name."""

    @property
    def model(self) -> str:
        """Model id."""

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        """Return one full assistant output."""

    async def chat_stream(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamEvent]:
        """Yield token/done/error stream events."""


def _normalize_response(payload: dict[str, Any]) -> ParsedLLMResponse | None:
    action_value = payload.get("action")
    if action_value not in {"tool", "answer"}:
        return None
    if action_value == "tool":
        name = payload.get("name")
        args_value = payload.get("args", {})
        if not isinstance(name, str) or not isinstance(args_value, dict):
            return None
        return ParsedLLMResponse(
            action="tool",
            name=name,
            args=args_value,
            reasoning=str(payload.get("reasoning", "")),
        )

    content = payload.get("content")
    citations_value = payload.get("citations", [])
    if not isinstance(content, str):
        return None
    if not isinstance(citations_value, list):
        return None
    return ParsedLLMResponse(
        action="answer",
        content=content,
        reasoning=str(payload.get("reasoning", "")),
        citations=[
            item for item in citations_value if isinstance(item, dict)
        ],
    )


def parse_llm_response(raw_text: str) -> ParsedLLMResponse:
    """Parse strict action JSON with deterministic fallback chain."""

    stripped = raw_text.strip()

    # Fallback level 1: parse the whole response as JSON.
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        normalized = _normalize_response(payload)
        if normalized is not None:
            return normalized

    # Fallback level 2: extract JSON object from free text.
    for candidate in _extract_json_candidates(raw_text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        normalized = _normalize_response(payload)
        if normalized is not None:
            return normalized

    # Fallback level 3: deterministic degraded answer with parse_failed.
    return ParsedLLMResponse(
        action="answer",
        content=stripped or raw_text,
        error_code=ERROR_PARSE_FAILED,
    )


def _extract_json_candidates(raw_text: str) -> list[str]:
    candidates: list[str] = []
    stack_depth = 0
    start_index: int | None = None
    in_string = False
    escaped = False

    for index, char in enumerate(raw_text):
        if char == "\\" and in_string and not escaped:
            escaped = True
            continue
        if char == '"' and not escaped:
            in_string = not in_string
        if escaped:
            escaped = False
            continue
        if in_string:
            continue
        if char == "{":
            if stack_depth == 0:
                start_index = index
            stack_depth += 1
        elif char == "}":
            if stack_depth == 0:
                continue
            stack_depth -= 1
            if stack_depth == 0 and start_index is not None:
                candidates.append(raw_text[start_index : index + 1])
                start_index = None
    return candidates
