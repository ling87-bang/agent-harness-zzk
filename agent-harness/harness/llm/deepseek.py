"""DeepSeek provider with SSE streaming."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from harness.config import Settings
from harness.errors import ERROR_LLM_ERROR
from harness.state import Message, StreamEvent


class DeepSeekProvider:
    """DeepSeek chat provider using HTTP streaming."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def model(self) -> str:
        return self._settings.deepseek_model

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._settings.deepseek_api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _serialize_messages(messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": item.role, "content": item.content} for item in messages]

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        """Return full assistant output (non-stream endpoint)."""

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._serialize_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        url = f"{self._settings.deepseek_base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self._settings.deepseek_timeout_seconds) as client:
                response = await client.post(url, headers=self._build_headers(), json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"{ERROR_LLM_ERROR}: deepseek_non_stream_failed") from exc

        try:
            data = response.json()
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError(f"{ERROR_LLM_ERROR}: deepseek_empty_choices")
            first = choices[0]
            if not isinstance(first, dict):
                raise RuntimeError(f"{ERROR_LLM_ERROR}: deepseek_invalid_choice")
            message = first.get("message")
            if not isinstance(message, dict):
                raise RuntimeError(f"{ERROR_LLM_ERROR}: deepseek_missing_message")
            content = message.get("content")
            if not isinstance(content, str):
                raise RuntimeError(f"{ERROR_LLM_ERROR}: deepseek_missing_content")
            return content
        except ValueError as exc:
            raise RuntimeError(f"{ERROR_LLM_ERROR}: deepseek_invalid_json") from exc

    async def chat_stream(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamEvent]:
        """Yield StreamEvent from DeepSeek SSE chunks."""

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._serialize_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        url = f"{self._settings.deepseek_base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self._settings.deepseek_timeout_seconds) as client:
                async with client.stream(
                    "POST",
                    url,
                    headers=self._build_headers(),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data_text = line[len("data:") :].strip()
                        if data_text == "[DONE]":
                            yield StreamEvent(event_type="done")
                            return
                        try:
                            chunk = json.loads(data_text)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        token = delta.get("content")
                        if isinstance(token, str) and token:
                            yield StreamEvent(event_type="token", content=token)
        except httpx.HTTPError as exc:
            yield StreamEvent(
                event_type="error",
                content=str(exc),
                error_code=ERROR_LLM_ERROR,
            )
