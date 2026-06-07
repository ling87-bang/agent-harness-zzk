"""Conversation context manager for multi-turn chat."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from harness.state import Message

if TYPE_CHECKING:
    from harness.llm.deepseek import DeepSeekProvider

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConversationManager:
    """Load, persist, and compress conversation history."""

    storage_dir: Path = Path.home() / ".zzk" / "conversations"
    max_messages: int = 20
    max_chars: int = 8_192

    def new_conversation_id(self) -> str:
        return f"conv-{uuid.uuid4().hex[:12]}"

    def load_history(self, conversation_id: str) -> list[Message]:
        path = self._conversation_path(conversation_id)
        if not path.exists():
            return []
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        messages: list[Message] = []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                role = row.get("role")
                content = row.get("content")
                if role in {"system", "user", "assistant", "tool"} and isinstance(content, str):
                    messages.append(Message(role=role, content=content))
        return messages

    def save_history(self, conversation_id: str, messages: list[Message]) -> bool:
        path = self._conversation_path(conversation_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = [{"role": item.role, "content": item.content} for item in messages]
            path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("conversation save failed: %s", exc)
            return False
        return True

    def compress_history(self, messages: list[Message]) -> list[Message]:
        """Deterministic compression (default; tests and llm fallback)."""
        return self._compress_deterministic(messages)

    async def compress_history_async(
        self,
        messages: list[Message],
        *,
        mode: Literal["deterministic", "llm"] = "deterministic",
        provider: DeepSeekProvider | None = None,
        summary_max_tokens: int = 512,
    ) -> list[Message]:
        """Compress history; optional LLM summary when mode is llm and provider is set."""
        if mode != "llm" or provider is None:
            return self._compress_deterministic(messages)
        return await self._compress_with_llm(
            messages,
            provider=provider,
            summary_max_tokens=max(64, min(summary_max_tokens, 2048)),
        )

    def _compress_deterministic(self, messages: list[Message]) -> list[Message]:
        non_system = [item for item in messages if item.role != "system"]
        if len(non_system) > self.max_messages:
            keep_recent = max(4, self.max_messages // 2)
            older = non_system[:-keep_recent]
            recent = non_system[-keep_recent:]
            summary = self._build_rule_summary(older)
            non_system = [
                Message(role="system", content=f"Conversation summary:\n{summary}"),
                *recent,
            ]
        return self._apply_char_budget(non_system)

    async def _compress_with_llm(
        self,
        messages: list[Message],
        *,
        provider: DeepSeekProvider,
        summary_max_tokens: int,
    ) -> list[Message]:
        non_system = [item for item in messages if item.role != "system"]
        if len(non_system) > self.max_messages:
            keep_recent = max(4, self.max_messages // 2)
            older = non_system[:-keep_recent]
            recent = non_system[-keep_recent:]
            summary = await self._summarize_older_messages(
                older,
                provider=provider,
                summary_max_tokens=summary_max_tokens,
            )
            non_system = [
                Message(role="system", content=f"Conversation summary:\n{summary}"),
                *recent,
            ]
        return self._apply_char_budget(non_system)

    def _build_rule_summary(self, older: list[Message]) -> str:
        summary_lines = [f"{item.role}: {item.content[:160]}" for item in older[-10:]]
        return "\n".join(summary_lines)[:1200]

    async def _summarize_older_messages(
        self,
        older: list[Message],
        *,
        provider: DeepSeekProvider,
        summary_max_tokens: int,
    ) -> str:
        if not older:
            return ""
        transcript = "\n".join(f"{item.role}: {item.content[:400]}" for item in older[-20:])
        prompt_messages = [
            Message(
                role="system",
                content=(
                    "Summarize the prior conversation for context continuity. "
                    "Keep facts, decisions, and open tasks. Use concise prose."
                ),
            ),
            Message(role="user", content=transcript),
        ]
        try:
            summary = await provider.chat(
                prompt_messages,
                temperature=0.0,
                max_tokens=summary_max_tokens,
            )
            cleaned = summary.strip()
            if cleaned:
                return cleaned[:4000]
        except (RuntimeError, OSError) as exc:
            LOGGER.warning("llm conversation summary failed, using rule fallback: %s", exc)
        return self._build_rule_summary(older)

    def _apply_char_budget(self, non_system: list[Message]) -> list[Message]:
        while sum(len(item.content) for item in non_system) > self.max_chars and len(non_system) > 1:
            removal_index = 0
            if non_system[0].role == "system":
                removal_index = 1
                if len(non_system) <= 2:
                    break
            non_system.pop(removal_index)
        return non_system

    def _conversation_path(self, conversation_id: str) -> Path:
        return self.storage_dir / f"{conversation_id}.json"
