from dataclasses import dataclass
from typing import Any

from harness.state import Message, StreamEvent


@dataclass(frozen=True, slots=True)
class FakeProvider:
    name: str = "fake"
    model: str = "fake-model"
    stream_text: str = ""
    should_error: bool = False
    scripted_texts: tuple[str, ...] = ()

    async def chat(self, messages: list[Message], *, temperature: float = 0.1, max_tokens: int = 4096) -> str:
        if self.should_error:
            raise RuntimeError("boom")
        if self.scripted_texts:
            calls = sum(1 for item in messages if item.role == "assistant")
            index = min(calls, len(self.scripted_texts) - 1)
            return self.scripted_texts[index]
        return self.stream_text

    async def chat_stream(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> Any:
        if self.should_error:
            yield StreamEvent(event_type="error", content="boom", error_code="llm_error")
            return
        if self.scripted_texts:
            calls = sum(1 for item in messages if item.role == "assistant")
            index = min(calls, len(self.scripted_texts) - 1)
            text = self.scripted_texts[index]
        else:
            text = self.stream_text
        yield StreamEvent(event_type="token", content=text)
        yield StreamEvent(event_type="done")
