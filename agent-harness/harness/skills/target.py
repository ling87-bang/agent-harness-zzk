"""Knowledge target abstraction and implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from harness.errors import ERROR_TARGET_UNREACHABLE


@dataclass(frozen=True, slots=True)
class KnowledgeResult:
    """Knowledge backend response."""

    items: list[dict[str, Any]]
    hit_count: int
    error: str | None = None
    error_code: str | None = None


class KnowledgeTarget(Protocol):
    """Knowledge backend protocol."""

    async def search(self, query: str, top_k: int = 5) -> KnowledgeResult:
        """Search knowledge source."""


@dataclass(frozen=True, slots=True)
class HttpKnowledgeTarget:
    """Knowledge target backed by HTTP API."""

    base_url: str
    api_key: str | None = None
    timeout_seconds: float = 10.0

    async def search(self, query: str, top_k: int = 5) -> KnowledgeResult:
        payload = {"query": query, "top_k": top_k}
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url.rstrip('/')}/search"

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return KnowledgeResult(
                items=[],
                hit_count=0,
                error=str(exc),
                error_code=ERROR_TARGET_UNREACHABLE,
            )

        items_value = data.get("items", [])
        items = [item for item in items_value if isinstance(item, dict)] if isinstance(items_value, list) else []
        return KnowledgeResult(items=items, hit_count=len(items))


@dataclass(frozen=True, slots=True)
class MockKnowledgeTarget:
    """Deterministic knowledge target for tests/offline."""

    items: list[dict[str, Any]] | None = None

    async def search(self, query: str, top_k: int = 5) -> KnowledgeResult:
        values = self.items or []
        sliced = values[:top_k]
        return KnowledgeResult(items=sliced, hit_count=len(sliced))
