"""Knowledge and search target abstractions."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx

from harness.errors import ERROR_SEARCH_MISCONFIGURED, ERROR_TARGET_UNREACHABLE


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
    """Knowledge target backed by HTTP API (POST {base_url}/search, Harness contract)."""

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


_DEFAULT_KNOWLEDGE_SEED: tuple[tuple[str, str], ...] = (
    ("zzk-overview", "Agent Harness (zzk) is a lightweight agent runtime with skills and trace."),
    ("zzk-chain", "Chain orchestration supports sequential and router pipelines."),
    ("zzk-eval", "Golden-case eval writes machine-readable reports for CI regression checks."),
)


@dataclass(frozen=True, slots=True)
class SqliteKnowledgeTarget:
    """Embedded SQLite knowledge store for offline demos (stdlib sqlite3)."""

    db_path: Path

    async def search(self, query: str, top_k: int = 5) -> KnowledgeResult:
        bounded_top_k = max(1, min(top_k, 20))
        items = await asyncio.to_thread(_sqlite_search_sync, self.db_path, query, bounded_top_k)
        return KnowledgeResult(items=items, hit_count=len(items))


def default_knowledge_db_path() -> Path:
    """Default path for the bundled knowledge SQLite database."""

    return Path.home() / ".zzk" / "knowledge.db"


def ensure_knowledge_database(db_path: Path) -> None:
    """Create and seed the knowledge database when missing."""

    if db_path.is_file():
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE documents (
                source TEXT NOT NULL,
                snippet TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO documents (source, snippet) VALUES (?, ?)",
            _DEFAULT_KNOWLEDGE_SEED,
        )
        connection.commit()
    finally:
        connection.close()


@dataclass(frozen=True, slots=True)
class AutoKnowledgeTarget:
    """Try HTTP knowledge first; fall back to SQLite when unreachable (sticky per process)."""

    http: HttpKnowledgeTarget
    sqlite: SqliteKnowledgeTarget
    _use_sqlite: list[bool] = field(default_factory=lambda: [False])

    async def search(self, query: str, top_k: int = 5) -> KnowledgeResult:
        if self._use_sqlite[0]:
            return await self.sqlite.search(query=query, top_k=top_k)
        result = await self.http.search(query=query, top_k=top_k)
        if result.error_code != ERROR_TARGET_UNREACHABLE:
            return result
        self._use_sqlite[0] = True
        return await self.sqlite.search(query=query, top_k=top_k)


def build_knowledge_target(
    *,
    provider: Literal["auto", "http", "sqlite", "mock"] = "auto",
    base_url: str = "http://127.0.0.1:8000/knowledge",
    api_key: str = "",
    timeout_seconds: float = 10.0,
    sqlite_path: str = "",
) -> KnowledgeTarget:
    """Construct knowledge target from settings."""

    if provider == "mock":
        return MockKnowledgeTarget(items=[])
    db_path = Path(sqlite_path) if sqlite_path.strip() else default_knowledge_db_path()
    if provider == "sqlite":
        ensure_knowledge_database(db_path)
        return SqliteKnowledgeTarget(db_path=db_path)
    http_target = HttpKnowledgeTarget(
        base_url=base_url,
        api_key=api_key or None,
        timeout_seconds=timeout_seconds,
    )
    if provider == "http":
        return http_target
    ensure_knowledge_database(db_path)
    return AutoKnowledgeTarget(http=http_target, sqlite=SqliteKnowledgeTarget(db_path=db_path))


def _sqlite_search_sync(db_path: Path, query: str, top_k: int) -> list[dict[str, Any]]:
    ensure_knowledge_database(db_path)
    pattern = f"%{query.strip()}%"
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            """
            SELECT source, snippet FROM documents
            WHERE snippet LIKE ? OR source LIKE ?
            LIMIT ?
            """,
            (pattern, pattern, top_k),
        ).fetchall()
        if not rows:
            rows = connection.execute(
                "SELECT source, snippet FROM documents LIMIT ?",
                (top_k,),
            ).fetchall()
    finally:
        connection.close()
    return [{"source": str(source), "snippet": str(snippet)} for source, snippet in rows]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Web search backend response."""

    items: list[dict[str, Any]]
    hit_count: int
    error: str | None = None
    error_code: str | None = None


class SearchTarget(Protocol):
    """Web search backend protocol."""

    async def search(self, query: str, top_k: int = 5) -> SearchResult:
        """Search the web."""


@dataclass(frozen=True, slots=True)
class DuckDuckGoTarget:
    """DuckDuckGo Instant Answer API (no API key)."""

    timeout_seconds: float = 10.0
    min_interval_seconds: float = 1.0
    # Mutable slot inside frozen dataclass: holds last perf_counter for rate limiting.
    _last_call: list[float] = field(default_factory=lambda: [0.0])

    async def search(self, query: str, top_k: int = 5) -> SearchResult:
        now = time.perf_counter()
        wait = self.min_interval_seconds - (now - self._last_call[0])
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_call[0] = time.perf_counter()

        params = {"q": query, "format": "json", "no_redirect": 1, "no_html": 1}
        url = "https://api.duckduckgo.com/"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return SearchResult(
                items=[],
                hit_count=0,
                error=str(exc),
                error_code=ERROR_TARGET_UNREACHABLE,
            )

        items = _duckduckgo_items(data, top_k)
        return SearchResult(items=items, hit_count=len(items), error=None)


@dataclass(frozen=True, slots=True)
class SerpApiTarget:
    """SerpAPI Google search backend."""

    api_key: str
    timeout_seconds: float = 10.0

    async def search(self, query: str, top_k: int = 5) -> SearchResult:
        if not self.api_key.strip():
            return SearchResult(
                items=[],
                hit_count=0,
                error="ZZK_SEARCH_API_KEY is required for serpapi provider",
                error_code=ERROR_SEARCH_MISCONFIGURED,
            )
        params = {"q": query, "api_key": self.api_key, "engine": "google", "num": top_k}
        url = "https://serpapi.com/search.json"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return SearchResult(
                items=[],
                hit_count=0,
                error=str(exc),
                error_code=ERROR_TARGET_UNREACHABLE,
            )

        items = _serpapi_items(data, top_k)
        return SearchResult(items=items, hit_count=len(items))


@dataclass(frozen=True, slots=True)
class MockSearchTarget:
    """Deterministic search target for tests/offline."""

    items: list[dict[str, Any]] | None = None

    async def search(self, query: str, top_k: int = 5) -> SearchResult:
        values = self.items or []
        sliced = values[:top_k]
        return SearchResult(items=sliced, hit_count=len(sliced))


def build_search_target(
    *,
    provider: Literal["duckduckgo", "serpapi"] = "duckduckgo",
    api_key: str = "",
    timeout_seconds: float = 10.0,
) -> SearchTarget:
    """Construct search target from settings."""

    if provider == "serpapi":
        return SerpApiTarget(api_key=api_key, timeout_seconds=timeout_seconds)
    return DuckDuckGoTarget(timeout_seconds=timeout_seconds)


def _duckduckgo_items(data: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    abstract = data.get("AbstractText")
    abstract_url = data.get("AbstractURL") or ""
    if isinstance(abstract, str) and abstract.strip():
        items.append(
            {
                "title": str(data.get("Heading", "DuckDuckGo")),
                "url": abstract_url,
                "snippet": abstract[:500],
            }
        )
    related = data.get("RelatedTopics", [])
    if isinstance(related, list):
        for entry in related:
            if len(items) >= top_k:
                break
            if not isinstance(entry, dict):
                continue
            text = entry.get("Text")
            if isinstance(text, str) and text.strip():
                items.append({"title": text[:80], "url": str(entry.get("FirstURL", "")), "snippet": text[:500]})
    return items[:top_k]


def _serpapi_items(data: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    organic = data.get("organic_results", [])
    if not isinstance(organic, list):
        return []
    items: list[dict[str, Any]] = []
    for entry in organic[:top_k]:
        if not isinstance(entry, dict):
            continue
        items.append(
            {
                "title": str(entry.get("title", "")),
                "url": str(entry.get("link", "")),
                "snippet": str(entry.get("snippet", ""))[:500],
            }
        )
    return items
