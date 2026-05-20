"""Chain orchestration protocols and shared models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from harness.engine.trace import TraceRecorder
from harness.llm.base import LLMProvider
from harness.skills.registry import SkillRegistry


@dataclass(frozen=True, slots=True)
class ChainResult:
    """Result of a chain or single node execution."""

    output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    sub_results: tuple[ChainResult, ...] = ()


@dataclass(frozen=True, slots=True)
class ChainContext:
    """Shared runtime context passed through chain nodes."""

    provider: LLMProvider | None = None
    registry: SkillRegistry | None = None
    cwd: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # Reserved for v1.5+: wire TraceRecorder steps without coupling to engine/loop.py.
    trace: TraceRecorder | None = None


class ChainNode(Protocol):
    """Single composable step inside a chain."""

    @property
    def name(self) -> str:
        """Node identifier for logging and metadata."""

    async def run(self, input_text: str, context: ChainContext) -> ChainResult:
        """Execute one step using the shared context."""


class Chain(Protocol):
    """Top-level chain orchestrator."""

    @property
    def name(self) -> str:
        """Chain name."""

    async def run(self, input_text: str, context: ChainContext | None = None) -> ChainResult:
        """Run the chain for the given input."""


_CHAIN_REGISTRY: dict[str, Chain] = {}


def register_chain(chain: Chain) -> None:
    """Register a chain by its name."""

    _CHAIN_REGISTRY[chain.name] = chain


def get_chain(name: str) -> Chain | None:
    """Return a registered chain or None."""

    return _CHAIN_REGISTRY.get(name)


def list_chains() -> tuple[str, ...]:
    """Return registered chain names in insertion order."""

    return tuple(_CHAIN_REGISTRY.keys())
