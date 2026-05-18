from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from harness.chain.base import ChainContext, ChainResult, get_chain, list_chains, register_chain
from harness.chain.sequential import SequentialChain
from harness.engine.trace import TraceRecorder


def test_chain_result_defaults_are_not_shared() -> None:
    first = ChainResult(output="a")
    second = ChainResult(output="b")
    first.metadata["x"] = 1
    assert "x" not in second.metadata


def test_chain_context_trace_placeholder() -> None:
    trace = TraceRecorder()
    context = ChainContext(trace=trace)
    assert context.trace is trace


def test_chain_registry_roundtrip() -> None:
    chain = SequentialChain(nodes=())
    register_chain(chain)
    assert get_chain("sequential") is chain
    assert "sequential" in list_chains()


def test_chain_context_is_frozen() -> None:
    context = ChainContext()
    with pytest.raises(FrozenInstanceError):
        context.cwd = "/tmp"  # type: ignore[misc]
