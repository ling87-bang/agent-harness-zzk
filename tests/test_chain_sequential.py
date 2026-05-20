from __future__ import annotations

import pytest

from harness.chain.base import ChainContext
from harness.chain.nodes import PassThroughNode, TransformNode
from harness.chain.sequential import SequentialChain
from harness.errors import ERROR_TOOL_CRASH


@pytest.mark.asyncio()
async def test_sequential_chain_empty_passthrough_input() -> None:
    chain = SequentialChain(nodes=())
    result = await chain.run("input", ChainContext())
    assert result.output == "input"
    assert result.error_code is None
    assert result.sub_results == ()


@pytest.mark.asyncio()
async def test_sequential_chain_pipes_output() -> None:
    chain = SequentialChain(
        nodes=(
            PassThroughNode(),
            TransformNode(transform=str.upper, node_name="upper"),
        )
    )
    result = await chain.run("hi", ChainContext())
    assert result.output == "HI"
    assert len(result.sub_results) == 2


@pytest.mark.asyncio()
async def test_sequential_chain_stops_on_error() -> None:
    def _boom(_: str) -> str:
        raise ValueError("fail")

    chain = SequentialChain(
        nodes=(
            PassThroughNode(),
            TransformNode(transform=_boom),
            PassThroughNode(),
        )
    )
    result = await chain.run("x", ChainContext())
    assert result.error_code == ERROR_TOOL_CRASH
    assert len(result.sub_results) == 2
