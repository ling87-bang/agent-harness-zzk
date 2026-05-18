from __future__ import annotations

import pytest

from harness.chain.base import ChainContext
from harness.chain.nodes import PassThroughNode, TransformNode
from harness.chain.router import RouterChain
from harness.errors import ERROR_CHAIN_ROUTE_MISS


@pytest.mark.asyncio()
async def test_router_chain_selects_branch() -> None:
    chain = RouterChain(
        routes={
            "file": TransformNode(transform=str.upper, node_name="upper"),
            "default": PassThroughNode(),
        },
        predicate=lambda text, _: "file" if "readme" in text.lower() else "default",
    )
    result = await chain.run("read readme", ChainContext())
    assert result.output == "READ README"
    assert result.metadata.get("route_key") == "file"


@pytest.mark.asyncio()
async def test_router_chain_uses_default_key() -> None:
    chain = RouterChain(
        routes={"only": PassThroughNode()},
        predicate=lambda _text, _: "missing",
        default_key="only",
    )
    result = await chain.run("x", ChainContext())
    assert result.output == "x"


@pytest.mark.asyncio()
async def test_router_chain_miss_returns_error_code() -> None:
    chain = RouterChain(
        routes={"a": PassThroughNode()},
        predicate=lambda _text, _: "b",
    )
    result = await chain.run("input", ChainContext())
    assert result.error_code == ERROR_CHAIN_ROUTE_MISS
