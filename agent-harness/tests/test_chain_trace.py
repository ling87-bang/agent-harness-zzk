from __future__ import annotations

import json

import pytest

from harness.chain import ChainContext
from harness.chain.nodes import PassThroughNode, TransformNode
from harness.chain.sequential import SequentialChain
from harness.chain.tracing import finish_chain_trace, start_chain_trace
from harness.engine.trace import TraceRecorder


@pytest.mark.asyncio()
async def test_sequential_chain_writes_chain_node_trace(tmp_path) -> None:
    trace = TraceRecorder(trace_dir=tmp_path)
    context = ChainContext(trace=trace)
    chain = SequentialChain(
        nodes=(
            PassThroughNode(),
            TransformNode(transform=str.upper, node_name="upper"),
        )
    )
    start_chain_trace(trace, "hello", context, chain_name=chain.name)
    result = await chain.run("hello", context)
    finish_chain_trace(trace, error_code=result.error_code)

    lines = trace.path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    run_header = next(row for row in records if row.get("record_type") == "run")
    assert run_header["query"] == "hello"
    steps = [row for row in records if row.get("record_type") == "step"]
    assert len(steps) == 2
    assert all(step["step_type"] == "chain_node" for step in steps)
    assert steps[0]["skill"] == "passthrough"
    assert steps[1]["skill"] == "upper"
    summary = next(row for row in records if row.get("record_type") == "run_summary")
    assert summary["final_status"] == "success"
