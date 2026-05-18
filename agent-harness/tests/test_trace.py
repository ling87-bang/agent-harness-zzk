import json

from harness.engine.trace import TraceRecorder
from harness.errors import ERROR_CODE_MAP
from harness.state import TraceStep


def test_trace_recorder_writes_records(tmp_path) -> None:
    recorder = TraceRecorder(trace_dir=tmp_path)
    recorder.start_run(query="q", llm_provider="fake", llm_model="m", conversation_id="conv-x")
    recorder.append_step(
        TraceStep(
            step=1,
            step_type="llm_call",
            run_id=recorder.run_id,
            status="failed",
            error_code="llm_error",
            error="boom",
        )
    )
    recorder.finish_run(final_status="error")

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["record_type"] == "run"
    assert parsed[0]["conversation_id"] == "conv-x"
    assert parsed[1]["record_type"] == "step"
    assert parsed[1]["step_type"] == "llm_call"


def test_trace_has_standard_error_codes() -> None:
    assert "parse_failed" in ERROR_CODE_MAP
    assert "llm_error" in ERROR_CODE_MAP
