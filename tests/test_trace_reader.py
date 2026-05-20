import json

from harness.engine.trace import TraceRecorder
from harness.engine.trace_reader import list_trace_runs, load_trace_run
from harness.state import TraceStep


def test_list_and_show_trace_run(tmp_path) -> None:
    recorder = TraceRecorder(trace_dir=tmp_path)
    recorder.start_run(query="hello", llm_provider="fake", llm_model="fake-model")
    recorder.append_step(
        TraceStep(
            step=1,
            step_type="skill_execution",
            run_id=recorder.run_id,
            skill="file_reader",
            status="succeeded",
        )
    )
    recorder.finish_run(final_status="success")

    entries, _ = list_trace_runs(trace_dir=tmp_path, limit=5)
    assert len(entries) == 1
    assert entries[0].run_id == recorder.run_id
    assert entries[0].final_status == "success"
    assert entries[0].step_count == 1

    loaded = load_trace_run(recorder.run_id, trace_dir=tmp_path)
    assert loaded.error_code is None
    assert len(loaded.records) >= 2
    step_rows = [row for row in loaded.records if row.get("record_type") == "step"]
    assert step_rows[0]["skill"] == "file_reader"


def test_load_trace_run_prefix_match(tmp_path) -> None:
    recorder = TraceRecorder(trace_dir=tmp_path)
    recorder.start_run(query="q", llm_provider="fake", llm_model="m")
    recorder.finish_run(final_status="success")

    prefix = recorder.run_id[:8]
    loaded = load_trace_run(prefix, trace_dir=tmp_path)
    assert loaded.error_code is None
    assert loaded.path.name == f"{recorder.run_id}.jsonl"


def test_load_trace_run_not_found(tmp_path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    loaded = load_trace_run("run-missing", trace_dir=tmp_path)
    assert loaded.error_code == "trace_not_found"


def test_load_trace_run_ambiguous_prefix(tmp_path) -> None:
    for index in range(2):
        path = tmp_path / f"run-abc{index}.jsonl"
        path.write_text(
            json.dumps({"record_type": "run", "run_id": f"run-abc{index}", "query": "q"}) + "\n",
            encoding="utf-8",
        )

    loaded = load_trace_run("run-abc", trace_dir=tmp_path)
    assert loaded.error_code == "trace_ambiguous"
