from typer.testing import CliRunner

from harness.cli.app import app
from harness.engine.trace import TraceRecorder
from harness.state import TraceStep


def test_cli_trace_list_and_show(tmp_path) -> None:
    recorder = TraceRecorder(trace_dir=tmp_path)
    recorder.start_run(query="trace cli", llm_provider="fake", llm_model="m")
    recorder.append_step(
        TraceStep(
            step=1,
            step_type="llm_call",
            run_id=recorder.run_id,
            status="succeeded",
            latency_ms=3.0,
        )
    )
    recorder.finish_run(final_status="success")

    runner = CliRunner()
    list_result = runner.invoke(
        app,
        ["trace", "list", "--trace-dir", str(tmp_path), "--limit", "5"],
    )
    assert list_result.exit_code == 0
    assert recorder.run_id in list_result.stdout

    show_result = runner.invoke(
        app,
        ["trace", "show", recorder.run_id, "--trace-dir", str(tmp_path)],
    )
    assert show_result.exit_code == 0
    assert "llm_call" in show_result.stdout


def test_cli_trace_show_missing_run(tmp_path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["trace", "show", "run-missing", "--trace-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "not found" in result.stdout
