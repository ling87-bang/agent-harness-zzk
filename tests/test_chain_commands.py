from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from harness.cli.app import app
from harness.cli.chain_commands import parse_chain_steps, run_chain_sequential
from harness.chain.nodes import LLMNode, PassThroughNode, SkillNode, TransformNode


def test_parse_chain_steps() -> None:
    nodes = parse_chain_steps("llm,skill:file_reader,passthrough,transform:upper")
    assert isinstance(nodes[0], LLMNode)
    assert isinstance(nodes[1], SkillNode)
    assert nodes[1].skill_name == "file_reader"
    assert isinstance(nodes[2], PassThroughNode)
    assert isinstance(nodes[3], TransformNode)


def test_parse_chain_steps_truncate() -> None:
    nodes = parse_chain_steps("transform:truncate:3")
    assert len(nodes) == 1
    node = nodes[0]
    assert isinstance(node, TransformNode)


def test_parse_chain_steps_empty_tokens_skipped() -> None:
    nodes = parse_chain_steps("passthrough,,llm")
    assert len(nodes) == 2


def test_parse_chain_steps_unknown_token_raises() -> None:
    with pytest.raises(ValueError, match="unknown step token"):
        parse_chain_steps("transform:upper_extra")


def test_parse_chain_steps_invalid_skill_raises() -> None:
    with pytest.raises(ValueError, match="invalid skill step"):
        parse_chain_steps("skill:")


def test_parse_chain_steps_truncate_invalid_int_raises() -> None:
    with pytest.raises(ValueError):
        parse_chain_steps("transform:truncate:abc")


def test_chain_list_cli() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["chain", "list"])
    assert result.exit_code == 0
    assert "llm" in result.stdout


def test_chain_run_sequential_without_api_key_passthrough_ok() -> None:
    runner = CliRunner()
    with patch("harness.cli.chain_commands.get_settings") as mock_settings:
        mock_settings.return_value.deepseek_api_key = ""
        mock_settings.return_value.app_name = "zzk"
        mock_settings.return_value.knowledge_provider = "mock"
        mock_settings.return_value.search_provider = "duckduckgo"
        mock_settings.return_value.search_api_key = ""
        result = runner.invoke(
            app,
            ["chain", "run", "sequential", "hi", "--steps", "passthrough,transform:upper"],
        )
    assert result.exit_code == 0
    assert "HI" in result.stdout
    assert "[chain:trace]" in result.stdout


def test_chain_run_sequential_llm_requires_api_key() -> None:
    runner = CliRunner()
    with patch("harness.cli.chain_commands.get_settings") as mock_settings:
        mock_settings.return_value.deepseek_api_key = ""
        mock_settings.return_value.app_name = "zzk"
        result = runner.invoke(
            app,
            ["chain", "run", "sequential", "hi", "--steps", "llm"],
        )
    assert result.exit_code == 1


def test_chain_run_sequential_passthrough_only() -> None:
    runner = CliRunner()
    with patch("harness.cli.chain_commands._build_runtime_context") as mock_context:
        from harness.chain.base import ChainContext
        from harness.skills.registry import SkillRegistry

        mock_context.return_value = ChainContext(registry=SkillRegistry.with_builtins_for_tests())
        result = runner.invoke(
            app,
            ["chain", "run", "sequential", "hello", "--steps", "passthrough,transform:upper"],
        )
    assert result.exit_code == 0
    assert "HELLO" in result.stdout


def test_run_chain_sequential_parse_error_exit_code() -> None:
    code = run_chain_sequential(steps="bogus", input_text="x")
    assert code == 1
