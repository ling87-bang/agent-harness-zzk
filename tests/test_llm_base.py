from harness.llm.base import messages_to_chat_api_payload, parse_llm_response
from harness.state import Message


def test_messages_to_chat_api_payload_maps_tool_role_to_user() -> None:
    payload = messages_to_chat_api_payload(
        [
            Message(role="assistant", content='{"action":"tool"}'),
            Message(role="tool", content='{"output":"ok"}'),
        ]
    )
    assert payload[-1]["role"] == "user"
    assert payload[-1]["content"].startswith("[tool_observation]")
    assert '{"output":"ok"}' in payload[-1]["content"]
    assert all(item["role"] != "tool" for item in payload)


def test_parse_llm_response_answer_schema() -> None:
    parsed = parse_llm_response(
        '{"action":"answer","content":"ok","reasoning":"r","citations":[{"source":"x"}]}'
    )
    assert parsed.action == "answer"
    assert parsed.content == "ok"
    assert parsed.error_code is None
    assert parsed.citations == [{"source": "x"}]


def test_parse_llm_response_tool_schema() -> None:
    parsed = parse_llm_response('{"action":"tool","name":"file_reader","args":{"path":"a.txt"}}')
    assert parsed.action == "tool"
    assert parsed.name == "file_reader"
    assert parsed.args == {"path": "a.txt"}


def test_parse_llm_response_with_embedded_json() -> None:
    parsed = parse_llm_response(
        'noise before {"action":"answer","content":"embedded","reasoning":"r","citations":[]} noise after'
    )
    assert parsed.action == "answer"
    assert parsed.content == "embedded"
    assert parsed.error_code is None


def test_parse_llm_response_fallback_parse_failed() -> None:
    parsed = parse_llm_response("plain text output")
    assert parsed.action == "answer"
    assert parsed.content == "plain text output"
    assert parsed.error_code == "parse_failed"


def test_parse_llm_response_invalid_tool_shape_falls_back() -> None:
    parsed = parse_llm_response('{"action":"tool","name":"bad","args":[]}')
    assert parsed.action == "answer"
    assert parsed.error_code == "parse_failed"


def test_parse_llm_response_embedded_nested_json_object() -> None:
    parsed = parse_llm_response(
        'prefix {"action":"answer","content":"ok","reasoning":"r","citations":[{"meta":{"a":1}}]} suffix'
    )
    assert parsed.action == "answer"
    assert parsed.content == "ok"
