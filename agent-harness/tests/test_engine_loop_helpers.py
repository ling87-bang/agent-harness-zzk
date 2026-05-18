from harness.engine.loop import _decode_json_string_partial, _extract_json_string_value


def test_decode_json_string_partial_handles_escapes_and_unicode() -> None:
    raw = '"line\\nvalue \\u4f60\\u597d \\\\ slash\\/"'
    decoded = _decode_json_string_partial(raw, 0)
    assert "line\nvalue" in decoded
    assert "你好" in decoded
    assert "\\" in decoded
    assert "/" in decoded


def test_decode_json_string_partial_invalid_unicode_is_tolerated() -> None:
    raw = '"x\\uZZZZy"'
    decoded = _decode_json_string_partial(raw, 0)
    assert decoded == "xy"


def test_extract_json_string_value_missing_or_non_string() -> None:
    assert _extract_json_string_value('{"a":1}', "a") is None
    assert _extract_json_string_value("{}", "missing") is None
