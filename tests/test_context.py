from harness.engine.context import ConversationManager
from harness.state import Message


def test_context_save_and_load(tmp_path) -> None:
    manager = ConversationManager(storage_dir=tmp_path)
    conversation_id = "conv-1"
    original = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="world"),
    ]
    assert manager.save_history(conversation_id, original) is True
    loaded = manager.load_history(conversation_id)
    assert loaded == original


def test_context_compress_message_count() -> None:
    manager = ConversationManager(max_messages=4, max_chars=10_000)
    messages = [Message(role="user", content=f"m{i}") for i in range(10)]
    compressed = manager.compress_history(messages)
    assert len(compressed) <= 5
    assert compressed[0].role == "system"


def test_context_compress_char_budget() -> None:
    manager = ConversationManager(max_messages=20, max_chars=20)
    messages = [
        Message(role="user", content="a" * 10),
        Message(role="assistant", content="b" * 10),
        Message(role="user", content="c" * 10),
        Message(role="assistant", content="d" * 10),
    ]
    compressed = manager.compress_history(messages)
    assert sum(len(item.content) for item in compressed) <= 20


def test_context_load_invalid_json_returns_empty(tmp_path) -> None:
    manager = ConversationManager(storage_dir=tmp_path)
    path = tmp_path / "conv-bad.json"
    path.write_text("{invalid", encoding="utf-8")
    loaded = manager.load_history("conv-bad")
    assert loaded == []


def test_context_save_io_error_returns_false(tmp_path, monkeypatch) -> None:
    manager = ConversationManager(storage_dir=tmp_path)
    monkeypatch.setattr(
        "pathlib.Path.write_text",
        lambda self, text, encoding=None: (_ for _ in ()).throw(OSError("disk full")),
    )
    ok = manager.save_history("conv-io", [Message(role="user", content="x")])
    assert ok is False


def test_context_compress_keeps_summary_message() -> None:
    manager = ConversationManager(max_messages=4, max_chars=25)
    messages = [Message(role="user", content=f"m{i}-" + ("x" * 8)) for i in range(10)]
    compressed = manager.compress_history(messages)
    assert compressed[0].role == "system"
