import importlib

import pytest

from harness.errors import ERROR_PATH_DENIED, ERROR_TOOL_CRASH
from harness.skills.builtins.file_reader import _is_in_blocked_dirs, file_reader

file_reader_module = importlib.import_module("harness.skills.builtins.file_reader")


@pytest.mark.asyncio()
async def test_file_reader_reads_allowed_file(tmp_path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")

    result = await file_reader.execute(path="a.txt", cwd=str(tmp_path))
    assert result.error_code is None
    assert result.output == "hello"


@pytest.mark.asyncio()
async def test_file_reader_denies_outside_allowlist(tmp_path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("x", encoding="utf-8")

    result = await file_reader.execute(path=str(outside), cwd=str(tmp_path))
    assert result.error_code == ERROR_PATH_DENIED
    assert result.metadata["reason"] == "path_outside_allowlist"


@pytest.mark.asyncio()
async def test_file_reader_denies_large_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(file_reader_module, "MAX_FILE_BYTES", 1)
    target = tmp_path / "big.txt"
    target.write_text("hello", encoding="utf-8")

    result = await file_reader.execute(path="big.txt", cwd=str(tmp_path))
    assert result.error_code == ERROR_PATH_DENIED
    assert result.metadata["reason"] == "file_too_large"


@pytest.mark.asyncio()
async def test_file_reader_invalid_path_arg_returns_tool_crash(tmp_path) -> None:
    result = await file_reader.execute(path=123, cwd=str(tmp_path))
    assert result.error_code == ERROR_TOOL_CRASH


@pytest.mark.asyncio()
async def test_file_reader_denies_blocked_home_ssh(monkeypatch, tmp_path) -> None:
    fake_home = tmp_path / "home"
    ssh_dir = fake_home / ".ssh"
    ssh_dir.mkdir(parents=True)
    target = ssh_dir / "id_rsa"
    target.write_text("secret", encoding="utf-8")

    monkeypatch.setattr(file_reader_module.Path, "home", staticmethod(lambda: fake_home))
    result = await file_reader.execute(path=str(target), cwd=str(fake_home))
    assert result.error_code == ERROR_PATH_DENIED
    assert result.metadata["reason"] == "path_in_blocked_directory"


@pytest.mark.asyncio()
async def test_file_reader_windows_blocked_dir_uses_system_drive(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SystemDrive", "E:")
    blocked = file_reader_module.Path("E:/Windows/secret.txt")
    assert _is_in_blocked_dirs(blocked) is True
