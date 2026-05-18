import pytest

from harness.errors import ERROR_PATH_DENIED, ERROR_TOOL_CRASH
from harness.skills.builtins.file_writer import file_writer


@pytest.mark.asyncio()
async def test_file_writer_creates_new_file(tmp_path) -> None:
    target = tmp_path / "out" / "summary.md"

    result = await file_writer.execute(
        path="out/summary.md",
        content="# Summary\nhello",
        mode="write",
        cwd=str(tmp_path),
    )
    assert result.error_code is None
    assert target.read_text(encoding="utf-8") == "# Summary\nhello"
    assert result.metadata["mode"] == "write"


@pytest.mark.asyncio()
async def test_file_writer_denies_existing_without_overwrite(tmp_path) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("old", encoding="utf-8")

    result = await file_writer.execute(
        path="existing.txt",
        content="new",
        mode="write",
        cwd=str(tmp_path),
    )
    assert result.error_code == ERROR_PATH_DENIED
    assert result.metadata["reason"] == "file_exists"
    assert target.read_text(encoding="utf-8") == "old"


@pytest.mark.asyncio()
async def test_file_writer_overwrite_existing(tmp_path) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("old", encoding="utf-8")

    result = await file_writer.execute(
        path="existing.txt",
        content="new",
        mode="write",
        overwrite=True,
        cwd=str(tmp_path),
    )
    assert result.error_code is None
    assert target.read_text(encoding="utf-8") == "new"


@pytest.mark.asyncio()
async def test_file_writer_append(tmp_path) -> None:
    target = tmp_path / "log.txt"
    target.write_text("a", encoding="utf-8")

    result = await file_writer.execute(
        path="log.txt",
        content="b",
        mode="append",
        cwd=str(tmp_path),
    )
    assert result.error_code is None
    assert target.read_text(encoding="utf-8") == "ab"


@pytest.mark.asyncio()
async def test_file_writer_denies_outside_allowlist(tmp_path) -> None:
    outside = tmp_path.parent / "outside.txt"

    result = await file_writer.execute(
        path=str(outside),
        content="x",
        mode="write",
        cwd=str(tmp_path),
    )
    assert result.error_code == ERROR_PATH_DENIED
    assert result.metadata["reason"] == "path_outside_allowlist"


@pytest.mark.asyncio()
async def test_file_writer_denies_directory_target(tmp_path) -> None:
    target_dir = tmp_path / "project"
    target_dir.mkdir()

    result = await file_writer.execute(
        path="project",
        content="x",
        mode="write",
        cwd=str(tmp_path),
    )
    assert result.error_code == ERROR_PATH_DENIED
    assert result.metadata["reason"] == "path_is_directory"


@pytest.mark.asyncio()
async def test_file_writer_invalid_args_return_tool_crash(tmp_path) -> None:
    result = await file_writer.execute(path="a.txt", content=123, cwd=str(tmp_path))
    assert result.error_code == ERROR_TOOL_CRASH
