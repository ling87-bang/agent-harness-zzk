"""Builtin file reader skill with path safety checks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from harness.errors import ERROR_PATH_DENIED, ERROR_TOOL_CRASH
from harness.skills.base import SkillResult

MAX_FILE_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _FileReaderSkill:
    """Read UTF-8 text files under working-directory allowlist."""

    @property
    def name(self) -> str:
        return "file_reader"

    @property
    def description(self) -> str:
        return "Read local text file content under workspace allowlist."

    async def execute(self, **kwargs: Any) -> SkillResult:
        path_value = kwargs.get("path")
        cwd_value = kwargs.get("cwd")
        if not isinstance(path_value, str) or not path_value.strip():
            return SkillResult(output="", error_code=ERROR_TOOL_CRASH)

        base_dir = Path(cwd_value).resolve() if isinstance(cwd_value, str) else Path.cwd().resolve()
        try:
            target = (base_dir / path_value).resolve() if not Path(path_value).is_absolute() else Path(path_value).resolve()
        except OSError:
            return SkillResult(output="", error_code=ERROR_PATH_DENIED)

        denied_reason = _validate_path(base_dir=base_dir, target=target)
        if denied_reason is not None:
            return SkillResult(
                output=f"path denied: {denied_reason}",
                metadata={"reason": denied_reason, "path": str(target)},
                error_code=ERROR_PATH_DENIED,
            )

        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return SkillResult(output="", error_code=ERROR_TOOL_CRASH)

        return SkillResult(
            output=content,
            metadata={"path": str(target), "bytes": target.stat().st_size},
        )


def _validate_path(base_dir: Path, target: Path) -> str | None:
    try:
        target.relative_to(base_dir)
    except ValueError:
        return "path_outside_allowlist"

    if _is_in_blocked_dirs(target):
        return "path_in_blocked_directory"

    if not target.exists() or not target.is_file():
        return "path_not_readable_file"
    try:
        if target.stat().st_size > MAX_FILE_BYTES:
            return "file_too_large"
    except OSError:
        return "path_not_readable_file"
    return None


def _is_in_blocked_dirs(target: Path) -> bool:
    home = Path.home().resolve()
    blocked: list[Path] = [home / ".ssh", home / ".zzk"]

    if target.drive:
        system_drive = os.environ.get("SystemDrive", target.drive or "C:")
        blocked.extend(
            [
                Path(f"{system_drive}/Windows"),
                Path(f"{system_drive}/Program Files"),
                Path(f"{system_drive}/Program Files (x86)"),
            ]
        )
    else:
        blocked.extend([Path("/etc"), Path("/sys"), Path("/proc"), Path("/dev")])

    for blocked_dir in blocked:
        blocked_resolved = blocked_dir.resolve()
        try:
            target.relative_to(blocked_resolved)
            return True
        except ValueError:
            continue
    return False


file_reader = _FileReaderSkill()
