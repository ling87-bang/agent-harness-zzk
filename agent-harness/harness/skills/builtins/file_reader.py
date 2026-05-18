"""Builtin file reader skill with path safety checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.errors import ERROR_PATH_DENIED, ERROR_TOOL_CRASH
from harness.skills.base import SkillResult
from harness.skills.builtins.path_policy import (
    MAX_FILE_BYTES,
    resolve_workspace_path,
    validate_read_path,
)

__all__ = ["MAX_FILE_BYTES", "file_reader"]


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

        resolved = resolve_workspace_path(path_value, cwd_value)
        if resolved is None:
            return SkillResult(output="", error_code=ERROR_PATH_DENIED)
        base_dir, target = resolved

        denied_reason = validate_read_path(base_dir=base_dir, target=target)
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


file_reader = _FileReaderSkill()
