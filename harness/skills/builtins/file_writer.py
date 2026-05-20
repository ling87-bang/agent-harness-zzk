"""Builtin file writer skill with path safety checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from harness.errors import ERROR_PATH_DENIED, ERROR_TOOL_CRASH
from harness.skills.base import SkillResult
from harness.skills.builtins.path_policy import resolve_workspace_path, validate_write_path

WriteMode = Literal["write", "append"]


@dataclass(frozen=True, slots=True)
class _FileWriterSkill:
    """Write or append UTF-8 text under workspace allowlist."""

    @property
    def name(self) -> str:
        return "file_writer"

    @property
    def description(self) -> str:
        return "Write or append UTF-8 text to a file under workspace allowlist."

    async def execute(self, **kwargs: Any) -> SkillResult:
        path_value = kwargs.get("path")
        content = kwargs.get("content")
        mode_value = kwargs.get("mode", "write")
        overwrite_value = kwargs.get("overwrite", False)
        cwd_value = kwargs.get("cwd")

        if not isinstance(path_value, str) or not path_value.strip():
            return SkillResult(output="", error_code=ERROR_TOOL_CRASH)
        if not isinstance(content, str):
            return SkillResult(output="", error_code=ERROR_TOOL_CRASH)
        if mode_value not in {"write", "append"}:
            return SkillResult(output="", error_code=ERROR_TOOL_CRASH)

        mode: WriteMode = mode_value
        overwrite = bool(overwrite_value) if isinstance(overwrite_value, bool) else False
        append = mode == "append"

        resolved = resolve_workspace_path(path_value, cwd_value)
        if resolved is None:
            return SkillResult(output="", error_code=ERROR_PATH_DENIED)
        _base_dir, target = resolved

        content_bytes = len(content.encode("utf-8"))
        denied_reason = validate_write_path(
            base_dir=_base_dir,
            target=target,
            content_bytes=content_bytes,
            overwrite=overwrite,
            append=append,
        )
        if denied_reason is not None:
            return SkillResult(
                output=f"path denied: {denied_reason}",
                metadata={"reason": denied_reason, "path": str(target)},
                error_code=ERROR_PATH_DENIED,
            )

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if append:
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(content)
            else:
                target.write_text(content, encoding="utf-8")
            written_bytes = target.stat().st_size
        except OSError:
            return SkillResult(output="", error_code=ERROR_TOOL_CRASH)

        action = "appended" if append else "wrote"
        return SkillResult(
            output=f"{action} {content_bytes} bytes to {target.name}",
            metadata={
                "path": str(target),
                "bytes": written_bytes,
                "mode": mode,
                "overwrite": overwrite,
            },
        )


file_writer = _FileWriterSkill()
