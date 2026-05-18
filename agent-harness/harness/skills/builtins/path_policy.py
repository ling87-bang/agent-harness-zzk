"""Shared path allowlist and blocked-directory policy for file skills."""

from __future__ import annotations

import os
from pathlib import Path

MAX_FILE_BYTES = 10 * 1024 * 1024


def resolve_workspace_path(
    path_value: str,
    cwd: str | Path | None = None,
) -> tuple[Path, Path] | None:
    """Resolve *path_value* under workspace *cwd*; return ``(base_dir, target)`` or ``None``."""

    base_dir = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
    try:
        if Path(path_value).is_absolute():
            target = Path(path_value).resolve()
        else:
            target = (base_dir / path_value).resolve()
    except OSError:
        return None
    return base_dir, target


def is_in_blocked_dirs(target: Path) -> bool:
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


def _within_allowlist(base_dir: Path, target: Path) -> bool:
    try:
        target.relative_to(base_dir)
        return True
    except ValueError:
        return False


def validate_read_path(base_dir: Path, target: Path) -> str | None:
    if not _within_allowlist(base_dir, target):
        return "path_outside_allowlist"
    if is_in_blocked_dirs(target):
        return "path_in_blocked_directory"
    if target.exists() and target.is_dir():
        return "path_is_directory"
    if not target.exists() or not target.is_file():
        return "path_not_readable_file"
    try:
        if target.stat().st_size > MAX_FILE_BYTES:
            return "file_too_large"
    except OSError:
        return "path_not_readable_file"
    return None


def validate_write_path(
    base_dir: Path,
    target: Path,
    *,
    content_bytes: int,
    overwrite: bool,
    append: bool,
) -> str | None:
    if content_bytes > MAX_FILE_BYTES:
        return "content_too_large"
    if not _within_allowlist(base_dir, target):
        return "path_outside_allowlist"
    if is_in_blocked_dirs(target):
        return "path_in_blocked_directory"
    if target.exists() and target.is_dir():
        return "path_is_directory"
    if target.exists() and target.is_file():
        if append:
            try:
                if target.stat().st_size + content_bytes > MAX_FILE_BYTES:
                    return "file_too_large"
            except OSError:
                return "path_not_writable"
        elif not overwrite:
            return "file_exists"
    else:
        parent = target.parent
        if not _within_allowlist(base_dir, parent) and parent != base_dir:
            return "path_outside_allowlist"
        if is_in_blocked_dirs(parent):
            return "path_in_blocked_directory"
    return None
