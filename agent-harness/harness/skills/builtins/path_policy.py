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


def _drive_letter(path: Path) -> str | None:
    """Return drive letter (e.g. ``C:``) for native or ``E:/...``-style paths."""

    if path.drive:
        return path.drive.upper()
    if path.parts:
        head = path.parts[0]
        if len(head) >= 2 and head[1] == ":" and head[0].isalpha():
            return f"{head[0].upper()}:"
    posix = path.as_posix().replace("\\", "/")
    if len(posix) >= 2 and posix[1] == ":" and posix[0].isalpha():
        return f"{posix[0].upper()}:"
    return None


def _posix_prefix_under(candidate: Path, root: Path) -> bool:
    candidate_posix = candidate.as_posix().replace("\\", "/").lower()
    root_posix = root.as_posix().replace("\\", "/").lower().rstrip("/")
    if not root_posix:
        return False
    return candidate_posix == root_posix or candidate_posix.startswith(f"{root_posix}/")


def _path_is_under(candidate: Path, root: Path) -> bool:
    cand_drive = _drive_letter(candidate)
    root_drive = _drive_letter(root)
    if cand_drive and root_drive and cand_drive == root_drive:
        if _posix_prefix_under(candidate, root):
            return True
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        pass
    return _posix_prefix_under(candidate, root)


def is_in_blocked_dirs(target: Path) -> bool:
    home = Path.home().resolve()
    blocked: list[Path] = [home / ".ssh", home / ".zzk"]

    drive = _drive_letter(target)
    if drive:
        system_drive = os.environ.get("SystemDrive", drive)
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
        if _path_is_under(target, blocked_dir):
            return True
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
