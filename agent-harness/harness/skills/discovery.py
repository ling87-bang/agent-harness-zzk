"""Discover and load user skills from filesystem directories."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

from harness.skills.base import Skill

LOGGER = logging.getLogger(__name__)

DEFAULT_USER_SKILLS_DIR = Path.home() / ".zzk" / "skills"


def default_user_skill_dirs() -> tuple[Path, ...]:
    """Return default directories scanned when user skills are enabled."""

    return (DEFAULT_USER_SKILLS_DIR,)


def discover_skills_from_directory(directory: Path) -> dict[str, Skill]:
    """Load skills from ``*.py`` files in *directory* (non-recursive)."""

    if not directory.is_dir():
        return {}

    discovered: dict[str, Skill] = {}
    for module_path in sorted(directory.glob("*.py")):
        if module_path.name.startswith("_"):
            continue
        skill = _load_skill_from_file(module_path)
        if skill is None:
            continue
        if skill.name in discovered:
            LOGGER.warning(
                "duplicate skill name %s in %s; keeping first definition",
                skill.name,
                module_path,
            )
            continue
        discovered[skill.name] = skill
    return discovered


def discover_user_skills(extra_dirs: tuple[Path, ...] | None = None) -> dict[str, Skill]:
    """Scan default and extra directories for user skills."""

    directories = list(default_user_skill_dirs())
    if extra_dirs:
        directories.extend(extra_dirs)

    merged: dict[str, Skill] = {}
    seen_dirs: set[Path] = set()
    for directory in directories:
        resolved = directory.expanduser().resolve()
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)
        for name, skill in discover_skills_from_directory(resolved).items():
            if name in merged:
                LOGGER.warning("duplicate skill name %s across dirs; keeping first", name)
                continue
            merged[name] = skill
    return merged


def _load_skill_from_file(module_path: Path) -> Skill | None:
    module_name = f"zzk_user_skill_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        LOGGER.warning("failed to create module spec for %s", module_path)
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        LOGGER.exception("failed to import user skill module %s", module_path)
        return None
    finally:
        sys.modules.pop(module_name, None)

    candidate = getattr(module, "skill", None)
    if _is_skill(candidate):
        return candidate

    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        value = getattr(module, attr_name)
        if _is_skill(value):
            return value

    LOGGER.warning("no skill export found in %s (expected module-level `skill`)", module_path)
    return None


def _is_skill(value: Any) -> bool:
    if value is None:
        return False
    return (
        hasattr(value, "name")
        and hasattr(value, "description")
        and callable(getattr(value, "execute", None))
    )
