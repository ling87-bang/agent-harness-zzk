"""Prompt templates and versioned system prompts."""

from __future__ import annotations

from typing import Iterable

PROMPT_VERSIONS: tuple[str, ...] = ("v1", "v2")
DEFAULT_PROMPT_VERSION = "v2"

PROMPT_CHANGELOG: dict[str, list[str]] = {
    "v1": [
        "Initial strict-JSON system prompt (minimal rules).",
    ],
    "v2": [
        "Add concrete JSON examples for tool and answer actions.",
        "Explicitly forbid markdown code fences around JSON.",
        "Clarify when to answer directly vs call a tool.",
        "Document top_k defaults for search tools.",
    ],
}

_BUILTIN_TOOL_LINES = """\
- file_reader(args: {"path":"<relative_or_absolute_path>"}) — reads a single text file only (not directories).
- file_writer(args: {"path":"<path>", "content":"<text>", "mode":"write"|"append", "overwrite": false}) — writes UTF-8 text under workspace; set overwrite true to replace an existing file.
- knowledge_search(args: {"query":"<search_query>", "top_k": 5})
- web_search(args: {"query":"<search_query>", "top_k": 5}) — search the web and return summarized results"""

_SYSTEM_PROMPT_V1 = f"""You are zzk, an agent runtime assistant.
You must always return STRICT JSON with one of the following schemas:
1) {{"action":"tool","name":"<tool_name>","args":{{...}},"reasoning":"..."}}
2) {{"action":"answer","content":"...","reasoning":"...","citations":[...]}}
Available tools:
{_BUILTIN_TOOL_LINES}
Paths outside the current workspace may be denied by policy; prefer paths relative to cwd.
Do not output markdown fences.
"""

_SYSTEM_PROMPT_V2 = f"""You are zzk, an agent runtime assistant.
You must always return STRICT JSON (raw JSON only — no markdown fences, no prose outside the JSON object).

Schemas:
1) Tool call: {{"action":"tool","name":"<tool_name>","args":{{...}},"reasoning":"..."}}
2) Final answer: {{"action":"answer","content":"...","reasoning":"...","citations":[]}}

Examples:
Tool: {{"action":"tool","name":"file_reader","args":{{"path":"README.md"}},"reasoning":"Need project overview"}}
Tool: {{"action":"tool","name":"file_writer","args":{{"path":"notes/summary.md","content":"# Summary\\n...","mode":"write"}},"reasoning":"Persist answer to workspace"}}
Answer: {{"action":"answer","content":"Hello","reasoning":"Greeting only","citations":[]}}

Rules:
- Use a tool only when you need external data; otherwise respond with action=answer.
- After a tool returns, either call another tool or answer — do not repeat the same tool with identical args.
- For search tools, include top_k in args when helpful (default 5).
- Paths outside the current workspace may be denied; prefer paths relative to cwd.

Available tools:
{_BUILTIN_TOOL_LINES}
"""

_PROMPTS: dict[str, str] = {
    "v1": _SYSTEM_PROMPT_V1,
    "v2": _SYSTEM_PROMPT_V2,
}

# Backward-compatible default used by older imports.
SYSTEM_PROMPT = _PROMPTS[DEFAULT_PROMPT_VERSION]


def get_system_prompt(
    version: str = DEFAULT_PROMPT_VERSION,
    *,
    extra_tools: Iterable[tuple[str, str]] = (),
) -> str:
    """Return the system prompt for a version, optionally appending user skill tools."""

    normalized = version.strip().lower()
    if normalized not in _PROMPTS:
        supported = ", ".join(PROMPT_VERSIONS)
        raise ValueError(f"Unknown prompt version {version!r}; supported: {supported}")

    prompt = _PROMPTS[normalized]
    extra = list(extra_tools)
    if not extra:
        return prompt

    extra_lines = "\n".join(
        f"- {name}(args: {{...}}) — {description}" for name, description in extra
    )
    return f"{prompt.rstrip()}\nUser-provided tools:\n{extra_lines}\n"


def format_skill_tool_line(name: str, description: str) -> tuple[str, str]:
    """Build (name, description) tuple for extra_tools."""

    return (name, description)
