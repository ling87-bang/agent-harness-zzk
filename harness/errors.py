"""Centralized error code definitions."""

from __future__ import annotations

from types import MappingProxyType

ERROR_PARSE_FAILED = "parse_failed"
ERROR_TOOL_TIMEOUT = "tool_timeout"
ERROR_TOOL_CRASH = "tool_crash"
ERROR_PATH_DENIED = "path_denied"
ERROR_TARGET_UNREACHABLE = "target_unreachable"
ERROR_MAX_STEPS = "max_steps"
ERROR_LLM_ERROR = "llm_error"
ERROR_UNKNOWN_TOOL = "unknown_tool"
ERROR_CHAIN_ROUTE_MISS = "chain_route_miss"
ERROR_SEARCH_MISCONFIGURED = "search_misconfigured"

ERROR_CODE_MAP = MappingProxyType(
    {
        ERROR_PARSE_FAILED: "LLM output format is invalid and degraded.",
        ERROR_TOOL_TIMEOUT: "Tool execution exceeded timeout.",
        ERROR_TOOL_CRASH: "Tool execution raised an exception.",
        ERROR_PATH_DENIED: "Path access denied by allowlist policy.",
        ERROR_TARGET_UNREACHABLE: "Target backend is not reachable.",
        ERROR_MAX_STEPS: "Agent reached maximum step limit.",
        ERROR_LLM_ERROR: "LLM request failed.",
        ERROR_UNKNOWN_TOOL: "Tool requested by model is not registered.",
        ERROR_CHAIN_ROUTE_MISS: "Router chain has no matching branch.",
        ERROR_SEARCH_MISCONFIGURED: "Search provider requires configuration.",
    }
)
