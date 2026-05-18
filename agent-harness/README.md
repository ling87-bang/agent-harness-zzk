# Agent Harness (zzk)

Lightweight agent runtime CLI with strict JSON ReAct protocol, builtin skills, and trace logging.

## Current Scope

- Phase 1: single-turn `zzk "hello"` with streaming output
- Phase 2: builtin `file_reader` skill + path safety + tool timeout
- Phase 3: `zzk chat` multi-turn conversations with persisted context
- Phase 4: `knowledge_search` via configurable knowledge backend target

## Install

```bash
python -m pip install .
```

For development:

```bash
python -m pip install ".[dev]"
```

## Configuration

Copy `.env.example` to `.env` and fill values:

```env
ZZK_DEEPSEEK_API_KEY=your_deepseek_api_key
ZZK_DEEPSEEK_BASE_URL=https://api.deepseek.com
ZZK_DEEPSEEK_MODEL=deepseek-chat
ZZK_DEEPSEEK_TIMEOUT_SECONDS=30.0

ZZK_KNOWLEDGE_BASE_URL=http://127.0.0.1:8000/knowledge
ZZK_KNOWLEDGE_API_KEY=
ZZK_KNOWLEDGE_TIMEOUT_SECONDS=10.0
```

## Usage

Single-turn:

```bash
zzk "你好"
zzk "读取当前目录 README.md"
zzk "查知识库 phase4 架构"
```

Interactive chat:

```bash
zzk chat
zzk chat --conversation-id conv-123456789abc
```

In chat mode, type `exit`, `quit`, or `:q` to leave.

## Trace and Conversation Files

- Traces: `~/.zzk/traces/*.jsonl`
- Conversations: `~/.zzk/conversations/*.json`

Trace contains `record_type` (`run`/`step`) and `step_type` (`llm_call`/`parse_result`/`skill_execution`) with standardized `error_code`.
