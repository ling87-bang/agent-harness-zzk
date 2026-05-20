# Agent Harness (zzk) — 架构设计与实施计划

> 这是对原有 RAG Optimization Studio 的转向。原评测框架的设计存档于 `archive/rag-optimization-studio/`。

---

## 1. 背景与目标

### 1.1 项目定位

从 RAG 后端（KnowledgeOps Copilot）向上走一层，构建 **Agent 运行时框架**。不是一个 API 后端，不是一个评测工具——是 Agent 的骨架本身。

### 1.2 目标岗位覆盖

| 岗位 | 匹配点 |
|------|--------|
| 美团 agent 开发 | Harness、Chain、Skills 插件系统、上下文管理 |
| 腾讯 AI 开发 | Agent 系统设计、Prompt 工程、知识库集成 |
| 字节 AI Agent | Trace 追踪、Skill 系统、CLI 工具链（部分匹配） |

### 1.3 设计原则

1. **你自己会用** — 如果 zzk 不能让你在日常工作中比打开 DeepSeek 网页更爽，就是失败
2. **零强制依赖** — 不需要 PostgreSQL、Redis、甚至不需要 KnowledgeOps 在后台
3. **Skill 是第一等公民** — 插件化不是后加的，而是核心架构
4. **渐进交付** — 每个 Phase 产出可演示的垂直切片
5. **协议刚性** — 工具调用走严格 JSON schema，解析失败走确定性 fallback

### 1.4 与 KnowledgeOps Copilot 的关系

```
KnowledgeOps Copilot          Agent Harness (zzk)
────────────────────          ───────────────────
FastAPI RAG 后端              Agent 运行时框架
(已存在)                      (新建)

关系：
  Harness 作为"大脑":
    - 理解用户意图
    - 规划步骤
    - 选择技能

  Copilot 作为"知识库":
    - knowledge_search 技能通过 HTTP 调用 copilot
    - 这是 Harness 的可选技能，非必须

  Harness 不依赖 copilot 进程存在:
    - 纯聊天、读文件等操作完全独立
    - 只有需要查知识库时才需要 copilot
```

---

## 2. 系统架构

### 2.1 分层结构

```
┌──────────────────────────────────────────────────────────────────┐
│                        CLI 层 (harness/cli/)                     │
│     zzk run "query"    zzk chat    zzk eval    zzk chain list   │
│                     typer 入口 · 流式渲染 · 错误展示              │
└──────────┬───────────────────────────────┬───────────────────────┘
           │ 调用                           │ 调用
┌──────────▼───────────────────────────────▼───────────────────────┐
│                   编排层 (双引擎)                                │
│                                                                │
│   ReAct 循环 (engine/loop.py)    Chain 编排 (chain/)            │
│   ──────────────────────────    ─────────────────────           │
│   自主决策: LLM→工具→观察→循环   确定性流程: 节点→节点→节点        │
│   上下文管理·Trace记录           顺序链·路由链·可嵌套             │
│   适合：LLM 自主判断             适合：固定流程复用               │
└──────┬──────────────────────────────┬────────────────────────────┘
       │ 调用 LLM                      │ 调用 Skill
┌──────▼──────────────┐   ┌───────────▼────────────────────────────┐
│   LLM 层            │   │   Skill 层                             │
│  (harness/llm/)     │   │  (harness/skills/)                    │
│                     │   │                                        │
│  Provider 协议      │   │  Skill 协议 · 自动发现 · 注册中心       │
│  DeepSeek 实现      │   │  ├─ builtins/     file_reader         │
│  (可扩展多模型)     │   │  │               knowledge_search     │
│                     │   │  │               web_search  ← 新增   │
│                     │   │  ├─ user/         (用户，需 --enable) │
│                     │   │  └─ Target 适配器层                    │
│                     │   │       ├─ KnowledgeTarget → Copilot    │
│                     │   │       ├─ SearchTarget   → Web         │
│                     │   │       └─ MockTarget     → 测试/离线   │
└─────────────────────┘   └────────────────────────────────────────┘
```

### 2.2 包结构

```
agent-harness-zzk/                        # 本仓库（Agent Harness）
├── pyproject.toml                        # [project.scripts] zzk = harness.cli.app:app
├── README.md
│
├── harness/
│   │   ├── __init__.py
│   │   ├── config.py                     # 集中配置 (pydantic-settings)
│   │   │
│   │   ├── chain/                        # ── Chain 编排层 (Phase 5 新增) ──
│   │   │   ├── __init__.py               # 公开接口：Chain, ChainResult
│   │   │   ├── base.py                   # 抽象 + 数据模型
│   │   │   ├── sequential.py             # 顺序链
│   │   │   ├── router.py                 # 路由链
│   │   │   └── nodes/                    # 预置节点
│   │   │       ├── __init__.py
│   │   │       ├── llm_node.py
│   │   │       ├── skill_node.py
│   │   │       └── transform_node.py
│   │   │
│   │   ├── cli/                          # ── CLI 层 ──
│   │   │   ├── __init__.py
│   │   │   ├── app.py                    # typer 应用定义 (zzk)
│   │   │   ├── commands.py               # 命令实现 (run/chat/trace)
│   │   │   └── formatter.py              # 终端渲染（引用高亮、流式输出）
│   │   │
│   │   ├── engine/                       # ── 引擎层 ──
│   │   │   ├── __init__.py
│   │   │   ├── loop.py                   # ReAct 主循环 (async generator)
│   │   │   ├── context.py                # 上下文管理器（压缩策略）
│   │   │   ├── state.py                  # Agent 状态模型 (dataclass)
│   │   │   └── trace.py                  # Trace 记录器
│   │   │
│   │   ├── llm/                          # ── LLM 层 ──
│   │   │   ├── __init__.py
│   │   │   ├── base.py                   # LLMProvider 协议 · JSON schema 解析
│   │   │   ├── deepseek.py               # DeepSeek 实现 (httpx 流式)
│   │   │   └── prompts.py                # 系统提示词模板
│   │   │
│   │   └── skills/                       # ── Skill 层 ──
│   │       ├── __init__.py
│   │       ├── base.py                   # Skill 协议 · SkillResult
│   │       ├── registry.py               # 自动发现 · 注册（安全边界）
│   │       ├── target.py                 # KnowledgeTarget 抽象适配器层
│   │       ├── builtins/                 # 内置技能（默认启用）
│   │       │   ├── __init__.py
│   │       │   ├── file_reader.py         # 读本地文件（路径安全策略）
│   │       │   ├── knowledge_search.py    # 知识库检索（通过 Target 层）
│   │       │   └── web_search.py          # Web 搜索 (Phase 5 新增)
│   │       └── user/                     # 用户技能目录（需 --enable-user-skills）
│   │
│   ├── tests/                            # ── 测试 ──
│   │   ├── __init__.py
│   │   ├── conftest.py                   # 共享 fake 实现
│   │   ├── test_state.py
│   │   ├── test_loop.py
│   │   ├── test_context.py
│   │   ├── test_trace.py
│   │   ├── test_skills_base.py
│   │   ├── test_skills_registry.py
│   │   ├── test_file_reader.py
│   │   ├── test_knowledge_search.py
│   │   ├── test_target.py
│   │   ├── test_chain_base.py            # Chain 抽象契约 (Phase 5)
│   │   ├── test_chain_sequential.py      # 顺序链 (Phase 5)
│   │   ├── test_chain_router.py          # 路由链 (Phase 5)
│   │   ├── test_chain_nodes.py           # 预置节点 (Phase 5)
│   │   └── test_web_search.py            # Web 搜索技能 (Phase 5)
│   │
│   └── examples/
│       └── skills/
│           └── weather_skill.py           # 示例：第三方技能
│
└── ARCHITECTURE.md                        # 本文件
```

---

## 3. 核心接口设计

### 3.1 LLMProvider 协议

```python
class LLMProvider(Protocol):
    """所有 LLM 提供者需实现的接口。"""
    @property
    def name(self) -> str: ...
    @property
    def model(self) -> str: ...

    async def chat(self, messages, *, temperature=0.1, max_tokens=4096) -> CompletionResult: ...
    async def chat_stream(self, messages, *, temperature=0.1, max_tokens=4096) -> AsyncIterator[StreamEvent]: ...
```

设计要点：
- 协议（Protocol）而非 ABC —— 测试 fake 不需要显式继承
- DeepSeek 实现用 httpx，支持 SSE 流式解析

### 3.2 工具调用协议：JSON Schema（P0 刚性约束）

LLM 的输出**必须**是严格 JSON，而非自由文本。引擎只认两种 schema：

```json
{
  "action": "tool",
  "name": "file_reader",
  "args": {
    "path": "/home/user/report.pdf"
  },
  "reasoning": "用户要求分析 report.pdf，我需要先读取它"
}
```

```json
{
  "action": "answer",
  "content": "report.pdf 的主要内容如下：...",
  "reasoning": "我已经读取了文件内容，可以回答用户了",
  "citations": [
    {"source": "report.pdf", "snippet": "关键段落..."}
  ]
}
```

**解析失败时的确定性 fallback（严格逐级降级）：**

```
1. 尝试 json.loads()   → 匹配 action 字段
2. 失败 → 正则提取 {} 内的 JSON 片段
3. 失败 → 整段 LLM 输出作为 answer 返回，但加 error_code=parse_failed
           CLI 前缀标注"⚠ 降级回答（格式异常）"，不伪装成正常成功
4. 步骤超时 → error 事件，继续下一轮（不挂死）
```

关键约束：降级到第 3 级时，**必须在 trace 中记录 `error_code=parse_failed`**，
且 CLI 输出时明确标注"降级回答"，避免协议错误被统计为成功。

### 3.3 Skill 协议

```python
class Skill(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    async def execute(self, **kwargs: Any) -> SkillResult: ...

@dataclass(frozen=True)
class SkillResult:
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)  # P0: 用 default_factory
```

设计要点：
- `metadata` 使用 `field(default_factory=dict)`，**不使用** `= {}`，避免默认值在多个实例间共享的可变对象陷阱
- 内置技能导出模块级单例（如 `file_reader = _FileReaderSkill()`）

### 3.4 插件安全边界（P0）

| 风险 | 措施 |
|------|------|
| 恶意技能代码 | 默认只启用内置技能；用户技能需传入 `--enable-user-skills` 标志 |
| 技能无限循环 | 每技能执行设 **timeout=30s** 硬限制 |
| 技能崩溃污染主进程 | **v1**: 同进程 try/except 隔离；**v1.5**: 子进程 subprocess 隔离 |
| 签名验证 | v1 不做，v2 可补技能目录签名白名单 |

**技能隔离升级路线：**

```
v1（同进程）:
  builtins + user 均在同一进程
  try/except 兜住抛异常
  缺点：OOM、死循环、段错误仍会拖垮主进程

v1.5（用户 skill 走子进程）:
  builtins（内置技能）→ 同进程（信任）
  user（用户技能）→ 子进程 subprocess 执行（不信任）
     - 超时 kill（subprocess timeout）
     - stdout/stderr 捕获回主进程
     - 子进程崩溃不影响主进程
  --enable-user-skills 开关仍然有效
```

自动发现路径：
```
默认扫描（总是启用）:
  harness/skills/builtins/*.py

条件扫描（仅带 --enable-user-skills 时）:
  ~/.zzk/skills/*.py
  extra_dirs 参数
```

### 3.5 file_reader 路径安全策略（P0）

```
路径安全规则（每一跳都校验）:
  1. 基于工作目录做 allowlist
  2. 路径解析（Path.resolve()）后检查是否在允许范围内
  3. 禁止访问:
     - /etc, /sys, /proc, /dev (类 Unix)
     - C:\Windows, C:\Program Files (Windows)
     - ~/.zzk/ 自身
     - ~/.ssh/ 等敏感目录
  4. 被拒绝的操作在 trace 中记录 "reason: path_outside_allowlist"

v1 策略: 只读 + 工作目录 allowlist + 默认阻塞列表
v2 可加: 自定义 allowlist 配置文件 (~/.zzk/allowed_paths.txt)
```

### 3.6 KnowledgeTarget 适配器层（P1）

不在 `skills/builtins/knowledge_search.py` 里写死 HTTP 调用，而是抽一层抽象：

```python
class KnowledgeTarget(Protocol):
    """知识库后端的抽象接口。"""
    async def search(self, query: str, top_k: int = 5) -> KnowledgeResult: ...

@dataclass(frozen=True)
class KnowledgeResult:
    items: list[dict[str, Any]]
    hit_count: int
    error: str | None = None

# 内置实现
class HttpKnowledgeTarget:
    """通过 HTTP 调用 KnowledgeOps Copilot。"""
    def __init__(self, base_url: str, api_key: str | None = None): ...

class MockKnowledgeTarget:
    """测试 / 离线模式，返回固定结果。"""
```

面试叙事增益：
> "我设计适配器层把后端实现与 Agent 框架解耦，因此可以用 Mock 跑测试、用 HTTP 接 Copilot、未来甚至可以内嵌一个本地检索引擎。"

### 3.7 上下文管理 & 压缩触发策略（P1）

v1 策略写死，不做配置化：

```
触发条件（按优先级）:
  1. 如果 len(messages) > max_messages:
     裁剪 tool raw output（只保留摘要）
  2. 如果估算 token > max_tokens:
     对最早的历史消息做 LLM 摘要
  3. 如果摘要失败:
     从最早的 message 开始截断（deterministic truncate）
     不阻塞主流程

预算上限（硬约束，避免摘要自身消耗过高）:
  - 每轮最多执行 1 次 LLM 摘要（防止摘要循环）
  - 单次摘要本身消耗的 token 上限 512（超出截断摘要输入）
  - 摘要后仍超限 → 走 deterministic truncate
```

### 3.8 Trace 数据模型（P1）

```python
@dataclass(frozen=True)
class TraceHeader:
    """Trace 文件的第一行，描述一次 agent run 的元信息。"""
    record_type: str = "run"     # 固定 "run"
    run_id: str = ""
    timestamp: str = ""
    query: str = ""
    conversation_id: str = ""
    llm_model: str = ""
    llm_provider: str = ""
    total_steps: int = 0
    total_latency_ms: float = 0.0
    final_status: str = ""       # "success" | "error" | "max_steps"


@dataclass(frozen=True)
class TraceStep:
    """Trace 文件的每一行记录一个 agent 步骤。"""
    record_type: str = "step"    # 固定 "step"，不与 step_type 冲突
    step: int = 0
    step_type: str = ""          # "llm_call" | "skill_execution" | "parse_result"
    run_id: str = ""             # 关联到一次 agent run
    request_id: str = ""         # 关联到用户请求
    conversation_id: str = ""    # 关联到多轮对话
    parent_step: int | None = None  # 子步骤关联（如 tool_call 内的重试）
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str | None = None
    skill: str | None = None
    status: str = "succeeded"    # "succeeded" | "failed" | "skipped"
    error: str | None = None
    error_code: str | None = None  # 见统一错误码字典
```

每条 trace 文件包含 1 行 header + N 行 step，均为 JSON Lines：

```jsonl
{"record_type":"run","run_id":"run-001","timestamp":"2026-05-17T10:00:00Z","query":"分析 report.pdf","conversation_id":"conv-abc","llm_model":"deepseek-chat","total_steps":3}
{"record_type":"step","step":1,"step_type":"llm_call","run_id":"run-001","status":"succeeded","latency_ms":3400}
{"record_type":"step","step":2,"step_type":"skill_execution","run_id":"run-001","skill":"file_reader","status":"failed","error_code":"path_denied","error":"/etc/passwd is outside allowlist"}
{"record_type":"step","step":3,"step_type":"llm_call","run_id":"run-001","status":"succeeded","latency_ms":2100}
```

注意：`record_type` 区分"这一行是 run 元数据还是 step 记录"，`step_type` 表示"这个 step 具体是什么类型"，两者语义不同，不会冲突。

**Trace 安全策略**：落盘失败（磁盘满、权限）catch + warn，不影响主流程。

### 3.9 统一错误码字典（所有降级路径都落 trace）

所有失败路径使用同一套错误码，trace 和 CLI 输出一致：

| error_code | 含义 | 触发场景 | trace 必填 |
|------------|------|---------|-----------|
| `parse_failed` | LLM 输出非 JSON，降级回答 | JSON fallback 第3级 | 是 |
| `tool_timeout` | 技能执行超时 | 执行 > 30s | 是 |
| `tool_crash` | 技能抛异常 | except 捕获 | 是 |
| `path_denied` | file_reader 路径不安全 | 路径安全检查拦截 | 是 |
| `target_unreachable` | 知识库服务不可达 | HTTP 连接失败 | 是 |
| `max_steps` | 达到最大步骤限制 | 循环退出 | 是 |
| `llm_error` | LLM 调用异常 | HTTP 4xx/5xx 或网络错误 | 是 |
| `unknown_tool` | LLM 请求了一个不存在的 skill | registry 中未找到 | 是 |

要求：
- 所有 `except` 路径必须 set `error_code`
- CLI 输出时，`error_code` 非空的行以 `[降级]` 前缀标记

### 3.10 最小评估闭环（Phase 4+，不回退到大 Studio）

不重建 RAG Optimization Studio。Harness 内置一个极简 `zzk eval` 子命令：

```
zzk eval --cases eval-cases.json

# eval-cases.json 格式（10~20 条 golden cases）:
[
  {"query": "总结这份报告", "expected_tools": ["file_reader"], "min_tokens": 50},
  {"query": "查知识库 xxx", "expected_tools": ["knowledge_search"], "min_tokens": 20}
]

# 输出:
{
  "total": 15,
  "passed": 13,
  "task_success_rate": 0.87,
  "tool_error_rate": 0.07,
  "avg_latency_ms": 4200,
  "failures": [
    {"case": 3, "error": "tool_timeout", "tool": "knowledge_search"},
    {"case": 7, "error": "parse_failed", "step": 2}
  ]
}
```

收益：
- 能从"能跑"升级为"可验证优化"
- 面试时可以说"我做了个 15 条 golden cases 的评估闭环，用来追踪每次改动是否引入回退"

---

## 4. 垂直切片实施计划

不按模块水平开发，每个 Phase 产出可演示的端到端命令：

```
Phase 1              Phase 2              Phase 3              Phase 4
zzk "hello"          zzk "read x"         zzk chat             zzk "查文档..."
│                    │                    │                    │
├─ CLI               ├─ CLI               ├─ CLI               ├─ CLI
├─ 引擎循环          ├─ 引擎循环          ├─ 引擎循环          ├─ 引擎循环
├─ LLM DeepSeek      ├─ LLM DeepSeek      ├─ LLM DeepSeek      ├─ LLM DeepSeek
├─ JSON schema 解析  ├─ JSON schema       ├─ JSON schema       ├─ JSON schema
├─ 无技能            ├─ file_reader       ├─ file_reader       ├─ file_reader
├─ trace 桩           │  (路径安全)        │                    ├─ KnowledgeTarget
│                    ├─ trace 完整        ├─ context manager   │  (HTTP adapter)
│                    │                    ├─ trace             ├─ trace
│                    │                    ├─ zzk chat 交互      │
│                    │                    │                    │
▼ 演示               ▼ 演示               ▼ 演示               ▼ 演示
zzk "你好"           zzk "读这块文件"     多轮对话保持上下文    zzk "查知识库xxx"
纯聊天               带引用 + trace        可连续提问            混合检索
```

### Phase 1：单轮聊天（无技能）

| 步骤 | 内容 | 关键文件 |
|------|------|---------|
| 1.1 | 项目初始化 + 配置 + 状态模型 | `pyproject.toml`, `config.py`, `state.py` |
| 1.2 | LLM Provider 协议 + JSON schema 解析 + fallback | `llm/base.py`, `llm/prompts.py` |
| 1.3 | DeepSeek 流式实现 | `llm/deepseek.py` |
| 1.4 | ReAct 循环（简化为单轮：LLM → 直接回答） | `engine/loop.py` |
| 1.5 | CLI zzk 入口 + 流式输出 | `cli/app.py`, `cli/commands.py`, `cli/formatter.py` |
| 1.6 | 集成测试 + trace 桩 | `tests/`, `engine/trace.py` |

**交付命令**：`zzk "你好"` → 流式回答

### Phase 2：带 file_reader 技能 + 完整 Trace

| 步骤 | 内容 | 关键文件 |
|------|------|---------|
| 2.1 | Skill 协议 + SkillResult | `skills/base.py` |
| 2.2 | 注册中心（仅内置，安全边界） | `skills/registry.py` |
| 2.3 | file_reader（路径安全策略） | `skills/builtins/file_reader.py` |
| 2.4 | ReAct 循环完整：TOOL_CALL → 执行 → 观察 → 继续 | `engine/loop.py` 完善 |
| 2.5 | Trace 完整记录（含 error_code） | `engine/trace.py` |
| 2.6 | 测试覆盖 + 错误路径 | `tests/` |

**交付命令**：`zzk "读这个 README.md"` → 调用 file_reader → 带引用 + trace 回答

### Phase 3：多轮对话 + 上下文管理

| 步骤 | 内容 | 关键文件 |
|------|------|---------|
| 3.1 | 上下文管理器（压缩策略） | `engine/context.py` |
| 3.2 | `zzk chat` 交互模式 | `cli/commands.py` |
| 3.3 | 对话持久化（JSON 文件） | `engine/context.py` |
| 3.4 | Trace 关联 conversation_id | `engine/trace.py` |
| 3.5 | 联调 + 测试 | `tests/` |

**交付命令**：`zzk chat` → 多轮对话，上下文保持

### Phase 4：KnowledgeOps 集成

| 步骤 | 内容 | 关键文件 |
|------|------|---------|
| 4.1 | KnowledgeTarget 适配器层 | `skills/target.py` |
| 4.2 | HTTPAdapter（→ KnowledgeOps） | `skills/target.py` |
| 4.3 | MockAdapter（测试用） | `skills/target.py` |
| 4.4 | knowledge_search 技能 | `skills/builtins/knowledge_search.py` |
| 4.5 | 用户技能目录发现（--enable-user-skills） | `skills/registry.py` 完善 |
| 4.6 | README + 面试叙事打磨 | `README.md` |

**交付命令**：`zzk "查知识库 xxx"` → 混合检索 + 引用

---

## 5. Phase 5：Chain 编排模块 + Web Search 技能

### 5.1 动机

当前 ReAct 循环是唯一的编排方式：LLM 自主决策每一步调用什么工具。但简历中提到"Chain"能力——需要增加一种**确定性编排**的抽象，让开发者可以定义固定的处理流程（如"查知识库 → 摘要 → 翻译"），与 ReAct 的自主决策互补。

同时增加 `web_search` 技能，补齐"搜索"能力，丰富 Skill 生态。

### 5.2 架构变更

新增 `harness/chain/` 顶层模块，与 `engine/`、`llm/`、`skills/` 同级：

```
harness/chain/
├── __init__.py          # 公开接口: Chain, ChainResult, register_chain
├── base.py              # Chain 抽象 + ChainResult + ChainContext
├── sequential.py        # 顺序链（按序执行多个节点）
├── router.py            # 路由链（根据条件选择分支）
└── nodes/
    ├── __init__.py
    ├── llm_node.py      # 调用 LLM
    ├── skill_node.py    # 调用 Skill
    ├── transform_node.py# 文本转换
    └── passthrough_node.py
```

Chain 模块的分层定位：

```
CLI → Engine (ReAct)  ←→  Chain (顺序/路由)
          ↓                      ↓
         LLM                   Skills
```

- Engine 和 Chain **同级且可互相调用**：Chain 内部可用 SkillNode 调用技能，也可用 LLMNode 调用 LLM；Engine 的 ReAct 循环也可把 Chain 注册为一个 skill
- Chain 不依赖 engine/loop.py 的内部细节，只依赖 llm/ 和 skills/ 的公开接口

### 5.3 核心设计

```python
# chain/base.py

@dataclass(frozen=True, slots=True)
class ChainResult:
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    sub_results: tuple[ChainResult, ...] = ()  # 子链结果，支持嵌套

class Chain(Protocol):
    @property
    def name(self) -> str: ...
    async def run(self, input: str, context: ChainContext | None = None) -> ChainResult: ...
```

**SequentialChain** — 前一个节点的输出作为后一个节点的输入：

```
用户输入 → [LLMNode → SkillNode(file_reader) → LLMNode] → 最终输出
```

**RouterChain** — 根据条件选择分支：

```
输入 → 判断条件 → 分支A / 分支B → 输出
```

### 5.4 预置节点

| 节点 | 用途 | 依赖 |
|------|------|------|
| `LLMNode` | 调用 LLM，复用 llm/ 模块 | `harness/llm/` |
| `SkillNode` | 调用注册的技能 | `harness/skills/` |
| `TransformNode` | 文本处理（截断、格式化、拼接） | 纯函数 |
| `PassThroughNode` | 调试用，原样输出 | 无 |

均复用现有 `state.py` 中的 `Message`、`StreamEvent` 等核心模型。

### 5.5 面试叙事增益

> "我设计了 Chain 抽象层，支持顺序链和路由链两种模式。顺序链把前一个节点的输出传给后一个节点，适合固定流程；路由链根据条件选择不同分支。Chain 和 ReAct 是互补关系——ReAct 适合 LLM 自主决策的场景，Chain 适合确定性流程。两者都可以调用底层的 LLM 和 Skills 层，架构上保持正交。"

对比 LangChain：展示了你理解 Chain 的本质是"可组合的编排单元"，而非简单照搬框架。

### 5.6 Web Search 技能

新增 `harness/skills/builtins/web_search.py`，遵循现有 `Skill` 协议：

```python
@dataclass(frozen=True, slots=True)
class _WebSearchSkill:
    target: SearchTarget  # 可替换后端

    async def execute(self, query: str, top_k: int = 5) -> SkillResult: ...
```

复用 `skills/target.py` 的适配器模式，三个后端：

| 后端 | 接入 | 场景 |
|------|------|------|
| `DuckDuckGoTarget` | httpx 直接请求 | 免费，零配置 |
| `SerpApiTarget` | httpx + API Key | 生产级稳定 |
| `MockSearchTarget` | 固定结果 | 测试用 |

配置新增：

```env
ZZK_SEARCH_PROVIDER=duckduckgo    # duckduckgo | serpapi
ZZK_SEARCH_API_KEY=
```

### 5.7 CLI 交互

```bash
# Chain
zzk chain list                         # 查看已注册 chain
zzk chain run sequential --steps llm,skill:file_reader,llm "输入"

# Web Search（通过 run/chat 的 ReAct 循环触发）
zzk run "搜索今天的 AI 新闻"
```

`zzk chain` 命令挂在现有 app 下，与 `run`、`chat`、`eval` 同级：

```python
# cli/app.py
chain_app = typer.Typer(help="Run and manage chains")
app.add_typer(chain_app, name="chain")
```

### 5.8 Prompt 更新

`prompts.py` 的 `Available tools:` 列表追加：

```
- web_search(args: {"query":"<search_query>", "top_k": 5}) — 搜索互联网并返回摘要结果
```

### 5.9 实现顺序

| 步骤 | 内容 | 涉及时长估计 | 关键文件 |
|------|------|-------------|---------|
| 5.9.1 | `base.py` — Chain 抽象 + ChainResult + ChainContext | 小 | `chain/base.py` |
| 5.9.2 | 预置节点 — LLMNode, SkillNode, TransformNode | 中 | `chain/nodes/*.py` |
| 5.9.3 | `SequentialChain` — 顺序链 | 中 | `chain/sequential.py` |
| 5.9.4 | `RouterChain` — 路由链 | 小 | `chain/router.py` |
| 5.9.5 | `web_search` 技能 + SearchTarget 适配器 | 中 | `skills/builtins/web_search.py`, `skills/target.py` 扩展 |
| 5.9.6 | CLI `zzk chain` 子命令 | 小 | `cli/app.py`, `cli/commands.py` |
| 5.9.7 | prompt + errors + registry 更新 | 小 | `prompts.py`, `errors.py`, `registry.py` |
| 5.9.8 | 全量测试 + ARCHITECTURE.md 更新 | 中 | `tests/test_chain_*.py`, `tests/test_web_search.py` |

### 5.10 测试规划

```
tests/
├── test_chain_base.py           # ChainResult, ChainContext, 契约校验
├── test_chain_sequential.py     # SequentialChain 编排 + 节点组合
├── test_chain_router.py         # RouterChain 分支选择
├── test_chain_nodes.py          # LLMNode, SkillNode, TransformNode 单元
└── test_web_search.py           # WebSearchSkill + MockSearchTarget
```

遵循现有测试模式：
- 使用 `pytest-asyncio`（已有）
- Mock LLM / Mock Skill 隔离外部依赖
- 覆盖正常路径 + 错误路径 + 边界条件

---

## 6. 风险与回滚

| 风险 | 概率 | 影响 | 缓解措施 | 回滚策略 |
|------|------|------|---------|---------|
| DeepSeek API 接口变更 | 低 | LLM 层全部不可用 | Provider 协议封装；测试用 MockTarget | 切到其他 Provider 或本地上次可用版本 |
| 动态技能加载崩溃 | 中 | 单个技能故障影响主进程 | 异常隔离（try/except）；超时 30s 硬限制 | 禁用出问题技能；默认只启用 builtins |
| 上下文膨胀 OOM | 中 | 长对话内存溢出 | v1 硬限制 max_messages=20 + max_tokens=8192；超出截断 | 重启对话；清除 ~/.zzk/conversations/ |
| file_reader 路径遍历 | 低 | 敏感文件泄露 | 路径安全检查 + allowlist + trace 审计 | 紧急禁用 file_reader 技能 |
| 用户恶意技能代码 | 低 | 任意代码执行 | 默认不加载；需 --enable-user-skills 显式启用 | 撤掉用户技能目录 |

---

## 6. 验证标准

### 功能验证
- `zzk "你好"` → 流式返回回答（Phase 1）
- `zzk "读这个 README.md"` → 调用 file_reader → 带引用回答（Phase 2）
- `zzk chat` → 多轮对话，上下文保持（Phase 3）
- `zzk "查知识库 xxx"` → 知识库检索 + 引用（Phase 4）
- Trace 记录保存到 `~/.zzk/traces/`（Phase 2+）

### 代码质量
- `ruff` 零警告
- pytest 覆盖率 > 80%

### 失败路径 SLA
- tool timeout 后 **2 秒内**给出可解释错误，不挂死
- `max_steps` 命中时返回受控结束（"已到达最大步骤数，以下是当前结果..."）
- JSON 解析失败按 fallback 链逐级降级，**不抛异常**；第3级降级时带 `error_code=parse_failed`
- 所有失败路径 trace 中必须有 `error_code`（见统一错误码字典）
- trace 落盘失败 catch + warn，**不影响主流程**
- LLM 调用异常 → error 事件 → 继续下一轮或友好提示
- 上下文摘要每轮最多 1 次，每次最多消费 512 tokens

### 面试叙事
- **"为什么不用 LangGraph？"**
  > "Copilot 里已经用 LangGraph 展示了我对这个框架的应用能力。Harness 更侧重 runtime/Skills/Trace 这套契约的设计，先用轻量循环降低迭代成本、提高可控性。两个项目是互补的——Copilot 展示了我对业界框架的理解，Harness 展示了我从零设计 Agent 系统的能力。"

- **"Skill 怎么加载的？"**
  > "协议驱动 + 文件系统自动发现。技能只是一个满足 Skill 协议的 Python 对象，注册中心扫描指定目录动态加载。内置技能默认启用，用户技能需要显式授权（--enable-user-skills），防止恶意代码自动执行。"

- **"Trace 用来做什么？"**
  > "每步的输入输出、耗时、token、错误码都结构化记录为一串 JSON Lines。一次 run 是一个文件，可以按 run_id / conversation_id 聚合分析，定位瓶颈——比如发现某技能调用频繁超时、LLM 响应逐渐变慢等。"

- **"KnowledgeOps Copilot 和 Harness 什么关系？"**
  > "Copilot 是 RAG 后端，Harness 是 Agent 运行时。Harness 通过适配器层调用 Copilot 的知识库搜索，两者通过接口解耦。这样 Harness 不依赖 Copilot 也能跑（纯聊天、文件操作），面试官不会混淆两个项目的能力边界。"
