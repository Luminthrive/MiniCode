# MiniCode

轻量级本地 AI Agent 系统。

## 特性

- **Agent Loop**：从零实现的 plan→act→observe 循环，无需 LangGraph/LangChain
- **子代理**：支持前台同步派生子 agent，嵌套深度限制为 1
- **上下文压缩**：当上下文使用量超过阈值时自动压缩为6段摘要

## 安装

```bash
pip install -e .
```

## 配置

复制 `.env.example` 为 `.env` 并填入 API 配置：

```bash
cp .env.example .env
```

环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | (必填) | OpenAI 兼容 API Key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API 基础 URL |
| `OPENAI_MODEL` | `gpt-4o` | 模型名称 |
| `MAX_STEPS` | `30` | 最大执行步数 |
| `COMPACT_THRESHOLD` | `0.80` | 上下文压缩触发阈值 |

## 使用

```bash
# 执行一次任务
minicode run "list files in current directory"

# 交互式聊天（Textual TUI，需交互终端）
minicode chat

# 查看版本
minicode --version
```

## 代码导读（第一次读代码从这里开始）

一次对话的完整旅程：

1. 你在输入框回车 → `ui/tui.py` 把文字交给 `application/agent_controller.py` 的 `submit()`
2. `submit()` 组装一个 `runner.py` 的 AgentRunner，进入 `loop.py` 的主循环
3. 主循环每做一件事（收到一段字、调用一个工具……）就往 `events/bus.py` 的总线上发一条事件
4. 事件有两类消费者，各干各的：
   - `ui/tui.py`（或 `ui/rich.py`）：把事件画到屏幕上
   - `events/writer.py`：把事件追加到 `.minicode/traces/` 落盘保存
5. 任务完成 → 结果返回给界面，`session/store.py` 把消息存进会话，下轮接着用

推荐阅读顺序：

```
cli.py                          入口：四个命令各自怎么组装
→ application/agent_controller.py   chat/run 的业务入口（两种模式只差 session_id）
→ loop.py                       AI 的 plan→act→observe 主循环
→ ui/tui.py                     事件怎么画到屏幕
→ tools/permissions.py          危险操作怎么审批
```

分层规则：`ui → application → runner/loop/tools → llm/session/events/config`。
上层可以调用下层，下层永远不知道上层的存在——这保证以后加 Web 界面、
加 Task Memory 都不用改核心。

## 项目结构

```
minicode/
  pyproject.toml
  src/minicode/
    __init__.py
    __main__.py
    cli.py           # CLI 入口（run/replay/stats 走 Rich 渲染）
    config.py         # 配置加载 + .minicode 工作区路径约定
    loop.py           # Agent Loop
    context.py        # 执行上下文
    runner.py         # Agent 运行器
    agents/           # Agent 角色配置
    application/      # 应用层（AgentController：chat/run 统一编排，run 即 session_id=None）
    ui/               # 渲染器：RichRenderer（终端）/ TUIRenderer（Textual）
    tools/            # 工具系统
    subagent/         # 子代理
    compact/          # 上下文压缩
    llm/              # LLM 提供者
    events/           # 事件总线（渲染器都是它的事件消费者）
    session/          # 会话存储
```
