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

# 交互式聊天
minicode chat

# 查看版本
minicode --version
```

## 项目结构

```
minicode/
  pyproject.toml
  src/minicode/
    __init__.py
    __main__.py
    cli.py           # CLI 入口
    config.py         # 配置加载
    loop.py           # Agent Loop
    context.py        # 执行上下文
    runner.py         # Agent 运行器
    agents/           # Agent 角色配置
    tools/            # 工具系统
    subagent/         # 子代理
    compact/          # 上下文压缩
    llm/              # LLM 提供者
    events/           # 事件总线
    session/          # 会话存储
```
