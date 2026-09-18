"""最小化事件总线：pub/sub 和核心事件类"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# 事件处理函数类型
type EventHandler = Callable[[BaseModel], Awaitable[None]]


# 返回当前 UTC 时间的 ISO 8601 字符串（事件时间戳统一用此格式）
def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


# 运行开始事件
class RunStartedEvent(BaseModel):
    type: Literal["run.started"] = "run.started"
    run_id: str
    goal: str
    ts: str


# 运行结束事件
class RunFinishedEvent(BaseModel):
    type: Literal["run.finished"] = "run.finished"
    run_id: str
    status: str
    reason: str | None = None
    steps: int
    ts: str


# LLM 流式文本增量事件
class LlmDeltaEvent(BaseModel):
    type: Literal["llm.delta"] = "llm.delta"
    run_id: str
    text: str
    # LLM 网络重试会从头重放流式输出，attempt 预留给渲染端按轮去重
    attempt: int = 1
    ts: str


# LLM 响应统计事件
class LlmUsageEvent(BaseModel):
    type: Literal["llm.usage"] = "llm.usage"
    run_id: str
    input_tokens: int
    output_tokens: int
    context_pct: float
    ts: str


# 工具调用开始事件
class ToolCallEvent(BaseModel):
    type: Literal["tool.call"] = "tool.call"
    run_id: str
    # 子 agent 发布的工具事件携带父 run_id，渲染端据此缩进嵌套展示
    parent_run_id: str | None = None
    tool_name: str
    args: dict[str, object]
    tool_call_id: str | None = None
    ts: str


# 工具调用结束事件
class ToolResultEvent(BaseModel):
    type: Literal["tool.result"] = "tool.result"
    run_id: str
    parent_run_id: str | None = None
    tool_name: str
    content: str
    is_error: bool
    # "runtime_error" | "timeout" | "schema_error" | "permission_denied"
    error_type: str | None = None
    elapsed_ms: int | None = None
    tool_call_id: str | None = None
    ts: str


# 上下文压缩完成事件
class ContextCompactedEvent(BaseModel):
    type: Literal["context.compacted"] = "context.compacted"
    run_id: str
    original_tokens: int
    summary_tokens: int
    ts: str


# 子代理开始事件
class SubagentStartedEvent(BaseModel):
    type: Literal["subagent.started"] = "subagent.started"
    run_id: str
    parent_run_id: str
    description: str
    ts: str


# 子代理结束事件
class SubagentFinishedEvent(BaseModel):
    type: Literal["subagent.finished"] = "subagent.finished"
    run_id: str
    parent_run_id: str
    status: str
    reason: str | None = None
    ts: str


# 全部事件的判别联合，供订阅端统一分发/序列化
Event = Annotated[
    RunStartedEvent
    | RunFinishedEvent
    | LlmDeltaEvent
    | LlmUsageEvent
    | ToolCallEvent
    | ToolResultEvent
    | ContextCompactedEvent
    | SubagentStartedEvent
    | SubagentFinishedEvent,
    Field(discriminator="type"),
]


# 最小化事件总线：注册订阅者并按顺序发布事件
class EventBus:
    # 初始化空事件总线
    def __init__(self) -> None:
        self._subscribers: list[EventHandler] = []

    # 注册一个事件处理函数
    def subscribe(self, handler: EventHandler) -> None:
        self._subscribers.append(handler)

    # 按注册顺序依次调用所有订阅者；单个订阅者异常不中断其余订阅者
    async def publish(self, event: BaseModel) -> None:
        for handler in self._subscribers:
            try:
                await handler(event)
            except Exception:
                logger.exception("event handler failed event=%s", type(event).__name__)
