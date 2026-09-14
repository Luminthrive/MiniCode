"""最小化事件总线：pub/sub 和4个核心事件类"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel

# 事件处理函数类型
type EventHandler = Callable[[BaseModel], Awaitable[None]]


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
    ts: str


# 最小化事件总线：注册订阅者并按顺序发布事件
class EventBus:
    # 初始化空事件总线
    def __init__(self) -> None:
        self._subscribers: list[EventHandler] = []

    # 注册一个事件处理函数
    def subscribe(self, handler: EventHandler) -> None:
        self._subscribers.append(handler)

    # 按注册顺序依次调用所有订阅者
    async def publish(self, event: BaseModel) -> None:
        for handler in self._subscribers:
            await handler(event)
