"""渲染器共享件：Rich/TUI 两个渲染器共用的子代理嵌套与聚合状态

子代理的三个身份字段（trace_id / run_id / parent_run_id）在事件里表达树结构；
渲染层只关心两件事：当前是否处于子代理块内（决定缩进/前缀），
以及子代理块内的 usage/tool 累计（结束时随结束行统一展示）。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from minicode.events.bus import (
    LlmUsageEvent,
    SubagentFinishedEvent,
    SubagentStartedEvent,
    ToolCallEvent,
)


class SubagentTracker:
    def __init__(self) -> None:
        self._aggs: dict[str, dict[str, Any]] = {}
        self.current_child: str | None = None

    # 事件是否来自子 agent：优先看 parent_run_id，
    # usage 等不带父字段的事件回落查注册表（该 run 是否以子代理身份开始过）
    def is_child(self, event: BaseModel) -> bool:
        if getattr(event, "parent_run_id", None) is not None:
            return True
        return event.run_id in self._aggs  # type: ignore[attr-defined]

    def start(self, event: SubagentStartedEvent) -> None:
        self._aggs[event.run_id] = {
            "description": event.description,
            "started": event.ts,
            "tools": 0,
            "in": 0,
            "out": 0,
        }
        self.current_child = event.run_id

    # 结束并取走聚合数据（渲染结束行用）；None 表示起点缺失（replay 从半截开始）
    def finish(self, event: SubagentFinishedEvent) -> dict[str, Any] | None:
        self.current_child = None
        return self._aggs.pop(event.run_id, None)

    # 子代理 usage 累计进结束行；返回 True 表示已消费，不逐条上屏
    def add_usage(self, event: LlmUsageEvent) -> bool:
        agg = self._aggs.get(event.run_id)
        if agg is None:
            return False
        agg["in"] += event.input_tokens
        agg["out"] += event.output_tokens
        return True

    # 记录工具调用计数；返回事件是否来自子代理（决定缩进）
    def add_tool_call(self, event: ToolCallEvent) -> bool:
        child = self.is_child(event)
        if child:
            agg = self._aggs.get(event.run_id)
            if agg is not None:
                agg["tools"] += 1
        return child
