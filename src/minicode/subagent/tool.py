"""子代理工具：SpawnAgentTool（仅前台同步模式）"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from minicode.agents.loader import AgentProfile, AgentProfileLoader
from minicode.context import ExecutionContext
from minicode.events.bus import (
    EventBus,
    SubagentFinishedEvent,
    SubagentStartedEvent,
    utc_now_iso,
)
from minicode.ids import create_id
from minicode.loop import AgentLoop
from minicode.tools.base import BaseTool, ToolResult
from minicode.tools.builtin.bash import BashTool
from minicode.tools.builtin.edit_file import EditFileTool
from minicode.tools.builtin.list_dir import ListDirTool
from minicode.tools.builtin.read_file import ReadFileTool
from minicode.tools.builtin.write_file import WriteFileTool
from minicode.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from minicode.llm.base import LLMProvider

_profile_loader = AgentProfileLoader()


# 派生子代理参数模型
class SpawnAgentParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    description: str
    prompt: str
    subagent_type: str = ""


# 在隔离的冷启动上下文中派生子 agent（仅前台同步模式）
class SpawnAgentTool(BaseTool):
    name = "spawn_agent"
    description = (
        "Spawn an isolated sub-agent to handle a self-contained sub-task. "
        "The sub-agent starts with a clean context containing only the provided prompt — "
        "it does not inherit the current conversation history. "
        "Runs synchronously and returns the result when complete."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "3-5 word task description shown in progress display",
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Complete task description including all context the sub-agent needs. "
                    "The sub-agent cannot see the parent conversation, so be explicit."
                ),
            },
            "subagent_type": {
                "type": "string",
                "description": (
                    "Agent role profile (planner/executor/reviewer). "
                    "Leave empty for default."
                ),
            },
        },
        "required": ["description", "prompt"],
    }
    params_model = SpawnAgentParams

    # 构造 SpawnAgentTool；depth=0 表示根 agent，最大允许嵌套深度为 1
    def __init__(
        self,
        provider: LLMProvider,
        parent_bus: EventBus,
        parent_trace_id: str,
        parent_run_id: str,
        max_steps: int,
        depth: int = 0,
    ) -> None:
        self._provider = provider
        self._parent_bus = parent_bus
        # 子 agent 不新建 Trace：整个任务的所有 run 共享同一 trace_id
        self._parent_trace_id = parent_trace_id
        self._parent_run_id = parent_run_id
        self._max_steps = max_steps
        self._depth = depth

    # 派生子 agent，前台同步执行并返回结果
    async def invoke(self, params: dict[str, Any]) -> ToolResult:
        p = SpawnAgentParams.model_validate(params)

        # 嵌套深度限制为 1
        if self._depth >= 1:
            return ToolResult(
                content="Subagent nesting limit (1) reached; cannot spawn further subagents.",
                is_error=True,
                error_type="runtime_error",
            )

        profile: AgentProfile | None = None
        if p.subagent_type:
            # TOML 读取为同步磁盘 IO，丢线程池避免阻塞事件循环
            profile = await asyncio.to_thread(_profile_loader.load, p.subagent_type)

        child_run_id = create_id("run")
        # planner 5步够用，executor 需要更多步（读文件+分析+写报告）
        max_child_steps = 5 if p.subagent_type == "planner" else min(self._max_steps, 20)
        child_context = ExecutionContext(
            run_id=child_run_id,
            goal=p.prompt,
            max_steps=max_child_steps,
            system_prompt_override=profile.system_prompt if profile else None,
        )

        child_registry = self._build_child_registry(profile)
        # 子 loop 直接向父 bus 发布事件，trace_id 继承父任务、
        # run_id 重新生成、parent_run_id 指向父——三个字段共同表达树的节点与边
        child_loop = AgentLoop(
            self._provider,
            child_registry,
            self._parent_bus,
            trace_id=self._parent_trace_id,
            parent_run_id=self._parent_run_id,
        )

        await self._parent_bus.publish(
            SubagentStartedEvent(
                trace_id=self._parent_trace_id,
                run_id=child_run_id,
                parent_run_id=self._parent_run_id,
                description=p.description,
                ts=utc_now_iso(),
            )
        )

        try:
            await asyncio.wait_for(child_loop.run(child_context), timeout=600.0)
        except TimeoutError:
            if not child_context.is_done():
                child_context.mark_failed("timeout")
        except Exception:
            if not child_context.is_done():
                child_context.mark_failed("error")

        await self._parent_bus.publish(
            SubagentFinishedEvent(
                trace_id=self._parent_trace_id,
                run_id=child_run_id,
                parent_run_id=self._parent_run_id,
                status=child_context.status,
                reason=child_context.reason,
                ts=utc_now_iso(),
            )
        )

        if child_context.status == "success":
            return ToolResult(
                content=child_context.result or "Subagent completed with no text output."
            )
        return ToolResult(
            content=(
                child_context.result
                or f"Subagent failed (status={child_context.status}, reason={child_context.reason})"
            ),
            is_error=True,
            error_type="runtime_error",
        )

    # 构造子 registry；基于角色配置过滤工具，不再注册嵌套 spawn_agent
    def _build_child_registry(
        self,
        profile: AgentProfile | None,
    ) -> ToolRegistry:
        allowed: set[str] | None = (
            set(profile.allowed_tools) if profile and profile.allowed_tools else None
        )

        # 判断工具名是否在角色允许列表中
        def _allowed(name: str) -> bool:
            return allowed is None or name in allowed

        registry = ToolRegistry()
        all_tools: list[BaseTool] = [
            ReadFileTool(),
            BashTool(),
            WriteFileTool(),
            EditFileTool(),
            ListDirTool(),
        ]
        for t in all_tools:
            if _allowed(t.name):
                registry.register(t)

        return registry
