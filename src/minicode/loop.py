"""Agent Loop：驱动 plan→act→observe 循环"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from minicode.context import ExecutionContext
from minicode.events.bus import (
    ContextCompactedEvent,
    EventBus,
    LlmDeltaEvent,
    LlmUsageEvent,
    ToolCallEvent,
    ToolResultEvent,
    utc_now_iso,
)
from minicode.tools.base import ToolResult
from minicode.tools.invocation import invoke_tool
from minicode.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from minicode.compact.compactor import Compactor
    from minicode.llm.base import DeltaSink, LLMProvider
    from minicode.tools.permissions import PermissionManager

logger = logging.getLogger(__name__)


class AgentLoop:
    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        bus: EventBus,
        *,
        compactor: Compactor | None = None,
        compact_threshold: float = 0.80,
        permission_manager: PermissionManager | None = None,
        parent_run_id: str | None = None,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._bus = bus
        self._compactor = compactor
        self._compact_threshold = compact_threshold
        self._permission_manager = permission_manager
        # 子 agent 的 loop 携带父 run_id，事件据此表达嵌套关系
        self._parent_run_id = parent_run_id

    async def run(self, context: ExecutionContext) -> None:
        import json as _json
        while not context.is_done():
            context.step += 1

            try:
                response = await self._provider.chat(
                    messages=context.messages,
                    tool_schemas=self._registry.tool_schemas(),
                    run_id=context.run_id,
                    step=context.step,
                    system=context.system_prompt(
                        "你是 MiniCode Agent，一个能操作文件系统的 AI 助手。\n\n"
                        "## 核心规则\n"
                        "1. 用户要求做某件事 → 用工具执行，不是只描述\n"
                        "2. 写文件/生成文档 → 调用 write_file，不是只输出文本\n"
                        "3. 纯闲聊时才直接回复\n\n"
                        "## 可用工具\n"
                        "- read_file / write_file / bash / list_dir：文件系统操作\n"
                        "- spawn_agent：派生子代理（planner/executor/reviewer）\n\n"
                        "## 决策：主 agent 直接做 vs 派子 agent\n\n"
                        "主 agent 直接做（用 read_file/write_file/bash/list_dir）：\n"
                        "- 读1-2个文件\n"
                        "- 写一个文件\n"
                        "- 执行一条命令\n"
                        "- 列目录\n"
                        "- 简单的单步或两步操作\n\n"
                        "派子 agent（用 spawn_agent）：\n"
                        "- 需要分析3个以上文件\n"
                        "- 需要多步骤执行（读→分析→写→审查）\n"
                        "- 代码审查、生成报告、重构等复杂任务\n"
                        "- 需要规划→执行→审查的完整流程\n\n"
                        "## spawn_agent 工作流\n"
                        "1. 派 planner → 制定计划（planner 只规划，不执行）\n"
                        "2. 派 executor → 按计划执行（executor 自己读文件、分析、写文件）\n"
                        "3. 派 reviewer → 审查结果\n"
                        "**主 agent 不要预先读文件，把工作完整委托给子 agent**\n\n"
                        "## bash 命令（Windows 环境）\n"
                        "- 用 PowerShell：Get-ChildItem/dir、Get-Content/type、Select-String\n"
                        "- 禁止路径遍历（..）"
                    ),
                    delta_sink=self._make_delta_sink(context.run_id),
                )
            except asyncio.CancelledError:
                context.mark_failed("cancelled")
                raise
            except Exception:
                logger.exception("LLM call failed run_id=%s step=%d", context.run_id, context.step)
                context.mark_failed("llm_error")
                break

            if response.usage is not None:
                await self._bus.publish(
                    LlmUsageEvent(
                        run_id=context.run_id,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        context_pct=response.usage.context_pct,
                        ts=utc_now_iso(),
                    )
                )

            openai_tool_calls: list[dict[str, object]] | None = None
            if response.tool_calls:
                openai_tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": _json.dumps(tc.input)},
                    }
                    for tc in response.tool_calls
                ]
            context.add_assistant_message(
                response.text, openai_tool_calls, response.reasoning
            )

            # 有 tool_calls 就必须执行；但输出被 max_tokens 截断时参数可能是残缺 JSON，不执行
            if response.tool_calls and response.stop_reason != "length":
                for tc in response.tool_calls:
                    await self._publish_tool_call(context.run_id, tc.name, tc.input, tc.id)
                    result = await invoke_tool(
                        self._registry, tc, context.run_id,
                        permission_manager=self._permission_manager,
                    )
                    await self._publish_tool_result(context.run_id, tc.name, tc.id, result)
                    context.add_tool_result(tc.id, result.content, is_error=result.is_error)

            # 终止判断
            if response.stop_reason == "length":
                # 输出被截断，内容不完整，不能当作成功
                logger.warning(
                    "output truncated by max_tokens run_id=%s step=%d",
                    context.run_id, context.step,
                )
                context.mark_failed("output_truncated")
            elif not response.tool_calls:
                context.result = response.text or ""
                context.mark_success()
            elif context.step >= context.max_steps:
                context.mark_failed("exceeded_max_steps")

            # 仅在追加完工具结果、且 run 继续时检查压缩，
            # 此时 messages 末尾是 tool 结果，替换成 [摘要, ack] 对下次调用是合法输入
            if (
                not context.is_done()
                and self._compactor is not None
                and self._compact_threshold > 0
                and response.usage is not None
                and response.usage.context_pct >= self._compact_threshold
            ):
                compacted = await self._compactor.compact(context, self._provider)
                if compacted is not None:
                    await self._bus.publish(
                        ContextCompactedEvent(
                            run_id=context.run_id,
                            original_tokens=compacted.original_token_estimate,
                            summary_tokens=compacted.summary_tokens,
                            ts=utc_now_iso(),
                        )
                    )

    # 构造流式增量出口：把 provider 的 token 片段转成 LlmDeltaEvent 发布到总线
    def _make_delta_sink(self, run_id: str) -> DeltaSink:
        async def sink(text: str) -> None:
            await self._bus.publish(
                LlmDeltaEvent(run_id=run_id, text=text, ts=utc_now_iso())
            )

        return sink

    # 发布工具调用开始事件
    async def _publish_tool_call(
        self, run_id: str, tool_name: str, args: dict[str, object], tool_call_id: str
    ) -> None:
        await self._bus.publish(
            ToolCallEvent(
                run_id=run_id,
                parent_run_id=self._parent_run_id,
                tool_name=tool_name,
                args=dict(args),
                tool_call_id=tool_call_id,
                ts=utc_now_iso(),
            )
        )

    # 发布工具调用结束事件（含耗时与错误分类）
    async def _publish_tool_result(
        self, run_id: str, tool_name: str, tool_call_id: str, result: ToolResult
    ) -> None:
        await self._bus.publish(
            ToolResultEvent(
                run_id=run_id,
                parent_run_id=self._parent_run_id,
                tool_name=tool_name,
                content=result.content,
                is_error=result.is_error,
                error_type=result.error_type,
                elapsed_ms=result.elapsed_ms,
                tool_call_id=tool_call_id,
                ts=utc_now_iso(),
            )
        )
