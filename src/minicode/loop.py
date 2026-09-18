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

# 主 agent 系统提示词：任务分类 → 委派决策 → 转述式信息交接
# （v1 Agent-mediated Handoff：子代理结果经 ToolResult 返回，由主 agent 携带进下一棒委派简报）
MAIN_SYSTEM_PROMPT = """你是 MiniCode Agent，一个能操作文件系统的 AI 编码助手。

## 核心规则
1. 用户要求做某件事 → 用工具执行，不是只描述
2. 写文件/生成文档 → 调用 write_file，不是只输出文本
3. 纯闲聊时才直接回复

## 可用工具
- read_file / write_file / edit_file / bash / list_dir：文件系统操作
- 定点修改用 edit_file（old_string/new_string 精确替换）；新建或整文件重写用 write_file
- spawn_agent：派生子代理（planner/executor/reviewer）

## 第一步：任务分类（直接做 vs 委派）

A 类 —— 主 agent 直接做（read_file/write_file/bash/list_dir）：
- 查看/修改 1-2 个文件
- 执行单条命令
- 简单的单步或两步操作

B 类 —— 满足任一条件即一律委派（spawn_agent）：
- 需要分析 3 个以上文件
- 多步骤修改（读→分析→改→测）
- 代码审查、报告、重构
- 需要"规划→执行→审查"完整流程
判定为 B 类后：不要自己预先读文件，把探索和分析完整交给子代理。

## B 类工作流：委派链与信息交接
1. spawn planner：简报写清任务与已知约束 → 返回 Findings + Plan
2. spawn executor：简报携带 planner 的 Findings 与 Plan → 返回变更文件清单 + 执行报告
3. spawn reviewer：简报携带原始任务 + planner 的 Findings/Plan + 变更文件清单 + executor 执行报告 → 返回审查结论
4. reviewer 发现问题时：把问题清单交 executor 修复后重新审查，不要自己跳进去修改

## 委派简报规则（子代理上下文是干净的）
子代理看不到你的对话历史，只能看到传入的 prompt。每份简报必须包含四要素：
- 目标：要完成什么
- 期望产出：返回什么内容、什么格式
- 边界：不许做什么、有哪些约束
- 输入材料：需要携带的上游结果（见工作流）

携带上游结果时：不得丢失具体文件路径、symbol、步骤、验收标准与关键约束；可省略无关解释与重复内容。

## 子代理失败处理
1. 优先判断是否可以重新委派
2. 子任务失败但任务仍可继续 → 重新委派，并在简报中补充更明确的说明
3. 仅当主 agent 直接处理成本明显更低时才自己接管
4. 不要因一次子代理失败就从头重新规划整个任务

## bash 命令（Windows 环境）
- 用 PowerShell：Get-ChildItem/dir、Get-Content/type、Select-String
- 禁止路径遍历（..）
"""


class AgentLoop:
    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        bus: EventBus,
        *,
        trace_id: str,
        compactor: Compactor | None = None,
        compact_threshold: float = 0.80,
        permission_manager: PermissionManager | None = None,
        parent_run_id: str | None = None,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._bus = bus
        # 本 run 所属的 Trace（子 agent 由 SpawnAgentTool 传入父的 trace_id）
        self._trace_id = trace_id
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
                    system=MAIN_SYSTEM_PROMPT,
                    delta_sink=self._make_delta_sink(context.run_id, context.step),
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
                        trace_id=self._trace_id,
                        run_id=context.run_id,
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        context_pct=response.usage.context_pct,
                        step=context.step,
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
                    await self._publish_tool_call(
                        context.run_id, context.step, tc.name, tc.input, tc.id
                    )
                    result = await invoke_tool(
                        self._registry, tc, context.run_id,
                        permission_manager=self._permission_manager,
                    )
                    await self._publish_tool_result(
                        context.run_id, context.step, tc.name, tc.id, result
                    )
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
                            trace_id=self._trace_id,
                            run_id=context.run_id,
                            original_tokens=compacted.original_token_estimate,
                            summary_tokens=compacted.summary_tokens,
                            step=context.step,
                            ts=utc_now_iso(),
                        )
                    )

    # 构造流式增量出口：把 provider 的 token 片段转成 LlmDeltaEvent 发布到总线
    def _make_delta_sink(self, run_id: str, step: int) -> DeltaSink:
        async def sink(text: str, attempt: int) -> None:
            await self._bus.publish(
                LlmDeltaEvent(
                    trace_id=self._trace_id,
                    run_id=run_id,
                    text=text,
                    attempt=attempt,
                    step=step,
                    ts=utc_now_iso(),
                )
            )

        return sink

    # 发布工具调用开始事件
    async def _publish_tool_call(
        self, run_id: str, step: int, tool_name: str, args: dict[str, object], tool_call_id: str
    ) -> None:
        await self._bus.publish(
            ToolCallEvent(
                trace_id=self._trace_id,
                run_id=run_id,
                parent_run_id=self._parent_run_id,
                tool_name=tool_name,
                args=dict(args),
                tool_call_id=tool_call_id,
                step=step,
                ts=utc_now_iso(),
            )
        )

    # 发布工具调用结束事件（含耗时与错误分类）
    async def _publish_tool_result(
        self, run_id: str, step: int, tool_name: str, tool_call_id: str, result: ToolResult
    ) -> None:
        await self._bus.publish(
            ToolResultEvent(
                trace_id=self._trace_id,
                run_id=run_id,
                parent_run_id=self._parent_run_id,
                tool_name=tool_name,
                content=result.content,
                is_error=result.is_error,
                error_type=result.error_type,
                elapsed_ms=result.elapsed_ms,
                tool_call_id=tool_call_id,
                step=step,
                ts=utc_now_iso(),
            )
        )
