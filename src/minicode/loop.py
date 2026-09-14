"""Agent Loop：驱动 plan→act→observe 循环"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from minicode.context import ExecutionContext
from minicode.tools.invocation import invoke_tool
from minicode.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from minicode.compact.compactor import Compactor
    from minicode.llm.base import DeltaCallback, LLMProvider
    from minicode.tools.permissions import PermissionManager

logger = logging.getLogger(__name__)


class AgentLoop:
    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        *,
        compactor: Compactor | None = None,
        compact_threshold: float = 0.80,
        permission_manager: PermissionManager | None = None,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._compactor = compactor
        self._compact_threshold = compact_threshold
        self._permission_manager = permission_manager

    async def run(
        self,
        context: ExecutionContext,
        *,
        on_delta: DeltaCallback | None = None,
        on_tool_call: Any | None = None,
        on_tool_result: Any | None = None,
    ) -> None:
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
                    on_delta=on_delta,
                )
            except asyncio.CancelledError:
                context.mark_failed("cancelled")
                raise
            except Exception:
                logger.exception("LLM call failed run_id=%s step=%d", context.run_id, context.step)
                context.mark_failed("llm_error")
                break

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
            context.add_assistant_message(response.text, openai_tool_calls)

            # 有 tool_calls 就必须执行；但输出被 max_tokens 截断时参数可能是残缺 JSON，不执行
            if response.tool_calls and response.stop_reason != "length":
                for tc in response.tool_calls:
                    if on_tool_call:
                        await on_tool_call(tc.name, tc.input)
                    result = await invoke_tool(
                        self._registry, tc, context.run_id,
                        permission_manager=self._permission_manager,
                    )
                    if on_tool_result:
                        await on_tool_result(tc.name, result.content, result.is_error)
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
                await self._compactor.compact(context, self._provider)
