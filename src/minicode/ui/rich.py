"""Rich 终端渲染器：run/replay/管道场景的线性事件渲染（可脚本化、可回放）"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from pydantic import BaseModel
from rich.console import Console
from rich.prompt import Confirm

from minicode.events.bus import (
    ContextCompactedEvent,
    EventBus,
    LlmDeltaEvent,
    LlmUsageEvent,
    PermissionDecidedEvent,
    SubagentFinishedEvent,
    SubagentStartedEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from minicode.tools.permissions import ApprovalCallback
from minicode.ui.format import describe_tool_result, format_duration, summarize_args
from minicode.ui.renderer import SubagentTracker


# 终端审批回调（Rich 场景的决定通道实现）：危险操作在终端 y/n 确认；
# 非交互环境直接拒绝，避免管道运行时挂死。阻塞式输入丢线程池，不卡事件循环
def make_terminal_approval_callback(console: Console) -> ApprovalCallback:
    async def _ask(tool_name: str, params: dict[str, Any]) -> bool:
        if not sys.stdin.isatty():
            console.print(
                f"  [yellow]⚠[/] [dim]{tool_name} 需审批但当前非交互终端，已拒绝"
                f"（设 MINICODE_AUTO_APPROVE=1 可跳过审批）[/]",
                highlight=False,
            )
            return False
        summary = summarize_args(tool_name, dict(params))
        try:
            return await asyncio.to_thread(
                Confirm.ask,
                f"  允许执行 [cyan]{tool_name}[/] [dim]{summary}[/]?",
                default=False,
            )
        except (EOFError, KeyboardInterrupt):
            return False

    return _ask


# Rich 渲染器：订阅 EventBus，将运行中的各类事件线性打印到终端
class RichRenderer:
    # console 可注入：测试用 Console(record=True) 捕获输出
    def __init__(self, bus: EventBus, console: Console | None = None) -> None:
        self._console = console or Console()
        self._subagents = SubagentTracker()
        self._inline = False  # 上一条 llm.delta 之后尚未换行
        self._at_line_start = True  # 子代理流式输出是否位于行首（决定补不补 │ 前缀）
        # run_id -> 已见最大 attempt：LLM 断流重试会重发增量，须重置渲染状态避免叠印
        self._last_attempt: dict[str, int] = {}
        bus.subscribe(self.handle)

    # 补一个换行，避免后续事件行接在未结束的流式文本后半行
    def _ensure_newline(self) -> None:
        if self._inline:
            self._console.print()
            self._inline = False

    # 流式输出：子代理文本带 │ 前缀与 dim 样式，与主 agent 输出区分层级
    def _print_delta(self, text: str) -> None:
        if self._subagents.current_child is None:
            self._console.print(text, end="", markup=False, highlight=False)
            self._inline = True
            return
        if not text:
            return
        if self._at_line_start:
            self._console.print("  │ ", end="", markup=False, highlight=False)
            self._at_line_start = False
        body = text.replace("\r\n", "\n")
        trailing = body.endswith("\n")
        if trailing:
            body = body[:-1]
        body = body.replace("\n", "\n  │ ")
        if body:
            self._console.print(body, end="", markup=False, highlight=False, style="dim")
        if trailing:
            self._console.print()
            self._at_line_start = True
            self._inline = False
        else:
            self._inline = True

    async def handle(self, event: BaseModel) -> None:
        if isinstance(event, LlmDeltaEvent):
            last = self._last_attempt.get(event.run_id, 1)
            if event.attempt > last:
                # 断流重试：上一轮半截输出已作废，换行提示后从新流重新接收
                self._ensure_newline()
                self._console.print(
                    f"  [dim]↻ retry attempt {event.attempt}, output restarted[/]",
                    highlight=False,
                )
                self._at_line_start = True
            self._last_attempt[event.run_id] = max(last, event.attempt)
            # 前台模式下父与子 agent 不会同时流式输出，token 原样追加即可
            self._print_delta(event.text)
            return

        self._ensure_newline()
        # 非流式事件打印都会另起一行，流式前缀标志随之复位
        # （否则子代理下一段文本会因标志滞留 False 而顶格漏出 │ gutter）
        self._at_line_start = True

        if isinstance(event, LlmUsageEvent):
            if self._subagents.add_usage(event):
                return  # 子代理 usage 已累计进结束行，不逐条打印
            self._console.print(
                f"  [dim]· ctx {event.context_pct:.0%}"
                f" | in {event.input_tokens:,} · out {event.output_tokens:,}[/]",
                highlight=False,
            )
        elif isinstance(event, ToolCallEvent):
            child = self._subagents.add_tool_call(event)
            indent = "  │   " if child else "  "
            summary = summarize_args(event.tool_name, dict(event.args))
            self._console.print(
                f"{indent}[bold yellow]⚡[/] [cyan]{event.tool_name}[/] [dim]{summary}[/]",
                highlight=False,
                no_wrap=True,
                overflow="ellipsis",
            )
        elif isinstance(event, ToolResultEvent):
            self._render_tool_result(event)
        elif isinstance(event, ContextCompactedEvent):
            indent = "  │   " if self._subagents.is_child(event) else "  "
            self._console.print(
                f"{indent}[bold magenta]📦[/] [dim]context compacted: "
                f"{event.original_tokens:,} → {event.summary_tokens:,} tokens[/]",
                highlight=False,
            )
        elif isinstance(event, SubagentStartedEvent):
            self._subagents.start(event)
            self._at_line_start = True
            self._console.print(
                f"  [bold blue]┌─[/] [bold]{event.description}[/] [dim]({event.run_id[:12]})[/]",
                highlight=False,
                no_wrap=True,
                overflow="ellipsis",
            )
        elif isinstance(event, SubagentFinishedEvent):
            agg = self._subagents.finish(event)
            if event.status == "success":
                mark = "[bold green]✓[/]"
            else:
                mark = f"[bold red]✗[/][dim] ({event.reason or event.status})[/]"
            stats = ""
            if agg is not None:
                duration = format_duration(agg["started"], event.ts)
                stats = (
                    f" [dim]{duration} · {agg['tools']} tools"
                    f" · in {agg['in']:,} · out {agg['out']:,}[/]"
                )
            self._console.print(
                f"  [bold blue]└─[/] {mark}{stats}",
                highlight=False, no_wrap=True, overflow="ellipsis",
            )
        elif isinstance(event, PermissionDecidedEvent):
            # 审批审计行：live 场景紧随交互确认之后，replay 场景是审批的唯一可见痕迹
            mark = "[bold green]✓[/]" if event.approved else "[bold red]✗[/]"
            verdict = "approved" if event.approved else "denied"
            indent = "  │   " if self._subagents.is_child(event) else "  "
            self._console.print(
                f"{indent}{mark} [dim]permission {verdict}: {event.tool_name}[/]",
                highlight=False,
                no_wrap=True,
                overflow="ellipsis",
            )
        # permission.request 不渲染：live 场景由审批回调（Confirm.ask）负责交互，
        # replay 场景由 decided 审计行呈现，避免同一审批显示两遍
        # run.started / run.finished 的展示由命令层负责（Goal 面板与状态行）

    def _render_tool_result(self, event: ToolResultEvent) -> None:
        indent = "  │     " if self._subagents.is_child(event) else "    "
        elapsed = f" [dim]({event.elapsed_ms}ms)[/]" if event.elapsed_ms is not None else ""
        mark, body, note = describe_tool_result(
            event.tool_name, event.content, event.is_error
        )
        if mark == "✗":
            self._console.print(
                f"{indent}[bold red]✗[/] [red]{body}{note}[/]{elapsed}",
                highlight=False, no_wrap=True, overflow="ellipsis",
            )
        else:
            self._console.print(
                f"{indent}[bold green]✓[/] [dim]{body}{note}[/]{elapsed}",
                highlight=False, no_wrap=True, overflow="ellipsis",
            )
