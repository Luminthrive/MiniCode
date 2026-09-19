"""日志流行：把事件转成带样式的 Static 行（动态内容用 Text spans，防 markup 注入）"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.widgets import Static

from minicode.events.bus import (
    ContextCompactedEvent,
    PermissionDecidedEvent,
    SubagentFinishedEvent,
    SubagentStartedEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from minicode.ui.format import describe_tool_result, format_duration, summarize_args


# 行构造器：spans = (文本, 样式) 序列；no_wrap 超宽省略，child 加缩进 class
def _line(*spans: tuple[str, str], child: bool = False) -> Static:
    text = Text(no_wrap=True, overflow="ellipsis")
    for content, style in spans:
        text.append(content, style=style or None)
    return Static(text, classes="log-line child" if child else "log-line")


def user_line(text: str) -> Static:
    return _line(("❯ ", "bold cyan"), (text, "bold"))


def info_line(text: str, style: str = "dim") -> Static:
    return _line((text, style))


def tool_call_line_from(name: str, args: dict[str, Any], child: bool = False) -> Static:
    summary = summarize_args(name, dict(args))
    return _line(
        ("  ⚡ ", "bold yellow"),
        (name, "cyan"),
        (f" {summary}", "dim"),
        child=child,
    )


def tool_call_line(event: ToolCallEvent, child: bool) -> Static:
    return tool_call_line_from(event.tool_name, dict(event.args), child)


def tool_result_line_from(
    name: str,
    content: str,
    is_error: bool = False,
    elapsed_ms: int | None = None,
    child: bool = False,
) -> Static:
    elapsed = f" ({elapsed_ms}ms)" if elapsed_ms is not None else ""
    mark, body, note = describe_tool_result(name, content, is_error)
    if is_error:
        return _line(("  ✗ ", "bold red"), (f"{body}{note}", "red"),
                     (elapsed, "dim"), child=child)
    return _line(("  ✓ ", "bold green"), (f"{body}{note}", "dim"),
                 (elapsed, "dim"), child=child)


def tool_result_line(event: ToolResultEvent, child: bool) -> Static:
    return tool_result_line_from(
        event.tool_name, event.content, event.is_error, event.elapsed_ms, child
    )


def compacted_line(event: ContextCompactedEvent, child: bool) -> Static:
    return _line(
        ("  📦 ", "bold magenta"),
        (f"context compacted: {event.original_tokens:,} → {event.summary_tokens:,} tokens", "dim"),
        child=child,
    )


def subagent_start_line(event: SubagentStartedEvent) -> Static:
    return _line(
        ("  ┌─ ", "bold blue"),
        (event.description, "bold"),
        (f" ({event.run_id[:12]})", "dim"),
    )


# 结束行带聚合统计：agg 为 None 时（如 replay 起点缺失）只显示结论
def subagent_finish_line(event: SubagentFinishedEvent, agg: dict[str, Any] | None) -> Static:
    spans: list[tuple[str, str]] = [("  └─ ", "bold blue")]
    if event.status == "success":
        spans.append(("✓", "bold green"))
    else:
        spans.append(("✗", "bold red"))
        spans.append((f" ({event.reason or event.status})", "dim"))
    if agg is not None:
        duration = format_duration(agg["started"], event.ts)
        spans.append((
            f" {duration} · {agg['tools']} tools"
            f" · in {agg['in']:,} · out {agg['out']:,}",
            "dim",
        ))
    return _line(*spans)


def permission_decided_line(event: PermissionDecidedEvent, child: bool) -> Static:
    mark, mark_style = ("  ✓ ", "bold green") if event.approved else ("  ✗ ", "bold red")
    verdict = "approved" if event.approved else "denied"
    return _line((mark, mark_style), (f"permission {verdict}: {event.tool_name}", "dim"),
                 child=child)
