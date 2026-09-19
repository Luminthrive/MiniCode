"""CLI 入口：argparse 子命令（rich 渲染 + 流式输出）"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from minicode import __version__
from minicode.events.bus import (
    ContextCompactedEvent,
    EventBus,
    LlmDeltaEvent,
    LlmUsageEvent,
    RunFinishedEvent,
    RunStartedEvent,
    SubagentFinishedEvent,
    SubagentStartedEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from minicode.tools.permissions import ApprovalCallback

console = Console()


# 配置日志级别
def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# CLI 主入口
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="minicode",
        description="MiniCode - 轻量级本地 AI Agent 系统",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示调试日志")
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="执行一次 agent run")
    run_parser.add_argument("goal", help="Task description")
    chat_parser = subparsers.add_parser("chat", help="交互式聊天模式")
    chat_parser.add_argument("-s", "--session", default="default", help="会话 ID（默认 default）")
    replay_parser = subparsers.add_parser("replay", help="离线回放一次历史 trace（不调用 LLM）")
    replay_parser.add_argument("trace_id", help="Trace ID（.minicode/traces 下的目录名）")
    stats_parser = subparsers.add_parser("stats", help="输出一次 trace 的运行指标汇总")
    stats_parser.add_argument("trace_id", help="Trace ID（.minicode/traces 下的目录名）")
    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command == "run":
        _run_command(args.goal)
    elif args.command == "chat":
        _chat_command(session_id=args.session)
    elif args.command == "replay":
        _replay_command(args.trace_id)
    elif args.command == "stats":
        _stats_command(args.trace_id)
    else:
        parser.print_help()


# 工具参数摘要：单行显示
def _summarize_args(name: str, args: dict[str, Any]) -> str:
    """生成工具参数的单行摘要；bash 完整显示命令，多行命令折叠为首行 + 行数"""
    if name == "bash":
        cmd = str(args.get("command", "?"))
        lines = cmd.splitlines() or ["?"]
        first = lines[0]
        if len(first) > 120:
            first = first[:120] + "..."
        if len(lines) > 1:
            first += f" (+{len(lines) - 1} lines)"
        return first
    if name == "edit_file":
        path = str(args.get("path", "?"))
        old_lines = str(args.get("old_string", "")).splitlines() or [""]
        new_lines = str(args.get("new_string", "")).splitlines() or [""]
        suffix = f" (+{len(old_lines) - 1} lines)" if len(old_lines) > 1 else ""

        # 取第一处差异行展示：old/new 首行常是相同的段落标记，差异行才有辨识度
        i = 0
        while i < min(len(old_lines), len(new_lines)) and old_lines[i] == new_lines[i]:
            i += 1
        old_first = old_lines[i] if i < len(old_lines) else ""
        new_first = new_lines[i] if i < len(new_lines) else ""

        def _clip(s: str) -> str:
            return s[:32] + "..." if len(s) > 32 else s

        return f"{path}{suffix}: {_clip(old_first)} -> {_clip(new_first)}"
    if name in ("read_file", "write_file", "list_dir"):
        return str(args.get("path", "."))
    if name == "spawn_agent":
        agent_type = args.get("subagent_type") or "default"
        return f"{agent_type} · {args.get('description', '')}"
    return str(args)[:60]


# 由事件 ts 计算时长（replay 回放时也能得到真实耗时）
def _format_duration(start: str, end: str) -> str:
    try:
        delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
        return f"{delta.total_seconds():.1f}s"
    except ValueError:
        return "-"


# 工具审批回调：危险操作在终端 y/n 确认；非交互环境直接拒绝，避免管道运行时挂死
def _make_approval_callback() -> ApprovalCallback:
    async def _ask(tool_name: str, params: dict[str, Any]) -> bool:
        if not sys.stdin.isatty():
            console.print(
                f"  [yellow]⚠[/] [dim]{tool_name} 需审批但当前非交互终端，已拒绝"
                f"（设 MINICODE_AUTO_APPROVE=1 可跳过审批）[/]",
                highlight=False,
            )
            return False
        summary = _summarize_args(tool_name, dict(params))
        try:
            # 阻塞式终端输入丢线程池执行，避免卡住事件循环
            return await asyncio.to_thread(
                Confirm.ask,
                f"  允许执行 [cyan]{tool_name}[/] [dim]{summary}[/]?",
                default=False,
            )
        except (EOFError, KeyboardInterrupt):
            return False

    return _ask


# 事件打印器：订阅 EventBus，将运行中的各类事件渲染到终端
class EventPrinter:
    def __init__(self, bus: EventBus) -> None:
        self._inline = False  # 上一条 llm.delta 之后尚未换行
        self._at_line_start = True  # 子代理流式输出是否位于行首（决定补不补 │ 前缀）
        self._current_child: str | None = None
        # run_id -> 已见最大 attempt：LLM 断流重试会重发增量，须重置渲染状态避免叠印
        self._last_attempt: dict[str, int] = {}
        # child_run_id -> 聚合信息：子代理块内的 usage/tool 累计，结束时统一展示
        self._subagents: dict[str, dict[str, Any]] = {}
        bus.subscribe(self.handle)

    # 补一个换行，避免后续事件行接在未结束的流式文本后半行
    def _ensure_newline(self) -> None:
        if self._inline:
            console.print()
            self._inline = False

    # 判断事件是否来自子 agent（决定缩进层级）
    def _is_child(self, event: BaseModel) -> bool:
        if getattr(event, "parent_run_id", None) is not None:
            return True
        return event.run_id in self._subagents  # type: ignore[attr-defined]

    # 流式输出：子代理文本带 │ 前缀与 dim 样式，与主 agent 输出区分层级
    def _print_delta(self, text: str) -> None:
        if self._current_child is None:
            console.print(text, end="", markup=False, highlight=False)
            self._inline = True
            return
        if not text:
            return
        if self._at_line_start:
            console.print("  │ ", end="", markup=False, highlight=False)
            self._at_line_start = False
        body = text.replace("\r\n", "\n")
        trailing = body.endswith("\n")
        if trailing:
            body = body[:-1]
        body = body.replace("\n", "\n  │ ")
        if body:
            console.print(body, end="", markup=False, highlight=False, style="dim")
        if trailing:
            console.print()
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
                console.print(
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
            agg = self._subagents.get(event.run_id)
            if agg is not None:
                # 子代理 usage 不逐条打印，累计进结束行
                agg["in"] += event.input_tokens
                agg["out"] += event.output_tokens
                return
            console.print(
                f"  [dim]· ctx {event.context_pct:.0%}"
                f" | in {event.input_tokens:,} · out {event.output_tokens:,}[/]",
                highlight=False,
            )
        elif isinstance(event, ToolCallEvent):
            child = self._is_child(event)
            if child and event.run_id in self._subagents:
                self._subagents[event.run_id]["tools"] += 1
            indent = "  │   " if child else "  "
            summary = _summarize_args(event.tool_name, dict(event.args))
            console.print(
                f"{indent}[bold yellow]⚡[/] [cyan]{event.tool_name}[/] [dim]{summary}[/]",
                highlight=False,
                no_wrap=True,
                overflow="ellipsis",
            )
        elif isinstance(event, ToolResultEvent):
            indent = "  │     " if self._is_child(event) else "    "
            elapsed = (
                f" [dim]({event.elapsed_ms}ms)[/]" if event.elapsed_ms is not None else ""
            )
            # 子代理最终报告已在块内流式输出过，这里只报大小不重复内容
            if event.tool_name == "spawn_agent" and not event.is_error:
                console.print(
                    f"{indent}[bold green]✓[/] [dim]subagent result: "
                    f"{len(event.content):,} chars{elapsed}[/]",
                    highlight=False,
                    no_wrap=True,
                    overflow="ellipsis",
                )
                return

            # 跳过开头空行，取第一个非空行做预览（PowerShell 输出常以空行开头）
            lines = event.content.splitlines()
            start = next((i for i, ln in enumerate(lines) if ln.strip()), None)
            if event.is_error:
                if start is None:
                    console.print(
                        f"{indent}[bold red]✗[/] [red](empty error)[/]{elapsed}",
                        highlight=False,
                        no_wrap=True,
                        overflow="ellipsis",
                    )
                    return
                kept = [ln.strip()[:100] for ln in lines[start:start + 2]]
                extra = len(lines) - start - len(kept)
                console.print(
                    f"{indent}[bold red]✗[/] [red]{kept[0]}[/]{elapsed}",
                    highlight=False,
                    no_wrap=True,
                    overflow="ellipsis",
                )
                for ln in kept[1:]:
                    console.print(
                        f"{indent}  [red]{ln}[/]",
                        highlight=False,
                        no_wrap=True,
                        overflow="ellipsis",
                    )
                if extra > 0:
                    console.print(
                        f"{indent}  [dim](+{extra} lines)[/]", highlight=False
                    )
            else:
                if start is None:
                    body, note = "(empty output)", ""
                else:
                    raw = lines[start].strip()
                    body = raw[:72] + "..." if len(raw) > 72 else raw
                    extra = len(lines) - start - 1
                    note = f" (+{extra} lines)" if extra > 0 else ""
                console.print(
                    f"{indent}[bold green]✓[/] [dim]{body}{note}{elapsed}[/]",
                    highlight=False,
                    no_wrap=True,
                    overflow="ellipsis",
                )
        elif isinstance(event, ContextCompactedEvent):
            indent = "  │   " if self._is_child(event) else "  "
            console.print(
                f"{indent}[bold magenta]📦[/] [dim]context compacted: "
                f"{event.original_tokens:,} → {event.summary_tokens:,} tokens[/]",
                highlight=False,
            )
        elif isinstance(event, SubagentStartedEvent):
            self._subagents[event.run_id] = {
                "description": event.description,
                "started": event.ts,
                "tools": 0,
                "in": 0,
                "out": 0,
            }
            self._current_child = event.run_id
            self._at_line_start = True
            console.print(
                f"  [bold blue]┌─[/] [bold]{event.description}[/] [dim]({event.run_id[:12]})[/]",
                highlight=False,
                no_wrap=True,
                overflow="ellipsis",
            )
        elif isinstance(event, SubagentFinishedEvent):
            agg = self._subagents.pop(event.run_id, None)
            self._current_child = None
            if event.status == "success":
                mark = "[bold green]✓[/]"
            else:
                mark = f"[bold red]✗[/][dim] ({event.reason or event.status})[/]"
            stats = ""
            if agg is not None:
                duration = _format_duration(agg["started"], event.ts)
                stats = (
                    f" [dim]{duration} · {agg['tools']} tools"
                    f" · in {agg['in']:,} · out {agg['out']:,}[/]"
                )
            console.print(
                f"  [bold blue]└─[/] {mark}{stats}",
                highlight=False, no_wrap=True, overflow="ellipsis",
            )
        # run.started / run.finished 的展示由命令层负责（Goal 面板与状态行）


def _run_command(goal: str) -> None:
    """执行一次 agent run"""
    from pathlib import Path

    from minicode.config import get_config
    from minicode.events.writer import EventWriter
    from minicode.runner import AgentRunner

    config = get_config()
    bus = EventBus()
    EventPrinter(bus)
    bus.subscribe(EventWriter(Path(".minicode/traces")))
    approval_callback = None if config.auto_approve else _make_approval_callback()
    runner = AgentRunner(config, bus=bus, approval_callback=approval_callback)

    t0 = time.time()
    console.print()
    console.print(Panel(goal, title="[bold]Goal[/bold]", border_style="blue"))

    outcome = asyncio.run(runner.run_and_capture(goal))

    elapsed = time.time() - t0
    console.print()
    status_style = "bold green" if outcome.status == "success" else "bold red"
    console.print(f"[{status_style}][{outcome.status}] {elapsed:.1f}s[/]", highlight=False)
    if outcome.status != "success":
        console.print(f"[red]Error: {outcome.reason}[/red]", highlight=False)
        sys.exit(1)


def _chat_command(session_id: str = "default") -> None:
    """交互式聊天模式（支持多会话持久化）"""
    from pathlib import Path

    from minicode.config import get_config
    from minicode.events.writer import EventWriter
    from minicode.runner import AgentRunner
    from minicode.session.model import Session
    from minicode.session.store import SessionStore

    config = get_config()
    bus = EventBus()
    EventPrinter(bus)
    # chat 多轮每轮新建 trace_id，写入端按 trace 自动分目录
    bus.subscribe(EventWriter(Path(".minicode/traces")))
    approval_callback = None if config.auto_approve else _make_approval_callback()
    runner = AgentRunner(config, bus=bus, approval_callback=approval_callback)
    store = SessionStore(Path(".minicode/sessions"))

    # 加载或创建会话
    session = store.read_meta(session_id)
    if session is None:
        session = Session(
            id=session_id,
            mode="chat",
            status="active",
            title=session_id,
            created_at=datetime.now(UTC).isoformat(),
            updated_at=datetime.now(UTC).isoformat(),
        )
        store.write_meta(session)
        console.print(f"[dim]✨ Created new session: {session_id}[/dim]")
    else:
        console.print(f"[dim]📂 Session: {session_id}[/dim]")

    history = store.read_messages(session.id)
    if history:
        console.print(f"[dim]📂 Loaded {len(history)} previous messages[/dim]")

    console.print()
    console.print(
        Panel("type 'quit' to exit", title="[bold]MiniCode Chat[/bold]", border_style="blue")
    )
    console.print()

    while True:
        try:
            user_input = console.input("[bold cyan]>[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if user_input.strip().lower() in ("quit", "exit"):
            console.print("[dim]Goodbye![/dim]")
            break
        if not user_input.strip():
            continue

        t0 = time.time()
        outcome = asyncio.run(runner.run_and_capture(
            user_input,
            session_id=session.id,
            store=store,
            prefill_messages=history,
        ))
        elapsed = time.time() - t0

        console.print()
        status_style = "bold green" if outcome.status == "success" else "bold red"
        console.print(f"[{status_style}][{outcome.status}] {elapsed:.1f}s[/]", highlight=False)
        console.print()

        # 落盘由 runner 负责；这里只需同步内存中的会话状态
        history = outcome.messages


def _replay_command(trace_id: str) -> None:
    """离线回放历史 trace：读取 events.jsonl 重新交给 EventPrinter 渲染（不调用 LLM）"""
    import asyncio
    from pathlib import Path

    from minicode.events.replay import TraceReadError, read_trace

    path = Path(".minicode/traces") / trace_id / "events.jsonl"
    if not path.exists():
        console.print(f"[red]trace 不存在: {trace_id}[/red]（未找到 {path}）")
        sys.exit(1)

    try:
        events = read_trace(path)
    except TraceReadError as e:
        console.print(f"[red]trace 文件损坏: {e}[/red]")
        sys.exit(1)

    # 头尾信息由命令层展示（与实时 run 的 Goal 面板/状态行同风格）
    started = next((e for e in events if isinstance(e, RunStartedEvent)), None)
    finished = next((e for e in events if isinstance(e, RunFinishedEvent)), None)
    goal = started.goal if started else "(未知)"
    status = finished.status if finished else "unknown"
    steps = finished.steps if finished else 0

    console.print()
    console.print(Panel(goal, title=f"[bold]Trace {trace_id}[/bold]", border_style="blue"))

    # 逐条重发到临时总线，零改动复用 EventPrinter 的渲染逻辑
    bus = EventBus()
    EventPrinter(bus)

    async def _replay() -> None:
        for event in events:
            await bus.publish(event)

    asyncio.run(_replay())

    console.print()
    status_style = "bold green" if status == "success" else "bold red"
    console.print(f"[{status_style}][replay] {status} · {steps} steps[/]", highlight=False)


def _stats_command(trace_id: str) -> None:
    """基于 trace 事件流计算运行指标（不调用 LLM）"""
    from pathlib import Path

    from minicode.events.metrics import summarize_trace
    from minicode.events.replay import TraceReadError, read_trace

    path = Path(".minicode/traces") / trace_id / "events.jsonl"
    if not path.exists():
        console.print(f"[red]trace 不存在: {trace_id}[/red]（未找到 {path}）")
        sys.exit(1)

    try:
        events = read_trace(path)
    except TraceReadError as e:
        console.print(f"[red]trace 文件损坏: {e}[/red]")
        sys.exit(1)

    if not events:
        console.print(f"[yellow]trace 为空: {trace_id}[/yellow]")
        return

    s = summarize_trace(events)
    duration = f"{s.duration_s:.1f}s" if s.duration_s is not None else "-"

    console.print()
    body = "\n".join(
        [
            f"status [bold]{s.status}[/bold] · steps {s.steps} · duration {duration}",
            f"LLM {s.llm_calls} calls · in {s.input_tokens:,} · out {s.output_tokens:,}"
            f" · retries {s.retries}",
            f"Tool {s.tool_calls} calls · failed {s.tool_errors}",
            f"Subagents {s.subagents} · Compactions {s.compactions}",
        ]
    )
    console.print(Panel(body, title=f"[bold]Trace {trace_id}[/bold]", border_style="blue"))

    if s.runs:
        run_table = Table(title="Runs")
        run_table.add_column("run_id")
        run_table.add_column("role")
        run_table.add_column("llm", justify="right")
        run_table.add_column("tokens in/out", justify="right")
        run_table.add_column("tool call/fail", justify="right")
        for r in s.runs:
            run_table.add_row(
                r.run_id,
                "subagent" if r.is_subagent else "root",
                str(r.llm_calls),
                f"{r.input_tokens:,}/{r.output_tokens:,}",
                f"{r.tool_calls}/{r.tool_errors}",
            )
        console.print(run_table)

    if s.tools:
        tool_table = Table(title="Tools")
        tool_table.add_column("tool")
        tool_table.add_column("calls", justify="right")
        tool_table.add_column("errors", justify="right")
        tool_table.add_column("avg ms", justify="right")
        for t in s.tools.values():
            avg = t.total_ms / t.calls if t.calls else 0.0
            tool_table.add_row(t.name, str(t.calls), str(t.errors), f"{avg:.0f}")
        console.print(tool_table)
