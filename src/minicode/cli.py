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

from minicode import __version__
from minicode.events.bus import (
    ContextCompactedEvent,
    EventBus,
    LlmDeltaEvent,
    LlmUsageEvent,
    SubagentFinishedEvent,
    SubagentStartedEvent,
    ToolCallEvent,
    ToolResultEvent,
)

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
    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command == "run":
        _run_command(args.goal)
    elif args.command == "chat":
        _chat_command(session_id=args.session)
    else:
        parser.print_help()


# 工具参数摘要：单行显示
def _summarize_args(name: str, args: dict[str, Any]) -> str:
    """生成工具参数的单行摘要"""
    if name == "read_file":
        return str(args.get("path", "?"))
    elif name == "write_file":
        return str(args.get("path", "?"))
    elif name == "bash":
        cmd = str(args.get("command", "?"))
        return cmd[:50] + ("..." if len(cmd) > 50 else "")
    elif name == "list_dir":
        return str(args.get("path", "."))
    elif name == "spawn_agent":
        return str(args.get("subagent_type", "default"))
    else:
        return str(args)[:40]


# 事件打印器：订阅 EventBus，将运行中的各类事件渲染到终端
class EventPrinter:
    def __init__(self, bus: EventBus) -> None:
        self._inline = False  # 上一条 llm.delta 之后尚未换行
        self._subagents: dict[str, str] = {}  # child_run_id -> description
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

    async def handle(self, event: BaseModel) -> None:
        if isinstance(event, LlmDeltaEvent):
            # 前台模式下父与子 agent 不会同时流式输出，token 原样追加即可
            console.print(event.text, end="", highlight=False)
            self._inline = True
            return

        self._ensure_newline()
        indent = "      " if self._is_child(event) else "  "

        if isinstance(event, ToolCallEvent):
            summary = _summarize_args(event.tool_name, dict(event.args))
            console.print(
                f"{indent}[bold yellow]⚡[/] [cyan]{event.tool_name}[/]([dim]{summary}[/])",
                highlight=False,
            )
        elif isinstance(event, ToolResultEvent):
            tag = "[bold red]✗[/]" if event.is_error else "[bold green]✓[/]"
            preview = event.content[:60].replace("\n", " ").strip()
            if len(event.content) > 60:
                preview += "..."
            elapsed = (
                f" [dim]({event.elapsed_ms}ms)[/]" if event.elapsed_ms is not None else ""
            )
            console.print(f"{indent}  {tag} [dim]{preview}[/]{elapsed}", highlight=False)
        elif isinstance(event, LlmUsageEvent):
            console.print(
                f"{indent}[dim]· ctx {event.context_pct:.0%}"
                f" | in {event.input_tokens:,} · out {event.output_tokens:,}[/]",
                highlight=False,
            )
        elif isinstance(event, ContextCompactedEvent):
            console.print(
                f"{indent}[bold magenta]📦[/] [dim]context compacted: "
                f"{event.original_tokens:,} → {event.summary_tokens:,} tokens[/]",
                highlight=False,
            )
        elif isinstance(event, SubagentStartedEvent):
            self._subagents[event.run_id] = event.description
            console.print(
                f"  [bold blue]┌─[/] [bold]{event.description}[/] [dim]({event.run_id})[/]",
                highlight=False,
            )
        elif isinstance(event, SubagentFinishedEvent):
            self._subagents.pop(event.run_id, None)
            if event.status == "success":
                mark = "[bold green]✓[/]"
            else:
                mark = f"[bold red]✗[/][dim] ({event.reason or event.status})[/]"
            console.print(f"  [bold blue]└─[/] {mark}", highlight=False)
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
    runner = AgentRunner(config, bus=bus)

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
    runner = AgentRunner(config, bus=bus)
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
