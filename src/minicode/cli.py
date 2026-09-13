"""CLI 入口：argparse 子命令（rich 渲染 + 流式输出）"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import UTC, datetime
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from minicode import __version__

console = Console()


def _setup_logging(verbose: bool) -> None:
    """配置日志级别"""
    import logging
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")


def main() -> None:
    """CLI 主入口"""
    parser = argparse.ArgumentParser(prog="minicode", description="MiniCode - 轻量级本地 AI Agent 系统")
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
        return args.get("path", "?")
    elif name == "write_file":
        return args.get("path", "?")
    elif name == "bash":
        cmd = args.get("command", "?")
        return cmd[:50] + ("..." if len(cmd) > 50 else "")
    elif name == "list_dir":
        return args.get("path", ".")
    elif name == "spawn_agent":
        return args.get("subagent_type", "default")
    else:
        return str(args)[:40]


# 流式文本收集器：边收边打印
def _make_delta_collector() -> tuple[Any, list[str]]:
    """创建流式文本回调，实时打印并累积"""
    buf: list[str] = []

    async def on_delta(text: str) -> None:
        buf.append(text)
        console.print(text, end="", highlight=False)

    return on_delta, buf


# 渲染 markdown 输出
def _render_markdown(text: str) -> None:
    """用 rich 渲染 markdown"""
    console.print()
    console.print(Markdown(text))
    console.print()


# 工具调用展示：单行
def _make_tool_call_printer() -> Any:
    """创建工具调用回调"""
    async def on_tool_call(name: str, args: dict[str, Any]) -> None:
        summary = _summarize_args(name, args)
        console.print()  # 确保工具调用另起一行
        console.print(f"  [bold yellow]⚡[/] [cyan]{name}[/]([dim]{summary}[/])", highlight=False)
    return on_tool_call


# 工具结果展示：单行
def _make_tool_result_printer() -> Any:
    """创建工具结果回调"""
    async def on_tool_result(name: str, content: str, is_error: bool) -> None:
        if is_error:
            tag = "[bold red]✗[/]"
        else:
            tag = "[bold green]✓[/]"
        preview = content[:60].replace("\n", " ").strip()
        if len(content) > 60:
            preview += "..."
        console.print(f"    {tag} [dim]{preview}[/]", highlight=False)
    return on_tool_result


def _run_command(goal: str) -> None:
    """执行一次 agent run"""
    from minicode.config import get_config
    from minicode.runner import AgentRunner

    config = get_config()
    runner = AgentRunner(config)

    t0 = time.time()
    console.print()
    console.print(Panel(goal, title="[bold]Goal[/bold]", border_style="blue"))

    on_delta, buf = _make_delta_collector()
    outcome = asyncio.run(runner.run_and_capture(
        goal,
        on_delta=on_delta,
        on_tool_call=_make_tool_call_printer(),
        on_tool_result=_make_tool_result_printer(),
    ))

    _render_markdown("".join(buf))

    elapsed = time.time() - t0
    status_style = "bold green" if outcome.status == "success" else "bold red"
    console.print(f"[{status_style}][{outcome.status}][/{status_style}] {elapsed:.1f}s")
    if outcome.status != "success":
        console.print(f"[red]Error: {outcome.reason}[/red]")
        sys.exit(1)


def _chat_command(session_id: str = "default") -> None:
    """交互式聊天模式（支持多会话持久化）"""
    from pathlib import Path
    from minicode.config import get_config
    from minicode.runner import AgentRunner
    from minicode.session.store import SessionStore
    from minicode.session.model import Session

    config = get_config()
    runner = AgentRunner(config)
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
    console.print(Panel("type 'quit' to exit", title="[bold]MiniCode Chat[/bold]", border_style="blue"))
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
        on_delta, buf = _make_delta_collector()
        outcome = asyncio.run(runner.run_and_capture(
            user_input,
            prefill_messages=history,
            on_delta=on_delta,
            on_tool_call=_make_tool_call_printer(),
            on_tool_result=_make_tool_result_printer(),
        ))
        elapsed = time.time() - t0

        # 渲染 markdown 回复
        _render_markdown("".join(buf))

        status_style = "bold green" if outcome.status == "success" else "bold red"
        console.print(f"[{status_style}][{outcome.status}][/{status_style}] {elapsed:.1f}s")
        console.print()

        # 持久化：保存新消息到 session
        history = history + outcome.new_messages
        if outcome.new_messages:
            store.append_messages(session.id, outcome.new_messages, run_id="chat")
