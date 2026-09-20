"""CLI 入口：四个子命令，每个命令一个 _xxx_command 组装函数

    run     一次性任务：AgentController(session_id=None) + RichRenderer 线性输出（可管道）
    chat    交互式聊天：AgentController + Textual TUI（全屏接管终端）
    replay  把历史 trace 的事件重放给 RichRenderer（离线，不调用 LLM）
    stats   从历史 trace 统计运行指标（离线，不调用 LLM）

两个约定：
    - 子命令的 import 写在函数体内：`minicode --version` 之类不必加载 rich/textual
    - chat 与 run 统一走 AgentController（Application 层），CLI 只负责选渲染器 + 装配
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from minicode import __version__
from minicode.events.bus import EventBus, RunFinishedEvent, RunStartedEvent

console = Console()


# 日志输出到 stderr（chat 模式例外：全屏渲染，_chat_command 内重定向到文件）
def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# 子命令解析器：新增命令 = 这里加一个 subparser + 一个 _command 函数 + main 里一行分发
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minicode",
        description="MiniCode - 轻量级本地 AI Agent 系统",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示调试日志")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="执行一次 agent run")
    run.add_argument("goal", help="Task description")

    chat = subparsers.add_parser("chat", help="交互式聊天模式（TUI）")
    chat.add_argument("-s", "--session", default="default", help="会话 ID（默认 default）")

    replay = subparsers.add_parser("replay", help="离线回放一次历史 trace（不调用 LLM）")
    replay.add_argument("trace_id", help="Trace ID（.minicode/traces 下的目录名）")

    stats = subparsers.add_parser("stats", help="输出一次 trace 的运行指标汇总")
    stats.add_argument("trace_id", help="Trace ID（.minicode/traces 下的目录名）")
    return parser


# CLI 主入口：只做分发；每个命令的组装逻辑在自己的 _command 里
def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "chat":
        _chat_command(args.session, verbose=args.verbose)
        return
    _setup_logging(args.verbose)
    if args.command == "run":
        _run_command(args.goal)
    elif args.command == "replay":
        _replay_command(args.trace_id)
    elif args.command == "stats":
        _stats_command(args.trace_id)


def _run_command(goal: str) -> None:
    """执行一次 agent run（RichRenderer 线性渲染，可管道）；装配统一在 AgentController"""
    from minicode.application.agent_controller import AgentController
    from minicode.config import get_config
    from minicode.ui.rich import RichRenderer, make_terminal_approval_callback

    config = get_config()
    # run 模式：session_id=None（一次性任务，无会话持久化）。
    # 审批回调由入口注入；auto_approve 时传 None = 跳过所有审批
    controller = AgentController(
        config,
        session_id=None,
        approval_callback=(
            None if config.auto_approve else make_terminal_approval_callback(console)
        ),
    )
    # 渲染器构造即订阅 controller.bus：run 期间事件实时打印，无需手动接线
    RichRenderer(controller.bus)

    t0 = time.time()
    console.print()
    console.print(Panel(goal, title="[bold]Goal[/bold]", border_style="blue"))

    outcome = asyncio.run(controller.submit(goal))

    elapsed = time.time() - t0
    console.print()
    status_style = "bold green" if outcome.status == "success" else "bold red"
    console.print(f"[{status_style}][{outcome.status}] {elapsed:.1f}s[/]", highlight=False)
    if outcome.status != "success":
        console.print(f"[red]Error: {outcome.reason}[/red]", highlight=False)
        sys.exit(1)


def _chat_command(session_id: str, verbose: bool) -> None:
    """交互式聊天：全屏 Textual TUI。组装与审批/事件接线都封装在 MiniCodeTuiApp 内，
    这里只负责：终端检查 → 日志重定向 → 建 controller → 建 App 并运行
    """
    from minicode.application.agent_controller import AgentController
    from minicode.config import TUI_LOG_PATH, get_config
    from minicode.ui.tui import MiniCodeTuiApp, setup_tui_logging

    # chat 需要交互终端（全屏接管）；脚本/管道场景请用 run
    if not sys.stdin.isatty():
        console.print(
            "[yellow]⚠[/yellow] [dim]chat 模式需要交互终端（Textual TUI）。"
            "脚本/管道场景请使用 minicode run。[/]",
            highlight=False,
        )
        sys.exit(1)
    setup_tui_logging(TUI_LOG_PATH, verbose=verbose)

    controller = AgentController(get_config(), session_id=session_id)
    MiniCodeTuiApp(controller).run()


def _replay_command(trace_id: str) -> None:
    """离线回放历史 trace：把落盘事件按顺序重发进总线，零改动复用 RichRenderer（不调 LLM）"""
    from minicode.config import TRACES_DIR
    from minicode.events.replay import TraceReadError, read_trace
    from minicode.ui.rich import RichRenderer

    path = TRACES_DIR / trace_id / "events.jsonl"
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

    # 逐条重发到临时总线，RichRenderer 像实时运行一样渲染
    bus = EventBus()
    RichRenderer(bus)

    async def _replay() -> None:
        for event in events:
            await bus.publish(event)

    asyncio.run(_replay())

    console.print()
    status_style = "bold green" if status == "success" else "bold red"
    console.print(f"[{status_style}][replay] {status} · {steps} steps[/]", highlight=False)


def _stats_command(trace_id: str) -> None:
    """基于 trace 事件流计算运行指标（不调用 LLM）"""
    from minicode.config import TRACES_DIR
    from minicode.events.metrics import summarize_trace
    from minicode.events.replay import TraceReadError, read_trace

    path = TRACES_DIR / trace_id / "events.jsonl"
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
