"""chat 模式的全屏界面（Textual TUI）

读这个文件前只需知道一条规则：Textual 规定，屏幕只能由它自己的消息循环来改。
AI 的事件（流式文字、工具调用……）产生于别的后台任务，不能直接伸手改屏幕，
只能打包成"消息"投进队列，由界面循环自己取出来处理。
所以你会看到：转发事件 → 打包消息 → 界面取出渲染，这条链贯穿全文件。

三条主线：
    界面更新：AI 产生事件 → 控制器的事件总线 → forward_events 转发进界面队列
              → on_agent_event_message 按事件类型画到屏幕
    用户输入：输入框回车 → run_worker 把 controller.submit 放到后台跑
              （后台等 AI 的时候，界面照常响应键盘和刷新）
    权限审批：AI 想执行危险操作 → 弹出对话框 → 用户按键 → 结果交还权限层
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from textual import events
from textual.app import App
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widget import Widget

from minicode.application.agent_controller import SLASH_HELP, AgentController, SlashAction
from minicode.events.bus import (
    ContextCompactedEvent,
    EventBus,
    LlmDeltaEvent,
    LlmUsageEvent,
    PermissionDecidedEvent,
    RunFinishedEvent,
    SubagentFinishedEvent,
    SubagentStartedEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from minicode.ui.renderer import SubagentTracker
from minicode.ui.widgets.input_area import ChatTextArea
from minicode.ui.widgets.lines import (
    compacted_line,
    info_line,
    permission_decided_line,
    subagent_finish_line,
    subagent_start_line,
    tool_call_line,
    tool_call_line_from,
    tool_result_line,
    tool_result_line_from,
    user_line,
)
from minicode.ui.widgets.permission import PermissionDialog
from minicode.ui.widgets.status_bar import StatusBar
from minicode.ui.widgets.stream_block import StreamBlock, markdown_block

logger = logging.getLogger(__name__)


# 把 AI 产生的一个事件，打包成界面能收的"消息"（见模块开头的规则说明）
class AgentEventMessage(Message):
    def __init__(self, event: BaseModel) -> None:
        self.event = event
        super().__init__()


def forward_events(bus: EventBus, app: MiniCodeTuiApp) -> None:
    """订阅控制器的事件总线：每来一个事件，就转发一条消息给界面"""
    async def handle(event: BaseModel) -> None:
        app.post_message(AgentEventMessage(event))

    bus.subscribe(handle)


def setup_tui_logging(log_path: Path, verbose: bool = False) -> None:
    """日志全部重定向到文件：任何 stderr 输出都会破坏全屏渲染"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG if verbose else logging.WARNING)
    root.addHandler(handler)


# ---------------------------------------------------------------- 界面主体

class MiniCodeTuiApp(App[None]):
    TITLE = "MiniCode"

    CSS = """
    #status-bar {
        dock: top;
        height: 1;
        padding: 0 1;
        background: $surface;
    }
    #log-view {
        height: 1fr;
        padding: 0 1;
    }
    /* 审批对话框与输入框同住此容器：容器整体 dock 底部，内部普通流式布局。
       不能让二者各自 dock:bottom——Textual 的 bottom 停靠件彼此重叠分层，
       输入框会盖住对话框的参数正文与 y/n 页脚 */
    #bottom-stack {
        dock: bottom;
        height: auto;
    }
    #prompt {
        height: 5;
        border: round $primary;
    }
    #permission-dialog {
        height: auto;
        max-height: 16;
        margin-bottom: 1;
        padding: 0 1;
        border: round $warning;
        background: $surface;
    }
    .log-line {
        margin: 0;
    }
    .md-block {
        margin-bottom: 1;
    }
    .child {
        padding-left: 6;
    }
    StreamBlock {
        margin-bottom: 1;
    }
    """

    BINDINGS = [("escape", "cancel_run", "取消")]

    def __init__(self, controller: AgentController) -> None:
        super().__init__()
        self._controller = controller
        self._status: StatusBar | None = None
        self._stream_block: StreamBlock | None = None
        self._subagents = SubagentTracker()
        # run_id -> 已见最大 attempt：断流重试重发增量时据此重置当前块
        self._last_attempt: dict[str, int] = {}
        self._pending_dialog: PermissionDialog | None = None
        self._approval_future: asyncio.Future[bool] | None = None
        # 流式跟随滚动：上次 feed 时是否贴底、以及当时的滚动位置（用户上翻过就不再强拉）
        self._stream_follow = False
        self._stream_anchor: float = 0

        # 接线（对外只依赖 controller 一个对象，其余都在类内完成）：
        #   1. 把自己的对话框方法注册为控制器的审批回调（TUI 要读 diff，给 300s 超时）
        #   2. 把控制器总线上的事件桥接给自己渲染
        controller.attach_approval(self.request_approval, approval_timeout=300.0)
        forward_events(controller.bus, self)

    # ---------------------------------------------------------------- 组装

    def compose(self) -> Iterator[Widget]:
        yield StatusBar(id="status-bar")
        yield VerticalScroll(id="log-view")
        with Vertical(id="bottom-stack"):
            yield ChatTextArea(id="prompt")

    def on_mount(self) -> None:
        self._status = self.query_one("#status-bar", StatusBar)
        controller = self._controller
        if controller.is_new_session:
            self._write(info_line(f"✨ new session: {controller.session_id}"))
        else:
            self._write(info_line(
                f"📂 session: {controller.session_id}"
                f" · {len(controller.history)} messages"
            ))
        self._write(info_line(SLASH_HELP))
        # 状态栏回填上次运行收尾时的 usage，而不是空白占位
        if controller.last_usage is not None:
            usage = controller.last_usage
            self._status.set_usage(
                usage.context_pct, usage.input_tokens, usage.output_tokens
            )
        # 首帧先画出来，随后把历史会话回放进日志区
        self.call_after_refresh(self._render_history)
        self.query_one("#prompt", ChatTextArea).focus()

    # 历史回放：把持久化的会话消息按对话形态重放（助手内容走 Markdown 渲染）。
    # 消息里没有耗时/嵌套信息，工具行按根层级展示；错误行按持久化的 [ERROR] 前缀还原
    def _render_history(self) -> None:
        tool_names: dict[str, str] = {}
        for msg in self._controller.history:
            role = msg.get("role")
            if role == "user":
                self._write(user_line(str(msg.get("content") or "")))
            elif role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    name = str(fn.get("name") or "tool")
                    tool_names[str(tc.get("id") or "")] = name
                    try:
                        args = json.loads(str(fn.get("arguments") or "{}"))
                    except ValueError:
                        args = {}
                    self._write(tool_call_line_from(
                        name, args if isinstance(args, dict) else {}
                    ))
                content = msg.get("content")
                if content and str(content).strip():
                    self._write(markdown_block(str(content), classes="md-block"))
            elif role == "tool":
                content = str(msg.get("content") or "")
                is_error = content.startswith("[ERROR] ")
                if is_error:
                    content = content[len("[ERROR] "):]
                name = tool_names.get(str(msg.get("tool_call_id") or ""), "tool")
                self._write(tool_result_line_from(name, content, is_error=is_error))

    def _write(self, widget: Widget) -> None:
        """写入日志流行；用户上翻查看历史时不强行拉回底部"""
        view = self.query_one("#log-view", VerticalScroll)
        at_bottom = view.scroll_y >= view.max_scroll_y - 2
        view.mount(widget)
        if at_bottom:
            # 布局完成后再滚：新挂载 widget 的高度此时才计入 max_scroll_y
            self.call_after_refresh(view.scroll_end, animate=False)

    def _set_input_enabled(self, enabled: bool) -> None:
        prompt = self.query_one("#prompt", ChatTextArea)
        prompt.disabled = not enabled
        if enabled:
            prompt.focus()

    # ---------------------------------------------------- 事件消费（渲染）

    def on_agent_event_message(self, message: AgentEventMessage) -> None:
        event = message.event
        if isinstance(event, LlmDeltaEvent):
            self._handle_delta(event)
            return
        # 任何非 token 事件先切段：流式块边界与工具/文本交错天然对齐
        self._break_stream()

        if isinstance(event, LlmUsageEvent):
            if self._subagents.add_usage(event):
                return  # 子代理 usage 已累计，结束行统一展示
            if self._status is not None:
                self._status.set_usage(
                    event.context_pct, event.input_tokens, event.output_tokens
                )
        elif isinstance(event, ToolCallEvent):
            self._write(tool_call_line(event, self._subagents.add_tool_call(event)))
        elif isinstance(event, ToolResultEvent):
            self._write(tool_result_line(event, self._subagents.is_child(event)))
        elif isinstance(event, ContextCompactedEvent):
            self._write(compacted_line(event, self._subagents.is_child(event)))
        elif isinstance(event, SubagentStartedEvent):
            self._subagents.start(event)
            self._write(subagent_start_line(event))
        elif isinstance(event, SubagentFinishedEvent):
            agg = self._subagents.finish(event)
            self._write(subagent_finish_line(event, agg))
        elif isinstance(event, PermissionDecidedEvent):
            self._write(permission_decided_line(event, self._subagents.is_child(event)))
        elif isinstance(event, RunFinishedEvent):
            if self._status is not None:
                self._status.end_run()
        # RunStarted / PermissionRequest 不渲染：
        # 入口行由提交时写入；审批交互由对话框负责，decided 事件补审计行

    # 判断事件是否来自子 agent 的逻辑在 SubagentTracker（与 Rich 渲染器共用）

    # 流式追加：连续 token 进同一块；断流重试时重置块文本
    def _handle_delta(self, event: LlmDeltaEvent) -> None:
        last = self._last_attempt.get(event.run_id, 1)
        if event.attempt > last:
            if self._stream_block is not None:
                self._stream_block.reset()
            self._write(info_line(f"↻ retry attempt {event.attempt}, output restarted"))
        self._last_attempt[event.run_id] = max(last, event.attempt)

        view = self.query_one("#log-view", VerticalScroll)
        # 记录 feed 时的贴底状态与位置：Markdown 块在节流 flush 时才长高，
        # 长高后的补滚动见 _keep_following（用户上翻过则不强拉）
        self._stream_follow = view.scroll_y >= view.max_scroll_y - 2
        self._stream_anchor = view.scroll_y
        if self._stream_block is None:
            self._stream_block = StreamBlock(
                on_flush=self._keep_following,
                classes="child" if self._subagents.current_child else None,
            )
            self._write(self._stream_block)
        self._stream_block.feed(event.text)
        if self._stream_follow:
            view.scroll_end(animate=False)

    # Markdown 块长高后的补滚动：仅在 feed 时贴底、且用户此后没有主动滚动时跟随
    def _keep_following(self) -> None:
        view = self.query_one("#log-view", VerticalScroll)
        if not (self._stream_follow and view.scroll_y >= self._stream_anchor):
            return
        # 等本轮布局完成再滚：新挂载的 Markdown 段落此时才计入 max_scroll_y
        self.call_after_refresh(view.scroll_end, animate=False)

    def _break_stream(self) -> None:
        if self._stream_block is not None:
            self._stream_block.finalize()
            self._stream_block = None

    # ---------------------------------------------------- 提交 / 取消

    def on_chat_text_area_submitted(self, message: ChatTextArea.Submitted) -> None:
        text = message.text
        action = self._controller.parse_slash(text)
        if action is not None:
            self._run_slash(action)
            return
        self._write(user_line(text))
        if self._status is not None:
            self._status.start_run()
        self._set_input_enabled(False)
        self._last_attempt.clear()
        self.run_worker(self._run_turn(text), name="run-turn", exclusive=False)

    # 斜杠命令（解析在 controller，执行在 UI）
    def _run_slash(self, action: SlashAction) -> None:
        if action is SlashAction.HELP:
            self._write(info_line(SLASH_HELP))
        elif action is SlashAction.CLEAR:
            self._break_stream()
            self.query_one("#log-view", VerticalScroll).remove_children()
        elif action is SlashAction.QUIT:
            self.exit()

    # 一轮对话：controller.submit 由 worker 调度，键盘/事件泵全程可用
    async def _run_turn(self, text: str) -> None:
        t0 = time.monotonic()
        try:
            outcome = await self._controller.submit(text)
        except asyncio.CancelledError:
            self._break_stream()
            self._write(info_line("⏹ run cancelled"))
            self._finish_turn()
            return
        except Exception as exc:
            logger.exception("run failed")
            self._break_stream()
            self._write(info_line(f"✗ run failed: {exc}", style="red"))
            self._finish_turn()
            return
        self._break_stream()
        elapsed = time.monotonic() - t0
        style = "green" if outcome.status == "success" else "red"
        self._write(info_line(f"[{outcome.status}] {elapsed:.1f}s", style=style))
        self._finish_turn()

    def _finish_turn(self) -> None:
        if self._status is not None:
            self._status.end_run()
        self._set_input_enabled(True)

    def action_cancel_run(self) -> None:
        if self._controller.cancel():
            self._write(info_line("⏹ cancelling..."))

    # ------------------------------------------------------------- 审批
    # 等待方式说明：AI 的后台任务调 request_approval 后会停下来等结果。
    # 做法是创建 Future（理解为"取件单"）：弹对话框 → 用户一按键，
    # 结果（同意/拒绝）就写进单据 → 后台任务拿到 True/False 继续跑。
    # 等待期间界面照常刷新，不会冻结。

    # 审批回调入口：在 agent worker 协程中被 await；挂对话框等用户决定
    async def request_approval(self, tool_name: str, params: dict[str, Any]) -> bool:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        dialog = PermissionDialog(tool_name, params, id="permission-dialog")
        self._approval_future = future
        self._pending_dialog = dialog
        # 非模态挂载：插在输入框上方（同一 bottom 容器内，普通流式布局不会互相遮挡），
        # 接管焦点但不抢整个屏幕
        self.query_one("#bottom-stack").mount(dialog, before="#prompt")
        dialog.focus()
        try:
            return await future
        finally:
            self._approval_future = None
            self._pending_dialog = None
            await dialog.remove()

    def on_permission_dialog_decided(self, message: PermissionDialog.Decided) -> None:
        future = self._approval_future
        if future is not None and not future.done():
            future.set_result(message.approved)

    # 焦点丢失兜底：对话框存在时 y/n/esc 无论焦点在哪都路由到决定
    def on_key(self, event: events.Key) -> None:
        dialog = self._pending_dialog
        if dialog is None or dialog.decided:
            return
        if event.key in ("y", "a", "n", "d", "escape"):
            event.stop()
            event.prevent_default()
            dialog.decide(event.key in ("y", "a"))
