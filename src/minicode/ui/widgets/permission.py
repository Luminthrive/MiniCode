"""权限审批对话框：完整参数 + edit_file diff 红绿对照，非模态插在输入框上方

非模态而非 ModalScreen：避免与日志区争抢焦点（KamaClaude 验证过的方案）。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from rich.syntax import Syntax as RichSyntax
from rich.text import Text
from textual import events
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

from minicode.ui.format import build_edit_diff, summarize_args

_MAX_PARAMS_CHARS = 2000


# can_focus：挂载时接管焦点，y/n/esc 直接路由到 decide
class PermissionDialog(VerticalScroll):
    can_focus = True

    # 决定消息：widget 命名空间 → App 层 on_permission_dialog_decided 消费
    class Decided(Message):
        def __init__(self, approved: bool) -> None:
            self.approved = approved
            super().__init__()

    def __init__(self, tool_name: str, params: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tool_name = tool_name
        self._params = params
        self._decided = False

    @property
    def decided(self) -> bool:
        return self._decided

    def compose(self) -> Iterator[Widget]:
        summary = summarize_args(self._tool_name, dict(self._params))
        title = Text(no_wrap=True)
        title.append("⚡ ", style="bold yellow")
        title.append(self._tool_name, style="cyan")
        title.append(f" {summary}", style="dim")
        yield Static(title)

        # edit_file 展示真实 diff 红绿对照；其余工具展示完整参数
        diff = build_edit_diff(self._params) if self._tool_name == "edit_file" else ""
        if diff:
            yield Static(RichSyntax(diff, "diff", theme="ansi_dark", word_wrap=False))
        else:
            body = json.dumps(self._params, ensure_ascii=False, indent=2, default=str)
            if len(body) > _MAX_PARAMS_CHARS:
                body = body[:_MAX_PARAMS_CHARS] + "\n... (truncated)"
            yield Static(Text(body))

        footer = Text(no_wrap=True)
        footer.append("y", style="bold green")
        footer.append(" 允许 · ", style="dim")
        footer.append("n", style="bold red")
        footer.append(" 拒绝 · esc 拒绝", style="dim")
        yield Static(footer)

    # 决定入口：幂等（首次决定生效）；App 层焦点丢失兜底也会调这里
    def decide(self, approved: bool) -> None:
        if self._decided:
            return
        self._decided = True
        self.post_message(self.Decided(approved))

    async def _on_key(self, event: events.Key) -> None:
        if event.key in ("y", "a"):
            event.stop()
            event.prevent_default()
            self.decide(True)
        elif event.key in ("n", "d", "escape"):
            event.stop()
            event.prevent_default()
            self.decide(False)
        # 其余按键（滚动等）交回基类
