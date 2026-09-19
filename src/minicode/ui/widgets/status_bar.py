"""状态栏：运行状态（spinner + 计时）与上下文占用进度条"""

from __future__ import annotations

import time
from typing import Any

from rich.text import Text
from textual.widgets import Static

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_BAR_CELLS = 10


# 状态栏 Widget：App 在事件路由中驱动它更新；空闲时定时器空转不计帧。
# 运行标志必须叫 _run_active，不能用 _running——那是 Textual MessagePump 的内部
# 属性（挂载时被框架置 True），撞名会让首帧误显 running、end_run 反写泵标志
class StatusBar(Static):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._run_active = False
        self._frame = 0
        self._started: float | None = None
        self._ctx_pct: float | None = None
        self._tokens_in = 0
        self._tokens_out = 0

    def on_mount(self) -> None:
        self.set_interval(0.15, self._tick)
        self._repaint()

    # 进入运行状态：usage 计数清零（一轮 run 一份统计）
    def start_run(self) -> None:
        self._run_active = True
        self._started = time.monotonic()
        self._ctx_pct = None
        self._tokens_in = 0
        self._tokens_out = 0
        self._repaint()

    def end_run(self) -> None:
        self._run_active = False
        self._started = None
        self._repaint()

    # LlmUsageEvent 驱动：ctx 占用与 token 计数
    def set_usage(self, ctx_pct: float, tokens_in: int, tokens_out: int) -> None:
        self._ctx_pct = ctx_pct
        self._tokens_in = tokens_in
        self._tokens_out = tokens_out
        self._repaint()

    def _tick(self) -> None:
        if self._run_active:
            self._frame += 1
            self._repaint()

    def _repaint(self) -> None:
        text = Text(no_wrap=True)
        if self._run_active:
            elapsed = time.monotonic() - self._started if self._started is not None else 0.0
            text.append(f"{_SPINNER[self._frame % len(_SPINNER)]} ", style="bold cyan")
            text.append(f"running {elapsed:.0f}s", style="bold")
        else:
            text.append("● ", style="bold green")
            text.append("ready", style="bold")
        text.append("   ctx ", style="dim")
        pct = self._ctx_pct
        if pct is None:
            text.append("-", style="dim")
        else:
            filled = min(_BAR_CELLS, int(pct * _BAR_CELLS))
            color = "red" if pct >= 0.85 else "yellow" if pct >= 0.70 else "green"
            text.append("█" * filled + "░" * (_BAR_CELLS - filled), style=color)
            text.append(f" {pct:.0%}", style=color)
        text.append("  ·  in ", style="dim")
        text.append(f"{self._tokens_in:,}")
        text.append(" · out ", style="dim")
        text.append(f"{self._tokens_out:,}")
        self.update(text)
