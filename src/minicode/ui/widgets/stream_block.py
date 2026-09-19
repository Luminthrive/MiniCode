"""流式 Markdown 块：AI 回复的"打字机"

feed() 收到一个词就先攒进缓冲区；每 0.2 秒把攒下的整段文字交给 Markdown
组件重新排版一次——逐字排版太浪费，攒一小批肉眼也看不出延迟。
块结束（finalize）时做最后一次排版，就是最终样子。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from textual.timer import Timer
from textual.widgets import Markdown

# 重渲染节流间隔：Markdown 全量重解析较贵，按 ~5Hz 批量合并 token
_FLUSH_INTERVAL = 0.2


# 流式块：内容用 Markdown 渲染；text 属性保持增量语义供断流重试/测试断言
class StreamBlock(Markdown):
    def __init__(
        self, on_flush: Callable[[], None] | None = None, **kwargs: Any
    ) -> None:
        super().__init__("", **kwargs)
        self._buffer = ""
        self._flush_timer: Timer | None = None
        # 每次 Markdown 重渲染（块长高）完成后的回调；App 用它做跟随滚动
        self._on_flush = on_flush

    @property
    def text(self) -> str:
        return self._buffer

    def feed(self, token: str) -> None:
        """累积一段文字（不叫 append：父类 Markdown.append 是另一个异步接口）"""
        self._buffer += token
        if self._flush_timer is None:
            self._flush_timer = self.set_timer(_FLUSH_INTERVAL, self._flush)

    # 断流重试：上一轮半截输出作废，块内文本整体重置
    def reset(self) -> None:
        self._buffer = ""
        self._cancel_flush()
        self._schedule_update()

    # 块结束钩子：取消待触发的节流，立即终稿渲染
    def finalize(self) -> None:
        self._cancel_flush()
        self._schedule_update()

    def _cancel_flush(self) -> None:
        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None

    def _flush(self) -> None:
        self._flush_timer = None
        self._schedule_update()

    def _schedule_update(self) -> None:
        # exclusive：合并积压的重渲染；每次都用全量 buffer，作废中的旧解析无害
        self.run_worker(self._update_and_notify(), group="stream-md", exclusive=True)

    async def _update_and_notify(self) -> None:
        await self.update(self._buffer)
        # 块长高（可能挂载了新段落）发生在这一刻之后，滚动逻辑挂在回调里才跟得上
        if self._on_flush is not None:
            self._on_flush()


# 静态 Markdown 块：历史回放等一次性渲染场景（组件挂载时自行渲染初值）
def markdown_block(text: str, classes: str | None = None) -> Markdown:
    return Markdown(text, classes=classes)
