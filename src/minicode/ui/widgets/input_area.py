"""多行输入框：Enter 提交，Alt/Shift/Cmd+Enter 换行"""

from __future__ import annotations

from textual import events
from textual.message import Message
from textual.widgets import TextArea


# 输入框：把换行键让位给提交键（对标主流 agent CLI 的输入习惯）
class ChatTextArea(TextArea):
    # 提交消息：widget 命名空间 → App 层 on_chat_text_area_submitted 消费
    class Submitted(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            value = self.text.strip()
            if value:
                self.clear()
                self.post_message(self.Submitted(value))
            return
        if event.key in ("alt+enter", "shift+enter", "cmd+enter"):
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)
