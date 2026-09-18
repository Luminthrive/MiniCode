"""trace 回放：从落盘的 JSONL 文件还原事件序列"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from minicode.events.bus import Event


# trace 文件损坏（坏行/半行）时抛出，携带定位信息
class TraceReadError(ValueError):
    pass


# 逐行反序列化 events.jsonl；跳过空行，坏行报错并携带行号
def read_trace(path: Path) -> list[Event]:
    adapter: TypeAdapter[Any] = TypeAdapter(Event)
    events: list[Event] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                events.append(adapter.validate_json(line))
            except ValidationError as e:
                raise TraceReadError(f"trace 文件第 {lineno} 行无法解析为事件: {e}") from e
    return events
