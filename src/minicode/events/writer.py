"""EventWriter：EventBus 订阅者，把事件流按 trace 落盘为 JSONL 文件"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel

from minicode.events.bus import BaseEvent, RunFinishedEvent, RunStartedEvent

logger = logging.getLogger(__name__)


# 事件流订阅者：每个 trace 一个 <root>/<trace_id>/ 目录（events.jsonl + meta.json）。
# Session 与 Trace 的归属关系是 trace 级元数据（meta.json），不体现在目录层级上：
# chat 模式的 trace 带 session_id，run 模式的 trace session_id 为 null。
# seq 与 schema_version 只存在于落盘行中（内存事件模型不感知），
# 读取端用 pydantic 默认的额外字段忽略即可无损反序列化。
class EventWriter:
    # 落盘行格式版本；Schema 不兼容变更时递增
    SCHEMA_VERSION: Final[int] = 1

    # root 为 traces 根目录（如 .minicode/traces），其下按 trace_id 分目录
    def __init__(self, root: Path) -> None:
        self._root = root
        # 每个 trace 独立的自增序号，表达事件真实发生顺序
        self._seqs: dict[str, int] = {}
        # 各 trace 的 meta.json 内容（run.started 创建，run.finished 补全收尾字段）
        self._metas: dict[str, dict[str, Any]] = {}

    # EventBus 订阅入口：串行 publish 保证落盘顺序与事件发生顺序一致
    async def __call__(self, event: BaseModel) -> None:
        if not isinstance(event, BaseEvent):
            logger.warning("event writer skipped non-trace event type=%s", type(event).__name__)
            return

        seq = self._seqs.get(event.trace_id, 0) + 1
        self._seqs[event.trace_id] = seq

        payload = event.model_dump()
        payload["seq"] = seq
        payload["schema_version"] = self.SCHEMA_VERSION
        line = json.dumps(payload, ensure_ascii=False, default=str)

        path = self._trace_path(event.trace_id)
        # 阻塞 IO 丢线程池执行，避免拖慢事件循环（单订阅者串行 await 保证行序）
        await asyncio.to_thread(self._append_line, path, line)

        # run 级事件顺带维护 meta.json：归属关系（session_id/mode）只记在 trace 级元数据里
        if isinstance(event, RunStartedEvent):
            self._metas[event.trace_id] = {
                "trace_id": event.trace_id,
                "session_id": event.session_id,
                "mode": "run" if event.session_id is None else "chat",
                "status": "running",
                "created_at": event.ts,
                "finished_at": None,
            }
            await asyncio.to_thread(self._write_meta, event.trace_id)
        elif isinstance(event, RunFinishedEvent):
            meta = self._metas.setdefault(
                event.trace_id,
                {
                    "trace_id": event.trace_id,
                    "session_id": event.session_id,
                    "mode": "run" if event.session_id is None else "chat",
                    "status": "running",
                    "created_at": event.ts,
                    "finished_at": None,
                },
            )
            meta["status"] = event.status
            meta["finished_at"] = event.ts
            await asyncio.to_thread(self._write_meta, event.trace_id)

    def _trace_path(self, trace_id: str) -> Path:
        return self._root / trace_id / "events.jsonl"

    def _meta_path(self, trace_id: str) -> Path:
        return self._root / trace_id / "meta.json"

    # 每行独立 open-append-close：进程异常退出时最多丢最后一行，已写入内容不受损
    @staticmethod
    def _append_line(path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # meta.json 整体重写（体量极小，无需追加语义）
    def _write_meta(self, trace_id: str) -> None:
        path = self._meta_path(trace_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._metas[trace_id], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
