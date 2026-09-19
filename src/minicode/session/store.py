"""会话存储：JSONL 消息持久化（OpenAI 格式）"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from minicode.session.model import Session

logger = logging.getLogger(__name__)

# 支持持久化的角色
_VALID_ROLES = {"user", "assistant", "tool"}


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


# 基于文件系统的会话存储
class SessionStore:
    # 初始化 session 文件存储根目录
    def __init__(self, root: Path) -> None:
        self._root = root.expanduser()
        self._root.mkdir(parents=True, exist_ok=True)

    # 返回指定 session 的目录路径
    def session_dir(self, sid: str) -> Path:
        return self._root / sid

    # 将 session meta 写入 meta.json
    def write_meta(self, session: Session) -> None:
        path = self.session_dir(session.id)
        path.mkdir(parents=True, exist_ok=True)
        (path / "meta.json").write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    # 从 meta.json 读取 session meta
    def read_meta(self, sid: str) -> Session | None:
        path = self.session_dir(sid) / "meta.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Session.from_dict(data)

    # 批量追加一次 run 新产生的消息到 thread.jsonl（完整保留 OpenAI 格式）
    def append_messages(
        self,
        sid: str,
        messages: list[dict[str, Any]],
        run_id: str,
    ) -> None:
        path = self.session_dir(sid)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "thread.jsonl").open("a", encoding="utf-8") as f:
            for msg in messages:
                role = msg.get("role", "")
                if role not in _VALID_ROLES:
                    continue
                row: dict[str, Any] = {
                    "ts": _now(),
                    "role": role,
                    "run_id": run_id,
                }
                # assistant 消息可能包含 tool_calls 与思考内容
                if role == "assistant":
                    row["content"] = msg.get("content", "")
                    if msg.get("tool_calls"):
                        row["tool_calls"] = msg["tool_calls"]
                    row["reasoning_content"] = msg.get("reasoning_content", "")
                # tool 消息包含 tool_call_id
                elif role == "tool":
                    row["tool_call_id"] = msg.get("tool_call_id", "")
                    row["content"] = msg.get("content", "")
                # user 消息
                else:
                    row["content"] = msg.get("content", "")
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 备份现有 thread.jsonl 后整体重写（压缩后使用，使磁盘状态与内存一致）
    def rewrite_messages(
        self,
        sid: str,
        messages: list[dict[str, Any]],
        run_id: str,
    ) -> None:
        path = self.session_dir(sid)
        path.mkdir(parents=True, exist_ok=True)
        thread = path / "thread.jsonl"
        if thread.exists():
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            thread.rename(path / f"thread_{ts}.jsonl.bak")
        self.append_messages(sid, messages, run_id)

    # 读取完整 thread 并返回 OpenAI 格式 messages（含 tool 角色）
    def read_messages(self, sid: str) -> list[dict[str, Any]]:
        path = self.session_dir(sid) / "thread.jsonl"
        if not path.exists():
            return []

        messages: list[dict[str, Any]] = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skip broken thread row sid=%s line=%s", sid, line_no)
                continue
            role = row.get("role")
            if role not in _VALID_ROLES:
                continue

            msg: dict[str, Any] = {"role": role}

            if role == "assistant":
                msg["content"] = row.get("content", "")
                if row.get("tool_calls"):
                    msg["tool_calls"] = row["tool_calls"]
                # 旧记录缺该字段时补空串，使历史消息在切换模型后仍可回放
                msg["reasoning_content"] = row.get("reasoning_content", "")
            elif role == "tool":
                msg["tool_call_id"] = row.get("tool_call_id", "")
                msg["content"] = row.get("content", "")
            else:
                msg["content"] = row.get("content", "")

            messages.append(msg)
        return messages
