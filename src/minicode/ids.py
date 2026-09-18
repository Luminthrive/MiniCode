"""运行时标识符生成：随机 UUID 型 ID，时间信息一律由事件的 ts 字段表达"""

from __future__ import annotations

import uuid


# 生成带前缀的随机 ID（如 trace_xxx / run_xxx），不编码时间与顺序
def create_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"
