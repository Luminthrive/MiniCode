"""写入文件工具"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from minicode.tools.base import BaseTool, ToolResult, check_path_safety

_MAX_BYTES = 1 * 1024 * 1024  # 1 MB


# 同步写盘（建父目录 + 写入）：经 asyncio.to_thread 调用，避免阻塞事件循环
def _write_sync(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# 写入文件参数模型
class WriteFileParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str
    content: str


# 写入文件工具：超 1MB 拒绝，禁止 .. 路径遍历，自动创建父目录
class WriteFileTool(BaseTool):
    params_model = WriteFileParams
    name = "write_file"
    description = (
        "Write text content to a file, creating it (and any parent directories) if it "
        "does not exist, or overwriting it if it does. "
        "Path must be relative to the current working directory. "
        "Content size is limited to 1 MB."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the file (relative to current working directory).",
            },
            "content": {
                "type": "string",
                "description": "Text content to write.",
            },
        },
        "required": ["path", "content"],
    }

    # 写入文件内容；超 1MB 拒绝；禁止 .. 路径遍历；自动创建父目录
    async def invoke(self, params: dict[str, Any]) -> ToolResult:
        p = WriteFileParams.model_validate(params)
        path_str = p.path
        content = p.content

        safety_error = check_path_safety(path_str)
        if safety_error:
            return ToolResult(content=safety_error, is_error=True, error_type="permission_denied")

        encoded = content.encode("utf-8")
        if len(encoded) > _MAX_BYTES:
            return ToolResult(
                content=f"content too large: {len(encoded)} bytes (limit 1 MB)",
                is_error=True,
                error_type="runtime_error",
            )

        path = Path(path_str)
        await asyncio.to_thread(_write_sync, path, content)

        return ToolResult(content=f"wrote {len(encoded)} bytes to {path_str}")
