"""读取文件工具"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from minicode.tools.base import BaseTool, ToolResult, check_path_safety

_MAX_BYTES = 512 * 1024  # 512 KB


# 读取文件参数模型
class ReadFileParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str


# 读取文件内容工具：超 512KB 截断，禁止 .. 路径遍历
class ReadFileTool(BaseTool):
    params_model = ReadFileParams
    name = "read_file"
    description = (
        "Read the text content of a file. "
        "Path must be relative to the current working directory. "
        "Files larger than 512 KB are truncated."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the file (relative to current working directory).",
            }
        },
        "required": ["path"],
    }

    # 读取文件内容；超 512KB 截断；禁止 .. 路径遍历
    async def invoke(self, params: dict[str, Any]) -> ToolResult:
        path_str = ReadFileParams.model_validate(params).path

        safety_error = check_path_safety(path_str)
        if safety_error:
            return ToolResult(content=safety_error, is_error=True, error_type="permission_denied")

        path = Path(path_str)
        raw = await asyncio.to_thread(path.read_bytes)
        truncated = len(raw) > _MAX_BYTES
        text = raw[:_MAX_BYTES].decode("utf-8", errors="replace")
        if truncated:
            text += "\n[truncated]"

        return ToolResult(content=text)
