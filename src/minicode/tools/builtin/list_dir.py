"""目录列表工具"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from minicode.tools.base import BaseTool, ToolResult, check_path_safety

_MAX_DEPTH = 4
_MAX_ENTRIES = 200


# 目录列表参数模型
class ListDirParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str = "."
    max_depth: int = Field(default=2, ge=1, le=_MAX_DEPTH)


# 以树状格式列出目录内容工具，深度和条数有上限
class ListDirTool(BaseTool):
    params_model = ListDirParams
    name = "list_dir"
    description = (
        "List the contents of a directory as a tree. "
        "Path must be relative to the current working directory. "
        "Hidden entries (starting with .) are included. "
        f"Maximum depth is {_MAX_DEPTH}, maximum total entries is {_MAX_ENTRIES}."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the directory (default '.').",
            },
            "max_depth": {
                "type": "integer",
                "description": f"How many levels deep to recurse (default 2, max {_MAX_DEPTH}).",
            },
        },
        "required": [],
    }

    # 以树状格式列出目录内容，深度和条数有上限
    async def invoke(self, params: dict[str, Any]) -> ToolResult:
        p = ListDirParams.model_validate(params)
        path_str = p.path
        max_depth = p.max_depth

        safety_error = check_path_safety(path_str)
        if safety_error:
            return ToolResult(content=safety_error, is_error=True, error_type="permission_denied")

        root = Path(path_str)
        if not root.exists():
            raise FileNotFoundError(f"no such directory: {path_str}")
        if not root.is_dir():
            raise NotADirectoryError(f"not a directory: {path_str}")

        # 递归遍历为同步磁盘 IO，丢线程池执行，避免阻塞事件循环
        lines = await asyncio.to_thread(_walk_tree, root, max_depth)
        return ToolResult(content="\n".join(lines))


# 递归遍历目录并生成树状结构（同步磁盘 IO，经 asyncio.to_thread 调用）
def _walk_tree(root: Path, max_depth: int) -> list[str]:
    lines: list[str] = [str(root) + "/"]
    count = 0

    def _walk(directory: Path, depth: int, prefix: str) -> None:
        nonlocal count
        if depth > max_depth or count >= _MAX_ENTRIES:
            return
        entries = sorted(directory.iterdir(), key=lambda e: (e.is_file(), e.name))
        for i, entry in enumerate(entries):
            if count >= _MAX_ENTRIES:
                lines.append(f"{prefix}... (truncated)")
                return
            connector = "└── " if i == len(entries) - 1 else "├── "
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{connector}{entry.name}{suffix}")
            count += 1
            if entry.is_dir() and depth < max_depth:
                extension = "    " if i == len(entries) - 1 else "│   "
                _walk(entry, depth + 1, prefix + extension)

    _walk(root, 1, "")
    return lines
