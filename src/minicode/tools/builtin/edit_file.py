"""精确字符串替换编辑工具：定点修改，文件其余内容原样保留"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from minicode.tools.base import BaseTool, ToolResult, check_path_safety

_MAX_BYTES = 1 * 1024 * 1024  # 1 MB


# 编辑文件参数模型
class EditFileParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str
    old_string: str
    new_string: str
    replace_all: bool = False


# 编辑文件工具：old_string 精确匹配替换，未命中/多处命中均报错，不做模糊匹配
class EditFileTool(BaseTool):
    params_model = EditFileParams
    name = "edit_file"
    description = (
        "Make a targeted edit to an existing text file by replacing an exact string. "
        "old_string must match the file content exactly and be unique — include enough "
        "surrounding lines to disambiguate. Matching normalizes line endings, so an "
        "old_string with LF also matches CRLF files. Only the matched span changes; the "
        "rest of the file is preserved byte-for-byte. "
        "Use write_file instead to create new files or rewrite a whole file."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the file (relative to current working directory).",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to replace. Must appear in the file verbatim.",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text.",
            },
            "replace_all": {
                "type": "boolean",
                "description": (
                    "Replace every occurrence instead of just the first "
                    "(default false; requires a unique match when false)."
                ),
            },
        },
        "required": ["path", "old_string", "new_string"],
    }

    # 替换目标片段；未命中/多处命中/文件缺失等情况均返回明确错误
    async def invoke(self, params: dict[str, Any]) -> ToolResult:
        p = EditFileParams.model_validate(params)

        safety_error = check_path_safety(p.path)
        if safety_error:
            return ToolResult(
                content=safety_error,
                is_error=True,
                error_type="permission_denied",
                retryable=False,
            )

        if p.old_string == p.new_string:
            return ToolResult(
                content="old_string and new_string are identical; nothing to edit",
                is_error=True,
                error_type="runtime_error",
                retryable=False,
            )

        path = Path(p.path)
        if not path.is_file():
            return ToolResult(
                content=(
                    f"file not found: {p.path} — "
                    "edit_file cannot create files, use write_file instead"
                ),
                is_error=True,
                error_type="runtime_error",
                retryable=False,
            )

        try:
            raw = await asyncio.to_thread(path.read_bytes)
        except OSError as exc:
            return ToolResult(
                content=f"failed to read {p.path}: {exc}",
                is_error=True,
                error_type="runtime_error",
            )
        if len(raw) > _MAX_BYTES:
            return ToolResult(
                content=f"file too large: {len(raw)} bytes (limit 1 MB)",
                is_error=True,
                error_type="runtime_error",
                retryable=False,
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return ToolResult(
                content=f"{p.path} is not valid UTF-8 text; cannot edit",
                is_error=True,
                error_type="runtime_error",
                retryable=False,
            )

        # 匹配在归一化换行后进行：LF 的 old_string 也能命中 CRLF 文件
        crlf_pairs = text.count("\r\n")
        eol = "\r\n" if crlf_pairs and crlf_pairs * 2 >= text.count("\n") else "\n"
        norm = text.replace("\r\n", "\n")
        old = p.old_string.replace("\r\n", "\n")
        new = p.new_string.replace("\r\n", "\n")

        occurrences = norm.count(old)
        if occurrences == 0:
            return ToolResult(
                content=(
                    f"old_string not found in {p.path} — the file may have changed; "
                    "read_file it and retry with the exact text"
                ),
                is_error=True,
                error_type="runtime_error",
                retryable=False,
            )
        if occurrences > 1 and not p.replace_all:
            return ToolResult(
                content=(
                    f"old_string matched {occurrences} times in {p.path}; include more "
                    "surrounding context to make it unique, or set replace_all=true"
                ),
                is_error=True,
                error_type="runtime_error",
                retryable=False,
            )

        if p.replace_all:
            merged = norm.replace(old, new)
        else:
            merged = norm.replace(old, new, 1)
        # 写回前还原文件的主导换行风格
        if eol == "\r\n":
            merged = merged.replace("\n", "\r\n")

        if len(merged.encode("utf-8")) > _MAX_BYTES:
            return ToolResult(
                content="resulting content too large (limit 1 MB)",
                is_error=True,
                error_type="runtime_error",
                retryable=False,
            )

        # 字节写回：不做文本模式换行翻译，保持上面还原后的字节
        await asyncio.to_thread(path.write_bytes, merged.encode("utf-8"))
        replaced = occurrences if p.replace_all else 1
        return ToolResult(content=f"replaced {replaced} occurrence(s) in {p.path}")
