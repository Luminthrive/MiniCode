"""Shell 命令执行工具（Windows 兼容 + 安全防护）"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from minicode.tools.base import BaseTool, ToolResult

_MAX_OUTPUT_BYTES = 64 * 1024  # 64 KB
_DEFAULT_TIMEOUT = 60
_DANGEROUS_PATTERNS = ("..", "&amp;", "&amp;&amp;", "|", ";", "$(", "`")


# Shell 命令参数模型
class BashParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    command: str
    timeout: int = Field(default=_DEFAULT_TIMEOUT, ge=1, le=120)


# 检查命令是否包含危险模式
def _is_dangerous(command: str) -> str | None:
    """检查命令是否包含路径遍历或注入风险，返回原因或 None"""
    cmd_lower = command.lower().strip()
    if ".." in cmd_lower:
        return "path traversal not allowed (..)"
    return None


# Shell 命令执行工具：Windows 用 PowerShell，其他平台用 bash
class BashTool(BaseTool):
    params_model = BashParams
    name = "bash"
    description = (
        "Execute a shell command and return its output (stdout + stderr combined). "
        "Non-interactive only — commands requiring user input will hang and time out. "
        "On Windows, uses PowerShell for proper Unicode support. "
        "Path traversal (..) is forbidden. Output is truncated at 64 KB."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": f"Maximum seconds to wait (default {_DEFAULT_TIMEOUT}, max 120).",
            },
        },
        "required": ["command"],
    }

    # 在子进程中执行 shell 命令，合并 stdout/stderr，超时或非零退出码时返回错误
    async def invoke(self, params: dict[str, Any]) -> ToolResult:
        p = BashParams.model_validate(params)
        command = p.command
        timeout = p.timeout

        # 安全检查
        danger = _is_dangerous(command)
        if danger:
            return ToolResult(content=danger, is_error=True, error_type="runtime_error")

        # Windows 使用 PowerShell 以支持中文输出
        if sys.platform == "win32":
            proc = await asyncio.create_subprocess_exec(
                "powershell", "-NoProfile", "-Command", command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        else:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

        try:
            stdout_bytes, _ = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return ToolResult(
                content=f"[timeout after {timeout}s]",
                is_error=True,
                error_type="timeout",
            )
        except Exception as exc:
            return ToolResult(content=str(exc), is_error=True, error_type="runtime_error")

        # Windows PowerShell 输出 UTF-8，cmd.exe 输出 GBK
        if sys.platform == "win32":
            output = stdout_bytes.decode("utf-8", errors="replace")
        else:
            output = stdout_bytes.decode("utf-8", errors="replace")

        truncated = len(stdout_bytes) > _MAX_OUTPUT_BYTES
        if truncated:
            output = output[:_MAX_OUTPUT_BYTES] + "\n[truncated]"

        returncode = proc.returncode or 0
        if returncode != 0:
            return ToolResult(
                content=f"[exit {returncode}]\n{output}",
                is_error=True,
                error_type="runtime_error",
            )
        return ToolResult(content=output or "[no output]")
