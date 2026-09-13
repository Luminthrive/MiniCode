"""工具调用入口：权限检查、校验参数、限时调用、重试，返回 ToolResult"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, TYPE_CHECKING

from pydantic import ValidationError

from minicode.llm.types import ToolCallBlock
from minicode.tools.base import ToolResult
from minicode.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from minicode.tools.permissions import PermissionManager

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT: float = 120.0
_MAX_RETRIES: int = 2
_RETRY_BASE_S: float = 2.0


# 校验参数、权限检查、限时调用工具，失败时指数退避重试，返回 ToolResult（不抛异常）
async def invoke_tool(
    registry: ToolRegistry,
    tool_call: ToolCallBlock,
    run_id: str,
    timeout: float = _DEFAULT_TIMEOUT,
    permission_manager: PermissionManager | None = None,
) -> ToolResult:
    t0 = time.monotonic()

    # 计算从调用开始到现在的毫秒数
    def elapsed() -> int:
        return int((time.monotonic() - t0) * 1000)

    tool = registry.get(tool_call.name)
    if tool is None:
        return ToolResult(
            content=f"unknown tool: {tool_call.name}",
            is_error=True,
            error_type="runtime_error",
        )

    # 权限检查
    if permission_manager is not None:
        verdict = await permission_manager.check(tool_call.name, dict(tool_call.input))
        if verdict.verdict == "deny":
            return ToolResult(
                content=f"permission denied: {verdict.reason}",
                is_error=True,
                error_type="permission_denied",
            )

    if tool.params_model is not None:
        try:
            tool.params_model.model_validate(dict(tool_call.input))
        except ValidationError as exc:
            return ToolResult(
                content=str(exc),
                is_error=True,
                error_type="schema_error",
            )

    last_error: str | None = None
    for attempt in range(1, _MAX_RETRIES + 2):
        try:
            result = await asyncio.wait_for(
                tool.invoke(dict(tool_call.input)), timeout=timeout
            )
            if result.is_error:
                last_error = result.content
                if attempt <= _MAX_RETRIES:
                    await asyncio.sleep(_RETRY_BASE_S * (2 ** (attempt - 1)))
                    continue
                return result
            return result

        except TimeoutError:
            return ToolResult(
                content=f"tool timed out after {timeout}s",
                is_error=True,
                error_type="timeout",
            )
        except Exception as exc:
            last_error = str(exc)
            if attempt <= _MAX_RETRIES:
                await asyncio.sleep(_RETRY_BASE_S * (2 ** (attempt - 1)))
                continue

    return ToolResult(
        content=last_error or "internal error",
        is_error=True,
        error_type="runtime_error",
    )
