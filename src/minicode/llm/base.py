"""LLM 调用协议定义"""

from __future__ import annotations

from typing import Any, Callable, Awaitable, Protocol

from minicode.llm.types import LlmResponse


# 流式增量回调：接收文本片段
DeltaCallback = Callable[[str], Awaitable[None]]


# LLM 提供者协议：定义 chat 方法接口
class LLMProvider(Protocol):
    # 调用 LLM 并返回完整响应；on_delta 用于流式输出时逐片段回调
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> LlmResponse: ...
