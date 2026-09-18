"""LLM 调用协议定义"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from minicode.llm.types import LlmResponse

# 流式增量出口：参数为 (文本片段, 重试轮次 attempt，从 1 开始)
# （provider 层内部机制，由调用方决定是否转成事件）
type DeltaSink = Callable[[str, int], Awaitable[None]]


# LLM 提供者协议：定义 chat 方法接口
class LLMProvider(Protocol):
    # 调用 LLM 并返回完整响应；delta_sink 用于流式输出时逐片段送达
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
        delta_sink: DeltaSink | None = None,
    ) -> LlmResponse: ...
