"""LLM 调用相关数据类型"""

from __future__ import annotations

from dataclasses import dataclass, field


# Token 使用统计
@dataclass
class UsageStats:
    input_tokens: int
    output_tokens: int
    context_pct: float = 0.0


# 工具调用块：id、函数名和参数
@dataclass
class ToolCallBlock:
    id: str
    name: str
    input: dict[str, object]


# LLM 响应：停止原因、工具调用列表、文本内容和使用统计
@dataclass
class LlmResponse:
    stop_reason: str  # "stop" | "tool_calls" | "length"（length 表示被 max_tokens 截断）
    tool_calls: list[ToolCallBlock] = field(default_factory=list)
    text: str = ""
    usage: UsageStats | None = None
