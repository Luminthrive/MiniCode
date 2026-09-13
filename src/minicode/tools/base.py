"""工具基础类和结果数据结构"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel


# 工具执行结果：包含内容、错误标志和错误类型
@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    # "runtime_error" | "timeout" | "schema_error"
    error_type: str | None = None


# 所有工具的抽象基类，定义工具接口和输入 schema
class BaseTool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any]
    params_model: ClassVar[type[BaseModel] | None] = None

    # 执行工具调用，返回结果或错误
    @abstractmethod
    async def invoke(self, params: dict[str, Any]) -> ToolResult: ...
