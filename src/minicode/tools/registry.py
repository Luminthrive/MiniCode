"""工具注册表：管理工具的注册和查询"""

from __future__ import annotations

from typing import Any

from minicode.tools.base import BaseTool


class ToolRegistry:
    # 初始化空注册表
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # 注册工具；同名覆盖
    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    # 按名称查找工具，不存在返回 None
    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    # 返回所有工具的 OpenAI function calling 格式 schema 列表
    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in self._tools.values()
        ]
