"""执行上下文：管理 agent run 的消息历史和状态"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# 执行上下文：维护消息历史、状态和终止条件
@dataclass
class ExecutionContext:
    run_id: str
    goal: str
    max_steps: int
    prefill_messages: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0
    status: str = "running"  # "running" | "success" | "failed"
    reason: str | None = None
    result: str = ""
    # subagent 角色可覆盖默认 system prompt
    system_prompt_override: str | None = None
    # 本轮是否发生过上下文压缩（决定收尾时重写还是追加会话文件）
    compacted: bool = False

    # 初始化消息历史：历史回放在前，本轮目标始终作为新的 user 消息追加在后
    def __post_init__(self) -> None:
        if self.prefill_messages:
            self.messages = [dict(m) for m in self.prefill_messages]
        # 不能写成 elif：有历史时也要追加本轮目标，否则模型看不到新问题
        self.messages.append({"role": "user", "content": self.goal})

    # 返回当前 run 的 system prompt；有 override 时跳过 base
    def system_prompt(self, base: str) -> str:
        if self.system_prompt_override:
            return self.system_prompt_override
        return base

    # 将 LLM 响应追加为 assistant 消息（OpenAI 格式）
    def add_assistant_message(
        self,
        text: str,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning: str = "",
    ) -> None:
        msg: dict[str, Any] = {"role": "assistant"}
        if tool_calls:
            msg["content"] = text or None
            msg["tool_calls"] = tool_calls
        else:
            msg["content"] = text
        # 统一带上该字段：思考模型要求它存在（空串即可），
        # 形状统一也避免中途切换模型后历史消息缺字段而被拒
        msg["reasoning_content"] = reasoning
        self.messages.append(msg)

    # 将工具调用结果追加为 role=tool 消息（OpenAI 格式）
    def add_tool_result(
        self, tool_call_id: str, content: str, is_error: bool = False
    ) -> None:
        result_content = content
        if is_error:
            result_content = f"[ERROR] {content}"
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result_content,
        })

    # 返回 True 表示 loop 应停止（状态不再是 running）
    def is_done(self) -> bool:
        return self.status != "running"

    # 将 run 标记为成功
    def mark_success(self) -> None:
        self.status = "success"

    # 将 run 标记为失败并记录原因
    def mark_failed(self, reason: str) -> None:
        self.status = "failed"
        self.reason = reason
