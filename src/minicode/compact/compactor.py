"""上下文压缩器：将长对话压缩为摘要"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from minicode.context import ExecutionContext
    from minicode.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# 压缩时原样保留的最近消息条数：刚拿到的工具结果不该被摘要抹掉，否则模型会重复读文件
_KEEP_RECENT = 4

_COMPACT_PROMPT = """\
You are compressing an agent conversation into a handoff summary.
Another LLM instance will continue this task from your summary alone — make it complete.

Structure your response with exactly these six sections:

## 1. Original Goal
One sentence describing what the user asked the agent to accomplish.

## 2. Completed Steps
Bullet list of what has been done. Be specific (file paths, commands run, decisions made).

## 3. Key Constraints & Discoveries
Facts learned during the run that affect future decisions.

## 4. Current File State
For each file that was created or modified: path, a one-line description of its current state.

## 5. Remaining TODOs
Ordered list of what still needs to be done to complete the original goal.

## 6. Critical Data
Any values the next LLM needs verbatim: IDs, tokens, exact error messages, config values.

Be concise. Omit reasoning steps and intermediate attempts. Keep conclusions.\
"""


# 压缩结果数据类
@dataclass
class CompactionResult:
    summary_text: str
    original_token_estimate: int
    summary_tokens: int


# 上下文压缩器：将长对话压缩为6段摘要
class Compactor:
    # 初始化压缩器，记录 session ID 用于日志
    def __init__(self, session_id: str = "") -> None:
        self._session_id = session_id

    # 压缩 ExecutionContext.messages：旧消息换成摘要，最近若干条原样保留
    async def compact(
        self,
        context: ExecutionContext,
        provider: LLMProvider,
        focus: str = "",
    ) -> CompactionResult | None:
        split = self._split_index(context.messages)
        head, tail = context.messages[:split], context.messages[split:]
        if not head:
            return None

        result = await self.compact_messages(head, provider, focus=focus)
        if result is None:
            return None

        context.messages = [
            {"role": "user", "content": result.summary_text},
            {"role": "assistant", "content": "Understood, I'll continue from this summary."},
        ] + [dict(m) for m in tail]
        # 标记本轮发生压缩，收尾时 runner 据此重写会话文件而非追加
        context.compacted = True
        logger.info(
            "context compacted session=%s run=%s original≈%d summary=%d tokens",
            self._session_id, context.run_id,
            result.original_token_estimate, result.summary_tokens,
        )
        return result

    # 计算压缩切分点：保留最近 _KEEP_RECENT 条，且不切断 tool_calls 与 tool 结果的配对
    def _split_index(self, messages: list[dict[str, Any]]) -> int:
        split = max(0, len(messages) - _KEEP_RECENT)
        # 若切在工具结果中间，这些 tool 消息会失去对应的 assistant tool_calls，API 会拒绝
        while split < len(messages) and messages[split].get("role") == "tool":
            split += 1
        return split

    # 纯函数式压缩：接收消息列表，返回 CompactionResult；失败时返回 None
    async def compact_messages(
        self,
        messages: list[dict[str, Any]],
        provider: LLMProvider,
        focus: str = "",
    ) -> CompactionResult | None:
        original_estimate = _estimate_tokens(messages)

        history_text = _messages_to_text(messages)
        prompt = _COMPACT_PROMPT
        if focus.strip():
            prompt += f"\n\nIMPORTANT: Pay special attention to: {focus.strip()}"

        compress_request: list[dict[str, object]] = [
            {"role": "user", "content": f"{prompt}\n\n---\n\n{history_text}"}
        ]

        try:
            response = await provider.chat(
                messages=compress_request,
                tool_schemas=[],
                run_id="compact",
                step=0,
                system="You are a helpful assistant that summarizes conversations.",
            )
        except Exception:
            logger.exception("compactor: LLM call failed, skipping compaction")
            return None

        summary_text = response.text.strip()
        if not summary_text:
            logger.warning("compactor: LLM returned empty summary, skipping compaction")
            return None

        summary_tokens = response.usage.output_tokens if response.usage else len(summary_text) // 4

        return CompactionResult(
            summary_text=summary_text,
            original_token_estimate=original_estimate,
            summary_tokens=summary_tokens,
        )


# 粗略估算消息列表的 token 数：content 与 tool_calls 参数都计入，按字符数 / 4 折算
def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    chars = 0
    for msg in messages:
        chars += len(str(msg.get("content") or ""))
        for tc in msg.get("tool_calls") or []:
            func = tc.get("function", {})
            chars += len(str(func.get("name", "")))
            chars += len(str(func.get("arguments", "")))
    return chars // 4


# 将消息列表序列化为可供 LLM 阅读的纯文本（OpenAI 格式）
def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = str(msg.get("content") or "")
        tool_call_id = msg.get("tool_call_id")

        chunks: list[str] = []
        if role == "TOOL" and tool_call_id:
            # 工具结果用成对标签包住内容，便于摘要模型分辨边界
            chunks.append(f"<tool_result id={tool_call_id}>")
            if content:
                chunks.append(content)
            chunks.append("</tool_result>")
        else:
            if content:
                chunks.append(content)
            for tc in msg.get("tool_calls") or []:
                func = tc.get("function", {})
                chunks.append(
                    f"<tool_call name={func.get('name')} id={tc.get('id')}>\n"
                    f"{func.get('arguments', '{}')}\n</tool_call>"
                )
        parts.append(f"[{role}]\n" + "\n".join(chunks))
    return "\n\n".join(parts)
