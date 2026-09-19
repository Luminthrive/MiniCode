"""OpenAI 兼容 LLM 提供者（支持流式输出）"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

import httpx

from minicode.llm.types import LlmResponse, ToolCallBlock, UsageStats

if TYPE_CHECKING:
    from minicode.llm.base import DeltaSink

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF_S = (1.0, 2.0, 4.0)
_DEFAULT_CONTEXT_WINDOW = 128_000


class OpenAIProvider:
    def __init__(self, model: str, base_url: str, api_key: str) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
        delta_sink: DeltaSink | None = None,
    ) -> LlmResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 8192,
            "stream_options": {"include_usage": True},
        }
        if system:
            payload["messages"] = [{"role": "system", "content": system}] + list(messages)
        if tool_schemas:
            payload["tools"] = tool_schemas

        # 始终使用流式调用
        async def _noop(text: str, _attempt: int) -> None:
            pass
        _fn = delta_sink or _noop
        return await self._chat_stream(payload, run_id, _fn)

    # 流式调用 API，逐片段送达文本，累积工具调用后返回完整响应
    async def _chat_stream(
        self,
        payload: dict[str, Any],
        run_id: str,
        delta_sink: DeltaSink,
    ) -> LlmResponse:
        payload["stream"] = True
        logger.debug("LLM stream request: model=%s msgs=%d tools=%d",
                      payload["model"], len(payload["messages"]),
                      len(payload.get("tools", [])))
        for attempt in range(1, _MAX_RETRIES + 1):
            # 累积缓冲必须每次尝试重置：断流重试时上一轮半截数据已不可信，
            # 保留会把重复内容叠加进最终响应
            text_parts: list[str] = []
            tool_calls: list[ToolCallBlock] = []
            usage: UsageStats | None = None
            stop_reason = "stop"
            tc_buffer: dict[int, dict[str, Any]] = {}
            # 思考模型的思考增量：不回传给用户，但要留存以便回传 API
            reasoning_parts: list[str] = []
            try:
                async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
                    if resp.status_code != 200:
                        # 保留响应体，否则调用方只能看到无信息的 "error"
                        raw = await resp.aread()
                        body = raw.decode("utf-8", errors="replace")[:500]
                        logger.error(
                            "LLM API %d error run_id=%s body=%s",
                            resp.status_code, run_id, body,
                        )
                        raise httpx.HTTPStatusError(
                            f"HTTP {resp.status_code}: {body}",
                            request=resp.request,
                            response=resp,
                        )

                    async for raw_line in resp.aiter_lines():
                        if not raw_line.startswith("data: "):
                            continue
                        data_str = raw_line[6:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            # usage 数据可能在空 choices 的 chunk 里
                            if chunk.get("usage"):
                                u = chunk["usage"]
                                usage = UsageStats(
                                    input_tokens=u.get("prompt_tokens", 0),
                                    output_tokens=u.get("completion_tokens", 0),
                                    context_pct=u.get("prompt_tokens", 0) / _DEFAULT_CONTEXT_WINDOW,
                                )
                            continue
                        choice = choices[0]
                        delta = choice.get("delta", {})
                        finish = choice.get("finish_reason")

                        if finish:
                            # tool_calls 只在确实有工具调用时才设为 "tool_calls"
                            if finish == "tool_calls" and bool(tc_buffer):
                                stop_reason = "tool_calls"
                            elif finish == "length":
                                # 输出被 max_tokens 截断，内容不完整
                                stop_reason = "length"
                            else:
                                stop_reason = "stop"

                        # 思考模型的思考增量：不回传给用户，但要留存以便回传 API
                        rc_delta = delta.get("reasoning_content")
                        if rc_delta:
                            reasoning_parts.append(rc_delta)

                        text_delta = delta.get("content", "")
                        if text_delta:
                            text_parts.append(text_delta)
                            await delta_sink(text_delta, attempt)

                        for tc_delta in delta.get("tool_calls") or []:
                            idx = tc_delta.get("index", 0)
                            if idx not in tc_buffer:
                                tc_buffer[idx] = {
                                    "id": tc_delta.get("id", ""),
                                    "name": "",
                                    "arguments": "",
                                }
                            buf = tc_buffer[idx]
                            if tc_delta.get("id"):
                                buf["id"] = tc_delta["id"]
                            func = tc_delta.get("function", {})
                            if func.get("name"):
                                buf["name"] = func["name"]
                            if func.get("arguments"):
                                buf["arguments"] += func["arguments"]

                        if chunk.get("usage"):
                            u = chunk["usage"]
                            usage = UsageStats(
                                input_tokens=u.get("prompt_tokens", 0),
                                output_tokens=u.get("completion_tokens", 0),
                                context_pct=u.get("prompt_tokens", 0) / _DEFAULT_CONTEXT_WINDOW,
                            )

                for idx in sorted(tc_buffer):
                    buf = tc_buffer[idx]
                    try:
                        args = json.loads(buf["arguments"]) if buf["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append(ToolCallBlock(
                        id=buf["id"] or f"call_{uuid.uuid4().hex[:8]}",
                        name=buf["name"],
                        input=args,
                    ))
                logger.debug("LLM stream result: stop=%s tool_calls=%d text_len=%d",
                             stop_reason, len(tool_calls), len("".join(text_parts)))
                break

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                # 4xx（除 408/429）是确定性错误，重试只会原样再失败
                if 400 <= status < 500 and status not in (408, 429):
                    raise
                logger.warning(
                    "LLM stream error (attempt %d/%d) run_id=%s: %s",
                    attempt, _MAX_RETRIES, run_id, exc,
                )
                if attempt == _MAX_RETRIES:
                    raise
                await asyncio.sleep(_RETRY_BACKOFF_S[attempt - 1])
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as exc:
                logger.warning(
                    "LLM stream error (attempt %d/%d) run_id=%s: %s",
                    attempt, _MAX_RETRIES, run_id, exc,
                )
                if attempt == _MAX_RETRIES:
                    raise
                await asyncio.sleep(_RETRY_BACKOFF_S[attempt - 1])

        return LlmResponse(
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            text="".join(text_parts),
            usage=usage,
            reasoning="".join(reasoning_parts),
        )

    async def close(self) -> None:
        await self._client.aclose()
