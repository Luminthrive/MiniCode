"""AgentRunner：组装依赖并执行 agent run"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from minicode.compact.compactor import Compactor
from minicode.config import MiniConfig
from minicode.context import ExecutionContext
from minicode.events.bus import (
    EventBus,
    RunFinishedEvent,
    RunStartedEvent,
    utc_now_iso,
)
from minicode.ids import create_id
from minicode.llm.provider import OpenAIProvider
from minicode.loop import AgentLoop
from minicode.session.store import SessionStore
from minicode.subagent.tool import SpawnAgentTool
from minicode.tools.builtin.bash import BashTool
from minicode.tools.builtin.list_dir import ListDirTool
from minicode.tools.builtin.read_file import ReadFileTool
from minicode.tools.builtin.write_file import WriteFileTool
from minicode.tools.permissions import PermissionManager
from minicode.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from minicode.llm.base import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class RunOutcome:
    status: str
    result: str
    reason: str | None
    # 本轮结束后的会话状态（含压缩后的形态），供调用方直接作为下一轮上下文
    messages: list[dict[str, Any]]
    # 本次任务的 Trace 标识（与事件流中的 trace_id 一致）
    trace_id: str


class AgentRunner:
    # bus 可由外部注入（CLI 预先订阅事件），未注入时每次 run 自建
    def __init__(self, config: MiniConfig, bus: EventBus | None = None) -> None:
        self._config = config
        self._bus = bus

    def _build_registry(
        self,
        *,
        provider: LLMProvider,
        bus: EventBus,
        trace_id: str,
        run_id: str,
        max_steps: int,
    ) -> ToolRegistry:
        registry = ToolRegistry()
        for t in [ReadFileTool(), BashTool(), WriteFileTool(), ListDirTool()]:
            registry.register(t)
        registry.register(
            SpawnAgentTool(
                provider=provider,
                parent_bus=bus,
                parent_trace_id=trace_id,
                parent_run_id=run_id,
                max_steps=max_steps,
                depth=0,
            )
        )
        return registry

    async def run_and_capture(
        self,
        goal: str,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
        store: SessionStore | None = None,
        system_prompt_override: str | None = None,
        prefill_messages: list[dict[str, Any]] | None = None,
    ) -> RunOutcome:
        # 一次 run_and_capture = 一次完整任务：trace_id 新建，子 agent 全部继承
        trace_id = create_id("trace")
        run_id = run_id or create_id("run")
        prefill_len = len(prefill_messages) if prefill_messages else 0

        bus = self._bus if self._bus is not None else EventBus()

        context = ExecutionContext(
            run_id=run_id,
            goal=goal,
            max_steps=self._config.max_steps,
            system_prompt_override=system_prompt_override,
            prefill_messages=prefill_messages or [],
        )

        # 压缩器会重新绑定 context.messages（而非原地修改），
        # 先留存原列表引用，确保压缩触发后仍能取到完整本轮消息用于持久化
        messages_log = context.messages

        await bus.publish(
            RunStartedEvent(
                trace_id=trace_id, run_id=run_id, goal=goal,
                session_id=session_id, ts=utc_now_iso(),
            )
        )

        cancelled = False
        try:
            provider = OpenAIProvider(
                model=self._config.llm_model,
                base_url=self._config.llm_base_url,
                api_key=self._config.llm_api_key,
            )

            registry = self._build_registry(
                provider=provider, bus=bus, trace_id=trace_id, run_id=run_id,
                max_steps=self._config.max_steps,
            )

            compactor = Compactor(session_id or "")
            permission_manager = PermissionManager()
            loop = AgentLoop(
                provider, registry, bus,
                trace_id=trace_id,
                compactor=compactor,
                compact_threshold=self._config.compact_threshold,
                permission_manager=permission_manager,
            )
            await loop.run(context)
        except asyncio.CancelledError:
            cancelled = True
            if not context.is_done():
                context.mark_failed("cancelled")
        except Exception:
            logger.exception("agent run failed run_id=%s step=%d", run_id, context.step)
            if not context.is_done():
                context.mark_failed("llm_error")

        await bus.publish(
            RunFinishedEvent(
                trace_id=trace_id,
                run_id=run_id,
                session_id=session_id,
                status=context.status,
                reason=context.reason,
                steps=context.step,
                ts=utc_now_iso(),
            )
        )

        # 压缩过则整体重写会话文件（原文件备份为 .bak），否则只追加本轮新消息；
        # 磁盘 IO 丢线程池执行，避免阻塞事件循环
        if session_id and store:
            if context.compacted:
                await asyncio.to_thread(
                    store.rewrite_messages, session_id, context.messages, run_id=run_id
                )
            else:
                await asyncio.to_thread(
                    store.append_messages, session_id, messages_log[prefill_len:], run_id=run_id
                )

        if cancelled:
            raise asyncio.CancelledError()

        return RunOutcome(
            status=context.status,
            result=context.result,
            reason=context.reason,
            messages=list(context.messages),
            trace_id=trace_id,
        )
