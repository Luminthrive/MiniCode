"""AgentRunner：组装依赖并执行 agent run"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from minicode.compact.compactor import Compactor
from minicode.config import MiniConfig
from minicode.context import ExecutionContext
from minicode.events.bus import EventBus, RunFinishedEvent, RunStartedEvent
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
    from minicode.llm.base import DeltaCallback, LLMProvider

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")


@dataclass
class RunOutcome:
    status: str
    result: str
    reason: str | None
    new_messages: list[dict[str, Any]]


class AgentRunner:
    def __init__(self, config: MiniConfig) -> None:
        self._config = config

    def _build_registry(
        self,
        *,
        provider: LLMProvider,
        bus: EventBus,
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
                parent_run_id=run_id,
                max_steps=max_steps,
                runs_dir=Path("runs"),
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
        on_delta: DeltaCallback | None = None,
        on_tool_call: Any | None = None,
        on_tool_result: Any | None = None,
    ) -> RunOutcome:
        run_id = run_id or _new_run_id()
        prefill_len = len(prefill_messages) if prefill_messages else 0

        bus = EventBus()

        context = ExecutionContext(
            run_id=run_id,
            goal=goal,
            max_steps=self._config.max_steps,
            system_prompt_override=system_prompt_override,
            prefill_messages=prefill_messages or [],
        )

        await bus.publish(RunStartedEvent(run_id=run_id, goal=goal, ts=_now()))

        cancelled = False
        try:
            provider = OpenAIProvider(
                model=self._config.llm_model,
                base_url=self._config.llm_base_url,
                api_key=self._config.llm_api_key,
            )

            registry = self._build_registry(
                provider=provider, bus=bus, run_id=run_id, max_steps=self._config.max_steps,
            )

            session_dir = Path("runs") / run_id
            compactor = Compactor(session_dir, session_id or "")
            permission_manager = PermissionManager()
            loop = AgentLoop(
                provider, registry,
                compactor=compactor,
                compact_threshold=self._config.compact_threshold,
                permission_manager=permission_manager,
            )
            await loop.run(context, on_delta=on_delta, on_tool_call=on_tool_call, on_tool_result=on_tool_result)
        except asyncio.CancelledError:
            cancelled = True
            if not context.is_done():
                context.mark_failed("cancelled")
        except Exception:
            logger.exception("agent run failed run_id=%s step=%d", run_id, context.step)
            if not context.is_done():
                context.mark_failed("llm_error")

        await bus.publish(RunFinishedEvent(run_id=run_id, status=context.status, reason=context.reason, steps=context.step, ts=_now()))

        if session_id and store:
            store.append_messages(session_id, context.messages[prefill_len:], run_id=run_id)

        if cancelled:
            raise asyncio.CancelledError()

        return RunOutcome(
            status=context.status,
            result=context.result,
            reason=context.reason,
            new_messages=context.messages[prefill_len:],
        )
