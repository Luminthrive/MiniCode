"""应用层控制器：chat（多轮会话）与 run（一次性任务）统一的编排入口

持有总线、落盘订阅与审批通道，向上承接 CLI/TUI（未来 Web API 同样适用），
向下装配 AgentRunner。本层不做 LLM/Tool 调用，也不拼接提示词。
两种模式只有会话维度不同：session_id=None 即 run 模式（无会话持久化），
其余装配规则（bus/EventWriter/审批/取消）完全一致。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from minicode.config import SESSIONS_DIR, TRACES_DIR, MiniConfig
from minicode.events.bus import EventBus, LlmUsageEvent
from minicode.events.metrics import last_session_usage
from minicode.events.writer import EventWriter
from minicode.runner import AgentRunner, RunOutcome
from minicode.session.model import Session
from minicode.session.store import SessionStore
from minicode.tools.permissions import ApprovalCallback

# 斜杠命令（补全/帮助文案由 UI 层基于此表生成）
SLASH_HELP = "命令: /help 帮助 · /clear 清屏 · /quit 退出"


# 斜杠命令动作
class SlashAction(Enum):
    HELP = "help"
    CLEAR = "clear"
    QUIT = "quit"


_SLASH_COMMANDS: dict[str, SlashAction] = {
    "help": SlashAction.HELP,
    "clear": SlashAction.CLEAR,
    "quit": SlashAction.QUIT,
    "exit": SlashAction.QUIT,
}


# 应用层控制器：入口层（CLI/TUI）只与本类对话，不直接持有 runner/store
#
# 用法（两种模式只差 session_id 一个参数）：
#     AgentController(config)                    # chat：多轮会话，历史持久化到 sessions/
#     AgentController(config, session_id=None)   # run：一次性任务，不落盘
#     outcome = await controller.submit("任务")   # 提交并等跑完
#     controller.cancel()                        # 取消进行中的一轮
class AgentController:
    # session_id=None 即 run 模式；审批回调可在构造时注入（run 的终端回调现成），
    # 也可延后 attach_approval（TUI 回调需要 App 先构造，对话框挂载目标）
    def __init__(
        self,
        config: MiniConfig,
        session_id: str | None = "default",
        *,
        approval_callback: ApprovalCallback | None = None,
        approval_timeout: float | None = None,
        traces_dir: Path | None = None,
        sessions_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._session_id = session_id
        traces = traces_dir or TRACES_DIR
        self.bus = EventBus()
        # 每轮 submit 新建 trace_id，写入端按 trace 自动分目录
        self.bus.subscribe(EventWriter(traces))
        self._approval_callback = approval_callback
        self._approval_timeout = approval_timeout
        self._current_task: asyncio.Task[RunOutcome] | None = None

        if session_id is None:
            # run 模式：一次性任务，不建会话存储，历史仅驻留内存
            self._store: SessionStore | None = None
            self._is_new = False
            self._history: list[dict[str, Any]] = []
            self._last_usage: LlmUsageEvent | None = None
        else:
            # chat 模式：加载或创建会话；历史消息作为后续每轮的预填上下文
            self._store = SessionStore(sessions_dir or SESSIONS_DIR)
            session = self._store.read_meta(session_id)
            self._is_new = session is None
            if session is None:
                now = datetime.now(UTC).isoformat()
                self._store.write_meta(Session(
                    id=session_id, mode="chat", status="active",
                    title=session_id, created_at=now, updated_at=now,
                ))
            self._history = self._store.read_messages(session_id)
            # 上次运行收尾时的 usage（ctx 占用 + token 数），供状态栏初始回填
            self._last_usage = last_session_usage(traces, session_id)

    # 注入审批决定通道。两个时机任选其一：
    #   - 构造时传 approval_callback=...（run 模式：终端 y/n 回调当时就绪）
    #   - 运行中调本方法（chat 模式：回调指向 TUI 对话框方法，得等 App 建好）
    def attach_approval(
        self, callback: ApprovalCallback, approval_timeout: float | None = None
    ) -> None:
        self._approval_callback = callback
        self._approval_timeout = approval_timeout

    # 属性：供入口层渲染欢迎信息（run 模式无会话，is_new_session 恒 False）
    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def is_new_session(self) -> bool:
        return self._is_new

    @property
    def history(self) -> list[dict[str, Any]]:
        return self._history

    # 上次运行收尾时的 usage（无 trace 的会话为 None）；供入口层回填状态栏
    @property
    def last_usage(self) -> LlmUsageEvent | None:
        return self._last_usage

    @property
    def busy(self) -> bool:
        return self._current_task is not None and not self._current_task.done()

    # 提交一轮任务并等它跑完；同一时刻只允许一个 run（UI 禁用输入是第一道防线，这里兜底）。
    # runner 每轮现建：AgentRunner 是无状态装配体，审批回调因此可以延迟注入
    async def submit(self, text: str) -> RunOutcome:
        if self.busy:
            raise RuntimeError("a run is already in progress")
        self._current_task = asyncio.current_task()
        try:
            runner = AgentRunner(
                self._config,
                bus=self.bus,
                approval_callback=self._approval_callback,
                approval_timeout=self._approval_timeout,
            )
            outcome = await runner.run_and_capture(
                text,
                session_id=self._session_id,
                store=self._store,
                prefill_messages=self._history,
            )
        finally:
            self._current_task = None
        self._history = outcome.messages
        return outcome

    # 取消当前 run；loop 捕获 CancelledError 后落盘失败状态（runner 已处理）
    def cancel(self) -> bool:
        task = self._current_task
        if task is not None and not task.done():
            task.cancel()
            return True
        return False

    # 解析斜杠命令；非命令文本返回 None（run 模式无输入回路，不会触达）
    def parse_slash(self, text: str) -> SlashAction | None:
        stripped = text.strip()
        if not stripped.startswith("/"):
            return None
        name = stripped.split(maxsplit=1)[0][1:].lower()
        return _SLASH_COMMANDS.get(name)
