"""权限沙箱：分层渐进式工具权限评估（路径边界 → 安全名单 → 黑名单 → 会话缓存 → 审批）"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minicode.events.bus import (
    EventBus,
    PermissionDecidedEvent,
    PermissionRequestEvent,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

# 权限评估结果
VERDICT_ALLOW = "allow"
VERDICT_DENY = "deny"
VERDICT_ASK = "ask"

# 默认拒绝模式（危险命令）
_DEFAULT_DENY_PATTERNS: list[str] = [
    "rm -rf /",
    "rm -rf /*",
    "format c:",
    "del /s /q",
    "shutdown",
    "reboot",
    "mkfs",
    ":(){ :|:& };:",  # fork bomb
]

# 默认安全工具（无需审批）
_SAFE_TOOLS: frozenset[str] = frozenset({"read_file", "list_dir", "spawn_agent"})


# 权限评估结果数据类
@dataclass
class PermissionVerdict:
    """权限评估结果"""
    verdict: str  # "allow" | "deny" | "ask"
    reason: str = ""
    # 是否已缓存（同一工具+参数的后续调用不再审批）
    cached: bool = False


# 权限审批回调类型
type ApprovalCallback = Callable[[str, dict[str, Any]], Awaitable[bool]]


# 权限管理器：分层渐进式评估
class PermissionManager:
    """分层渐进式工具权限评估：路径边界 → 安全名单 → deny_patterns → 会话缓存 → 审批"""

    # 初始化权限管理器
    def __init__(
        self,
        cwd: Path | None = None,
        deny_patterns: list[str] | None = None,
        safe_tools: frozenset[str] | None = None,
        approval_callback: ApprovalCallback | None = None,
        approval_timeout: float = 30.0,
        bus: EventBus | None = None,
        trace_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self._cwd = (cwd or Path.cwd()).resolve()
        self._deny_patterns = deny_patterns or _DEFAULT_DENY_PATTERNS
        self._safe_tools = safe_tools or _SAFE_TOOLS
        self._approval_callback = approval_callback
        self._approval_timeout = approval_timeout
        # 审批事件化：注入 bus + 身份后，审批请求/决定发布为 trace 事件（否则审批对 trace 不可见）
        self._bus = bus
        self._trace_id = trace_id
        self._default_run_id = run_id
        # 会话级缓存：key = (tool_name, frozen_params) → verdict
        self._cache: dict[tuple[str, str], PermissionVerdict] = {}

    # 评估工具调用权限（3层渐进式）；run_id 由调用点传入以正确归属子代理事件
    async def check(
        self,
        tool_name: str,
        params: dict[str, Any],
        run_id: str | None = None,
    ) -> PermissionVerdict:
        # 第1层：文件操作先做路径边界检查——安全名单不能越过工作目录边界，
        # 否则 read_file/list_dir 凭安全名单即可读任意绝对路径
        if tool_name in ("read_file", "write_file", "edit_file", "list_dir"):
            outside = self._check_outside_cwd(params.get("path", ""))
            if outside:
                return PermissionVerdict(verdict=VERDICT_DENY, reason=outside)

        # 第2层：安全工具直接放行
        if tool_name in self._safe_tools:
            return PermissionVerdict(verdict=VERDICT_ALLOW, reason="safe tool")

        # 第3层：命令黑名单检查（bash 工具）
        if tool_name == "bash":
            deny = self._check_deny_patterns(params.get("command", ""))
            if deny:
                return PermissionVerdict(verdict=VERDICT_DENY, reason=deny)

        # 第4层：会话缓存检查
        cache_key = self._cache_key(tool_name, params)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return PermissionVerdict(
                verdict=cached.verdict,
                reason=cached.reason,
                cached=True,
            )

        # 第5层：需要审批（异步等待用户确认）；注入了 bus + trace 身份时审批同步发布为 trace 事件
        if self._approval_callback:
            event_run_id = run_id or self._default_run_id or ""
            bus, trace_id = self._bus, self._trace_id
            if bus is not None and trace_id is not None:
                await bus.publish(
                    PermissionRequestEvent(
                        trace_id=trace_id, run_id=event_run_id,
                        tool_name=tool_name, args=dict(params),
                        ts=utc_now_iso(),
                    )
                )
            try:
                approved = await asyncio.wait_for(
                    self._approval_callback(tool_name, params),
                    timeout=self._approval_timeout,
                )
                verdict = VERDICT_ALLOW if approved else VERDICT_DENY
                reason = "approved by user" if approved else "denied by user"
            except TimeoutError:
                verdict = VERDICT_DENY
                reason = f"approval timed out after {self._approval_timeout}s"
            if bus is not None and trace_id is not None:
                await bus.publish(
                    PermissionDecidedEvent(
                        trace_id=trace_id, run_id=event_run_id,
                        tool_name=tool_name, approved=verdict == VERDICT_ALLOW,
                        ts=utc_now_iso(),
                    )
                )
        else:
            # 无审批回调时默认允许
            verdict = VERDICT_ALLOW
            reason = "no approval callback, default allow"

        result = PermissionVerdict(verdict=verdict, reason=reason)
        self._cache[cache_key] = result
        return result

    # 检查命令是否匹配拒绝模式
    def _check_deny_patterns(self, command: str) -> str | None:
        """检查命令是否匹配拒绝模式，返回原因或 None"""
        cmd_lower = command.lower().strip()
        for pattern in self._deny_patterns:
            if fnmatch.fnmatch(cmd_lower, pattern.lower()):
                return f"command matches deny pattern: {pattern}"
        return None

    # 检查路径是否在工作目录外
    def _check_outside_cwd(self, path_str: str) -> str | None:
        """检查路径是否在工作目录外，返回原因或 None"""
        if not path_str:
            return None
        try:
            resolved = (self._cwd / path_str).resolve()
            # 用 is_relative_to 而非 startswith：后者对前缀同名的兄弟目录（D:\work vs D:\work2）误判
            if not resolved.is_relative_to(self._cwd):
                return f"path outside working directory: {path_str}"
        except (ValueError, OSError):
            return f"invalid path: {path_str}"
        return None

    # 生成缓存键
    def _cache_key(self, tool_name: str, params: dict[str, Any]) -> tuple[str, str]:
        """生成缓存键：工具名 + 参数的稳定字符串表示"""
        import json
        param_str = json.dumps(params, sort_keys=True, default=str)
        return (tool_name, param_str)

    # 清除会话缓存
    def clear_cache(self) -> None:
        """清除会话级权限缓存"""
        self._cache.clear()

    # 获取缓存统计
    def cache_stats(self) -> dict[str, int]:
        """返回缓存统计信息"""
        return {
            "total": len(self._cache),
            "allow": sum(1 for v in self._cache.values() if v.verdict == VERDICT_ALLOW),
            "deny": sum(1 for v in self._cache.values() if v.verdict == VERDICT_DENY),
        }
