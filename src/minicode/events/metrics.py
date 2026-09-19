"""trace 运行指标：从事件序列计算 Run Summary（纯函数，不触碰 Agent）"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from minicode.events.bus import (
    ContextCompactedEvent,
    Event,
    LlmDeltaEvent,
    LlmUsageEvent,
    RunFinishedEvent,
    SubagentStartedEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from minicode.events.replay import TraceReadError, read_trace


# 单个 run（一个 Agent 实例）的明细指标
@dataclass
class RunStats:
    run_id: str
    # 是否子 agent（由 subagent.started 声明的 run_id 判定）
    is_subagent: bool
    # llm.usage 事件次数（每次 LLM 响应一条）
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # tool.call / tool.result(is_error) 计数
    tool_calls: int = 0
    tool_errors: int = 0


# 按工具名聚合的调用统计
@dataclass
class ToolStat:
    name: str
    calls: int = 0
    errors: int = 0
    total_ms: int = 0


# 整个 trace 的汇总指标
@dataclass
class TraceSummary:
    trace_id: str
    status: str = "unknown"
    # 取 run.finished 的 steps（主 agent 总步数）
    steps: int = 0
    # 首末事件时间差（秒）；事件缺时间戳时为 None
    duration_s: float | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # LLM 网络重试轮数：按 (run_id, step) 分组取最大 attempt 再减 1 后求和
    retries: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    subagents: int = 0
    compactions: int = 0
    # 按首次出现顺序排列的 run 明细
    runs: list[RunStats] = field(default_factory=list)
    # 按工具名聚合（名称即键）
    tools: dict[str, ToolStat] = field(default_factory=dict)


# 解析 ISO 8601 时间戳；格式异常返回 None（统计不应因坏时间戳而中断）
def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


# 遍历一次事件序列得到全部指标；空序列返回全零汇总
def summarize_trace(events: Sequence[Event]) -> TraceSummary:
    subagent_runs = {e.run_id for e in events if isinstance(e, SubagentStartedEvent)}
    run_map: dict[str, RunStats] = {}
    tools: dict[str, ToolStat] = {}

    # 惰性建明细行，保证 runs 按首次出现顺序输出
    def _run(run_id: str) -> RunStats:
        stats = run_map.get(run_id)
        if stats is None:
            stats = RunStats(run_id=run_id, is_subagent=run_id in subagent_runs)
            run_map[run_id] = stats
        return stats

    summary = TraceSummary(trace_id=events[0].trace_id if events else "")
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    # (run_id, step) -> 该次 LLM 调用见过的最大 attempt（一步一次调用）
    max_attempt: dict[tuple[str, int], int] = {}

    for event in events:
        ts = _parse_ts(event.ts)
        if ts is not None:
            if first_ts is None:
                first_ts = ts
            last_ts = ts

        if isinstance(event, RunFinishedEvent):
            summary.status = event.status
            summary.steps = max(summary.steps, event.steps)
        elif isinstance(event, LlmUsageEvent):
            run = _run(event.run_id)
            run.llm_calls += 1
            run.input_tokens += event.input_tokens
            run.output_tokens += event.output_tokens
        elif isinstance(event, LlmDeltaEvent):
            key = (event.run_id, event.step)
            max_attempt[key] = max(max_attempt.get(key, 1), event.attempt)
        elif isinstance(event, ToolCallEvent):
            _run(event.run_id).tool_calls += 1
            stat = tools.setdefault(event.tool_name, ToolStat(name=event.tool_name))
            stat.calls += 1
        elif isinstance(event, ToolResultEvent):
            run = _run(event.run_id)
            stat = tools.setdefault(event.tool_name, ToolStat(name=event.tool_name))
            if event.is_error:
                run.tool_errors += 1
                stat.errors += 1
            if event.elapsed_ms is not None:
                stat.total_ms += event.elapsed_ms
        elif isinstance(event, SubagentStartedEvent):
            _run(event.run_id)  # 无 LLM/工具活动的子 agent 也应出现在明细中
        elif isinstance(event, ContextCompactedEvent):
            summary.compactions += 1

    if first_ts is not None and last_ts is not None:
        summary.duration_s = (last_ts - first_ts).total_seconds()

    summary.runs = list(run_map.values())
    summary.tools = tools
    summary.retries = sum(v - 1 for v in max_attempt.values() if v > 1)
    summary.subagents = len(subagent_runs)
    summary.llm_calls = sum(r.llm_calls for r in summary.runs)
    summary.input_tokens = sum(r.input_tokens for r in summary.runs)
    summary.output_tokens = sum(r.output_tokens for r in summary.runs)
    summary.tool_calls = sum(r.tool_calls for r in summary.runs)
    summary.tool_errors = sum(r.tool_errors for r in summary.runs)
    return summary


# 该 session 最近一次 run 的最后一条 usage 事件（chat 进入时回填状态栏用）
def last_session_usage(traces_dir: Path, session_id: str) -> LlmUsageEvent | None:
    """按 meta.json 找到该 session 最新的 trace，取其最后一条 llm.usage；没有则 None"""
    latest: tuple[str, Path] | None = None  # (created_at, events.jsonl 路径)，ISO 串可比较
    for meta_path in traces_dir.glob("*/meta.json"):
        try:
            meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if meta.get("session_id") != session_id:
            continue
        created = str(meta.get("created_at") or "")
        if latest is None or created > latest[0]:
            latest = (created, meta_path.parent / "events.jsonl")
    if latest is None or not latest[1].exists():
        return None

    usage: LlmUsageEvent | None = None
    try:
        for event in read_trace(latest[1]):
            if isinstance(event, LlmUsageEvent):
                usage = event
    except TraceReadError:
        return None
    return usage
