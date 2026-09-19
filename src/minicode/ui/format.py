"""UI 共享格式化工具：Rich/TUI 两个渲染器与审批对话框共用，不依赖任何框架"""

from __future__ import annotations

import difflib
from datetime import datetime
from typing import Any


# 工具参数摘要：单行显示
def summarize_args(name: str, args: dict[str, Any]) -> str:
    """生成工具参数的单行摘要；bash 完整显示命令，多行命令折叠为首行 + 行数"""
    if name == "bash":
        cmd = str(args.get("command", "?"))
        lines = cmd.splitlines() or ["?"]
        first = lines[0]
        if len(first) > 120:
            first = first[:120] + "..."
        if len(lines) > 1:
            first += f" (+{len(lines) - 1} lines)"
        return first
    if name == "edit_file":
        path = str(args.get("path", "?"))
        old_lines = str(args.get("old_string", "")).splitlines() or [""]
        new_lines = str(args.get("new_string", "")).splitlines() or [""]
        suffix = f" (+{len(old_lines) - 1} lines)" if len(old_lines) > 1 else ""

        # 取第一处差异行展示：old/new 首行常是相同的段落标记，差异行才有辨识度
        i = 0
        while i < min(len(old_lines), len(new_lines)) and old_lines[i] == new_lines[i]:
            i += 1
        old_first = old_lines[i] if i < len(old_lines) else ""
        new_first = new_lines[i] if i < len(new_lines) else ""

        def _clip(s: str) -> str:
            return s[:32] + "..." if len(s) > 32 else s

        return f"{path}{suffix}: {_clip(old_first)} -> {_clip(new_first)}"
    if name in ("read_file", "write_file", "list_dir"):
        return str(args.get("path", "."))
    if name == "spawn_agent":
        agent_type = args.get("subagent_type") or "default"
        return f"{agent_type} · {args.get('description', '')}"
    return str(args)[:60]


# 由事件 ts 计算时长（replay 回放时也能得到真实耗时）
def format_duration(start: str, end: str) -> str:
    try:
        delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
        return f"{delta.total_seconds():.1f}s"
    except ValueError:
        return "-"


# 工具结果单行预览：→ (标记, 主体, 附注)。Rich/TUI 两个渲染器共用
def describe_tool_result(name: str, content: str, is_error: bool) -> tuple[str, str, str]:
    if name == "spawn_agent" and not is_error:
        # 子代理最终报告已在流式块内输出过，只报大小
        return "✓", f"subagent result: {len(content):,} chars", ""
    # 跳过开头空行，取第一个非空行做预览（PowerShell 输出常以空行开头）
    lines = content.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if start is None:
        return ("✗", "(empty error)", "") if is_error else ("✓", "(empty output)", "")
    first = lines[start].strip()
    limit = 100 if is_error else 72
    body = first[:limit] + "..." if len(first) > limit else first
    extra = len(lines) - start - 1
    note = f" (+{extra} lines)" if extra > 0 else ""
    return ("✗" if is_error else "✓"), body, note


# edit_file 参数转 unified diff 文本（审批对话框展示红绿对照用）
def build_edit_diff(args: dict[str, Any], context_lines: int = 2) -> str:
    """old_string/new_string → unified diff；参数缺失或无差异时返回空串"""
    path = str(args.get("path", "?"))
    old_lines = str(args.get("old_string", "")).splitlines(keepends=True)
    new_lines = str(args.get("new_string", "")).splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=context_lines,
    )
    return "".join(diff)
