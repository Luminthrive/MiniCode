"""Agent 角色配置加载器"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from minicode.config import AGENTS_DIR

logger = logging.getLogger(__name__)


# Agent 角色配置数据类
@dataclass
class AgentProfile:
    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str] = field(default_factory=list)
    model: str = ""


# 按两级优先级（项目本地 > 内建）查找并解析角色配置
class AgentProfileLoader:
    _BUILTIN_DIR = Path(__file__).parent / "builtin"

    # 查找指定角色配置；未找到返回 None
    def load(self, name: str) -> AgentProfile | None:
        for path in self._search_paths(name):
            if path.exists():
                try:
                    return self._parse(path, name)
                except Exception:
                    # 本地配置损坏要留痕，且继续尝试内建回退，而不是静默放弃
                    logger.warning("agent profile parse failed: %s", path, exc_info=True)
                    continue
        return None

    # 返回 [项目本地, 内建] 路径；load() 返回第一个存在的，项目本地优先级最高
    def _search_paths(self, name: str) -> list[Path]:
        builtin = self._BUILTIN_DIR / f"{name}.toml"
        local = AGENTS_DIR / f"{name}.toml"
        return [local, builtin]

    # 解析 TOML 角色配置文件
    def _parse(self, path: Path, name: str) -> AgentProfile:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        agent = data.get("agent", {})
        return AgentProfile(
            name=name,
            description=agent.get("description", ""),
            system_prompt=agent.get("system_prompt", "").strip(),
            allowed_tools=agent.get("allowed_tools", []),
            model=agent.get("model", ""),
        )
