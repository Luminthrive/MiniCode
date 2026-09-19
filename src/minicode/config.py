"""运行时配置：从环境变量加载"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# 运行时配置数据类
@dataclass
class MiniConfig:
    llm_model: str = "gpt-4o"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    max_steps: int = 30
    compact_threshold: float = 0.80
    # True 时跳过终端审批（bash/写文件等直接放行）；False 时危险工具需 y/n 确认
    auto_approve: bool = False


# 从 minicode 包目录的 .env 加载配置（不受当前工作目录影响）
def get_config() -> MiniConfig:
    # 优先加载 minicode 包目录下的 .env，其次加载当前目录的 .env
    pkg_dir = Path(__file__).parent
    load_dotenv(pkg_dir / ".env", override=False)
    load_dotenv(".env", override=False)

    config = MiniConfig()

    model = os.environ.get("OPENAI_MODEL")
    if model is not None:
        config.llm_model = model

    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url is not None:
        config.llm_base_url = base_url

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is not None:
        config.llm_api_key = api_key

    max_steps_str = os.environ.get("MAX_STEPS")
    if max_steps_str is not None:
        try:
            val = int(max_steps_str)
            if val > 0:
                config.max_steps = val
        except ValueError:
            pass

    compact_str = os.environ.get("COMPACT_THRESHOLD")
    if compact_str is not None:
        try:
            ratio = float(compact_str)
            if 0.0 <= ratio <= 1.0:
                config.compact_threshold = ratio
        except ValueError:
            pass

    approve_str = os.environ.get("MINICODE_AUTO_APPROVE")
    if approve_str is not None:
        config.auto_approve = approve_str.strip().lower() in ("1", "true", "yes", "on")

    return config
