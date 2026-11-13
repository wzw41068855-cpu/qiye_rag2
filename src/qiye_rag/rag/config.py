"""项目 .env 优先，系统环境变量作为回退。"""
import os
from pathlib import Path
from dotenv import dotenv_values

# 通过项目标记定位根目录，避免模块移动一层后误读 src/.env。
PROJECT_ROOT = next((parent for parent in Path(__file__).resolve().parents
                     if (parent / "pyproject.toml").is_file()), Path(__file__).resolve().parents[3])


def load_config(path: Path | None = None) -> dict[str, str]:
    """读取配置，不修改全局环境，不输出密钥。

    Args:
        path: .env 路径，默认项目根目录。
    Returns:
        合并后的配置，文件中非空值优先。
    """
    local = dotenv_values(path or PROJECT_ROOT / ".env", encoding="utf-8-sig", interpolate=False)
    return {**os.environ, **{k: v for k, v in local.items() if v is not None and v.strip()}}


