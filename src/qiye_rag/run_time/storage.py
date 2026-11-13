"""Web 知识库目录约定与单写入协调。"""
from pathlib import Path
import re
from filelock import FileLock
from qiye_rag import setting
from qiye_rag.rag.config import PROJECT_ROOT


def kb_root(kb: str) -> Path:
    """定位 Web 管理的知识库，禁止目录穿越。

    Args:
        kb: 简单的知识库名称。
    Returns:
        返回kb的绝对根目录，用于存储该知识库的文档、索引和状态。
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", kb):
        raise ValueError("知识库名仅支持字母、数字、下划线和短横线")
    return PROJECT_ROOT / setting.WEB_DATA_DIR / kb


def kb_lock(kb: str) -> FileLock:
    """避免 worker 写入和 Web 查询同时持有不同索引快照。

    Args:
        kb: 知识库名称。
    Returns:
        跨进程本地文件锁。只锁定同一个 kb 的索引更新，允许不同 kb 并行。
    """
    root = kb_root(kb)
    root.mkdir(parents=True, exist_ok=True)
    return FileLock(str(root / "runtime.lock"), timeout=setting.WEB_LOCK_TIMEOUT_SECONDS)


def knowledge_id(kb: str) -> str:
    """生成 Web 专用编号，不覆盖命令行演示库。

    Args:
        kb: Web 名称。
    Returns:
        Milvus 元数据中的知识库编号。
    """
    kb_root(kb)
    return "web_" + kb
