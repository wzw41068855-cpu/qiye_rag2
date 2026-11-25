"""使用官方离线 tokenizer 规划上下文，API usage 才是实际计费消耗。"""
from functools import lru_cache
import json
from typing import Any
from qiye_rag import setting
from qiye_rag.rag.config import PROJECT_ROOT


@lru_cache(maxsize=1)
def tokenizer() -> Any:
    """加载本地官方 tokenizer；未安装时保守按 UTF-8 字节预算。

    Returns:
        tokenizer 对象或 None，不自动联网下载。
    """
    path = PROJECT_ROOT / setting.CHAT_TOKENIZER_PATH
    if not path.is_file():
        return None
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    return Tokenizer.from_file(str(path))


def count_text(text: str) -> int:
    """计算实际 tokenizer 的文本 Token 数，不使用中文分词数。

    Args:
        text: 待计数文本。
    Returns:
        离线文本 Token 数；缺 tokenizer 时为字节上界估算。
    """
    codec = tokenizer()
    return len(codec.encode(text, add_special_tokens=False).ids) if codec else len(text.encode("utf-8"))


def count_json(value: Any) -> int:
    """计数实际 JSON 消息文本。

    Args:
        value: 待序列化数据。
    Returns:
        离线 Token 数。
    """
    return count_text(json.dumps(value, ensure_ascii=False, default=str))


def select_evidence(chunks: list[dict]) -> list[dict]:
    """按完整证据装入预算，保留顺序和连续引用编号。

    Args:
        chunks: 已排序、已授权的证据。
    Returns:
        满足证据预算的完整块。
    """
    selected: list[dict] = []
    for chunk in chunks:
        trial = selected + [chunk]
        payload = [{"number": i, "text": c["text"], "source": c["citation"]}
                   for i, c in enumerate(trial, 1)]
        if count_json(payload) <= setting.EVIDENCE_TOKEN_BUDGET:
            selected = trial
        if len(selected) >= setting.MAX_EVIDENCE_CHUNKS:
            break
    return selected
