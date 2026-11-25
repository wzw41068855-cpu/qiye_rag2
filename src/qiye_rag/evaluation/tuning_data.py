"""参数分组、长历史探针与临时参数上下文。"""
from contextlib import contextmanager
from copy import deepcopy
from typing import Iterator
from qiye_rag import setting

NAMES = ("CHUNK_SIZE", "CHUNK_OVERLAP", "RETRIEVAL_TOP_K", "RETRIEVAL_RECALL", "MAX_RETRIEVAL_K",
         "MAX_RETRIEVAL_ROUNDS", "MAX_SUBQUERIES", "MAX_EVIDENCE_CHUNKS", "HISTORY_TURNS")
BASELINE = dict(zip(NAMES, (1000, 200, 5, 20, 100, 2, 3, 30, 6)))
DEV_IDS = {"Q01", "Q06", "Q10", "Q14", "Q16", "Q22", "Q29", "Q30"}


@contextmanager
def parameters(values: dict) -> Iterator[None]:
    """临时应用参数并恢复，避免多个试验互相污染。

    Args:
        values: 本次完整参数。
    Yields:
        无返回值。
    """
    previous = {k: getattr(setting, k) for k in NAMES + ("GRAPH_RECURSION_LIMIT",)}
    try:
        for key in NAMES:
            setattr(setting, key, values[key])
        setting.GRAPH_RECURSION_LIMIT = max(20, values["MAX_RETRIEVAL_ROUNDS"] * 3 + 5)
        setting.validate()
        yield
    finally:
        for key, value in previous.items():
            setattr(setting, key, value)


def history_probes(cases: list[dict], validation: bool = False) -> list[dict]:
    """从真实事实构造需要三轮/五轮历史的独立指代题。

    Args:
        cases: 已映射的原始题集。
        validation: 是否生成未参与搜索的探针。
    Returns:
        两道带真实Gold事实的长历史题。
    """
    sources = [("Q07", "SS-330", "主要材质", 3), ("Q09", "OF-420", "标准生产交期", 5)] if validation else [
        ("Q29", "BT-210", "标准生产交期", 3), ("Q08", "TX-620", "规格与尺寸", 5)]
    result = []
    for i, (source, sku, attribute, depth) in enumerate(sources, 1):
        case = deepcopy(next(c for c in cases if c["id"] == source))
        history = [{"question": f"先记下我关注的产品型号：{sku}。", "answer": "已记录，具体产品事实请以资料为准。"}]
        fillers = ["这些文档用于教学吗？", "回答时能注明资料来源吗？", "不要补充资料外信息。", "我们继续按知识库信息讨论。"]
        history.extend({"question": text, "answer": "好的。"} for text in fillers[:depth - 1])
        case.update(id=f"H{'V' if validation else 'S'}{i}", category="长历史指代探针",
                    question=f"我最初关注的那款产品，{attribute}是什么？", history=history, role="user")
        result.append(case)
    return result
