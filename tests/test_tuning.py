"""参数实验有效性、隔离及历史探针回归。"""
import json
import pytest
from pathlib import Path
from types import SimpleNamespace
from qiye_rag import setting
from qiye_rag.evaluation.dataset import corpus
from qiye_rag.evaluation.tuning_data import BASELINE, DEV_IDS, history_probes, parameters

pytest.importorskip("optuna", reason="参数实验使用可选tuning依赖")

from qiye_rag.evaluation.tuning import choose, suggest
from optuna.trial import TrialState


def test_parameters_restore_and_chunk_binding() -> None:
    """临时参数确实改变切块，并在退出后恢复。"""
    root = Path(__file__).resolve().parents[1] / "data/远澜外贸公司_RAG知识库"
    old = setting.CHUNK_SIZE
    with parameters({**BASELINE, "CHUNK_SIZE": 256, "CHUNK_OVERLAP": 25}):
        small, digest = corpus(root, "prefix")
        assert max(len(p["text"]) for p in small) <= 256
    assert setting.CHUNK_SIZE == old
    with parameters(BASELINE):
        large, larger = corpus(root, "prefix")
    assert digest != larger and len(small) >= len(large)


def test_history_probes_and_holdout() -> None:
    """探针需要不同历史深度，保留题不进入开发集合。"""
    path = Path(__file__).resolve().parents[1] / "eval/enterprise_30.json"
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    probes = history_probes(cases)
    assert [len(c["history"]) for c in probes] == [3, 5]
    assert "BT-210" not in str(probes[0]["history"][-1:])
    assert "BT-210" in str(probes[0]["history"][-3:])
    assert all(c["id"] not in DEV_IDS for c in probes)
    assert len([c for c in cases if c["id"] not in DEV_IDS]) == 22
    assert {c["id"] for c in history_probes(cases, True)}.isdisjoint({c["id"] for c in probes})


def test_selection_uses_coverage_then_tokens() -> None:
    """不以更低Token换取丢失事实，也不读取验证成绩。"""
    trials = [SimpleNamespace(state=TrialState.COMPLETE, values=values,
                             user_attrs={"full_parameters": BASELINE}, number=i)
              for i, values in enumerate(((0.8, 1), (1.0, 100), (1.0, 90)))]
    assert choose(SimpleNamespace(trials=trials)).number == 2


def test_search_space_is_valid() -> None:
    """所有参数区间合法，并保留安全上限约束。"""
    class FirstChoice:
        """提供确定性最小范围采样。"""
        def suggest_categorical(self, name: str, choices: list) -> object:
            """返回最小项。

            Args:
                name: 参数名。
                choices: 合法取值。
            Returns:
                第一项。
            """
            return choices[0]
    for group in ("A", "B", "C"):
        values = suggest(FirstChoice(), group, BASELINE)
        with parameters(values):
            assert setting.RETRIEVAL_TOP_K <= setting.RETRIEVAL_RECALL <= setting.MAX_RETRIEVAL_K
