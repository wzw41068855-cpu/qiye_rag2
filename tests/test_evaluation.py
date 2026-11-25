"""评测金标准、阶段记录及错误路径的回归测试。"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from qiye_rag.evaluation.dataset import build, corpus
from qiye_rag.evaluation.runner import metrics, summarize
from qiye_rag.graph.nodes import Nodes
from qiye_rag.graph.state import Plan
from qiye_rag.rag.catalog import Catalog
from qiye_rag.rag.retrieval import search
from qiye_rag.telemetry import capture, stage
from test_v1 import FakeVectors


def test_gold_snapshot(tmp_path: Path) -> None:
    """每个 Gold 都能在实际资料中定位，八类题和权限对齐全。

    Args:
        tmp_path: 临时目录。
    """
    root = Path(__file__).resolve().parents[1] / "data/远澜外贸公司_RAG知识库"
    generated = build(root, tmp_path / "questions.json")
    checked = json.loads((root.parents[1] / "eval/enterprise_30.json").read_text(encoding="utf-8"))
    assert generated == checked
    parts, _ = corpus(root)
    by_id = {p["id"]: p for p in parts}
    assert len(generated["cases"]) == 30
    assert len({c["category"] for c in generated["cases"]}) == 8
    for case in generated["cases"]:
        assert case["answerable"] == bool(case["gold_chunk_ids"])
        for citation in case["expected_citations"]:
            part = by_id[citation["chunk_id"]]
            assert part["metadata"]["document_id"] == citation["document_id"]
            assert "".join(citation["fact"].split()) in "".join(part["text"].split())


def test_usage_and_capture_isolation() -> None:
    """记录服务端真实 usage，嵌套采集和异常不相互污染。"""
    chat = MagicMock()
    chat.with_structured_output.return_value.invoke.return_value = {
        "raw": SimpleNamespace(usage_metadata={"input_tokens": 11, "output_tokens": 7}),
        "parsed": Plan(kind="simple", queries=["问题"]), "parsing_error": None}
    with capture() as events:
        result = Nodes(chat, lambda q: []).structured(Plan, "说明", {})
        with capture() as inner:
            with pytest.raises(ValueError):
                with stage("failure"):
                    raise ValueError("不应记录的秘密")
    assert result.kind == "simple"
    assert events[0]["input_tokens"] == 11 and events[0]["output_tokens"] == 7
    assert events[0]["model_calls"] == 1
    assert len(events) == len(inner) == 1
    assert inner[0]["error"] == "ValueError" and "秘密" not in str(inner)


def test_rerank_length_and_authorization_trace(tmp_path: Path) -> None:
    """重排不能静默丢分数，阶段日志仅含授权候选。

    Args:
        tmp_path: 临时目录。
    """
    cat = Catalog(tmp_path / "state.json")
    parts = [{"id": i, "text": "年假规则", "metadata": {"kb_id": "kb", "document_id": i,
              "sha256": "x", "access_level": level, "status": "现行"}}
             for i, level in (("allowed", "internal"), ("secret", "restricted"))]
    cat.save("kb", "a.md", "digest", parts)
    vec = FakeVectors()
    vec.rows = {p["id"]: p for p in parts}
    rank = MagicMock()
    rank.score.return_value = []
    with capture() as events:
        with pytest.raises(ValueError, match="数量"):
            search("年假", "kb", cat, vec, rank)
    assert events[-1]["error"] == "ValueError"
    assert "secret" not in json.dumps(events)
    assert {e["stage"] for e in events} == {"authorization", "bm25", "dense", "fusion", "rerank"}


def test_metrics_denominator_and_failures() -> None:
    """不可回答题不计算召回；错误题必须显示在汇总中。"""
    case = {"gold_document_ids": [], "gold_chunk_ids": [], "forbidden_document_ids": ["secret"], "answerable": False}
    result = metrics(case, [], {"approved": False, "citations": []})
    assert result["document_recall"] is None and result["refusal_correct"]
    summary = summarize([{"category": "拒答", "metrics": result}, {"category": "拒答", "error": "Timeout"}])
    assert summary["overall"]["count"] == 2 and summary["overall"]["errors"] == 1
