"""演示入口交互回归测试。"""
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock
import pytest


def test_demo_loop(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """空输入跳过、多次查询复用模型、退出关闭清单并输出正文来源。"""
    path = Path(__file__).resolve().parents[1] / "run_demo_v1.py"
    spec = importlib.util.spec_from_file_location("demo_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    catalog = MagicMock()
    monkeypatch.setattr(module, "Catalog", MagicMock(return_value=catalog))
    vector_factory, rank_factory = MagicMock(), MagicMock()
    monkeypatch.setattr(module, "MilvusVectors", vector_factory)
    monkeypatch.setattr(module, "Reranker", rank_factory)
    sync_mock = MagicMock(return_value={"updated": 1})
    monkeypatch.setattr(module, "sync", sync_mock)
    retrieve = MagicMock(return_value=[{"text": "试用期为三个月", "score": 0.8,
        "citation": {"source_path": "HR001.pdf", "location": "页 1", "chunk_id": "abc"}}])
    monkeypatch.setattr(module, "search", retrieve)
    inputs = iter(["", "员工试用期多久", "年假", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    module.main([])
    assert retrieve.call_count == 2
    vector_factory.assert_called_once()
    rank_factory.assert_called_once()
    sync_mock.assert_called_once()
    catalog.close.assert_called_once()
    output = capsys.readouterr().out
    assert "试用期为三个月" in output and "HR001.pdf" in output and "页 1" in output
