"""v1 模块出口测试。"""
from pathlib import Path
import json
import pytest
from langchain_core.documents import Document
from qiye_rag.rag.loader import load_directory
from qiye_rag.rag.splitter import split_documents
from qiye_rag.rag.catalog import Catalog


def test_loader_and_splitter(tmp_path: Path) -> None:
    """批量加载隔离坏文件并继承标题、来源和权限。"""
    (tmp_path / "README.md").write_text("说明", encoding="utf-8")
    (tmp_path / "EVAL_问题设计建议.md").write_text("评估", encoding="utf-8")
    (tmp_path / "bad.md").write_text("---\nbad", encoding="utf-8")
    (tmp_path / "empty.md").write_text("  ", encoding="utf-8")
    (tmp_path / "ok.md").write_text("---\naccess_level: internal\n---\n# 制度\n## 年假\n申请年假。", encoding="utf-8")
    docs, errors = load_directory(tmp_path, "demo")
    assert len(docs) == 1 and set(errors) == {"bad.md", "empty.md"}
    assert isinstance(docs[0], Document)
    parts = split_documents(docs, 100, 10)
    assert parts[0].metadata["section"] == "制度 / 年假"
    assert parts[0].metadata["knowledge_base_id"] == "demo"
    assert parts[0].metadata["token_count"] > 0
    assert parts[0].metadata["source_path"] == "ok.md"
    with pytest.raises(ValueError):
        split_documents(docs, 10, 10)


def test_page_boundary() -> None:
    """PDF 切分不能跨页。"""
    docs = [Document(page_content="制度" * 100, metadata={"file_type": "pdf", "page": i,
        "knowledge_base_id": "demo", "source_path": "a.pdf"}) for i in (1, 2)]
    parts = split_documents(docs, 50, 5)
    assert {d.metadata["page"] for d in parts} == {1, 2}
    assert all(len(d.page_content) <= 50 for d in parts)


def test_json_atomic_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """原子替换失败时保留磁盘和内存状态。"""
    path = tmp_path / "state.json"
    cat = Catalog(path)
    parts = [{"id": "a", "text": "内容", "metadata": {"sha256": "hash", "document_id": "a"}}]
    cat.save("demo", "a.md", "digest", parts)
    before = path.read_bytes()

    def fail(*args: object) -> None:
        """模拟原子替换失败。

        Args:
            args: 替换参数。
        """
        raise OSError("模拟写入失败")

    monkeypatch.setattr("qiye_rag.rag.catalog.os.replace", fail)
    with pytest.raises(OSError):
        cat.delete("demo", "a.md")
    assert path.read_bytes() == before
    assert "a.md" in cat.records("demo")
    assert json.loads(before)["demo"]["a.md"]["status"] == "indexed"
