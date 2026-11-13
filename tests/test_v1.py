"""不下载模型、不依赖外部服务的 v1 回归测试。"""
import json
from pathlib import Path
import pytest
from docx import Document
from pypdf import PdfWriter
from qiye_rag.rag.catalog import Catalog
from qiye_rag.rag.indexing import sync
from qiye_rag.rag.parsing import parse, chunks
from qiye_rag.rag.retrieval import search


class FakeVectors:
    """保留真实同步语义的测试存储。"""
    signature = "test-v1"

    def __init__(self) -> None:
        """初始化测试存储。"""
        self.rows: dict[str, dict] = {}
        self.fail = False

    def upsert(self, parts: list[dict]) -> None:
        """写入片段。

        Args:
            parts: 文本片段。
        """
        if self.fail:
            raise RuntimeError("模拟网络失败")
        self.rows.update({p["id"]: p for p in parts})

    def delete(self, ids: list[str]) -> None:
        """移除旧块。

        Args:
            ids: 块编号。
        """
        for key in ids:
            self.rows.pop(key, None)

    def search(self, *args: object) -> list[str]:
        """故意返回全部编号，验证检索端防御性过滤。

        Args:
            args: 查询参数。
        Returns:
            所有测试块编号。
        """
        return list(self.rows)


class FakeReranker:
    """检查权限后返回确定性分数。"""

    def score(self, query: str, parts: list[dict]) -> list[float]:
        """按关键词打分。

        Args:
            query: 查询。
            parts: 候选块。
        Returns:
            分数。
        """
        assert all(c["metadata"]["access_level"] != "restricted" for c in parts)
        return [float(query in c["text"]) for c in parts]


def test_incremental_and_permissions(tmp_path: Path) -> None:
    """验证新增、跳过、修改、删块、权限及多知识库隔离。"""
    root = tmp_path / "docs"
    root.mkdir()
    a = root / "a.md"
    a.write_text("---\naccess_level: internal\n---\n年假制度。" * 1 + "休假。" * 300, encoding="utf-8")
    b = root / "b.md"
    b.write_text("---\naccess_level: restricted\n---\n员工秘密", encoding="utf-8")
    cat, vec = Catalog(tmp_path / "db"), FakeVectors()
    assert sync(root, "a", cat, vec)["updated"] == 2
    assert sync(root, "a", cat, vec)["skipped"] == 2
    results = search("年假", "a", cat, vec, FakeReranker())
    assert results and results[0]["citation"]["source_path"] == "a.md"
    assert search("年假", "other", cat, vec, FakeReranker()) == []
    a.write_text("---\naccess_level: internal\n---\n新版年假", encoding="utf-8")
    assert sync(root, "a", cat, vec)["updated"] == 1
    assert len(vec.rows) == 2
    b.unlink()
    assert sync(root, "a", cat, vec)["deleted"] == 1
    vec.fail = True
    a.write_text("变化", encoding="utf-8")
    previous = cat.records("a")
    with pytest.raises(RuntimeError):
        sync(root, "a", cat, vec)
    assert cat.records("a") == previous
    cat.close()


def test_sample_corpus(tmp_path: Path) -> None:
    """真实解析仓库中全部三种格式，验证重复同步和权限过滤。"""
    root = Path(__file__).resolve().parents[1] / "data/远澜外贸公司_RAG知识库"
    cat, vec = Catalog(tmp_path / "db"), FakeVectors()
    entries = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["documents"]
    assert sync(root, "demo", cat, vec)["updated"] == len(entries)
    assert sync(root, "demo", cat, vec)["skipped"] == len(entries)
    results = search("年假", "demo", cat, vec, FakeReranker())
    assert results
    assert all(r["citation"]["status"] == "现行" for r in results)
    assert all(r["citation"]["access_level"] != "restricted" for r in results)
    cat.close()


def test_metadata_change(tmp_path: Path) -> None:
    """验证清单权限变化触发更新，并默认保护缺失权限的文档。"""
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("年假制度", encoding="utf-8")
    manifest = root / "manifest.json"
    entry = {"source_path": "a.md", "access_level": "internal"}
    manifest.write_text(json.dumps({"documents": [entry]}), encoding="utf-8")
    cat, vec = Catalog(tmp_path / "db"), FakeVectors()
    assert sync(root, "a", cat, vec)["updated"] == 1
    assert search("年假", "a", cat, vec, FakeReranker())
    entry.pop("access_level")
    manifest.write_text(json.dumps({"documents": [entry]}), encoding="utf-8")
    assert sync(root, "a", cat, vec)["updated"] == 1
    assert search("年假", "a", cat, vec, FakeReranker()) == []
    cat.close()


def test_formats_and_path_safety(tmp_path: Path) -> None:
    """验证 DOCX 表格、空 PDF、非法格式及越界清单。"""
    doc = Document()
    doc.add_paragraph("产品资料")
    doc.add_table(rows=1, cols=1).cell(0, 0).text = "产品参数"
    path = tmp_path / "a.docx"
    doc.save(path)
    assert "产品参数" in parse(path)[0][-1][0]
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    pdf = tmp_path / "a.pdf"
    writer.write(pdf)
    with pytest.raises(ValueError, match="无文本"):
        chunks(pdf, {"kb_id": "a", "source_path": "a.pdf"})
    with pytest.raises(ValueError):
        parse(tmp_path / "a.exe")
    root = tmp_path / "root"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"documents": [
        {"source_path": "../a.docx"}]}), encoding="utf-8")
    cat = Catalog(tmp_path / "db")
    with pytest.raises(ValueError, match="不安全"):
        sync(root, "a", cat, FakeVectors())
    cat.close()
