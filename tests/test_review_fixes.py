"""版本切换、失败恢复及检索缓存回归测试。"""
from pathlib import Path
from unittest.mock import patch
import pytest
from langchain_core.documents import Document
from qiye_rag.rag.catalog import Catalog
from qiye_rag.rag.indexing import sync
from qiye_rag.rag.retrieval import search
from qiye_rag.rag.splitter import split_documents
from test_v1 import FakeVectors, FakeReranker


class PartialVectors(FakeVectors):
    """可在写入部分块后失败的存储。"""

    def upsert(self, parts: list[dict]) -> None:
        """模拟已产生残留的失败。

        Args:
            parts: 新版本块。
        """
        if self.fail:
            self.rows[parts[0]["id"]] = parts[0]
            raise RuntimeError("部分写入失败")
        super().upsert(parts)


def test_failure_preserves_old_and_cleans_deleted_source(tmp_path: Path) -> None:
    """部分新版本写入失败后，旧版本可检索；源文件随后删除也能清理残留。"""
    root = tmp_path / "docs"
    root.mkdir()
    path = root / "a.md"
    path.write_text("---\naccess_level: internal\n---\n旧年假", encoding="utf-8")
    cat, vec = Catalog(tmp_path / "state.json"), PartialVectors()
    sync(root, "kb", cat, vec)
    old = cat.records("kb")
    old_ids = set(vec.rows)
    path.write_text("---\naccess_level: internal\n---\n新版年假", encoding="utf-8")
    vec.fail = True
    with pytest.raises(RuntimeError):
        sync(root, "kb", cat, vec)
    assert old_ids < set(vec.rows) and cat.records("kb") == old
    assert search("年假", "kb", cat, vec, FakeReranker())[0]["text"] == "旧年假"
    # 用重新打开的清单模拟进程重启，证明恢复日志不只在内存中。
    cat = Catalog(tmp_path / "state.json")
    path.unlink()
    assert sync(root, "kb", cat, vec)["deleted"] == 1
    assert not vec.rows and not cat.records("kb")


def test_manifest_failure_and_cleanup_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """清单提交失败不切换；提交后清理失败仍可查询新版本，并在重启后清理。"""
    root = tmp_path / "docs"
    root.mkdir()
    path = root / "a.md"
    path.write_text("---\naccess_level: internal\n---\n旧年假", encoding="utf-8")
    state_path = tmp_path / "state.json"
    cat, vec = Catalog(state_path), PartialVectors()
    sync(root, "kb", cat, vec)
    old = cat.records("kb")
    path.write_text("---\naccess_level: internal\n---\n新版年假", encoding="utf-8")
    with patch.object(cat, "save", side_effect=OSError("清单失败")):
        with pytest.raises(OSError):
            sync(root, "kb", cat, vec)
    assert cat.records("kb") == old
    sync(root, "kb", cat, vec)
    assert len(vec.rows) == 1
    path.write_text("---\naccess_level: internal\n---\n第三版年假", encoding="utf-8")
    original_delete = vec.delete

    def fail_cleanup(ids: list[str]) -> None:
        """只在实际清理旧块时失败。

        Args:
            ids: 待清理块。
        """
        if ids:
            raise RuntimeError("清理失败")

    monkeypatch.setattr(vec, "delete", fail_cleanup)
    with pytest.raises(RuntimeError):
        sync(root, "kb", cat, vec)
    assert search("年假", "kb", cat, vec, FakeReranker())[0]["text"] == "第三版年假"
    monkeypatch.setattr(vec, "delete", original_delete)
    assert sync(root, "kb", Catalog(state_path), vec)["skipped"] == 1
    assert len(vec.rows) == 1


def test_validation_before_write(tmp_path: Path) -> None:
    """空 Chunk、空文档和重复 document_id 都应明确拒绝。"""
    root = tmp_path / "docs"
    root.mkdir()
    for name in ("a.md", "b.md"):
        (root / name).write_text("---\ndoc_id: duplicated\n---\n正文", encoding="utf-8")
    cat, vec = Catalog(tmp_path / "state.json"), FakeVectors()
    with pytest.raises(ValueError, match="唯一"):
        sync(root, "kb", cat, vec)
    assert not vec.rows
    with pytest.raises(ValueError, match="空"):
        cat.save("kb", "a.md", "digest", [])
    (root / "b.md").unlink()
    (root / "a.md").write_text("  ", encoding="utf-8")
    with pytest.raises(ValueError, match="无文本"):
        sync(root, "kb", cat, vec)
    assert not vec.rows


def test_bm25_cache_and_zero_score(tmp_path: Path) -> None:
    """复用索引、按权限隔离、提交后失效，未匹配语料不进入候选。"""
    cat = Catalog(tmp_path / "state.json")
    parts = [{"id": str(i), "text": "年假制度", "metadata": {"sha256": "hash",
        "document_id": "a", "kb_id": "kb", "access_level": "internal", "status": "现行"}}
        for i in range(30)]
    cat.save("kb", "a.md", "digest", parts)
    vec, rank = FakeVectors(), FakeReranker()
    from qiye_rag.rag.retrieval import BM25Plus
    with patch("qiye_rag.rag.retrieval.BM25Plus", wraps=BM25Plus) as factory:
        assert len(search("年假", "kb", cat, vec, rank, top_k=25)) == 25
        assert search("完全无关产品", "kb", cat, vec, rank) == []
        assert factory.call_count == 1
        assert search("年假", "kb", cat, vec, rank, levels=("public",)) == []
        cat.save("kb", "a.md", "new", parts)
        search("年假", "kb", cat, vec, rank)
        assert factory.call_count == 2


def test_content_identity() -> None:
    """在前面新增段落不会改变后续未改正文的内容编号；重复正文也有独立编号。"""
    meta = {"file_type": "docx", "knowledge_base_id": "kb", "source_path": "a.docx"}
    docs = [Document(page_content=t, metadata=meta) for t in ("年假制度", "报销制度", "年假制度")]
    before = split_documents(docs)
    after = split_documents([Document(page_content="新增段落", metadata=meta)] + docs)
    assert [d.metadata["chunk_id"] for d in before] == [d.metadata["chunk_id"] for d in after[1:]]
    assert len({d.metadata["chunk_id"] for d in before}) == 3


def test_journal_failure_prevents_vector_write(tmp_path: Path) -> None:
    """恢复日志无法落盘时，不允许开始向量写入。"""
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("正文", encoding="utf-8")
    cat, vec = Catalog(tmp_path / "state.json"), FakeVectors()
    with patch("qiye_rag.rag.indexing.stage", side_effect=OSError("日志失败")):
        with pytest.raises(OSError):
            sync(root, "kb", cat, vec)
    assert not vec.rows and not cat.records("kb")
