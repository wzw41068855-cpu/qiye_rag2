"""显式启用的真实 Milvus/Embedding 冒烟测试。"""
import os
from pathlib import Path
from uuid import uuid4
import pytest
from qiye_rag.rag.catalog import Catalog
from qiye_rag.rag.indexing import sync
from qiye_rag.rag.vectors import MilvusVectors


@pytest.mark.skipif(os.getenv("RAG_SMOKE") != "1", reason="设置 RAG_SMOKE=1 启用真实服务测试")
def test_real_milvus(tmp_path: Path) -> None:
    """真实向量查询命中 HR001，错误知识库无结果，并清理测试集合。"""
    collection = "smoke_" + uuid4().hex
    vec = MilvusVectors(os.getenv("MILVUS_URI", "http://localhost:19530"),
        os.getenv("RAG_EMBEDDING", "BAAI/bge-small-zh-v1.5"), collection,
        os.getenv("MILVUS_TOKEN", ""))
    cat = Catalog(tmp_path / "state.json")
    root = Path(__file__).resolve().parents[1] / "data/远澜外贸公司_RAG知识库"
    try:
        sync(root, "smoke", cat, vec)
        hits = vec.search_results("员工试用期多久", "smoke", ("internal", "public"), "现行", 5)
        assert any(h["metadata"]["document_id"] == "HR001" for h in hits)
        assert hits[0]["text"] and isinstance(hits[0]["score"], float)
        assert hits[0]["metadata"]["source_path"]
        assert not vec.search_results("员工试用期多久", "wrong", ("internal", "public"), "现行", 5)
    finally:
        if vec.store.col is not None:
            vec.store.col.drop()
        cat.close()
