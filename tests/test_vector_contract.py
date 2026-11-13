"""无需模型的官方存储调用契约测试。"""
from types import SimpleNamespace
import sys
from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document
from qiye_rag.rag.vectors import MilvusVectors


def test_vector_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """维度不符拒绝复用，并验证写入、过滤、删除接口。"""
    embedding = MagicMock()
    embedding.embed_query.return_value = [0.0, 1.0]
    provider = MagicMock(return_value=embedding)
    store = MagicMock()
    store.col.schema.to_dict.return_value = {"enable_dynamic_field": False, "auto_id": False,
        "fields": [{"name": "id", "type": 21, "is_primary": True},
                   {"name": "text", "type": 21}, {"name": "vector", "type": 101},
                   {"name": "metadata", "type": 23}]}
    store.col.schema.fields = [SimpleNamespace(name="vector", params={"dim": 3})]
    factory = MagicMock(return_value=store)
    monkeypatch.setitem(sys.modules, "langchain_huggingface", SimpleNamespace(HuggingFaceEmbeddings=provider))
    monkeypatch.setitem(sys.modules, "langchain_milvus", SimpleNamespace(Milvus=factory))
    with pytest.raises(ValueError, match="维度"):
        MilvusVectors("http://localhost:19530", "test")
    store.col.schema.fields = [SimpleNamespace(name="vector", params={"dim": 2})]
    vec = MilvusVectors("http://localhost:19530", "test")
    part = {"id": "a", "text": "内容", "metadata": {"chunk_id": "a", "source_path": "a.md"}}
    vec.upsert([part])
    assert store.add_texts.call_args.kwargs["texts"] == ["内容"]
    store.similarity_search_with_score.return_value = [(Document(page_content="内容", metadata=part["metadata"]), 0.9)]
    hits = vec.search_results("问题", "wrong", ("internal",), "现行", 5)
    assert hits[0]["score"] == 0.9 and hits[0]["text"] == "内容"
    assert 'metadata["knowledge_base_id"] == "wrong"' in store.similarity_search_with_score.call_args.kwargs["expr"]
    vec.delete_document("kb", "doc")
    assert 'metadata["document_id"] == "doc"' in store.delete.call_args.kwargs["expr"]
