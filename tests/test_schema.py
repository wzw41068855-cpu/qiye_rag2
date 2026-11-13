"""旧 JSON 布局与删除失败的回归测试。"""
from types import SimpleNamespace
from unittest.mock import MagicMock
import sys
import pytest
from langchain_core.documents import Document
from qiye_rag.rag.schema import collection_layout
from qiye_rag.rag.vectors import MilvusVectors


def test_legacy_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """复现真实 pk/metadata 集合，验证初始化、JSON 过滤和删除失败中止。"""
    schema = {"auto_id": False, "enable_dynamic_field": False, "fields": [
        {"name": "pk", "type": 21, "is_primary": True}, {"name": "text", "type": 21},
        {"name": "vector", "type": 101}, {"name": "metadata", "type": 23}]}
    assert collection_layout(schema) == ("pk", "metadata")
    store, encoder = MagicMock(), MagicMock()
    store.col.schema.to_dict.return_value = schema
    store.col.schema.fields = [SimpleNamespace(name="vector", params={"dim": 2})]
    encoder.embed_query.return_value = [0.0, 1.0]
    factory = MagicMock(return_value=store)
    monkeypatch.setitem(sys.modules, "langchain_huggingface", SimpleNamespace(HuggingFaceEmbeddings=MagicMock(return_value=encoder)))
    monkeypatch.setitem(sys.modules, "langchain_milvus", SimpleNamespace(Milvus=factory))
    vec = MilvusVectors("http://localhost:19530", "test")
    assert factory.call_args.kwargs["primary_field"] == "pk"
    assert factory.call_args.kwargs["metadata_field"] == "metadata"
    assert factory.call_args.kwargs["enable_dynamic_field"] is False
    vec.delete_document("demo_v1", "HR001")
    assert 'metadata["knowledge_base_id"]' in store.delete.call_args.kwargs["expr"]
    store.similarity_search_with_score.return_value = [(Document(page_content="正文", metadata={"chunk_id": "a"}), 0.9)]
    assert vec.search("问题", "demo_v1", ("internal",), "现行", 5) == ["a"]
    expr = store.similarity_search_with_score.call_args.kwargs["expr"]
    assert store.similarity_search_with_score.call_args.kwargs["consistency_level"] == "Strong"
    assert 'metadata["access_level"]' in expr and 'metadata["status"]' in expr
    store.delete.return_value = False
    with pytest.raises(RuntimeError, match="中止"):
        vec.delete_document("demo_v1", "HR001")
    with pytest.raises(RuntimeError, match="中止"):
        vec.delete(["a"])
    schema["fields"] = [f for f in schema["fields"] if f["name"] != "metadata"]
    with pytest.raises(ValueError, match="schema 不兼容"):
        collection_layout(schema)
