"""真实 Milvus 的旧 schema 回归，不调用云端、不读取业务资料。"""
import os
from uuid import uuid4
from unittest.mock import patch
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from qiye_rag.rag.config import load_config
from qiye_rag.rag.vectors import MilvusVectors


@pytest.mark.skipif(os.getenv("RAG_SCHEMA_SMOKE") != "1", reason="显式启用本地 schema 冒烟")
def test_real_legacy_schema() -> None:
    """重建用户的 pk/metadata 布局，验证写入、权限、跨库隔离和按文档删除。"""
    from pymilvus import MilvusClient, DataType
    config = load_config()
    uri = config.get("MILVUS_URI", "http://localhost:19530")
    token = config.get("MILVUS_TOKEN", "")
    client = MilvusClient(uri=uri, token=token, timeout=10)
    name = "schema_test_" + uuid4().hex
    new_name = name + "_new"
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=8)
    schema.add_field("metadata", DataType.JSON)
    index = client.prepare_index_params()
    index.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    try:
        client.create_collection(name, schema=schema, index_params=index)
        # 使用官方测试 Embedding 提供商，仅测试真实数据库协议，不调用 API。
        with patch("langchain_openai.OpenAIEmbeddings", return_value=DeterministicFakeEmbedding(size=8)), \
             patch("qiye_rag.rag.vectors.load_config", return_value={"DASHSCOPE_API_KEY": "test-only"}):
            vec = MilvusVectors(uri, "text-embedding-v3", name, token)
        vec.upsert([{"id": "chunk1", "text": "schema test", "metadata": {
            "knowledge_base_id": "test", "document_id": "doc1", "access_level": "internal",
            "status": "现行", "chunk_id": "chunk1", "source_path": "synthetic.md"}}])
        hits = vec.search_results("schema test", "test", ("internal",), "现行", 3)
        assert hits[0]["id"] == "chunk1" and hits[0]["text"] == "schema test"
        row = client.query(name, filter='pk == "chunk1"', output_fields=["pk", "text", "metadata"],
                           consistency_level="Strong")[0]
        assert isinstance(row["metadata"], dict)
        assert row["metadata"]["knowledge_base_id"] == "test"
        assert set(row) == {"pk", "text", "metadata"}
        assert not vec.search_results("schema test", "wrong", ("internal",), "现行", 3)
        assert not vec.search_results("schema test", "test", ("public",), "现行", 3)
        assert vec.search_active("schema test", "test", ("internal",), "现行", 3, ["chunk1"]) == ["chunk1"]
        assert vec.search_active("schema test", "test", ("internal",), "现行", 3, ["uncommitted"]) == []
        vec.delete_document("test", "doc1")
        assert not vec.search_results("schema test", "test", ("internal",), "现行", 3)
        with patch("langchain_openai.OpenAIEmbeddings", return_value=DeterministicFakeEmbedding(size=8)), \
             patch("qiye_rag.rag.vectors.load_config", return_value={"DASHSCOPE_API_KEY": "test-only"}):
            fresh = MilvusVectors(uri, "text-embedding-v3", new_name, token)
        fresh.upsert([{"id": "new1", "text": "new JSON collection", "metadata": {
            "chunk_id": "new1", "knowledge_base_id": "test", "tags": ["中文"], "optional": None}}])
        description = client.describe_collection(new_name)
        assert not description["enable_dynamic_field"]
        assert {f["name"] for f in description["fields"]} == {"pk", "text", "vector", "metadata"}
        row = client.query(new_name, filter='pk == "new1"', output_fields=["metadata"],
                           consistency_level="Strong")[0]
        assert row["metadata"]["tags"] == ["中文"] and row["metadata"]["optional"] is None
    finally:
        if client.has_collection(new_name):
            client.drop_collection(new_name)
        if client.has_collection(name):
            client.drop_collection(name)
        client.close()
