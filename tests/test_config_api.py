"""配置优先级和云端请求契约测试，不发送真实请求。"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import sys
import httpx
import pytest
from qiye_rag.rag.config import load_config
from qiye_rag.rag.vectors import MilvusVectors, Reranker


def test_config_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """文件覆盖系统值，空值及缺失文件回退到系统。"""
    path = tmp_path / ".env"
    path.write_text("MODEL=file\nFALLBACK=\n", encoding="utf-8")
    monkeypatch.setenv("MODEL", "system")
    monkeypatch.setenv("FALLBACK", "fallback")
    assert load_config(path)["MODEL"] == "file"
    assert load_config(path)["FALLBACK"] == "fallback"
    assert load_config(tmp_path / "missing")["MODEL"] == "system"


def test_cloud_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """云端模型走兼容 API，传入纯文本并限制单次批量。"""
    monkeypatch.setattr("qiye_rag.rag.vectors.load_config", lambda: {
        "DASHSCOPE_API_KEY": "test-only", "EMBEDDING_BATCH_SIZE": "32"})
    provider, store = MagicMock(), MagicMock()
    store.col = None
    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(OpenAIEmbeddings=provider))
    monkeypatch.setitem(sys.modules, "langchain_milvus", SimpleNamespace(Milvus=MagicMock(return_value=store)))
    vec = MilvusVectors("http://localhost:19530", "text-embedding-v3")
    assert provider.call_args.kwargs["chunk_size"] == 10
    assert provider.call_args.kwargs["check_embedding_ctx_length"] is False
    from langchain_milvus import Milvus
    assert Milvus.call_args.kwargs["metadata_field"] == "metadata"
    assert Milvus.call_args.kwargs["enable_dynamic_field"] is False


def test_remote_rerank(monkeypatch: pytest.MonkeyPatch) -> None:
    """分数按 index 还原顺序，密钥回退到百炼密钥，失败不回显响应体。"""
    monkeypatch.setattr("qiye_rag.rag.vectors.load_config", lambda: {"DASHSCOPE_API_KEY": "test-only"})
    post = MagicMock(return_value=httpx.Response(200, json={"output": {"results": [
        {"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]}}))
    monkeypatch.setattr("qiye_rag.rag.vectors.httpx.post", post)
    reranker = Reranker("qwen3-rerank")
    assert reranker.score("问题", [{"text": "a"}, {"text": "b"}]) == [0.1, 0.9]
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-only"
    assert post.call_args.kwargs["json"]["input"]["documents"] == ["a", "b"]
    post.return_value = httpx.Response(401, text="secret-response")
    with pytest.raises(RuntimeError, match="HTTP 401") as error:
        reranker.score("问题", [{"text": "a"}])
    assert "secret-response" not in str(error.value)
