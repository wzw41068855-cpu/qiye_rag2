"""中央调优参数的行为回归测试。"""
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from dotenv import dotenv_values
from qiye_rag import setting
from qiye_rag.graph import Conversation, build_graph
from qiye_rag.graph.nodes import clean_queries
from qiye_rag.graph.runtime import create_chat
from qiye_rag.rag.catalog import Catalog
from qiye_rag.rag.retrieval import search
from test_graph import FakeChat, responses, CHUNK
from test_v1 import FakeVectors, FakeReranker


def test_model_settings_ignore_old_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """模型重试次数读取 setting，忽略旧环境配置。"""
    monkeypatch.setattr(setting, "CHAT_MAX_RETRIES", 4)
    monkeypatch.setattr(setting, "CHAT_TEMPERATURE", 0.2)
    with patch("qiye_rag.graph.runtime.ChatOpenAI") as factory:
        create_chat({"DEEPSEEK_API_KEY": "fake", "CHAT_MAX_RETRIES": "99"})
    assert factory.call_args.kwargs["max_retries"] == 4
    assert factory.call_args.kwargs["temperature"] == 0.2


def test_graph_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """轮数、历史长度和子查询限制确实控制执行。"""
    monkeypatch.setattr(setting, "MAX_RETRIEVAL_ROUNDS", 1)
    data = responses()
    data["Research"] = [{"sufficient": False, "answer": "", "queries": ["补查"]}]
    retrieve = MagicMock(return_value=[])
    result = Conversation(build_graph(FakeChat(data), retrieve), {}).ask("问题")
    assert result["attempts"] == 1 and not result["approved"]
    retrieve.assert_called_once()
    monkeypatch.setattr(setting, "HISTORY_TURNS", 1)
    chat = FakeChat(responses(3))
    conversation = Conversation(build_graph(chat, lambda q: [CHUNK]), {})
    for question in ("问题一", "问题二", "问题三"):
        result = conversation.ask(question)
    assert len(result["history"]) == 1 and len(chat.calls[6][1]["history"]) == 1
    monkeypatch.setattr(setting, "MAX_QUERY_CHARS", 2)
    monkeypatch.setattr(setting, "MAX_SUBQUERIES", 1)
    assert clean_queries(["问题一", "问题二"], "fallback") == ["问题"]


def test_retrieval_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """召回数传递到存储，BM25 参数变化自动隔离缓存。"""
    cat, vec = Catalog(tmp_path / "state.json"), FakeVectors()
    cat.save("kb", "a.md", "digest", [{"id": "a", "text": "年假制度", "metadata": {
        "document_id": "a", "sha256": "h", "kb_id": "kb", "access_level": "internal", "status": "现行"}}])
    monkeypatch.setattr(setting, "RETRIEVAL_RECALL", 7)
    vec.search = MagicMock(return_value=["a"])
    search("年假", "kb", cat, vec, FakeReranker(), top_k=1)
    assert vec.search.call_args.args[-1] == 7
    monkeypatch.setattr(setting, "BM25_K1", 2.0)
    search("年假", "kb", cat, vec, FakeReranker(), top_k=1)
    assert len(cat.bm25_cache) == 2
    assert any(index.k1 == 2.0 for _, index in cat.bm25_cache.values())


def test_env_only_connection_settings() -> None:
    """示例环境文件只保留 API、模型与 URI。"""
    keys = dotenv_values(Path(__file__).resolve().parents[1] / ".env.example")
    allowed = {"DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "CHAT_MODEL", "DASHSCOPE_API_KEY",
               "DASHSCOPE_BASE_URL", "EMBEDDING_MODEL", "MILVUS_URI", "MILVUS_TOKEN",
               "RERANK_MODEL", "RERANK_API_BASE", "RERANK_API_KEY", "REDIS_URI", "DATABASE_URI"}
    assert set(keys) <= allowed
    assert "CHAT_MODEL" in keys and "MILVUS_URI" in keys


def test_invalid_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    """无效历史窗口和重叠大小应明确报错。"""
    setting.validate()
    with monkeypatch.context() as context:
        context.setattr(setting, "HISTORY_TURNS", 0)
        with pytest.raises(ValueError, match="正数"):
            setting.validate()
    monkeypatch.setattr(setting, "CHUNK_OVERLAP", setting.CHUNK_SIZE)
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        setting.validate()
