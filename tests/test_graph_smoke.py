"""显式启用的 v2 真实聊天 API 冒烟，仅发送合成资料。"""
import os
import pytest
from qiye_rag.rag.config import load_config
from qiye_rag.graph import Conversation, build_graph
from qiye_rag.graph.runtime import create_chat


@pytest.mark.skipif(os.getenv("RAG_V2_SMOKE") != "1", reason="设置 RAG_V2_SMOKE=1 调用真实聊天 API")
def test_chat_grounded() -> None:
    """真实 DeepSeek 分类、引用生成与审核，不读取业务知识库。"""
    chunk = {"text": "合成演示规则：测试员工的试用期为3个月。", "score": 1.0,
             "citation": {"chunk_id": "synthetic1", "document_id": "TEST001",
                          "source_path": "synthetic.md", "location": "正文"}}
    chat = create_chat(load_config())
    graph = build_graph(chat, lambda query: [chunk])
    result = Conversation(graph, {"kb": "synthetic"}).ask("按照合成演示规则，测试员工的试用期多久？")
    assert result["kind"] != "chat" and result["approved"]
    assert result["citations"][0]["document_id"] == "TEST001"
    assert "[1]" in result["answer"]
