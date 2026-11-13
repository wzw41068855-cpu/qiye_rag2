"""真实 LangGraph 调度和 Checkpointer 测试，模型与检索使用替身。"""
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from qiye_rag.graph import Conversation, build_graph
from qiye_rag.graph.nodes import REFUSAL
from qiye_rag.graph.runtime import create_chat

CHUNK = {"text": "试用期一般为3个月。", "score": 0.8, "citation": {
    "chunk_id": "hr001", "document_id": "HR001", "source_path": "hr001.pdf", "location": "页 1"}}


class FakeChat:
    """按结构化类型返回预设结果，记录实际节点输入。"""

    def __init__(self, responses: dict[str, list]) -> None:
        """初始化响应队列。

        Args:
            responses: Pydantic 模型名到结果/异常队列。
        """
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    def with_structured_output(self, schema: type, **kwargs: Any) -> Any:
        """生成对应节点的调用替身。

        Args:
            schema: 输出类型。
            kwargs: 结构化方法配置。
        Returns:
            有 invoke 方法的对象。
        """
        def invoke(messages: list) -> Any:
            """记录请求并取下一个预设响应。

            Args:
                messages: 模型消息。
            Returns:
                预设输出。
            """
            self.calls.append((schema.__name__, json.loads(messages[-1][1])))
            response = self.responses[schema.__name__].pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        return MagicMock(invoke=invoke)


def responses(turns: int = 1) -> dict[str, list]:
    """生成正常回答的响应。

    Args:
        turns: 轮次数。
    Returns:
        模型响应队列。
    """
    return {"Plan": [{"kind": "simple", "queries": ["试用期"]}] * turns,
            "Research": [{"sufficient": True, "answer": "试用期一般为3个月。[1]"}] * turns,
            "Verdict": [{"grounded": True, "reason": "数字有直接证据支持"}] * turns}


def test_simple_and_multi_turn_scope() -> None:
    """正确引用、追问历史、跨轮状态清零和会话范围隔离。"""
    chat = FakeChat(responses(3))
    retrieve = MagicMock(return_value=[CHUNK])
    graph = build_graph(chat, retrieve)
    conversation = Conversation(graph, {"kb": "a", "levels": ["internal"]})
    first = conversation.ask("试用期多久", "session")
    assert first["approved"] and first["citations"][0]["document_id"] == "HR001"
    second = conversation.ask("那如何转正？", "session")
    assert second["attempts"] == 1 and len(second["history"]) == 2
    assert chat.calls[3][1]["history"][0]["question"] == "试用期多久"
    other = Conversation(graph, {"kb": "b", "levels": ["public"]})
    assert len(other.ask("试用期多久", "session")["history"]) == 1


def test_complex_and_insufficient_retry() -> None:
    """复杂问题拆分，检索不足后重写查询，证据去重。"""
    data = responses()
    data["Plan"] = [{"kind": "complex", "queries": ["试用期", "转正流程"]}]
    data["Research"].insert(0, {"sufficient": False, "answer": "", "queries": ["转正考核"]})
    retrieve = MagicMock(side_effect=[[], [], [CHUNK]])
    state = Conversation(build_graph(FakeChat(data), retrieve), {}).ask("试用期多久，如何转正？")
    assert state["attempts"] == 2 and state["approved"]
    assert [c.args[0] for c in retrieve.call_args_list] == ["试用期", "转正流程", "转正考核"]
    assert len(state["evidence"]) == 1


@pytest.mark.parametrize("bad_citation", [True, False])
def test_rejected_answer_is_not_published(bad_citation: bool) -> None:
    """引用越界或语义审核失败，最多两轮后拒答。"""
    data = responses()
    data["Research"] = [{"sufficient": True, "answer": "试用期30年。[99]" if bad_citation else "试用期30年。[1]"}] * 2
    data["Verdict"] = [{"grounded": False, "reason": "数字不受证据支持", "queries": ["试用期"]}] * 2
    chat = FakeChat(data)
    state = Conversation(build_graph(chat, lambda q: [CHUNK]), {}).ask("试用期多久")
    assert state["answer"] == REFUSAL and not state["approved"] and not state["citations"]
    assert state["attempts"] == 2 and "30年" not in state["history"][-1]["answer"]
    if bad_citation:
        assert all(name != "Verdict" for name, _ in chat.calls)


def test_no_evidence_and_chat() -> None:
    """空召回强制拒答，纯问候不检索。"""
    data = responses()
    data["Research"] *= 2
    state = Conversation(build_graph(FakeChat(data), lambda q: []), {}).ask("不存在的制度")
    assert state["answer"] == REFUSAL and state["attempts"] == 2
    retrieve = MagicMock()
    chat = FakeChat({"Plan": [{"kind": "chat", "queries": []}]})
    state = Conversation(build_graph(chat, retrieve), {}).ask("你好")
    assert state["attempts"] == 0 and state["answer"] and not state["citations"]
    retrieve.assert_not_called()


def test_sqlite_restart_and_resume(tmp_path: Path) -> None:
    """真实 SQLite 跨图重建保留未完成节点，恢复后仍保留对话历史。"""
    path = str(tmp_path / "checkpoint.sqlite")
    data = responses()
    data["Research"] = [RuntimeError("模拟 API 失败")]
    with SqliteSaver.from_conn_string(path) as saver:
        conversation = Conversation(build_graph(FakeChat(data), lambda q: [CHUNK], saver), {"kb": "a"})
        with pytest.raises(RuntimeError, match="模拟 API"):
            conversation.ask("试用期多久", "session")
        with pytest.raises(ValueError, match="未完成"):
            conversation.ask("新问题", "session")
    with SqliteSaver.from_conn_string(path) as saver:
        chat = FakeChat(responses(2))
        retrieve = MagicMock(return_value=[CHUNK])
        conversation = Conversation(build_graph(chat, retrieve, saver), {"kb": "a"})
        state = conversation.resume("session")
        assert state["approved"] and state["attempts"] == 1
        retrieve.assert_not_called()
        state = conversation.ask("再说一下", "session")
        assert len(state["history"]) == 2
        with pytest.raises(ValueError, match="没有待恢复"):
            conversation.resume("session")


def test_deepseek_config_and_input() -> None:
    """模型使用传入的 .env 配置，拒绝缺失密钥和空问题。"""
    with patch("qiye_rag.graph.runtime.ChatOpenAI") as factory:
        create_chat({"DEEPSEEK_API_KEY": "fake-key", "DEEPSEEK_BASE_URL": "https://example.test/v1", "CHAT_MODEL": "configured-model"})
        assert factory.call_args.kwargs["model"] == "configured-model"
        assert factory.call_args.kwargs["base_url"] == "https://example.test/v1"
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        create_chat({})
    with pytest.raises(ValueError):
        Conversation(build_graph(FakeChat({}), lambda q: []), {}).ask(" ")


def test_misclassified_chat_still_retrieves() -> None:
    """模型误分类知识问题为闲聊时，确定性保护仍执行检索。"""
    data = responses()
    data["Plan"] = [{"kind": "chat", "queries": []}]
    retrieve = MagicMock(return_value=[CHUNK])
    state = Conversation(build_graph(FakeChat(data), retrieve), {}).ask("员工试用期多久？")
    assert state["kind"] == "simple" and state["approved"]
    retrieve.assert_called_once()
