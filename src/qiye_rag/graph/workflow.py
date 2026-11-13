"""有界条件路由和按可信范围隔离的会话入口。"""
from qiye_rag import setting
import hashlib
import json
from collections.abc import Callable, Iterator
from typing import Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from qiye_rag.graph.nodes import Nodes
from qiye_rag.graph.state import GraphState


def build_graph(chat: Any, retrieve: Callable[[str], list[dict]], checkpointer: Any = None) -> Any:
    """编译四节点图，最多两轮检索。

    Args:
        chat: 聊天模型。
        retrieve: 绑定权限的检索函数。
        checkpointer: 官方保存器，默认进程内保存。
    Returns:
        已编译图。
    """
    nodes = Nodes(chat, retrieve)
    graph = StateGraph(GraphState)
    for name in ("router", "retriever", "researcher", "reviewer", "finish"):
        graph.add_node(name, getattr(nodes, name))
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", lambda s: "finish" if s["kind"] == "chat" else "retriever")
    graph.add_edge("retriever", "researcher")
    graph.add_conditional_edges("researcher", lambda s: "retriever"
        if not s["sufficient"] and s["attempts"] < setting.MAX_RETRIEVAL_ROUNDS else "reviewer")
    graph.add_conditional_edges("reviewer", lambda s: "retriever"
        if not s["approved"] and s["attempts"] < setting.MAX_RETRIEVAL_ROUNDS else "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


class Conversation:
    """一个固定可信知识库/权限范围内的会话。"""

    def __init__(self, graph: Any, scope: dict) -> None:
        """绑定图和范围。

        Args:
            graph: 已编译图。
            scope: 由调用方确定的知识库、权限及索引目标。
        """
        self.graph = graph
        self.scope = scope

    def config(self, session: str) -> dict:
        """生成范围隔离的 Checkpointer 线程键。

        Args:
            session: 用户提供的会话名。
        Returns:
            LangGraph 执行配置。
        """
        key = json.dumps([self.scope, session], sort_keys=True, ensure_ascii=False)
        return {"configurable": {"thread_id": hashlib.sha256(key.encode()).hexdigest()}, "recursion_limit": setting.GRAPH_RECURSION_LIMIT}

    def ask(self, text: str, session: str = "default") -> GraphState:
        """开始新轮次，重置证据、计数和输出，保留同线程历史。

        Args:
            text: 用户问题。
            session: 会话名。
        Returns:
            本轮最终状态。
        """
        inputs, config = self.prepare(text, session)
        return self.graph.invoke(inputs, config)

    def prepare(self, text: str, session: str) -> tuple[GraphState, dict]:
        """校验新轮次并生成干净输入，供同步与 SSE 入口共用。

        Args:
            text: 用户问题。
            session: 会话名。
        Returns:
            本轮输入与执行配置。
        """
        #校验输入问题合法性：
        #不能为空白；
        # 长度不能超过上限。
        # 这样可以避免无效请求和超长输入拖垮后续 LLM/检索流程。
        if not text.strip() or len(text) > setting.MAX_QUESTION_CHARS:
            raise ValueError(f"问题须为 1–{setting.MAX_QUESTION_CHARS} 字符")
        #生成thread_id，确保同一会话名在同一范围内的执行共享历史；不同范围或不同会话名不会互相污染。
        config = self.config(session)
        if self.graph.get_state(config).next:#next不为空，说明上次执行未完成，不能直接开始新轮次。
            raise ValueError("该会话有未完成执行，请先 resume，或使用新会话名")
        return {"question": text.strip(), "kind": "", "queries": [],
            "evidence": [], "attempts": 0, "sufficient": False, "draft": "", "approved": False,
            "reason": "", "answer": "", "citations": []}, config

    def stream(self, text: str, session: str = "default", resume: bool = False) -> Iterator[dict]:
        """按节点推送进度，最终只公开审核后的答案。

        Args:
            text: 新问题，恢复时忽略。
            session: 会话名。
            resume: 是否恢复失败的执行。
        Yields:
            progress 或 answer 事件。
        """
        if resume:
            config = self.config(session)
            if not self.graph.get_state(config).next:
                raise ValueError("没有待恢复的执行")
            inputs = None
        else:
            inputs, config = self.prepare(text, session)
        for update in self.graph.stream(inputs, config, stream_mode="updates"):
            for node in update:
                yield {"event": "progress", "data": {"node": node}}
        state = self.graph.get_state(config).values
        yield {"event": "answer", "data": {key: state[key] for key in
            ("answer", "citations", "kind", "attempts", "approved", "reason")}}

    def resume(self, session: str = "default") -> GraphState:
        """从最近失败/中断节点继续执行。

        Args:
            session: 已存在的会话名。
        Returns:
            恢复执行的最终状态。
        """
        config = self.config(session)
        if not self.graph.get_state(config).next:
            raise ValueError("没有待恢复的执行")
        return self.graph.invoke(None, config)

