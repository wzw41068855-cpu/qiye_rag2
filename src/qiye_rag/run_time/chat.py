"""将 v2 节点事件转换为 SSE。"""
import json
import logging
from collections.abc import Iterator
from qiye_rag.graph.runtime import open_conversation
from qiye_rag.run_time.storage import kb_root, kb_lock, knowledge_id
from qiye_rag.rag.config import PROJECT_ROOT
from qiye_rag import setting


def encode_event(event: str, data: dict) -> str:
    """编码合法 SSE 帧，JSON 中的换行不会破坏帧边界。

    Args:
        event: 固定事件名。
        data: JSON 事件内容。
    Returns:
        以空行结尾的 SSE 帧。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def chat_events(kb: str, text: str, session: str, resume: bool = False,
                levels: tuple[str, ...] = ("public", "internal"), identity: str = "local") -> Iterator[str]:
    """按节点流式输出；异常不回显服务密钥或模型错误响应。

    Args:
        kb: Web 知识库。
        text: 问题。
        session: 会话名。
        resume: 恢复未完成执行。
        levels: 服务端根据角色授予的访问等级。
        identity: 账号与角色的会话隔离标识。
    Yields:
        start、progress、answer、error、done 事件。
    """
    yield encode_event("start", {"session": session})
    try:
        with kb_lock(kb), open_conversation(kb_root(kb) / "state.json",
            PROJECT_ROOT / setting.WEB_DATA_DIR / "sessions.sqlite", knowledge_id(kb), levels=levels) as conversation:
            # 用户与角色加入会话键，禁止同名 session 跨账号或降权后复用旧证据。
            for event in conversation.stream(text, identity + ":" + session, resume):
                yield encode_event(event["event"], event["data"])
    except Exception:
        logging.getLogger(__name__).exception("Web 问答失败")
        yield encode_event("error", {"message": "问答失败，请检查服务日志；中断会话可使用恢复按钮继续。"})
    yield encode_event("done", {})
