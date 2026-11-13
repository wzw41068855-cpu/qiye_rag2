"""v3 HTTP、SSE 和上传索引回归，不调用云端。"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from qiye_rag import setting
from qiye_rag.web.app import create_app
from qiye_rag.web import routes
from qiye_rag.run_time import storage, ingestion
from qiye_rag.run_time.chat import encode_event
from qiye_rag.graph import Conversation, build_graph
from test_graph import FakeChat, responses, CHUNK
from test_v1 import FakeVectors
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from qiye_rag.models import Base
from qiye_rag.run_time.auth import create_user


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """创建隔离存储的 HTTP 客户端。

    Args:
        tmp_path: 测试目录。
        monkeypatch: 补丁工具。
    Returns:
        不连接外部服务的客户端。
    """
    monkeypatch.setattr(storage, "PROJECT_ROOT", tmp_path)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    with sessions() as db:
        create_user(db, "test_hr", "password123", "hr")
    client = TestClient(create_app(sessions))
    token = client.post("/api/auth/login", json={"username": "test_hr", "password": "password123"}).json()["access_token"]
    client.headers["Authorization"] = "Bearer " + token
    return client


def test_upload_and_status(client: TestClient) -> None:
    """上传返回任务编号，HTTP 阶段不做解析；任务完成后可取预览。"""
    job_id = str(uuid4())
    enqueue = MagicMock(return_value=SimpleNamespace(id=job_id))
    client.app.state.enqueue = enqueue
    response = client.post("/api/documents?kb=demo", files={"file": ("规则.md", b"# policy")})
    assert response.status_code == 202 and response.json()["task_id"] == job_id
    args = enqueue.call_args.args
    assert args[0] == "demo" and args[2:] == ("规则.md", ".md")
    assert (storage.kb_root("demo") / "inbox" / (args[1] + ".md")).read_bytes() == b"# policy"
    client.app.state.task_result = lambda task_id: SimpleNamespace(state="SUCCESS", result={"chunks": 1})
    assert client.get("/api/tasks/" + job_id).json()["result"]["chunks"] == 1
    client.app.state.task_result = lambda task_id: SimpleNamespace(state="FAILURE", result="secret-key")
    assert "secret-key" not in client.get("/api/tasks/" + job_id).text
    assert client.get("/api/tasks/not-a-task").status_code == 422
    assert client.get("/").status_code == 200


def test_upload_validation(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """拒绝空文件、不支持格式、路径穿越和超限文件，清理不完整上传。"""
    client.app.state.enqueue = MagicMock()
    for filename, content in (("a.exe", b"x"), ("a.md", b""), ("../a.md", b"x")):
        assert client.post("/api/documents", files={"file": (filename, content)}).status_code == 400
    assert client.post("/api/documents?kb=../bad", files={"file": ("a.md", b"x")}).status_code == 422
    monkeypatch.setattr(setting, "UPLOAD_MAX_BYTES", 3)
    assert client.post("/api/documents", files={"file": ("a.md", b"1234")}).status_code == 413
    client.app.state.enqueue.assert_not_called()
    assert not list((storage.kb_root("demo") / "inbox").glob("*.md"))


def test_submit_failure_cleans_upload(client: TestClient) -> None:
    """队列提交失败时清理未被接受的上传文件。"""
    client.app.state.enqueue = MagicMock(side_effect=RuntimeError("broker unavailable"))
    response = client.post("/api/documents", files={"file": ("a.md", b"content")})
    assert response.status_code == 503
    assert not list((storage.kb_root("demo") / "inbox").glob("*.md"))


def test_chat_sse_and_validation(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """校验输入、缺失索引，检查 SSE 类型和事件边界。"""
    assert client.post("/api/chat", json={"text": ""}).status_code == 422
    assert client.post("/api/chat", json={"text": "问题"}).status_code == 409
    root = storage.kb_root("demo")
    root.mkdir(parents=True)
    (root / "state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(routes, "chat_events", lambda *args: iter([
        encode_event("progress", {"node": "router"}),
        encode_event("answer", {"answer": "答案\n第二行", "citations": []}), encode_event("done", {})]))
    response = client.post("/api/chat", json={"text": "问题"})
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
    assert "event: progress" in response.text and "event: answer" in response.text
    assert len(response.text.strip().split("\n\n")) == 3


def test_ingestion_reuses_v1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """真实解析、切分和 JSON 清单；向量服务替换为内存替身。"""
    monkeypatch.setattr(storage, "PROJECT_ROOT", tmp_path)
    vec = FakeVectors()
    monkeypatch.setattr(ingestion, "MilvusVectors", lambda *args: vec)
    file_id = uuid4().hex
    root = storage.kb_root("demo")
    (root / "inbox").mkdir(parents=True)
    (root / "inbox" / (file_id + ".md")).write_text("# 规则\n试用期为3个月。", encoding="utf-8")
    result = ingestion.ingest("demo", file_id, "规则.md", ".md")
    assert result["chunks"] >= 1 and result["stats"]["updated"] == 1
    assert vec.rows and result["preview"]
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    assert "web_demo" in state
    assert not (root / "inbox" / (file_id + ".md")).exists()


def test_stream_hides_drafts() -> None:
    """真实 LangGraph 按节点流动，但 SSE 数据中不公开模型草稿。"""
    conversation = Conversation(build_graph(FakeChat(responses()), lambda q: [CHUNK]), {})
    events = list(conversation.stream("试用期多久"))
    assert [e["data"]["node"] for e in events[:-1]] == ["router", "retriever", "researcher", "reviewer", "finish"]
    assert all(set(e["data"]) == {"node"} for e in events[:-1])
    assert events[-1]["event"] == "answer" and events[-1]["data"]["approved"]
