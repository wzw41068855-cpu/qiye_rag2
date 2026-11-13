"""SQLAlchemy ORM 登录、角色与批量上传回归测试。"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from qiye_rag.models import Base, LoginToken, User, UploadJob
from qiye_rag.run_time.auth import create_user
from qiye_rag.run_time import storage
from qiye_rag.web.app import create_app
from qiye_rag.web import routes
from qiye_rag.run_time.chat import encode_event
from qiye_rag.run_time import ingestion
from test_v1 import FakeVectors


@pytest.fixture
def auth_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """使用真实 ORM 测试库，外部服务使用替身。

    Args:
        tmp_path: 测试目录。
        monkeypatch: 补丁工具。
    Returns:
        已初始化两种角色的 HTTP 客户端。
    """
    monkeypatch.setattr(storage, "PROJECT_ROOT", tmp_path)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    with sessions() as db:
        create_user(db, "hr", "password123", "hr")
        create_user(db, "reader", "password123", "user")
    client = TestClient(create_app(sessions))
    client.app.state.enqueue = MagicMock(side_effect=lambda *args: SimpleNamespace(id=str(uuid4())))
    return client


def sign_in(client: TestClient, name: str) -> dict:
    """取得认证头。

    Args:
        client: 测试客户端。
        name: 用户名。
    Returns:
        Bearer 头。
    """
    response = client.post("/api/auth/login", json={"username": name, "password": "password123"})
    assert response.status_code == 200
    return {"Authorization": "Bearer " + response.json()["access_token"]}


def test_token_lifecycle(auth_client: TestClient) -> None:
    """密码摘要、Token 摘要、过期和注销生效，SQL 注入字符串无法登录。"""
    client = auth_client
    assert client.get("/api/auth/me").status_code == 401
    assert client.post("/api/auth/login", json={"username": "hr' OR 1=1 --", "password": "x"}).status_code == 401
    header = sign_in(client, "hr")
    with client.app.state.sessions() as db:
        token = db.scalar(select(LoginToken))
        assert token.token_hash not in header["Authorization"]
        assert db.scalar(select(User).where(User.username == "hr")).password_hash != "password123"
        token.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.commit()
    assert client.get("/api/auth/me", headers=header).status_code == 401
    header = sign_in(client, "hr")
    assert client.post("/api/auth/logout", headers=header).status_code == 200
    assert client.get("/api/auth/me", headers=header).status_code == 401
    assert client.get("/login").status_code == 200


def test_roles_batch_and_task_ownership(auth_client: TestClient) -> None:
    """普通用户禁止上传和读取他人任务；文件夹逐文件提交并保留受限等级。"""
    client = auth_client
    user, hr = sign_in(client, "reader"), sign_in(client, "hr")
    assert client.post("/api/documents", headers=user, files={"file": ("a.md", b"x")}).status_code == 403
    response = client.post("/api/documents/batch", headers=hr, data={"access_level": "restricted"}, files=[
        ("files", ("folder/a.md", b"# a")), ("files", ("folder/sub/b.docx", b"example")),
        ("files", ("../bad.md", b"x")), ("files", ("folder/a.exe", b"x"))])
    assert response.status_code == 202
    assert len(response.json()["tasks"]) == 2 and len(response.json()["errors"]) == 2
    assert all(call.args[-1] == "restricted" for call in client.app.state.enqueue.call_args_list)
    job = response.json()["tasks"][0]["task_id"]
    client.app.state.task_result = lambda task_id: SimpleNamespace(state="SUCCESS", result={"chunks": 1})
    assert client.get("/api/tasks/" + job, headers=user).status_code == 404
    assert client.get("/api/tasks/" + job, headers=hr).status_code == 200
    assert client.get("/api/documents", headers=user).json() == []
    assert len(client.get("/api/documents", headers=hr).json()) == 2
    with client.app.state.sessions() as db:
        account = db.scalar(select(User).where(User.username == "hr"))
        account.role = "user"
        db.commit()
    assert client.get("/api/tasks/" + job, headers=hr).status_code == 404


def test_server_controls_scope(auth_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """服务端控制检索权限与用户会话隔离，显式越权文档返回403。"""
    client = auth_client
    user, hr = sign_in(client, "reader"), sign_in(client, "hr")
    root = storage.kb_root("demo")
    root.mkdir(parents=True)
    (root / "state.json").write_text("{}", encoding="utf-8")
    events = MagicMock(side_effect=lambda *args: iter([encode_event("done", {})]))
    monkeypatch.setattr(routes, "chat_events", events)
    client.post("/api/chat", headers=user, json={"text": "员工情况", "levels": ["restricted"], "role": "hr"})
    assert events.call_args.args[4] == ("public", "internal")
    identity_user = events.call_args.args[5]
    client.post("/api/chat", headers=hr, json={"text": "员工情况"})
    assert "restricted" in events.call_args.args[4] and events.call_args.args[5] != identity_user
    with client.app.state.sessions() as db:
        hr_id = db.scalar(select(User.id).where(User.username == "hr"))
        db.add(UploadJob(id=str(uuid4()), user_id=hr_id, document_id="a" * 32, kb="demo",
                         filename="private.md", access_level="restricted"))
        db.commit()
    assert client.post("/api/chat", headers=user, json={"text": "查询", "document_id": "a" * 32}).status_code == 403
    assert client.get("/api/documents/" + "a" * 32, headers=user).status_code == 403


def test_worker_preserves_restricted_level(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """HR 指定的受限等级进入最终 Chunk，不被文档头部覆盖。"""
    monkeypatch.setattr(storage, "PROJECT_ROOT", tmp_path)
    vectors = FakeVectors()
    monkeypatch.setattr(ingestion, "MilvusVectors", lambda *args: vectors)
    root, file_id = storage.kb_root("private"), uuid4().hex
    (root / "inbox").mkdir(parents=True)
    (root / "inbox" / (file_id + ".md")).write_text(
        "---\naccess_level: public\n---\n受限资料", encoding="utf-8")
    ingestion.ingest("private", file_id, "private.md", ".md", "restricted")
    assert vectors.rows and all(row["metadata"]["access_level"] == "restricted" for row in vectors.rows.values())
