"""显式启用的 PostgreSQL ORM 登录测试。"""
import os
import secrets
from uuid import uuid4
import pytest
from sqlalchemy import select
from qiye_rag.models import User, LoginToken
from qiye_rag.run_time.database import create_sessions
from qiye_rag.run_time.auth import create_user, login, authenticate


@pytest.mark.skipif(os.getenv("RAG_DB_SMOKE") != "1", reason="设置 RAG_DB_SMOKE=1 验证 PostgreSQL")
def test_postgres_login_roundtrip() -> None:
    """真实 PostgreSQL 上创建临时用户、校验 Token，最后仅清理本次记录。"""
    sessions = create_sessions()
    username = "test_" + uuid4().hex
    password = secrets.token_urlsafe(20)
    user_id = None
    try:
        with sessions() as db:
            user = create_user(db, username, password, "user")
            user_id = user.id
            result = login(db, username, password)
            assert result and authenticate(db, result[0]).id == user_id
            assert login(db, username, "wrong-password") is None
            token = db.scalar(select(LoginToken).where(LoginToken.user_id == user_id))
            db.delete(token)
            db.commit()
            assert authenticate(db, result[0]) is None
    finally:
        if user_id is not None:
            with sessions() as db:
                for token in db.scalars(select(LoginToken).where(LoginToken.user_id == user_id)):
                    db.delete(token)
                user = db.get(User, user_id)
                if user:
                    db.delete(user)
                db.commit()
