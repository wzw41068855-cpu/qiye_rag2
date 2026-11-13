"""登录路由和服务端角色依赖。"""
import hashlib
from collections.abc import Iterator
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from qiye_rag.models import User, LoginToken
from qiye_rag.run_time.database import create_sessions
from qiye_rag.run_time.auth import authenticate, login

router = APIRouter(prefix="/api/auth")
bearer = HTTPBearer(auto_error=False)


def get_db(request: Request) -> Iterator[Session]:
    """为一个请求打开 ORM Session。

    Args:
        request: 当前应用。
    Yields:
        自动关闭的数据库会话。
    """
    if request.app.state.sessions is None:
        request.app.state.sessions = create_sessions()
    with request.app.state.sessions() as db:
        yield db


def current_user(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
                 db: Annotated[Session, Depends(get_db)]) -> User:
    """从 Bearer Token 确认登录用户。

    Args:
        credentials: 认证头。
        db: 数据库会话。
    Returns:
        当前有效用户。
    """
    user = authenticate(db, credentials.credentials) if credentials else None
    if not user:
        raise HTTPException(401, "请登录，或登录状态已失效", headers={"WWW-Authenticate": "Bearer"})
    return user


def require_hr(user: Annotated[User, Depends(current_user)]) -> User:
    """仅允许 HR 执行文档写入操作。

    Args:
        user: 当前用户。
    Returns:
        HR 用户。
    """
    if user.role != "hr":
        raise HTTPException(403, "仅 HR 可以上传文档")
    return user


class LoginRequest(BaseModel):
    """用户名和密码输入。"""

    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=256)


@router.post("/login")
def sign_in(body: LoginRequest, db: Annotated[Session, Depends(get_db)]) -> dict:
    """登录成功后返回 Token 和角色。

    Args:
        body: 登录信息。
        db: ORM 会话。
    Returns:
        登录状态。
    """
    result = login(db, body.username, body.password)
    if not result:
        raise HTTPException(401, "用户名或密码错误")
    token, user = result
    return {"access_token": token, "token_type": "bearer", "username": user.username, "role": user.role}


@router.get("/me")
def me(user: Annotated[User, Depends(current_user)]) -> dict:
    """返回当前账号信息。

    Args:
        user: 当前用户。
    Returns:
        用户名与角色。
    """
    return {"username": user.username, "role": user.role}


@router.post("/logout")
def logout(credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer)],
           user: Annotated[User, Depends(current_user)], db: Annotated[Session, Depends(get_db)]) -> dict:
    """删除当前登录 Token。

    Args:
        credentials: 当前 Token。
        user: 已认证账号。
        db: ORM 会话。
    Returns:
        注销结果。
    """
    token = db.get(LoginToken, hashlib.sha256(credentials.credentials.encode()).hexdigest())
    if token:
        db.delete(token)
        db.commit()
    return {"logged_out": True}
