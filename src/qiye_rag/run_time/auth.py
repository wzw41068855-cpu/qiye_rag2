"""密码校验和数据库 Token 登录状态。"""
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, InvalidHashError
from sqlalchemy import select
from sqlalchemy.orm import Session
from qiye_rag import setting
from qiye_rag.models import User, LoginToken

hasher = PasswordHasher()


def create_user(db: Session, username: str, password: str, role: str = "user") -> User:
    """由本地管理命令创建账号，禁止前端自行授予 HR。

    Args:
        db: ORM 会话。
        username: 用户名。
        password: 初始密码。
        role: user 或 hr。
    Returns:
        新用户。
    """
    if role not in {"user", "hr"} or not username.strip() or len(username) > 80 or not 8 <= len(password) <= 256:
        raise ValueError("用户名不为空且最多80字符，密码至少8字符，角色为 user/hr")
    if db.scalar(select(User).where(User.username == username.strip())):
        raise ValueError("用户名已存在")
    user = User(username=username.strip(), password_hash=hasher.hash(password), role=role)
    db.add(user)
    db.commit()
    return user


def login(db: Session, username: str, password: str) -> tuple[str, User] | None:
    """校验密码并签发随机 Token，仅摘要进入数据库。

    Args:
        db: ORM 会话。
        username: 用户名。
        password: 密码。
    Returns:
        Token 和用户，失败返回 None。
    """
    user = db.scalar(select(User).where(User.username == username))
    if not user or not user.active:
        return None
    try:
        hasher.verify(user.password_hash, password)
    except (VerificationError, InvalidHashError):
        return None
    token = secrets.token_urlsafe(32)
    db.add(LoginToken(token_hash=hashlib.sha256(token.encode()).hexdigest(), user_id=user.id,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=setting.LOGIN_TOKEN_TTL_SECONDS)))
    db.commit()
    return token, user


def authenticate(db: Session, token: str) -> User | None:
    """从数据库查询仍有效的登录状态。

    Args:
        db: ORM 会话。
        token: Bearer Token。
    Returns:
        有效用户或 None。
    """
    return db.scalar(select(User).join(LoginToken, User.id == LoginToken.user_id).where(
        LoginToken.token_hash == hashlib.sha256(token.encode()).hexdigest(),
        LoginToken.expires_at > datetime.now(timezone.utc), User.active.is_(True)))
