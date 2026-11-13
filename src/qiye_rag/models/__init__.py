"""SQLAlchemy ORM 表定义。"""
from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """统一 ORM 元数据。"""


class User(Base):
    """学习系统用户，不保存明文密码。"""

    __tablename__ = "rag_users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    role: Mapped[str] = mapped_column(String(16), default="user")
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class LoginToken(Base):
    """仅保存不透明 Token 的摘要与有效期。"""

    __tablename__ = "rag_login_tokens"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("rag_users.id"), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class UploadJob(Base):
    """上传任务归属及文档访问等级。"""

    __tablename__ = "rag_upload_jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("rag_users.id"), index=True)
    document_id: Mapped[str] = mapped_column(String(32), unique=True)
    kb: Mapped[str] = mapped_column(String(64), index=True)
    filename: Mapped[str] = mapped_column(String(200))
    access_level: Mapped[str] = mapped_column(String(16), default="internal")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
