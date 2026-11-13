"""PostgreSQL 引擎与 ORM Session，不直接拼接 SQL。"""
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker
from qiye_rag.models import Base
from qiye_rag.rag.config import load_config


def create_sessions() -> sessionmaker:
    """读取 DATABASE_URI 并创建本项目的 ORM 表。

    Returns:
        可创建短生命周期 Session 的工厂。
    """
    uri = load_config().get("DATABASE_URI", "")
    if not uri or make_url(uri).get_backend_name() != "postgresql":
        raise ValueError("请在 .env 配置 DATABASE_URI=postgresql+psycopg://用户:密码@localhost:5432/数据库")
    url = make_url(uri).set(drivername="postgresql+psycopg")
    engine = create_engine(url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
