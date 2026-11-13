"""FastAPI 应用工厂；启动时不连接 LLM 或 Milvus。"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from qiye_rag.run_time.celery_app import celery_app
from qiye_rag.run_time.tasks import ingest_document
from qiye_rag.web.routes import router
from qiye_rag.web.auth import router as auth_router
from qiye_rag.web.documents import router as documents_router
from sqlalchemy.orm import sessionmaker


def create_app(sessions: sessionmaker | None = None) -> FastAPI:
    """组装 HTML、上传任务和 SSE 路由。

    Args:
        sessions: 可注入的 ORM Session 工厂，默认从 DATABASE_URI 创建。
    Returns:
        FastAPI 应用。
    """
    app = FastAPI(title="Qiye RAG v4")
    app.state.sessions = sessions
    app.state.enqueue = ingest_document.delay
    app.state.task_result = celery_app.AsyncResult
    app.include_router(router)
    app.include_router(auth_router)
    app.include_router(documents_router)

    @app.get("/login", include_in_schema=False)
    def login_page() -> FileResponse:
        """返回登录页面。

        Returns:
            登录 HTML。
        """
        return FileResponse(Path(__file__).with_name("static") / "login.html")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """返回自包含演示网页。

        Returns:
            HTML 文件响应。
        """
        return FileResponse(Path(__file__).with_name("static") / "index.html")

    return app


app = create_app()
