"""文件上传、异步任务状态和 SSE 问答接口。"""
from pathlib import Path
from uuid import UUID, uuid4
from typing import Literal
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator
from starlette.concurrency import run_in_threadpool
from qiye_rag import setting
from qiye_rag.run_time.chat import chat_events
from qiye_rag.run_time.storage import kb_root
from qiye_rag.models import User, UploadJob
from qiye_rag.web.auth import current_user, require_hr, get_db

router = APIRouter(prefix="/api")


class ChatRequest(BaseModel):
    """问答输入；v3 没有角色或权限切换参数。"""

    kb: str = Field(default="demo", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    session: str = Field(default="default", min_length=1, max_length=80)#用户与角色隔离的会话名，允许同一账号多轮对话。
    text: str = Field(default="", max_length=setting.MAX_QUESTION_CHARS)
    resume: bool = False
    document_id: str | None = None

    @model_validator(mode="after")
    def validate_question(self) -> "ChatRequest":
        """新轮次必须提供非空问题。

        Returns:
            通过校验的请求。
        """
        if not self.resume and not self.text.strip():
            raise ValueError("问题不能为空")
        return self


async def save_upload(request: Request, file: UploadFile, kb: str, user: User, db: Session,
                      access_level: str, relative: bool = False) -> dict:
    """保存上传文件并提交 Celery，不在 HTTP 请求内调用模型。

    Args:
        request: 当前应用。
        file: PDF、DOCX 或 Markdown。
        kb: Web 知识库名。
        user: 已由服务端验证的 HR。
        db: ORM 会话。
        access_level: 服务端接受的文档访问等级。
        relative: 是否允许文件夹中的安全相对路径。
    Returns:
        异步任务编号。
    """
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    display_name = filename if len(filename) <= 200 else filename[:200 - len(suffix)] + suffix
    parts = filename.replace("\\", "/").split("/")
    invalid_path = any(p in {"", ".", ".."} or ":" in p for p in parts)#它不仅限制后缀，还阻止目录穿越（..）和盘符注入（:）
    if suffix not in {".pdf", ".docx", ".md"} or invalid_path or (len(parts) > 1 and not relative):
        raise HTTPException(400, "仅支持普通文件名的 PDF、DOCX、Markdown")
    inbox = kb_root(kb) / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    file_id = uuid4().hex#不会用用户文件名直接落盘，而是生成服务器侧 file_id：
    path = inbox / (file_id + suffix)
    size = 0
    try:
        with path.open("xb") as stream:
            #x = exclusive creation（独占创建）：文件必须不存在；如果已存在会抛 FileExistsError
            #b = binary（二进制）：按字节写入，不做文本编码转换
            while data := await file.read(setting.UPLOAD_READ_BYTES):
                size += len(data)
                if size > setting.UPLOAD_MAX_BYTES:
                    raise HTTPException(413, "文件超过上传大小限制")
                stream.write(data)
        if size == 0:
            raise HTTPException(400, "不能上传空文件")
        args = (kb, file_id, display_name, suffix)
        if access_level == "restricted":
            args += (access_level,)

        #await run_in_threadpool(...) 是为了避免阻塞事件循环，确保 Celery 任务提交在后台线程中执行，从而提高响应速度和吞吐量。
        #app.state.enqueue = ingest_document.delay,提交ingest任务，但不等待 Celery 完成，而是立即返回任务 ID，允许前端轮询任务状态。
        job = await run_in_threadpool(request.app.state.enqueue, *args)
        db.add(UploadJob(id=str(job.id), user_id=user.id, document_id=file_id, kb=kb,
                         filename=display_name, access_level=access_level))
        db.commit()
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(503, "任务提交失败，请检查 Celery 队列配置") from exc
    finally:
        await file.close()
    return {"task_id": job.id, "filename": filename, "kb": kb}


@router.post("/documents", status_code=202)
async def upload_document(request: Request, file: UploadFile = File(...),
    kb: str = Query(default="demo", pattern=r"^[A-Za-z0-9_-]{1,64}$"),
    access_level: Literal["internal", "restricted"] = Form(default="internal"),
    user: User = Depends(require_hr), db: Session = Depends(get_db)) -> dict:
    """HR 上传单文件，服务端记录文档级别及任务归属。

    Args:
        request: 应用请求。
        file: 上传文件。
        kb: 知识库。
        access_level: 文档等级。
        user: HR 身份。
        db: ORM 会话。
    Returns:
        任务编号。
    """
    return await save_upload(request, file, kb, user, db, access_level)


@router.post("/documents/batch", status_code=202)
async def upload_batch(request: Request, files: list[UploadFile] = File(...),
    kb: str = Query(default="demo", pattern=r"^[A-Za-z0-9_-]{1,64}$"),
    access_level: Literal["internal", "restricted"] = Form(default="internal"),
    user: User = Depends(require_hr), db: Session = Depends(get_db)) -> dict:
    """文件夹按文件提交 Celery，个别文件失败不阻塞其他文件。

    Args:
        request: 当前请求。
        files: 浏览器文件夹选择产生的文件列表。
        kb: 知识库。
        access_level: 本批文档级别。
        user: HR。
        db: ORM 会话。
    Returns:
        成功任务列表和逐文件错误。
    """
    if len(files) > setting.BATCH_UPLOAD_MAX_FILES:
        raise HTTPException(413, "本次批量文件数量超过限制")
    tasks, errors = [], []
    for file in files:
        try:
            tasks.append(await save_upload(request, file, kb, user, db, access_level, relative=True))
        except HTTPException as exc:
            db.rollback()
            errors.append({"filename": file.filename, "error": exc.detail})
    return {"tasks": tasks, "errors": errors}


@router.get("/tasks/{task_id}")
def task_status(task_id: UUID, request: Request, user: User = Depends(current_user),
                db: Session = Depends(get_db)) -> dict:
    """查询 Celery 执行状态；失败不向网页暴露原始异常。

    Args:
        task_id: 任务编号。
        request: 当前应用。
        user: 当前登录账号。
        db: ORM 会话。
    Returns:
        状态和成功结果，或通用错误说明。
    """
    owned = db.get(UploadJob, str(task_id))
    if (not owned or owned.user_id != user.id
            or (owned.access_level == "restricted" and user.role != "hr")):
        raise HTTPException(404, "任务不存在")
    task = request.app.state.task_result(str(task_id))
    result = {"task_id": str(task_id), "state": task.state}
    if task.state == "SUCCESS":
        result["result"] = task.result
    elif task.state in {"FAILURE", "REVOKED"}:
        result["error"] = "解析或索引失败，请检查 worker 日志后重新上传"
    return result


@router.post("/chat")
def chat(body: ChatRequest, user: User = Depends(current_user), db: Session = Depends(get_db)) -> StreamingResponse:
    """启动 SSE 节点进度及最终答案流。

    Args:
        body: 问题或恢复请求。
        user: 当前登录账号，权限来自数据库角色。
        db: ORM 会话。
    Returns:
        同步生成器在线程池执行的 SSE 响应。
    """
    if not (kb_root(body.kb) / "state.json").is_file():
        raise HTTPException(409, "该 Web 知识库尚未索引成功，请先上传并等待任务完成")
    if body.document_id:
        document = db.scalar(select(UploadJob).where(UploadJob.document_id == body.document_id,
                                                     UploadJob.kb == body.kb))
        if document and document.access_level == "restricted" and user.role != "hr":
            raise HTTPException(403, "无权访问指定文档")
    levels = ("public", "internal", "restricted") if user.role == "hr" else ("public", "internal")
    return StreamingResponse(chat_events(body.kb, body.text, body.session, body.resume,
                             levels, f"user:{user.id}:{user.role}"),
        media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
