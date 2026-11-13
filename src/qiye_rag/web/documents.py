"""已上传文档的授权目录与 HR 原文件查看。"""
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from qiye_rag.models import UploadJob, User
from qiye_rag.web.auth import current_user, require_hr, get_db
from qiye_rag.run_time.storage import kb_root

router = APIRouter(prefix="/api/documents")


@router.get("")
def list_documents(kb: str = Query(default="demo", pattern=r"^[A-Za-z0-9_-]{1,64}$"),
                   user: User = Depends(current_user), db: Session = Depends(get_db)) -> list[dict]:
    """只列出当前角色可查询的文档信息。

    Args:
        kb: 知识库。
        user: 登录用户。
        db: ORM 会话。
    Returns:
        文档编号与名称。
    """
    query = select(UploadJob).where(UploadJob.kb == kb)
    if user.role != "hr":
        query = query.where(UploadJob.access_level == "internal")
    return [{"document_id": row.document_id, "filename": row.filename, "access_level": row.access_level}
            for row in db.scalars(query)]


@router.get("/{document_id}")
def read_document(document_id: str, user: User = Depends(require_hr),
                   db: Session = Depends(get_db)) -> FileResponse:
    """HR 查看已上传的原始文件，普通用户只允许问答查询。

    Args:
        document_id: 服务器生成的文档编号。
        user: HR 用户。
        db: ORM 会话。
    Returns:
        原始文档。
    """
    row = db.scalar(select(UploadJob).where(UploadJob.document_id == document_id))
    if not row:
        raise HTTPException(404, "文档不存在")
    path = kb_root(row.kb) / "documents" / (row.document_id + Path(row.filename).suffix.lower())
    if not path.is_file():
        raise HTTPException(404, "文件尚未解析或不存在")
    return FileResponse(path, filename=Path(row.filename).name)
