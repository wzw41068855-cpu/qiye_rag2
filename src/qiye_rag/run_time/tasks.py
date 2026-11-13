"""异步索引任务入口。"""
from qiye_rag.run_time.celery_app import celery_app
from qiye_rag.run_time.ingestion import ingest


@celery_app.task(name="qiye_rag.ingest")
def ingest_document(kb: str, file_id: str, filename: str, suffix: str, access_level: str = "internal") -> dict:
    """在 worker 中执行解析与向量化。

    Args:
        kb: Web 知识库名称。
        file_id: 上传文件编号。
        filename: 展示名称。
        suffix: 文件类型。
        access_level: 文档等级，由上传接口验证。
    Returns:
        索引统计与切块预览。
    """
    if access_level == "internal":
        return ingest(kb, file_id, filename, suffix)
    return ingest(kb, file_id, filename, suffix, access_level)
