"""Celery worker 内的文档解析、切分和增量索引。"""
import re
from qiye_rag import setting
from qiye_rag.rag.catalog import Catalog
from qiye_rag.rag.config import load_config
from qiye_rag.rag.indexing import sync
from qiye_rag.rag.loader import load_file
from qiye_rag.rag.splitter import split_documents
from qiye_rag.rag.vectors import MilvusVectors
from qiye_rag.run_time.storage import kb_root, kb_lock, knowledge_id


def ingest(kb: str, file_id: str, filename: str, suffix: str, access_level: str = "internal") -> dict:
    """解析单次上传并同步该 Web 知识库。

    Args:
        kb: Web 知识库名称。
        file_id: 服务器生成的文件编号。
        filename: 原始文件名，只用于展示。
        suffix: 已验证的文件扩展名。
        access_level: 来自 HR 上传请求的文档等级。
    Returns:
        切块预览、总块数与同步统计。
    """
    #约束了 file_id 格式、允许的后缀和访问级别
    if (not re.fullmatch(r"[a-f0-9]{32}", file_id) or suffix not in {".pdf", ".docx", ".md"}
            or access_level not in {"internal", "restricted"}):
        raise ValueError("非法上传任务参数")

    root = kb_root(kb)#kb_root() 只返回路径，不会创建目录。目录创建在 Web 端完成，避免 Worker 直接操作磁盘。
    source = file_id + suffix
    staged = root / "inbox" / source
    #关键：
    #含义是：同一个 kb 的索引更新串行执行，避免多个任务同时改 manifest/state 或同时同步向量导致竞态。
    with kb_lock(kb):
        #先从 inbox 读取并标准化为文档对象，再切分成 chunk：
        docs = load_file(staged, {"knowledge_base_id": knowledge_id(kb), "document_id": file_id,
                                  "source_path": filename, "title": filename, "access_level": access_level})
        parts = split_documents(docs)
        if not parts:
            raise ValueError("文档未生成有效 Chunk")
        documents = root / "documents"
        documents.mkdir(exist_ok=True)
        # 文件名由服务器生成；原始名称不参与磁盘路径构造。
        #注意它使用服务器生成的 source（file_id + suffix）作为磁盘文件名，而不是用户文件名：
        (documents / source).write_bytes(staged.read_bytes())
        manifest = Catalog(documents / "manifest.json")#文件指纹
        entries = [e for e in manifest.state.get("documents", []) if e["source_path"] != source]
        entries.append({"source_path": source, "doc_id": file_id, "document_id": file_id, "title": filename,
                        "access_level": access_level, "status": "现行"})
        manifest._commit({"documents": entries})
        config = load_config()
        vectors = MilvusVectors(config.get("MILVUS_URI", "http://localhost:19530"),
            config.get("EMBEDDING_MODEL", "text-embedding-v3"), setting.MILVUS_COLLECTION,
            config.get("MILVUS_TOKEN", ""))
        catalog = Catalog(root / "state.json")
        try:
            # Milvus 向量库的 collection 可能不存在，sync() 会自动创建。
            #调用 sync(...) 做真正的增量索引同步：
            stats = sync(documents, knowledge_id(kb), catalog, vectors,
                         size=setting.CHUNK_SIZE, overlap=setting.CHUNK_OVERLAP)
        finally:
            catalog.close()
        staged.unlink(missing_ok=True)#最后删除暂存文件并返回结果摘要：
        return {"filename": filename, "chunks": len(parts), "stats": stats,
                "preview": [{"text": p.page_content[:200], "length": p.metadata["length"],
                             "location": p.metadata.get("location", "")} for p in parts[:3]]}
