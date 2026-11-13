"""文件内容与元数据驱动的增量同步。"""
from qiye_rag import setting
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from qiye_rag.rag.catalog import Catalog
from qiye_rag.rag.parsing import SUPPORTED, chunks, parse
from qiye_rag.rag.loader import discover


def recover(catalog: Catalog, journal: Catalog, vectors: Any, kb: str) -> None:
    """清理未激活版本；清理失败时保留日志以便重试。

    Args:
        catalog: 当前可检索版本清单。
        journal: 写入前持久化的块编号日志。
        vectors: 向量存储。
        kb: 知识库编号。
    """
    active = {p["id"] for _, parts in catalog.records(kb).values() for p in parts}
    for source, ids in list(journal.state.get(kb, {}).items()):
        vectors.delete(sorted(set(ids) - active))
        journal.delete(kb, source)


def stage(journal: Catalog, kb: str, source: str, ids: list[str]) -> None:
    """在向量操作前记录全部可能残留的编号。

    Args:
        journal: 恢复日志。
        kb: 知识库编号。
        source: 来源路径。
        ids: 新旧版本的块编号。
    """
    state = {k: dict(v) for k, v in journal.state.items()}
    state.setdefault(kb, {})[source] = ids
    journal._commit(state)


def sync(root: Path, kb: str, catalog: Catalog, vectors: Any,
         size: int = setting.CHUNK_SIZE, overlap: int = setting.CHUNK_OVERLAP) -> dict[str, int]:
    """同步新增、修改及删除文件；失败时不提交本地清单。

    Args:
        root: 完整知识库目录，必须存在。
        kb: 知识库编号。
        catalog: 本地清单。
        vectors: 向量存储。
        size: 块大小。
        overlap: 重叠字符数。
    Returns:
        更新、跳过、删除的文档数。
    """
    root = root.resolve(strict=True)
    if not root.is_dir() or not kb.strip():
        raise ValueError("知识库目录或编号无效")
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("要求 0 <= overlap < size")
    entries = discover(root)
    prepared = {}
    document_ids: set[str] = set()
    for entry in entries:
        path = (root / entry["source_path"]).resolve()
        # 防止清单中的路径跳出知识库，读取任意本地文件。
        if not path.is_relative_to(root) or path.suffix.lower() not in SUPPORTED:
            raise ValueError("清单包含不安全或不支持的路径")
        source = path.relative_to(root).as_posix()
        if source in prepared:
            raise ValueError("清单包含重复来源")
        # 无清单时读取 Markdown 权限；缺失权限默认 restricted，避免意外公开。
        if not path.exists():
            continue  # 清单仍引用但实际已移除的文件，也按删除处理。
        _, header = parse(path)
        meta = {**header, **entry, "source_path": source, "kb_id": kb}
        meta.setdefault("access_level", "restricted")
        meta.setdefault("status", "现行")
        meta.setdefault("title", path.stem)
        meta.setdefault("doc_id", source)
        meta["sha256"] = sha256(path.read_bytes()).hexdigest()
        meta["knowledge_base_id"] = kb
        raw_id = meta.get("document_id", meta["doc_id"])
        document_id = raw_id.strip() if isinstance(raw_id, str) else ""
        if not document_id or document_id in document_ids:
            raise ValueError(f"知识库内 document_id 必须非空且唯一: {document_id}")
        document_ids.add(document_id)
        meta["doc_id"] = meta["document_id"] = document_id
        payload = json.dumps([meta, size, overlap, setting.CHUNK_SEPARATORS, setting.MARKDOWN_HEADERS,
                              vectors.signature, "versioned-v1"], sort_keys=True,
                             ensure_ascii=False, default=str).encode()
        digest = sha256(path.read_bytes() + payload).hexdigest()
        prepared[source] = (path, meta, digest)
    old = catalog.records(kb)
    journal = Catalog(catalog.path.with_name(catalog.path.name + ".pending.json"))
    recover(catalog, journal, vectors, kb)
    stats = {"updated": 0, "skipped": 0, "deleted": 0}
    for source, (path, meta, digest) in prepared.items():
        if source in old and old[source][0] == digest:
            stats["skipped"] += 1
            continue
        parts = chunks(path, meta, size, overlap)
        if not parts or any(not p["text"].strip() for p in parts):
            raise ValueError(f"文档不能生成空 Chunk: {source}")
        for part in parts:
            content_id = part["id"]
            part["id"] = sha256(f"{content_id}:{digest}".encode()).hexdigest()
            part["metadata"].update(content_id=content_id, chunk_id=part["id"], index_version=digest)
        old_ids = [c["id"] for c in old.get(source, ("", []))[1]]
        stage(journal, kb, source, old_ids + [c["id"] for c in parts])
        # 写入完整新版本后，以原子清单提交作为唯一激活点；失败不影响旧版本。
        vectors.upsert(parts)
        catalog.save(kb, source, digest, parts)
        recover(catalog, journal, vectors, kb)
        stats["updated"] += 1
    for source in old.keys() - prepared.keys():
        stage(journal, kb, source, [c["id"] for c in old[source][1]])
        catalog.delete(kb, source)
        recover(catalog, journal, vectors, kb)
        stats["deleted"] += 1
    return stats
