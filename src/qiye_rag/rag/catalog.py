"""原子 JSON 增量清单，保留轻量的旧调用接口。"""
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


class Catalog:
    """保存每份成功索引的文件及片段。"""

    def __init__(self, path: Path) -> None:
        """打开 JSON 清单。

        Args:
            path: 状态文件，不再兼容旧 SQLite 文件。
        """
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.revision = 0
        self.bm25_cache: dict = {}

    def records(self, kb: str) -> dict[str, tuple[str, list[dict]]]:
        """读取知识库成功记录。

        Args:
            kb: 知识库编号。
        Returns:
            来源到摘要及片段的映射。
        """
        return {s: (r["digest"], r["chunks"]) for s, r in self.state.get(kb, {}).items()}

    def _commit(self, state: dict) -> None:
        """通过同目录临时文件原子替换，成功后才更新内存。

        Args:
            state: 待写入状态。
        """
        name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             delete=False) as stream:
                name = stream.name
                json.dump(state, stream, ensure_ascii=False, indent=2, default=str)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
            self.state = state
            self.revision += 1
            self.bm25_cache.clear()
        finally:
            if name and os.path.exists(name):
                os.unlink(name)

    def save(self, kb: str, source: str, digest: str, parts: list[dict]) -> None:
        """提交成功文档。

        Args:
            kb: 知识库编号。
            source: 规范来源路径。
            digest: 内容及配置摘要。
            parts: 新片段。
        """
        if not parts or any(not p.get("text", "").strip() for p in parts):
            raise ValueError("不能提交空文档或空 Chunk")
        state = {k: dict(v) for k, v in self.state.items()}
        meta = parts[0]["metadata"]
        state.setdefault(kb, {})[source] = {"digest": digest, "sha256": meta["sha256"],
            "document_id": meta["document_id"], "knowledge_base_id": kb,
            "source_path": source, "indexed_at": datetime.now(timezone.utc).isoformat(),
            "status": "indexed", "chunks": parts}
        self._commit(state)

    def delete(self, kb: str, source: str) -> None:
        """删除成功清理的记录。

        Args:
            kb: 知识库编号。
            source: 来源路径。
        """
        state = {k: dict(v) for k, v in self.state.items()}
        state.get(kb, {}).pop(source, None)
        self._commit(state)

    def close(self) -> None:
        """兼容旧入口；JSON 清单无需关闭连接。"""
