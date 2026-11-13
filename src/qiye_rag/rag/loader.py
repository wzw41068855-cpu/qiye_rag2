"""统一文档加载与目录失败隔离。"""
from pathlib import Path
import json
from langchain_core.documents import Document
from qiye_rag.rag.parsing import parse, SUPPORTED

#忽略的文件名，避免 README、评估和说明文档被索引。
IGNORED = {"readme.md", "eval_问题设计建议.md", "manifest.md"}


def load_file(path: Path, metadata: dict | None = None) -> list[Document]:
    """加载单文件，生成统一核心元数据。

    Args:
        path: 源文件。
        metadata: 可信导入信息，覆盖文档头部。
    Returns:
        保留位置的 LangChain 文档。
    """
    parts, header = parse(path)
    meta = {**header, **(metadata or {})}
    meta.setdefault("source_path", path.name)
    meta.setdefault("knowledge_base_id", meta.get("kb_id", "default"))
    meta.setdefault("document_id", meta.get("doc_id", meta["source_path"]))
    meta.setdefault("title", path.stem)
    meta.setdefault("access_level", "restricted")
    meta.setdefault("status", "现行")
    meta.setdefault("version", "")
    meta.update(kb_id=meta["knowledge_base_id"], doc_id=meta["document_id"],
                source=meta["source_path"], file_type=path.suffix.lower()[1:])
    meta = json.loads(json.dumps(meta, default=str))  # YAML 日期转为 JSON 可保存的字符串。
    docs = []
    section = ""
    for text, location in parts:
        if not text.strip():
            continue
        details = {**meta, "location": location}
        if location.startswith("页 "):
            details["page"] = int(location.split()[1])
        if location.startswith("标题 "):
            section = text
        if section:
            details["section"] = section
        docs.append(Document(page_content=text, metadata=details))
    if not docs:
        raise ValueError(f"文档无文本，扫描 PDF 需要 OCR: {path}")
    return docs


def discover(root: Path) -> list[dict]:
    """获取业务文件清单，排除说明和评估文件。

    Args:
        root: 知识库目录。
    Returns:
        包含 source_path 的条目。
    """
    manifest = root / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8-sig"))["documents"]
    return [{"source_path": p.relative_to(root).as_posix()} for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix.lower() in SUPPORTED
            and p.name.lower() not in IGNORED and not p.name.lower().startswith("eval_")]


def load_directory(root: Path, kb: str = "default") -> tuple[list[Document], dict[str, str]]:
    """批量加载，单文件失败不影响其他文件。

    Args:
        root: 知识库目录。
        kb: 知识库编号。
    Returns:
        成功文档与按来源记录的错误。
    """
    root = root.resolve(strict=True)
    docs, errors = [], {}
    for entry in discover(root):
        source = entry["source_path"]
        try:
            path = (root / source).resolve()
            if not path.is_relative_to(root):
                raise ValueError("不安全路径")
            docs.extend(load_file(path, {**entry, "knowledge_base_id": kb}))
        except Exception as exc:
            errors[source] = str(exc)
    return docs, errors
