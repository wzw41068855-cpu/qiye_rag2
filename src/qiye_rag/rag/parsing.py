"""文档解析、元数据合并和文本切分。"""
from qiye_rag import setting
from pathlib import Path
import yaml
from pypdf import PdfReader
from docx import Document
from docx.text.paragraph import Paragraph

SUPPORTED = {".pdf", ".docx", ".md"}


def parse(path: Path) -> tuple[list[tuple[str, str]], dict]:
    """解析正文与头部。

    Args:
        path: 原始文件。
    Returns:
        文本位置对和头部元数据。
    Raises:
        ValueError: 格式或头部不合法。
    """
    #对于 PDF，它按“页”读取。每页生成一个 (text, location)，location 是类似 "页 1" 的标识；
    if path.suffix.lower() == ".pdf":
        return [(p.extract_text() or "", f"页 {i}")
                for i, p in enumerate(PdfReader(path).pages, 1)], {}
    #对于 DOCX，它按段落和表格读取。每个段落/表格生成一个 (text, location)，location 是类似 "标题 1"、"段落 2"、"表格 3" 的标识；
    if path.suffix.lower() == ".docx":
        doc = Document(path)
        parts = []
        for i, item in enumerate(doc.iter_inner_content(), 1):
            if isinstance(item, Paragraph):
                label = "标题" if item.style.name.startswith("Heading") or item.style.name == "Title" else "段落"
                level = item.style.name.removeprefix("Heading ")
                suffix = f" 级 {level}" if label == "标题" and level.isdigit() else ""
                parts.append((item.text, f"{label} {i}{suffix}"))
            else:
                parts.append(("\n".join(" | ".join(c.text for c in r.cells) for r in item.rows), f"表格 {i}"))
        return parts, {}
    if path.suffix.lower() != ".md":
        raise ValueError("仅支持 PDF/DOCX/Markdown")
    #md, Markdown 解析：剥离 YAML Front Matter，保留正文为一体。
    text = path.read_text(encoding="utf-8-sig")
    lines, meta = text.splitlines(), {}
    if lines and lines[0].strip() == "---":
        # 查找 YAML Front Matter 的结束标记
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end is None:
            raise ValueError("Markdown 头部缺少结束标记")
        meta = yaml.safe_load("\n".join(lines[1:end])) or {}
        if not isinstance(meta, dict):
            raise ValueError("Markdown 头部必须为键值表")
        text = "\n".join(lines[end + 1:])
    return [(text, "正文")], meta


def chunks(path: Path, metadata: dict, size: int = setting.CHUNK_SIZE, overlap: int = setting.CHUNK_OVERLAP) -> list[dict]:
    """兼容原接口，加载并切分文档。

    Args:
        path: 文件路径。
        metadata: 可信元数据。
        size: 块字符数。
        overlap: 重叠字符数。
    Returns:
        带稳定编号的片段字典。
    """
    from qiye_rag.rag.loader import load_file
    from qiye_rag.rag.splitter import split_documents
    docs = split_documents(load_file(path, metadata), size, overlap, mode=setting.SPLITTER_MODE)
    return [{"id": d.metadata["chunk_id"], "text": d.page_content, "metadata": d.metadata} for d in docs]
