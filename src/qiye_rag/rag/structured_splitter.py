"""结构切分、标题前缀与官方离线 tokenizer 文本计数。"""
from hashlib import sha256
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from qiye_rag import setting
from qiye_rag.rag.legacy_splitter import split_documents as legacy_split
from qiye_rag.rag.structure import sections
from qiye_rag.tokens import count_text, tokenizer


def split_documents(documents: list[Document], size: int = setting.CHUNK_SIZE,
                    overlap: int = setting.CHUNK_OVERLAP, mode: str = "legacy") -> list[Document]:
    """先识别结构再切分，legacy 用于原版对照。

    Args:
        documents: 已加载文档。
        size: 包括前缀的字符上限。
        overlap: 字符重叠。
        mode: legacy、section 或 prefix。
    Returns:
        带位置、内容身份及明确计数来源的块。
    """
    if size <= 0 or not 0 <= overlap < size or mode not in ("legacy", "section", "prefix"):
        raise ValueError("无效切分模式或要求 0 <= overlap < size")
    if mode == "legacy":
        result = legacy_split(documents, size, overlap)
    else:
        result = []
        counters: dict[str, int] = {}
        occurrences: dict[str, int] = {}
        headers = MarkdownHeaderTextSplitter(headers_to_split_on=list(setting.MARKDOWN_HEADERS), strip_headers=False)
        for doc in sections(documents):
            units = headers.split_text(doc.page_content) if doc.metadata["file_type"] in ("md", "pdf") else [doc]
            for unit in units:
                meta = {**doc.metadata, **unit.metadata}
                names = [str(meta[h]) for _, h in setting.MARKDOWN_HEADERS if h in meta]
                if names:
                    meta["section"] = " / ".join(names)
                prefix = ""
                # 无标题章节且没有独立文档编号的裸文本保持原文，不合成空身份。
                if mode == "prefix" and (meta.get("section") or meta.get("document_id") != meta.get("source_path")):
                    title = meta.get("title", meta.get("document_id", meta["source_path"]))
                    prefix = f"文档：{meta.get('document_id', '')}｜{title}\n章节：{meta.get('section', '')}\n"
                    prefix = prefix[:max(0, size // 3)]
                budget = size - len(prefix)
                splitter = RecursiveCharacterTextSplitter(chunk_size=budget, chunk_overlap=min(overlap, budget - 1),
                    separators=list(setting.CHUNK_SEPARATORS))
                texts = splitter.split_text(unit.page_content)
                if " | " in unit.page_content and len(unit.page_content) > budget:
                    rows = unit.page_content.splitlines()
                    head, current = rows[0], rows[0]
                    texts = []
                    for row in rows[1:]:
                        if len(current) + len(row) + 1 > budget and current != head:
                            texts.append(current)
                            current = head
                        if len(head) + len(row) + 1 > budget:
                            raise ValueError("表格单行超过块上限，请增大块大小或整理原表")
                        current += "\n" + row
                    texts.append(current)
                for body in texts:
                    text = prefix + body
                    if not body.strip():
                        continue
                    key = f"{meta['knowledge_base_id']}:{meta['source_path']}"
                    content = f"{key}:{sha256(text.encode()).hexdigest()}"
                    occurrence = occurrences.get(content, 0)
                    occurrences[content] = occurrence + 1
                    index = counters.get(key, 0)
                    counters[key] = index + 1
                    chunk_id = sha256(f"{content}:{occurrence}".encode()).hexdigest()
                    result.append(Document(page_content=text, metadata={**meta, "chunk_id": chunk_id,
                        "chunk_index": index, "length": len(text)}))
    codec = "deepseek-v4-official-offline" if tokenizer() else "utf8-byte-upper-bound"
    for doc in result:
        doc.metadata.update(token_count=count_text(doc.page_content), tokenizer=codec,
                            token_count_is_api_usage=False, splitter_mode=mode)
    return result
