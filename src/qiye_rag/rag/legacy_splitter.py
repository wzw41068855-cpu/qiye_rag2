"""历史切分与Gold快照兼容；jieba字段不用于真实模型消耗或预算。"""
from qiye_rag import setting
from hashlib import sha256
import jieba
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def split_documents(documents: list[Document], size: int = setting.CHUNK_SIZE,
                    overlap: int = setting.CHUNK_OVERLAP) -> list[Document]:
    """保持页边界和标题结构切分。
        把一批 Document 再切成更小的片段，同时补齐每个片段的可追踪元数据。
        输出里的每个文档都变成“chunk 级别”。
    Args:
        documents: 已加载文档。
        size: 字符块大小。
        overlap: 重叠字符数，必须小于 size。
    Returns:
        带块编号、字符数、token 数的片段。
    """
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("要求 size > 0 且 0 <= overlap < size")
    recursive = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
        separators=list(setting.CHUNK_SEPARATORS))
    headers = MarkdownHeaderTextSplitter(headers_to_split_on=list(setting.MARKDOWN_HEADERS), strip_headers=False)
    result = []
    counters: dict[str, int] = {}
    occurrences: dict[str, int] = {}
    for doc in documents:
        # 按标题切分 Markdown，PDF/DOCX暂时不按标题切分，保留页/段落/表格位置。
        sections = headers.split_text(doc.page_content) if doc.metadata["file_type"] == "md" else [doc]
        for section in sections:
            #合并元数据：原文档元数据 + section 级元数据
            meta = {**doc.metadata, **section.metadata}
            #比如文档有
            # 公司制度
            ## 请假
            ### 年假
            #对应section的元数据可能带：
            #{"h1": "公司制度", "h2": "请假", "h3": "年假"}
            headings = [str(meta[h]) for _, h in setting.MARKDOWN_HEADERS if h in meta]
            #headings的结果是[公司制度.请假.年假]，
            # 然后用 / 连接成“公司制度 / 请假 / 年假”，作为 section 的总标题。
            if headings:
                meta["section"] = " / ".join(headings)

            #section.page_content 可能很长，再按 size/overlap 切分成更小的 chunk。
            for text in recursive.split_text(section.page_content):
                if not text.strip():#空白文本不生成 chunk，避免无意义的空片段。
                    continue
                # 生成稳定的 chunk_id，便于后续去重和更新。
                key = f"{meta['knowledge_base_id']}:{meta['source_path']}"
                index = counters.get(key, 0)
                counters[key] = index + 1
                content_key = f"{key}:{sha256(text.encode()).hexdigest()}"
                occurrence = occurrences.get(content_key, 0)
                occurrences[content_key] = occurrence + 1
                # 相同正文的出现次数用于区分重复块，不依赖全文序号。
                chunk_id = sha256(f"{content_key}:{occurrence}".encode()).hexdigest()
                result.append(Document(page_content=text, metadata={**meta,
                    "chunk_index": index, "chunk_id": chunk_id, "length": len(text),
                    "token_count": len(list(jieba.cut(text))), "tokenizer": "jieba"}))
    return result
