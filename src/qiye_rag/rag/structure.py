"""聚合 DOCX 章节，保留原始位置与标题层级。"""
from itertools import groupby
import re
from langchain_core.documents import Document


def sections(documents: list[Document]) -> list[Document]:
    """按来源与章节聚合，其他格式保留边界。

    Args:
        documents: 原文顺序的文档片段。
    Returns:
        章节级片段，不跨文件、权限或版本。
    """
    result: list[Document] = []
    def scope(doc: Document) -> tuple:
        """取得合并边界。

        Args:
            doc: 原始片段。
        Returns:
            可信范围标识。
        """
        return tuple(doc.metadata.get(k) for k in ("knowledge_base_id", "source_path", "access_level", "version", "file_type"))
    for _, batch in groupby(documents, scope):
        group = list(batch)
        if group[0].metadata["file_type"] != "docx" or "location" not in group[0].metadata:
            result.extend(group)
            continue
        if group[0].page_content.strip() == "---":
            end = next((i for i, d in enumerate(group[1:], 1) if d.page_content.strip() == "---"), None)
            if end is not None and any(d.page_content.strip().startswith("doc_id:") for d in group[1:end]):
                group = group[end + 1:]
        headings: dict[int, str] = {}
        buffer: list[Document] = []
        def flush() -> None:
            """提交完整章节和原始位置。"""
            if buffer:
                locations = [d.metadata["location"] for d in buffer]
                meta = {**buffer[0].metadata, "locations": locations,
                        "location": locations[0] if len(locations) == 1 else f"{locations[0]}—{locations[-1]}",
                        "section": " / ".join(headings.values())}
                result.append(Document(page_content="\n".join(d.page_content for d in buffer), metadata=meta))
                buffer.clear()
        for doc in group:
            text = doc.page_content.strip()
            if doc.metadata["location"].startswith("表格 "):
                flush()
                result.append(Document(page_content=doc.page_content,
                    metadata={**doc.metadata, "section": " / ".join(headings.values())}))
                continue
            match = re.match(r"^(#{1,6})\s+(.+)$", text)
            native = doc.metadata["location"].startswith("标题 ")
            if match or native:
                flush()
                number = re.search(r"级 (\d+)", doc.metadata["location"])
                level = len(match[1]) if match else int(number[1]) if number else 1
                headings = {k: v for k, v in headings.items() if k < level}
                headings[level] = match[2] if match else text
            buffer.append(doc)
        flush()
    return result
