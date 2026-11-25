"""切分、预算、真实 usage、全局重排及 MMR 的功能回归。"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import httpx
import pytest
from langchain_core.documents import Document
from qiye_rag import setting
from qiye_rag.rag.splitter import split_documents
from qiye_rag.rag.catalog import Catalog
from qiye_rag.rag.multi_retrieval import HybridRetriever
from qiye_rag.tokens import count_text, count_json, select_evidence
from qiye_rag.telemetry import capture, embedding_response
from qiye_rag.rag.loader import load_file
from qiye_rag.graph.nodes import Nodes
from qiye_rag.graph.state import Plan


def test_section_identity_and_location() -> None:
    """文字标题、相邻短段落及前缀构成完整事实，不跨文档合并。"""
    meta = {"knowledge_base_id": "kb", "source_path": "a.docx", "document_id": "A", "title": "产品A",
            "file_type": "docx", "access_level": "internal"}
    docs = [Document(page_content=t, metadata={**meta, "location": f"段落 {i}"})
            for i, t in enumerate(("# 产品A", "## 基础信息", "SKU：A", "交期25天", "## 风险", "含电池"), 1)]
    for mode in ("section", "prefix"):
        parts = split_documents(docs, mode=mode)
        basic = next(p for p in parts if "交期25天" in p.page_content)
        assert "SKU：A" in basic.page_content and "含电池" not in basic.page_content
        assert basic.metadata["section"] == "产品A / 基础信息"
        assert "段落 2" in basic.metadata["location"] and "段落 4" in basic.metadata["location"]
        assert basic.metadata["token_count"] == count_text(basic.page_content)
        assert basic.metadata["token_count_is_api_usage"] is False
        if mode == "prefix":
            assert basic.page_content.startswith("文档：A｜产品A")


def test_budget_and_real_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """预算按 JSON 实际序列化计数；Embedding usage 来自响应。

    Args:
        monkeypatch: 配置替换。
    """
    a = {"text": "规则" * 20, "citation": {"chunk_id": "a"}}
    b = {"text": "另一规则" * 200, "citation": {"chunk_id": "b"}}
    payload = [{"number": 1, "text": a["text"], "source": a["citation"]}]
    monkeypatch.setattr(setting, "EVIDENCE_TOKEN_BUDGET", count_json(payload))
    assert select_evidence([a, b]) == [a]
    response = httpx.Response(200, json={"data": [], "usage": {"total_tokens": 37}})
    with capture() as events:
        embedding_response(response)
    assert events[0]["input_tokens"] == 37 and events[0]["total_tokens"] == 37


def test_global_pool_and_mmr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """后一个查询的关键候选参与统一重排，旧证据重新验证权限。

    Args:
        tmp_path: 临时目录。
        monkeypatch: 检索替身。
    """
    cat = Catalog(tmp_path / "state.json")
    rows = [{"id": i, "text": t, "metadata": {"kb_id": "kb", "knowledge_base_id": "kb",
             "access_level": "internal", "status": "现行", "document_id": i, "sha256": "x"}}
            for i, t in (("a", "相似规则一"), ("b", "相似规则二"), ("c", "互补事实"))]
    cat.save("kb", "a.md", "x", rows)
    rankings = [[{"text": p["text"], "citation": {**p["metadata"], "chunk_id": p["id"]}} for p in batch]
                for batch in (rows[:2], rows[2:])]
    monkeypatch.setattr("qiye_rag.rag.multi_retrieval.search", MagicMock(side_effect=rankings))
    rank = MagicMock()
    rank.score.return_value = [0.95, 0.9, 0.8]  # RRF 保序为 a、c、b。
    embedding = SimpleNamespace(embed_documents=lambda texts: [[1, 0], [0.99, 0.01], [0, 1]],
                                embed_query=lambda query: [1, 0])
    retrieval = HybridRetriever("kb", cat, SimpleNamespace(embedding=embedding), rank, ("internal",), mmr=True)
    monkeypatch.setattr(setting, "RETRIEVAL_TOP_K", 2)
    monkeypatch.setattr(setting, "MMR_LAMBDA", 0.3)
    hits = retrieval.retrieve_many(["第一部分", "第二部分"], "完整问题", [])
    assert rank.score.call_count == 1 and rank.score.call_args.args[0] == "完整问题"
    assert {p["id"] for p in rank.score.call_args.args[1]} == {"a", "b", "c"}
    assert [h["citation"]["chunk_id"] for h in hits] == ["a", "b"]


def test_table_header_and_pdf_pages() -> None:
    """长表按行拆且重复表头，PDF 不跨页。"""
    meta = {"knowledge_base_id": "kb", "source_path": "a.docx", "document_id": "A", "file_type": "docx", "location": "表格 1"}
    text = "字段 | 值\n" + "\n".join(f"项目{i} | 数值{i}" for i in range(10))
    parts = split_documents([Document(page_content=text, metadata=meta)], size=50, overlap=0, mode="section")
    assert len(parts) > 1 and all(p.page_content.startswith("字段 | 值") for p in parts)
    assert all(len(p.page_content) <= 50 for p in parts)
    assert all(f"项目{i}" in "\n".join(p.page_content for p in parts) for i in range(10))
    pages = [Document(page_content="规则" * 100, metadata={**meta, "file_type": "pdf", "page": i, "location": f"页 {i}"}) for i in (1, 2)]
    pdf = split_documents(pages, size=50, overlap=5, mode="prefix")
    assert {p.metadata["page"] for p in pdf} == {1, 2}
    assert all(len(p.page_content) <= 50 for p in pdf)


def test_native_headings_and_input_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """原生Word标题保留层级，超预算在调用模型前停止。

    Args:
        tmp_path: 临时目录。
        monkeypatch: 配置替换。
    """
    from docx import Document as WordDocument
    document = WordDocument()
    document.add_heading("制度", 1)
    document.add_heading("申请", 2)
    document.add_paragraph("提前两天申请")
    path = tmp_path / "a.docx"
    document.save(path)
    parts = split_documents(load_file(path), mode="prefix")
    rule = next(p for p in parts if "提前两天" in p.page_content)
    assert rule.metadata["section"] == "制度 / 申请"
    chat = MagicMock()
    monkeypatch.setattr(setting, "INPUT_TOKEN_BUDGET", 1)
    with capture() as events:
        with pytest.raises(ValueError, match="预算"):
            Nodes(chat, lambda q: []).structured(Plan, "说明", {})
    chat.with_structured_output.assert_not_called()
    assert events[-1]["model_calls"] == 0


def test_previous_evidence_is_reauthorized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """旧证据失去授权时不参与全局重排，不能相信其携带正文。

    Args:
        tmp_path: 临时目录。
        monkeypatch: 检索替换。
    """
    cat = Catalog(tmp_path / "state.json")
    rows = [{"id": key, "text": key, "metadata": {"kb_id": "kb", "document_id": key, "sha256": "x",
             "access_level": level, "status": "现行"}} for key, level in (("ok", "internal"), ("secret", "restricted"))]
    cat.save("kb", "a.md", "x", rows)
    hit = {"text": "ok", "citation": {**rows[0]["metadata"], "chunk_id": "ok"}}
    monkeypatch.setattr("qiye_rag.rag.multi_retrieval.search", lambda *a, **kw: [hit])
    rank = MagicMock()
    rank.score.return_value = [1.0]
    retriever = HybridRetriever("kb", cat, object(), rank, ("internal",))
    previous = [{"text": "伪造旧正文", "citation": {"chunk_id": "secret"}}]
    result = retriever.retrieve_many(["问题"], "问题", previous)
    assert [c["citation"]["chunk_id"] for c in result] == ["ok"]
    assert all(c["id"] != "secret" for c in rank.score.call_args.args[1])


def test_upload_preview_matches_committed_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """上传预览与已提交的实际结构块一致。

    Args:
        tmp_path: 临时目录。
        monkeypatch: 外部服务替换。
    """
    from uuid import uuid4
    from qiye_rag.run_time import ingestion, storage
    from test_v1 import FakeVectors
    monkeypatch.setattr(storage, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(setting, "SPLITTER_MODE", "prefix")
    vectors = FakeVectors()
    monkeypatch.setattr(ingestion, "MilvusVectors", lambda *args: vectors)
    file_id = uuid4().hex
    root = storage.kb_root("preview")
    (root / "inbox").mkdir(parents=True)
    (root / "inbox" / (file_id + ".md")).write_text("# 产品\n## 参数\n交期25天", encoding="utf-8")
    result = ingestion.ingest("preview", file_id, "产品.md", ".md")
    parts = Catalog(root / "state.json").records("web_preview")[file_id + ".md"][1]
    assert result["chunks"] == len(parts)
    assert result["preview"][0]["text"] == parts[0]["text"][:200]
