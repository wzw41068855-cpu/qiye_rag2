"""中文 BM25 与向量召回融合、重排和来源引用。"""
from qiye_rag import setting
from typing import Any
import jieba
from rank_bm25 import BM25Plus
from qiye_rag.rag.catalog import Catalog
from qiye_rag.telemetry import stage


def search(query: str, kb: str, catalog: Catalog, vectors: Any, reranker: Any,
           levels: tuple[str, ...] = ("public", "internal"),
           status: str | None = "现行", top_k: int = setting.RETRIEVAL_TOP_K, recall: int | None = None) -> list[dict]:
    """先权限过滤，再混合召回和重排。

    Args:
        query: 非空问题。
        kb: 知识库编号。
        catalog: 本地索引。
        vectors: Milvus 适配器。
        reranker: CrossEncoder 适配器。
        levels: 可信调用方授权的权限等级。
        status: 文档状态。
        top_k: 最终结果数。
        recall: 每路召回数，未指定时取 max(setting.RETRIEVAL_RECALL, top_k)。
    Returns:
        正文、重排分数及引用信息。
    """
    recall = max(setting.RETRIEVAL_RECALL, top_k) if recall is None else recall
    if not query.strip() or not 1 <= top_k <= recall <= setting.MAX_RETRIEVAL_K:
        raise ValueError(f"问题不能为空，且要求 1 <= top_k <= recall <= {setting.MAX_RETRIEVAL_K}")
    # BM25 的语料和重排输入也必须受权限限制，不能只过滤最终输出。
    with stage("authorization", kb=kb, levels=list(levels), status=status) as event:
        key = (kb, tuple(sorted(set(levels))), status, setting.BM25_K1, setting.BM25_B, setting.BM25_DELTA)
        event["cache_hit"] = key in catalog.bm25_cache
        if key not in catalog.bm25_cache:
            parts = [c for _, cs in catalog.records(kb).values() for c in cs
                     if c["metadata"].get("kb_id") == kb
                     and c["metadata"].get("access_level") in levels
                     and (status is None or c["metadata"].get("status") == status)]
            tokens = [list(jieba.cut(c["text"])) for c in parts]
            # Plus 的 IDF 为正；delta=0 保证不匹配片段得分为 0。
            bm25 = BM25Plus(tokens, k1=setting.BM25_K1, b=setting.BM25_B, delta=setting.BM25_DELTA) if parts else None
            if len(catalog.bm25_cache) >= setting.BM25_CACHE_SCOPES:
                catalog.bm25_cache.clear()
            catalog.bm25_cache[key] = (parts, bm25)
        parts, bm25 = catalog.bm25_cache[key]
        event["chunk_ids"] = [c["id"] for c in parts]
    if not parts:
        return []
    by_id = {c["id"]: c for c in parts}
    query_tokens = [t for t in jieba.cut(query) if t.isalnum()]
    with stage("bm25", query=query) as event:
        scores = bm25.get_scores(query_tokens)
        selected = [i for i in sorted(range(len(parts)), key=lambda i: scores[i], reverse=True)
                    if scores[i] > setting.BM25_MIN_SCORE
                    and any(t in bm25.doc_freqs[i] for t in query_tokens)][:recall]
        event["results"] = [{"chunk_id": parts[i]["id"], "score": float(scores[i])} for i in selected]
        lexical = [parts[i]["id"] for i in selected]
    with stage("dense", query=query, model_calls=1) as event:
        if hasattr(vectors, "search_active"):
            dense = vectors.search_active(query, kb, levels, status, recall, list(by_id))
        else:
            dense = vectors.search(query, kb, levels, status, recall)
        event["chunk_ids"] = [i for i in dense if i in by_id]
    fused: dict[str, float] = {}
    # RRF 按排名融合，避免直接相加 BM25 与向量的不同分数量纲。
    with stage("fusion", query=query) as event:
        for ranking in (lexical, dense):
            for rank, chunk_id in enumerate(dict.fromkeys(ranking), 1):
                if chunk_id in by_id:
                    fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (setting.RRF_K + rank)
        event["results"] = [{"chunk_id": i, "score": fused[i]}
                            for i in sorted(fused, key=fused.get, reverse=True)[:recall]]
    candidates = [by_id[i] for i in sorted(fused, key=fused.get, reverse=True)[:recall]]
    if not candidates:
        return []
    with stage("rerank", query=query, model_calls=1) as event:
        scores = reranker.score(query, candidates)
        if len(scores) != len(candidates):
            raise ValueError("重排分数数量与候选数量不一致")
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
        event["results"] = [{"chunk_id": c["id"], "score": float(s)} for c, s in ranked]
    return [{"text": c["text"], "score": float(score),
             "citation": {**c["metadata"], "chunk_id": c["id"]}} for c, score in ranked]
