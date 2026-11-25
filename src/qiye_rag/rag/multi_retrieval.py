"""多查询候选全局融合、统一重排与可选语义 MMR。"""
from typing import Any
import numpy as np
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from qiye_rag import setting
from qiye_rag.rag.catalog import Catalog
from qiye_rag.rag.retrieval import search
from qiye_rag.telemetry import stage


class HybridRetriever:
    """绑定可信授权范围，支持单查询兼容及 Graph 多查询。"""

    def __init__(self, kb: str, catalog: Catalog, vectors: Any, reranker: Any,
                 levels: tuple[str, ...], mmr: bool = False) -> None:
        """初始化范围及策略。

        Args:
            kb: 知识库。
            catalog: 激活清单。
            vectors: 向量适配器。
            reranker: 重排器。
            levels: 服务端授权等级。
            mmr: 是否启用语义多样性选择。
        """
        self.kb, self.catalog, self.vectors, self.reranker = kb, catalog, vectors, reranker
        self.levels, self.mmr = levels, mmr

    def __call__(self, query: str) -> list[dict]:
        """兼容单查询调用。

        Args:
            query: 查询。
        Returns:
            已排序证据。
        """
        return self.retrieve_many([query], query, [])

    def retrieve_many(self, queries: list[str], question: str, previous: list[dict]) -> list[dict]:
        """全局融合后围绕完整问题重排，第二轮不固守旧顺序。

        Args:
            queries: 当前子查询。
            question: 包含指代所需上下文的完整问题。
            previous: 上轮选中证据。
        Returns:
            仅来自本次可信范围的 Top K 证据。
        """
        recall = setting.RETRIEVAL_RECALL
        rankings = [search(q, self.kb, self.catalog, self.vectors, self.reranker, levels=self.levels,
                           top_k=setting.RETRIEVAL_TOP_K, candidate_only=True) for q in dict.fromkeys(queries)]
        # 旧证据只在仍为激活且仍被授权时参与，不信任外部传入的正文。
        allowed = {p["id"]: p for _, cs in self.catalog.records(self.kb).values() for p in cs
                   if p["metadata"].get("kb_id") == self.kb and p["metadata"].get("access_level") in self.levels
                   and p["metadata"].get("status") == "现行"}
        if previous:
            rankings.append([{"text": allowed[c["citation"]["chunk_id"]]["text"],
                "citation": {**allowed[c["citation"]["chunk_id"]]["metadata"], "chunk_id": c["citation"]["chunk_id"]}}
                for c in previous if c["citation"]["chunk_id"] in allowed])
        with stage("global_fusion") as event:
            scores: dict[str, float] = {}
            chunks: dict[str, dict] = {}
            for ranking in rankings:
                for rank, c in enumerate(ranking, 1):
                    key = c["citation"]["chunk_id"]
                    chunks[key] = c
                    scores[key] = scores.get(key, 0) + 1 / (setting.RRF_K + rank)
            pool = [chunks[i] for i in sorted(scores, key=scores.get, reverse=True)[:recall]]
            event["chunk_ids"] = [c["citation"]["chunk_id"] for c in pool]
        if not pool:
            return []
        with stage("global_rerank", model_calls=1, query=question) as event:
            raw = [{"id": c["citation"]["chunk_id"], "text": c["text"], "metadata": c["citation"]} for c in pool]
            values = self.reranker.score(question, raw)
            if len(values) != len(pool):
                raise ValueError("重排分数数量与候选数量不一致")
            pool = [dict(c, score=float(s)) for c, s in sorted(zip(pool, values), key=lambda pair: pair[1], reverse=True)]
            event["results"] = [{"chunk_id": c["citation"]["chunk_id"], "score": c["score"]} for c in pool]
        if self.mmr:
            with stage("mmr", model_calls=2) as event:
                embeddings = self.vectors.embedding.embed_documents([c["text"] for c in pool])
                query_vector = self.vectors.embedding.embed_query(question)
                ids = maximal_marginal_relevance(np.array(query_vector), embeddings,
                    lambda_mult=setting.MMR_LAMBDA, k=setting.RETRIEVAL_TOP_K)
                selected = [pool[i] for i in ids]
                event["chunk_ids"] = [c["citation"]["chunk_id"] for c in selected]
            return selected
        return pool[:setting.RETRIEVAL_TOP_K]
