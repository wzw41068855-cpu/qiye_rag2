"""复用 LangChain 官方 Milvus 集成。"""
from qiye_rag import setting
import json
from typing import Any
import httpx
from qiye_rag.rag.config import load_config
from qiye_rag.rag.schema import collection_layout
from qiye_rag.telemetry import stage


class MilvusVectors:
    """最薄的向量写入与检索服务。"""

    def __init__(self, uri: str, model: str, collection: str = setting.MILVUS_COLLECTION, token: str = "") -> None:
        """加载单一 Embedding 提供商并检查集合的字段和维度是否匹配。

        Args:
            uri: Milvus 地址。
            model: 百炼 Embedding 模型，或可选的本地 HuggingFace 模型。
            collection: 集合名称。
            token: 认证令牌。
        """
        from langchain_milvus import Milvus
        config = load_config()
        #优先使用DASHSCOPE_API系列向量嵌入模型
        if model.startswith("text-embedding-"):
            from langchain_openai import OpenAIEmbeddings
            key = config.get("DASHSCOPE_API_KEY", "")
            if not key:
                raise ValueError("请在 .env 或环境变量设置 DASHSCOPE_API_KEY")
            self.embedding = OpenAIEmbeddings(model=model, api_key=key,
                base_url=config.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                check_embedding_ctx_length=False,
                chunk_size=min(setting.EMBEDDING_API_BATCH_LIMIT, setting.EMBEDDING_BATCH_SIZE),
                timeout=setting.EMBEDDING_TIMEOUT_SECONDS, max_retries=setting.EMBEDDING_MAX_RETRIES)
        #使用本地 HuggingFace 模型
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embedding = HuggingFaceEmbeddings(model_name=model,
                encode_kwargs={"normalize_embeddings": setting.NORMALIZE_EMBEDDINGS})
        #向量库参数：
        options = dict(embedding_function=self.embedding, collection_name=collection,
            connection_args={"uri": uri, "token": token, "timeout": setting.MILVUS_TIMEOUT_SECONDS},
            primary_field="pk", auto_id=False, metadata_field="metadata", enable_dynamic_field=False,
            consistency_level="Strong", index_params={"index_type": "AUTOINDEX", "metric_type": setting.VECTOR_METRIC})
        self.store = Milvus(**options)
        self.primary_field = "pk"
        if self.store.col is not None:
            primary, _ = collection_layout(self.store.col.schema.to_dict())
            self.primary_field = primary
            if primary != "pk":
                # 仅兼容已有 JSON 集合的主键名；元数据存储始终是 metadata JSON。
                options.update(primary_field=primary)
                self.store = Milvus(**options)
            expected = len(self.embedding.embed_query("维度检查"))
            field = next(f for f in self.store.col.schema.fields if f.name == "vector")
            if int(field.params["dim"]) != expected:
                raise ValueError("Embedding 维度与 Collection 不兼容，请使用新集合")
        #它把连接地址、集合名、模型、API 基地址和元数据布局拼在一起，常用于缓存键、变更检测或调试定位：只要其中一项变了，签名就会变，说明“当前向量运行环境”已经不是同一个。
        self.signature = (f"{uri}:{collection}:{model}:{config.get('DASHSCOPE_BASE_URL', '')}:metadata:"
                          f"{setting.NORMALIZE_EMBEDDINGS}:{setting.VECTOR_METRIC}")

    def _field(self, name: str) -> str:
        """生成固定 metadata JSON 字段的过滤路径。

        Args:
            name: 程序内固定的元数据键。
        Returns:
            JSON 路径。
        """
        return f'metadata["{name}"]'

    def upsert(self, parts: list[dict]) -> None:
        """批量写入带独立编号的新版本，不覆盖已激活块。

        Args:
            parts: 片段列表。
        """
        if not parts or any(not isinstance(p.get("metadata"), dict) for p in parts):
            raise ValueError("写入的 Chunk 非空，且 metadata 必须为字典")
        # 先验证 JSON 可序列化并复制数据；SDK 将字典编码进原生 JSON 字段。
        # 不传 json.dumps 后的字符串，避免 metadata 被二次编码而无法按键过滤。
        metadatas = json.loads(json.dumps([p["metadata"] for p in parts], ensure_ascii=False, allow_nan=False))
        with stage("index_embedding", model_calls=1, chunks=len(parts)):
            self.store.add_texts(texts=[p["text"] for p in parts],
                metadatas=metadatas, ids=[p["id"] for p in parts])

    def delete(self, ids: list[str]) -> None:
        """删除指定片段。

        Args:
            ids: 片段编号。
        """
        if ids and self.store.col is not None:
            if self.store.delete(ids=ids) is False:
                raise RuntimeError("Milvus 删除旧 Chunk 失败，已中止索引，清单未提交")

    def delete_document(self, kb: str, document_id: str) -> None:
        """按文档及知识库清理所有块，包括上次部分写入的块。

        Args:
            kb: 知识库编号。
            document_id: 文档编号。
        """
        if self.store.col is not None:
            expr = f"{self._field('knowledge_base_id')} == {json.dumps(kb)} and {self._field('document_id')} == {json.dumps(document_id)}"
            if self.store.delete(expr=expr) is False:
                raise RuntimeError("Milvus 删除旧文档失败，已中止索引，清单未提交")

    def search_results(self, query: str, kb: str, levels: tuple[str, ...],
                       status: str | None, limit: int, active_ids: list[str] | None = None) -> list[dict]:
        """返回过滤后的内容、分数和来源。

        Args:
            query: 查询。
            kb: 知识库编号。
            levels: 授权等级。
            status: 文档状态。
            limit: 结果数。
            active_ids: 当前清单激活的块编号；None 表示不限制版本。
        Returns:
            带原始相似度的检索结果。
        """
        if not levels or self.store.col is None or active_ids == []:
            return []
        # JSON 转义阻止查询值注入过滤表达式。
        expr = f"{self._field('knowledge_base_id')} == {json.dumps(kb)} and {self._field('access_level')} in {json.dumps(levels)}"
        if status is not None:
            expr += f" and {self._field('status')} == {json.dumps(status, ensure_ascii=False)}"
        if active_ids is not None:
            expr += f" and {self.primary_field} in {json.dumps(active_ids)}"
        # 已有集合可能采用 Bounded；逐次查询显式强一致，立即看见新写入和删除。
        hits = self.store.similarity_search_with_score(query, k=limit, expr=expr, consistency_level="Strong")
        return [{"id": d.metadata["chunk_id"], "text": d.page_content,
                 "score": float(s), "metadata": d.metadata} for d, s in hits]

    def search(self, query: str, kb: str, levels: tuple[str, ...],
               status: str | None, limit: int) -> list[str]:
        """提供混合召回兼容接口。

        Args:
            query: 查询。
            kb: 知识库编号。
            levels: 授权等级。
            status: 文档状态。
            limit: 结果数。
        Returns:
            排序后的片段编号。
        """
        return [r["id"] for r in self.search_results(query, kb, levels, status, limit)]

    def search_active(self, query: str, kb: str, levels: tuple[str, ...],
                      status: str | None, limit: int, active_ids: list[str]) -> list[str]:
        """在 ANN Top K 之前限定激活版本，防止旧块挤占召回名额。

        Args:
            query: 查询。
            kb: 知识库编号。
            levels: 授权等级。
            status: 状态。
            limit: 召回数。
            active_ids: 清单快照中的授权块编号。
        Returns:
            当前版本块编号。
        """
        return [r["id"] for r in self.search_results(query, kb, levels, status, limit, active_ids)]


class Reranker:
    """通过百炼 API 或可选本地 CrossEncoder 做重排。"""

    def __init__(self, model: str) -> None:
        """加载模型。

        Args:
            model: 百炼重排模型，或本地模型路径/名称。
        """
        config = load_config()
        self.name = model
        self.remote = model == "qwen3-rerank" or model.startswith("gte-rerank")
        if self.remote:
            self.key = config.get("RERANK_API_KEY") or config.get("DASHSCOPE_API_KEY", "")
            if not self.key:
                raise ValueError("请在 .env 或环境变量设置 RERANK_API_KEY 或 DASHSCOPE_API_KEY")
            self.url = config.get("RERANK_API_BASE", "https://dashscope.aliyuncs.com/api/v1").rstrip("/") + "/services/rerank/text-rerank/text-rerank"
            self.timeout = setting.RERANK_TIMEOUT_SECONDS
        else:
            from sentence_transformers import CrossEncoder
            self.model: Any = CrossEncoder(model)

    def score(self, query: str, parts: list[dict]) -> list[float]:
        """计算相关性分数。

        Args:
            query: 查询。
            parts: 授权候选片段。
        Returns:
            与候选顺序一致的分数。
        """
        if not parts:
            return []
        if not self.remote:
            return self.model.predict([(query, c["text"]) for c in parts]).tolist()
        with stage("rerank_api", model_calls=0) as event:
            response = httpx.post(self.url, headers={"Authorization": f"Bearer {self.key}"},
                json={"model": self.name, "input": {"query": query, "documents": [c["text"] for c in parts]},
                      "parameters": {"top_n": len(parts), "return_documents": False}}, timeout=self.timeout)
            if response.is_success:
                usage = response.json().get("usage", {})
                event["total_tokens"] = usage.get("total_tokens")
        if not response.is_success:
            raise RuntimeError(f"Rerank API 请求失败，HTTP {response.status_code}")
        # API 按相关性排序返回，必须根据 index 还原为候选原顺序。
        scores: list[float | None] = [None] * len(parts)
        for item in response.json()["output"]["results"]:
            index = item["index"]
            if not isinstance(index, int) or not 0 <= index < len(parts) or scores[index] is not None:
                raise ValueError("Rerank API 返回非法候选编号")
            scores[index] = float(item["relevance_score"])
        if any(s is None for s in scores):
            raise ValueError("Rerank API 未返回全部候选分数")
        return [float(s) for s in scores]

