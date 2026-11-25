"""真实服务参数实验，复用内容相同的索引并记录全部开销。"""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from time import perf_counter
from uuid import uuid4
from qiye_rag import setting
from qiye_rag.evaluation.experiments import remap_gold
from qiye_rag.evaluation.tuning_data import DEV_IDS, history_probes, parameters
from qiye_rag.graph.runtime import create_chat
from qiye_rag.graph.workflow import Conversation, build_graph
from qiye_rag.rag.catalog import Catalog
from qiye_rag.rag.config import PROJECT_ROOT, load_config
from qiye_rag.rag.indexing import sync
from qiye_rag.rag.multi_retrieval import HybridRetriever
from qiye_rag.rag.vectors import MilvusVectors, Reranker
from qiye_rag.telemetry import capture, write_result
from qiye_rag.tokens import count_json, select_evidence


def api_tokens(events: list[dict]) -> int:
    """仅累计API叶子事件的真实Token。

    Args:
        events: 阶段事件。
    Returns:
        已知实际消耗，不重复计算父事件。
    """
    return sum(e.get("total_tokens") or ((e.get("input_tokens") or 0) + (e.get("output_tokens") or 0))
               for e in events if e["stage"] in ("embedding_http", "rerank_api", "plan", "research", "verdict"))


class TuningEngine:
    """单进程顺序试验引擎，不向生产集合写入。"""

    def __init__(self, output: Path) -> None:
        """准备固定数据、服务与可追踪缓存。

        Args:
            output: 实验根目录。
        """
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.root = PROJECT_ROOT / "data/远澜外贸公司_RAG知识库"
        self.original = json.loads((PROJECT_ROOT / "eval/enterprise_30.json").read_text(encoding="utf-8"))
        self.config = load_config()
        self.chat = create_chat(self.config)
        self.reranker = Reranker(self.config.get("RERANK_MODEL", "qwen3-rerank"))
        self.indices: dict[str, tuple] = {}
        self.memo: dict[str, dict] = {}
        self.serial = 0
        metadata = {"packages": {n: version(n) for n in ("optuna", "numpy", "sqlalchemy", "langchain-openai", "langgraph")},
                    "models": {k: self.config.get(k) for k in ("CHAT_MODEL", "EMBEDDING_MODEL", "RERANK_MODEL")},
                    "source_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in sorted((PROJECT_ROOT / "src").rglob("*.py")))).hexdigest(),
                    "splitter": "prefix", "mmr": False, "evidence_token_budget": setting.EVIDENCE_TOKEN_BUDGET,
                    "input_token_budget": setting.INPUT_TOKEN_BUDGET, "chat_max_tokens": setting.CHAT_MAX_TOKENS}
        (output / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def index(self) -> tuple:
        """取得与本次实际切块内容一致的独立索引。

        Returns:
            当前Gold、清单及向量适配器。
        """
        gold = remap_gold(self.original, self.root, "prefix")
        digest = gold["corpus_sha256"]
        if digest not in self.indices:
            folder = self.output / "indices" / digest[:16]
            folder.mkdir(parents=True, exist_ok=True)
            name = "qiye_tune_" + uuid4().hex
            write_result(self.output / "collections.jsonl", {"collection": name, "digest": digest})
            catalog = Catalog(folder / "state.json")
            vectors = None
            try:
                with capture() as events:
                    vectors = MilvusVectors(self.config.get("MILVUS_URI", "http://localhost:19530"),
                        self.config.get("EMBEDDING_MODEL", "text-embedding-v3"), name, self.config.get("MILVUS_TOKEN", ""))
                    self.indices[digest] = (catalog, vectors)
                    sync(self.root, "eval_v5", catalog, vectors, size=setting.CHUNK_SIZE, overlap=setting.CHUNK_OVERLAP)
                write_result(self.output / "setup.jsonl", {"collection": name, "digest": digest,
                    "size": setting.CHUNK_SIZE, "overlap": setting.CHUNK_OVERLAP, "tokens": api_tokens(events), "events": events})
            except Exception:
                if vectors is not None and vectors.store.col is not None:
                    vectors.store.col.drop()
                raise
        catalog, vectors = self.indices[digest]
        return gold, catalog, vectors

    def evaluate(self, values: dict, mode: str, label: str, final: bool = False) -> dict:
        """测量一个参数组合，记录真实Token及明确命名的筛选代理。

        Args:
            values: 完整参数。
            mode: retrieval或agent。
            label: 结果标识。
            final: 是否强制运行原30题及保留探针，不复用试验成绩。
        Returns:
            双目标指标与原始结果路径。
        """
        with parameters(values):
            gold, catalog, vectors = self.index()
            cases = gold["cases"] + history_probes(gold["cases"], validation=True) if final else [
                c for c in gold["cases"] if c["id"] in DEV_IDS] + history_probes(gold["cases"])
            effective = {**values, "MAX_RETRIEVAL_K": 100,
                         "MAX_EVIDENCE_CHUNKS": min(values["MAX_EVIDENCE_CHUNKS"], values["RETRIEVAL_TOP_K"])}
            effective.pop("CHUNK_SIZE")
            effective.pop("CHUNK_OVERLAP")
            key = json.dumps([gold["corpus_sha256"], effective, mode], sort_keys=True)
            if not final and key in self.memo:
                return {**self.memo[key], "cache_hit": True}
            self.serial += 1
            folder = self.output / "measurements" / f"{self.serial:03}_{label}"
            folder.mkdir(parents=True, exist_ok=False)
            (folder / "config.json").write_text(json.dumps(values, indent=2), encoding="utf-8")
            (folder / "gold.json").write_text(json.dumps({**gold, "cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
            rows = []
            for case in cases:
                started = perf_counter()
                levels = ("public", "internal", "restricted") if case["role"] == "hr" else ("public", "internal")
                retriever = HybridRetriever("eval_v5", catalog, vectors, self.reranker, levels, mmr=False)
                row = {"id": case["id"], "category": case["category"]}
                with capture() as events:
                    if mode == "agent":
                        conversation = Conversation(build_graph(self.chat, retriever), {"kb": "eval_v5", "levels": levels})
                        inputs, execution = conversation.prepare(case["question"], case["id"])
                        inputs["history"] = case["history"]
                        state = conversation.graph.invoke(inputs, execution)
                        hits = state["evidence"]
                        row.update(answer=state["answer"], approved=state["approved"], reason=state["reason"], citations=state["citations"])
                    else:
                        query = " ".join(h["question"] for h in case["history"]) + " " + case["question"]
                        hits = select_evidence(retriever(query.strip()))
                    facts = case["expected_citations"]
                    row["coverage"] = mean(any(h["citation"]["document_id"] == f["document_id"] and
                        "".join(f["fact"].split()) in "".join(h["text"].split()) for h in hits) for f in facts) if facts else None
                    if any(h["citation"]["access_level"] not in levels for h in hits):
                        raise RuntimeError("实验发现未授权证据")
                    payload = [{"number": i, "text": h["text"], "source": h["citation"]} for i, h in enumerate(hits, 1)]
                    row["evidence_tokens"] = count_json(payload)
                    row["evidence"] = hits
                row.update(api_tokens=api_tokens(events), events=events, elapsed_ms=(perf_counter() - started) * 1000)
                row["usage_complete"] = all(e.get("input_tokens") is not None or e.get("total_tokens") is not None
                    for e in events if e["stage"] in ("embedding_http", "rerank_api", "plan", "research", "verdict"))
                if not row["usage_complete"]:
                    raise RuntimeError("真实API usage缺失，不能作为完整Token目标")
                rows.append(row)
                write_result(folder / "results.jsonl", row)
            coverage = mean(r["coverage"] for r in rows if r["coverage"] is not None)
            actual = sum(r["api_tokens"] for r in rows)
            objective_tokens = actual if mode == "agent" else actual + 2 * sum(r["evidence_tokens"] for r in rows)
            result = {"coverage": coverage, "tokens": objective_tokens, "actual_api_tokens": actual,
                      "token_kind": "actual_api" if mode == "agent" else "retrieval_api_plus_two_context_estimates",
                      "questions": len(rows), "path": str(folder), "cache_hit": False, "corpus_sha256": gold["corpus_sha256"]}
            (folder / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            self.memo[key] = result
            return result

    def close(self) -> None:
        """只清理本引擎创建的随机集合和持有的客户端。"""
        for catalog, vectors in self.indices.values():
            try:
                if vectors.store.col is not None:
                    vectors.store.col.drop()
            finally:
                catalog.close()
                vectors.close()
