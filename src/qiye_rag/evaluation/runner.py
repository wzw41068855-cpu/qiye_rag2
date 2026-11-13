"""运行离线 BM25 基线或隔离集合中的真实服务评测。"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from importlib.metadata import version
from tempfile import TemporaryDirectory
from uuid import uuid4
from statistics import mean
from qiye_rag import setting
from qiye_rag.evaluation.dataset import KB, build, corpus
from qiye_rag.graph.runtime import create_chat
from qiye_rag.graph.workflow import Conversation, build_graph
from qiye_rag.rag.catalog import Catalog
from qiye_rag.rag.config import PROJECT_ROOT, load_config
from qiye_rag.rag.indexing import sync
from qiye_rag.rag.retrieval import search
from qiye_rag.rag.vectors import MilvusVectors, Reranker
from qiye_rag.telemetry import capture, write_result


class NoDense:
    """离线实验明确禁用向量召回，不模拟语义质量。"""

    def search(self, *args: object) -> list[str]:
        """返回空向量候选。

        Args:
            args: 兼容检索调用的参数。
        Returns:
            空列表。
        """
        return []


class KeepOrder:
    """离线实验保持融合排序，不伪造模型重排。"""

    def score(self, query: str, parts: list[dict]) -> list[float]:
        """生成仅用于保持已有顺序的分数。

        Args:
            query: 查询。
            parts: 已排序候选。
        Returns:
            单调递减的顺序分数。
        """
        return [float(len(parts) - i) for i in range(len(parts))]


def metrics(case: dict, hits: list[dict], answer: dict | None = None) -> dict:
    """计算检索指标，拒答题不混入召回率分母。

    Args:
        case: 单题金标准。
        hits: 检索证据。
        answer: Agent 最终状态，离线时为 None。
    Returns:
        指标字典，未测量值为 None。
    """
    documents = [h["citation"]["document_id"] for h in hits]
    chunks = {h["citation"].get("content_id", h["citation"]["chunk_id"]) for h in hits}
    gold_docs, gold_chunks = set(case["gold_document_ids"]), set(case["gold_chunk_ids"])
    output = {"document_recall": len(gold_docs & set(documents)) / len(gold_docs) if gold_docs else None,
              "chunk_recall": len(gold_chunks & chunks) / len(gold_chunks) if gold_chunks else None,
              "all_documents": gold_docs <= set(documents) if gold_docs else None,
              "mrr": next((1 / (i + 1) for i, d in enumerate(documents) if d in gold_docs), 0) if gold_docs else None,
              "forbidden_hits": len(set(documents) & set(case["forbidden_document_ids"])),
              "refusal_correct": None, "citation_document_recall": None,
              "fact_coverage_manual": None, "citation_support_manual": None}
    if answer is not None:
        output["refusal_correct"] = bool(answer["approved"]) == case["answerable"]
        cited = {c["document_id"] for c in answer["citations"]}
        output["citation_document_recall"] = len(cited & gold_docs) / len(gold_docs) if gold_docs else None
    return output


def summarize(rows: list[dict]) -> dict:
    """按类别汇总成功题和失败题，不静默丢弃错误。

    Args:
        rows: 逐题实验结果。
    Returns:
        整体与分类统计。
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups["overall"].append(row)
        groups[row["category"]].append(row)
    result = {}
    for category, cases in groups.items():
        values = {"count": len(cases), "errors": sum("error" in c for c in cases)}
        for name in ("document_recall", "chunk_recall", "all_documents", "mrr", "refusal_correct", "citation_document_recall"):
            measured = [c["metrics"][name] for c in cases if c.get("metrics", {}).get(name) is not None]
            values[name] = mean(measured) if measured else None
        values["forbidden_hits"] = sum(c.get("metrics", {}).get("forbidden_hits", 0) for c in cases)
        result[category] = values
    return result


def run(root: Path, dataset: dict, mode: str, output: Path) -> dict:
    """执行全部题目并记录阶段日志；真实模式只操作随机临时集合。

    Args:
        root: 企业资料目录。
        dataset: 已校验金标准。
        mode: bm25、hybrid 或 agent。
        output: 新建运行目录。
    Returns:
        汇总结果。
    """
    parts, digest = corpus(root)
    if digest != dataset["corpus_sha256"]:
        raise ValueError("资料或切块配置已变化，请人工复核并重建金标准")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"mode": mode, "corpus_sha256": digest, "python": platform.python_version(),
                "packages": {name: version(name) for name in ("langchain-openai", "langgraph", "jieba", "rank-bm25")},
                "settings": {name: getattr(setting, name) for name in ("RETRIEVAL_TOP_K", "RETRIEVAL_RECALL", "RRF_K", "MAX_RETRIEVAL_ROUNDS", "MAX_SUBQUERIES")}}
    rows = []
    vectors, reranker = NoDense(), KeepOrder()
    config = load_config() if mode != "bm25" else {}
    metadata["models"] = {name: config.get(name) for name in ("CHAT_MODEL", "EMBEDDING_MODEL", "RERANK_MODEL")}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    chat = create_chat(config) if mode == "agent" else None
    with TemporaryDirectory(prefix="eval-", dir=output) as temporary:
        catalog = Catalog(Path(temporary) / "state.json")
        setup_events: list[dict] = []
        try:
            if mode == "bm25":
                # 仅在内存构建真实切块视图，不借助向量替身同步生产集合。
                catalog.state = {KB: {str(i): {"digest": "offline", "chunks": [p]} for i, p in enumerate(parts)}}
            else:
                collection = "qiye_eval_" + uuid4().hex
                with capture() as setup_events:
                    vectors = MilvusVectors(config.get("MILVUS_URI", "http://localhost:19530"),
                        config.get("EMBEDDING_MODEL", "text-embedding-v3"), collection, config.get("MILVUS_TOKEN", ""))
                    reranker = Reranker(config.get("RERANK_MODEL", "qwen3-rerank"))
                    sync(root, KB, catalog, vectors)
            for case in dataset["cases"]:
                row = {"id": case["id"], "category": case["category"], "mode": mode}
                levels = ("public", "internal", "restricted") if case["role"] == "hr" else ("public", "internal")
                with capture() as events:
                    try:
                        def retrieve(query: str) -> list[dict]:
                            """按本题可信角色检索。

                            Args:
                                query: 查询文本。
                            Returns:
                                授权范围内的证据。
                            """
                            return search(query, KB, catalog, vectors, reranker, levels=levels)
                        state = None
                        if mode == "agent":
                            conversation = Conversation(build_graph(chat, retrieve), {"kb": KB, "levels": levels})
                            inputs, execution = conversation.prepare(case["question"], case["id"])
                            inputs["history"] = case["history"]
                            state = conversation.graph.invoke(inputs, execution)
                            hits = state["evidence"]
                            row.update(answer=state["answer"], citations=state["citations"], approved=state["approved"], reason=state["reason"])
                        else:
                            # 检索模式不评价指代消解：明示采用历史问题拼接，不使用 Gold 改写查询。
                            query = " ".join(h["question"] for h in case["history"]) + " " + case["question"]
                            hits = retrieve(query.strip())
                        row["metrics"] = metrics(case, hits, state)
                        row["results"] = [{"chunk_id": h["citation"]["chunk_id"],
                                           "content_id": h["citation"].get("content_id", h["citation"]["chunk_id"]),
                                           "document_id": h["citation"]["document_id"], "score": h["score"]} for h in hits]
                    except Exception as exc:
                        row["error"] = type(exc).__name__
                if mode == "bm25":
                    for event in events:
                        event["model_calls"] = 0
                        if event["stage"] in ("dense", "rerank"):
                            event["disabled"] = True
                row["events"] = events
                by_id = {p["id"]: p for _, cs in catalog.records(KB).values() for p in cs}
                seen = {i for e in events for i in e.get("chunk_ids", [])}
                seen.update(r["chunk_id"] for e in events for r in e.get("results", []))
                row["unauthorized_stage_hits"] = sum(by_id[i]["metadata"]["access_level"] not in levels for i in seen if i in by_id)
                row["logical_model_calls"] = sum(e["model_calls"] for e in events)
                row["known_input_tokens"] = sum(e["input_tokens"] or 0 for e in events)
                row["known_output_tokens"] = sum(e["output_tokens"] or 0 for e in events)
                row["known_rerank_tokens"] = sum(e.get("total_tokens") or 0 for e in events)
                row["token_usage_complete"] = all(e["input_tokens"] is not None for e in events if e["model_calls"])
                rows.append(row)
                write_result(output / "results.jsonl", row)
        finally:
            if mode != "bm25":
                write_result(output / "setup.jsonl", {"events": setup_events})
            catalog.close()
            if isinstance(vectors, MilvusVectors) and vectors.store.col is not None:
                vectors.store.col.drop()  # 仅删除本函数生成的随机评测集合。
    summary = {"mode": mode, "corpus_sha256": digest, "python": platform.python_version(),
               "created_at": datetime.now(timezone.utc).isoformat(), "top_k": setting.RETRIEVAL_TOP_K,
               "recall": setting.RETRIEVAL_RECALL, "statistics": summarize(rows)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """解析评测参数，默认只运行不联网的 BM25 基线。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bm25", "hybrid", "agent"), default="bm25")
    parser.add_argument("--build", action="store_true", help="根据人工事实锚点重建 Gold；数据变动后需人工复核")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ids", nargs="+", help="仅复测指定题号，例如 Q16 Q18")
    args = parser.parse_args()
    root = PROJECT_ROOT / "data/远澜外贸公司_RAG知识库"
    target = PROJECT_ROOT / "eval/enterprise_30.json"
    dataset = build(root, target) if args.build else json.loads(target.read_text(encoding="utf-8"))
    if args.ids:
        if set(args.ids) - {c["id"] for c in dataset["cases"]}:
            parser.error("存在无效题号")
        dataset["cases"] = [c for c in dataset["cases"] if c["id"] in args.ids]
    output = args.output or PROJECT_ROOT / "eval/runs" / (args.mode + "_" + uuid4().hex[:8])
    print(json.dumps(run(root, dataset, args.mode, output), ensure_ascii=False, indent=2))
