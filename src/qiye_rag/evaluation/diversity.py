"""事后用相同 Embedding 比较选中证据的语义冗余；开销单独记录。"""
import argparse
import json
from pathlib import Path
from statistics import mean
import httpx
import numpy as np
from langchain_openai import OpenAIEmbeddings
from qiye_rag import setting
from qiye_rag.rag.config import load_config
from qiye_rag.telemetry import capture, embedding_response


def evaluate(paths: list[Path], target: Path) -> dict:
    """按唯一证据文本复用向量，比较每题平均余弦相似度。

    Args:
        paths: 同一切分的对照组路径。
        target: 输出 JSON，不修改问答原始结果。
    Returns:
        逐题相似度、汇总和本次评价开销。
    """
    runs = {p.name: [json.loads(line) for line in (p / "results.jsonl").read_text(encoding="utf-8").splitlines()] for p in paths}
    texts = sorted({h["text"] for rows in runs.values() for r in rows for h in r.get("evidence_texts", [])})
    config = load_config()
    with capture() as events, httpx.Client(event_hooks={"response": [embedding_response]}) as client:
        model = OpenAIEmbeddings(model=config.get("EMBEDDING_MODEL", "text-embedding-v3"),
            api_key=config["DASHSCOPE_API_KEY"], base_url=config.get("DASHSCOPE_BASE_URL"),
            check_embedding_ctx_length=False, chunk_size=setting.EMBEDDING_API_BATCH_LIMIT,
            timeout=setting.EMBEDDING_TIMEOUT_SECONDS, max_retries=setting.EMBEDDING_MAX_RETRIES,
            http_client=client)
        vectors = model.embed_documents(texts)
    by_text = dict(zip(texts, vectors))
    result = {"runs": {}, "evaluation_events": events,
              "unique_texts": len(texts), "evaluation_tokens": sum(e.get("total_tokens") or 0 for e in events)}
    for name, rows in runs.items():
        values = []
        per_question = []
        for row in rows:
            selected = row.get("evidence_texts", [])
            similarity = None
            if len(selected) > 1:
                matrix = np.array([by_text[h["text"]] for h in selected])
                matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
                cosine = matrix @ matrix.T
                similarity = float(cosine[np.triu_indices(len(selected), 1)].mean())
                values.append(similarity)
            per_question.append({"id": row["id"], "chunks": len(selected), "mean_pair_cosine": similarity})
        result["runs"][name] = {"mean_pair_cosine": mean(values) if values else None, "questions": per_question}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.paths, args.output), ensure_ascii=False, indent=2))
