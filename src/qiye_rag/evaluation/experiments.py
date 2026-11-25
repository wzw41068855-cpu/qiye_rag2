"""my_plan 的可复现真实服务实验；每个方案独立结果与 Gold。"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from qiye_rag.evaluation.dataset import corpus
from qiye_rag.evaluation.runner import run
from qiye_rag.rag.config import PROJECT_ROOT


def remap_gold(original: dict, root: Path, split_mode: str) -> dict:
    """按冻结的原文事实锚点映射新切分，不从检索结果反推 Gold。

    Args:
        original: 原版金标准。
        root: 资料路径。
        split_mode: 切分方式。
    Returns:
        独立的新切分金标准。
    """
    parts, digest = corpus(root, split_mode)
    data = deepcopy(original)
    data.update(corpus_sha256=digest, splitter_mode=split_mode)
    for case in data["cases"]:
        for fact in case["expected_citations"]:
            matches = [p for p in parts if p["metadata"]["document_id"] == fact["document_id"]
                       and "".join(fact["fact"].split()) in "".join(p["text"].split())]
            if not matches:
                raise ValueError(f"原文事实在新切分中丢失：{case['id']}")
            fact.update(chunk_id=matches[0]["id"], location=matches[0]["metadata"]["location"])
        case["gold_chunk_ids"] = sorted({f["chunk_id"] for f in case["expected_citations"]})
    data["chunk_statistics"] = {"count": len(parts), "characters": sum(len(p["text"]) for p in parts)}
    return data


def main() -> None:
    """按指定切分与检索策略执行三十题真实问答。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("legacy", "section", "prefix"), required=True)
    parser.add_argument("--retrieval", choices=("legacy", "global"), default="legacy")
    parser.add_argument("--mmr", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mmr and args.retrieval != "global":
        parser.error("MMR 对照需要固定为 global 检索")
    root = PROJECT_ROOT / "data/远澜外贸公司_RAG知识库"
    original = json.loads((PROJECT_ROOT / "eval/enterprise_30.json").read_text(encoding="utf-8"))
    gold = remap_gold(original, root, args.split)
    result = run(root, gold, "agent", args.output, args.split, args.retrieval, args.mmr)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
