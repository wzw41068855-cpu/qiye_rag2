"""v1 同步及检索命令行。"""
from qiye_rag import setting
import argparse
import json
import logging
from pathlib import Path
from qiye_rag.rag.config import load_config
from qiye_rag.rag.catalog import Catalog
from qiye_rag.rag.indexing import sync
from qiye_rag.rag.retrieval import search
from qiye_rag.rag.vectors import MilvusVectors, Reranker
from qiye_rag.rag.loader import load_directory
from qiye_rag.rag.splitter import split_documents


def main() -> None:
    """解析参数并执行知识库同步或带引用的检索。"""
    config = load_config()
    logging.basicConfig(level=setting.LOG_LEVEL)
    parser = argparse.ArgumentParser(description="企业 RAG v1")
    parser.add_argument("--kb", default="demo")
    parser.add_argument("--catalog", type=Path, default=Path(".rag/state.json"))
    parser.add_argument("--uri", default=config.get("MILVUS_URI", "http://localhost:19530"))
    parser.add_argument("--collection", default=setting.MILVUS_COLLECTION)
    parser.add_argument("--embedding", default=config.get("EMBEDDING_MODEL", "text-embedding-v3"))
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("sync")
    ingest.add_argument("root", type=Path)
    ingest.add_argument("--chunk-size", type=int, default=setting.CHUNK_SIZE)
    ingest.add_argument("--chunk-overlap", type=int, default=setting.CHUNK_OVERLAP)
    inspect = sub.add_parser("inspect", help="预览 Chunk，无需模型或 Milvus")
    inspect.add_argument("root", type=Path)
    inspect.add_argument("--chunk-size", type=int, default=setting.CHUNK_SIZE)
    inspect.add_argument("--chunk-overlap", type=int, default=setting.CHUNK_OVERLAP)
    ask = sub.add_parser("search")
    ask.add_argument("query")
    ask.add_argument("--top-k", type=int, default=setting.RETRIEVAL_TOP_K)
    ask.add_argument("--restricted", action="store_true", help="模拟获授权用户，仅限本地学习")
    ask.add_argument("--all-versions", action="store_true")
    ask.add_argument("--reranker", default=config.get("RERANK_MODEL", "qwen3-rerank"))
    args = parser.parse_args()
    if args.command == "search" and not 1 <= args.top_k <= setting.MAX_RETRIEVAL_K:
        parser.error("top-k 必须在 1 到 100 之间")
    if args.command == "inspect":
        docs, errors = load_directory(args.root, args.kb)
        parts = split_documents(docs, args.chunk_size, args.chunk_overlap)
        print(json.dumps({"chunks": [{"summary": d.page_content[:100], **d.metadata}
                                      for d in parts], "errors": errors},
                         ensure_ascii=False, indent=2, default=str))
        return
    catalog = Catalog(args.catalog)
    try:
        vectors = MilvusVectors(args.uri, args.embedding, args.collection,
                                config.get("MILVUS_TOKEN", ""))
        if args.command == "sync":
            result = sync(args.root, args.kb, catalog, vectors, args.chunk_size, args.chunk_overlap)
        else:
            levels = ("public", "internal", "restricted") if args.restricted else ("public", "internal")
            result = search(args.query, args.kb, catalog, vectors, Reranker(args.reranker),
                            levels, None if args.all_versions else "现行", args.top_k)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        catalog.close()


if __name__ == "__main__":
    main()
