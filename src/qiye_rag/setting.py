"""RAG 与 Agent 的调优参数；修改后重启程序，不从 .env 读取。

API 密钥、模型名称和服务 URI 仍由 rag.config.load_config 读取。
命令行显式参数可覆盖相应的切块大小和最终返回数量。
"""

# 切块：单位为字符，overlap 必须小于 size；修改后需重新同步索引。
# 九项参数TPE对照后保留原默认，依据见 eval/tuning/参数选择结论.md。
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SPLITTER_MODE = "prefix"  # legacy / section / prefix；实验保留三种策略。
SPLITTER_VERSION = "structure-v1"
CHAT_TOKENIZER_PATH = ".rag/tokenizers/deepseek_v4/deepseek_v4_tokenizer/tokenizer.json"
EVIDENCE_TOKEN_BUDGET = 6000
INPUT_TOKEN_BUDGET = 12000
USE_MMR = False
MMR_LAMBDA = 0.7
CHUNK_SEPARATORS = ("\n\n", "\n", "。", "；", " ", "")
MARKDOWN_HEADERS = (("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4"))

# 检索：每路先召回 RECALL 数量，经融合、重排后保留 TOP_K。
RETRIEVAL_TOP_K = 5
RETRIEVAL_RECALL = 20
MAX_RETRIEVAL_K = 100
RRF_K = 60  # 越大，融合时对头部排名的偏好越弱。
BM25_K1 = 1.5
BM25_B = 0.75
BM25_DELTA = 0.0  # 保持不匹配片段得分为零。
BM25_MIN_SCORE = 0.0
BM25_CACHE_SCOPES = 16

# 模型请求：API 重试次数不等于 Agent 二次检索轮数。
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_API_BATCH_LIMIT = 10  # 百炼兼容接口单次限制。
EMBEDDING_TIMEOUT_SECONDS = 60
EMBEDDING_MAX_RETRIES = 2
NORMALIZE_EMBEDDINGS = True  # 本地 Embedding 使用；云端由提供商处理。
CHAT_TEMPERATURE = 0.0
CHAT_TIMEOUT_SECONDS = 60
CHAT_MAX_RETRIES = 1
CHAT_MAX_TOKENS: int | None = 2000
RERANK_TIMEOUT_SECONDS = 30

# Agent 执行预算：限制调用成本与无效循环。
MAX_RETRIEVAL_ROUNDS = 2
MAX_SUBQUERIES = 3
MAX_EVIDENCE_CHUNKS = 30
HISTORY_TURNS = 6
MAX_QUERY_CHARS = 1000
MAX_QUESTION_CHARS = 2000
MAX_ANSWER_CHARS = 6000
MAX_REVIEW_REASON_CHARS = 1000
GRAPH_RECURSION_LIMIT = max(20, MAX_RETRIEVAL_ROUNDS * 3 + 5)

# 本地运行参数；知识权限属于调用方授权，不在这里配置。
MILVUS_COLLECTION = "qiye_rag_chunks"
MILVUS_TIMEOUT_SECONDS = 10
VECTOR_METRIC = "COSINE"
LOG_LEVEL = "INFO"

# v3 本地 Web 和异步任务；不涉及认证或热点缓存。
WEB_HOST = "127.0.0.1"
WEB_PORT = 8000
WEB_DATA_DIR = ".rag/web"
UPLOAD_MAX_BYTES = 10 * 1024 * 1024
UPLOAD_READ_BYTES = 64 * 1024
WEB_LOCK_TIMEOUT_SECONDS = 120
CELERY_QUEUE_NAME = "qiye_rag"
CELERY_RESULT_EXPIRES_SECONDS = 24 * 60 * 60
REDIS_TIMEOUT_SECONDS = 5
LOGIN_TOKEN_TTL_SECONDS = 8 * 60 * 60
BATCH_UPLOAD_MAX_FILES = 100


def validate() -> None:
    """检查调优参数之间的基本约束，防止无效预算和分块配置。"""
    positive = (CHUNK_SIZE, RETRIEVAL_TOP_K, RETRIEVAL_RECALL, MAX_RETRIEVAL_K,
                RRF_K, BM25_K1, BM25_CACHE_SCOPES, EMBEDDING_BATCH_SIZE, EMBEDDING_API_BATCH_LIMIT,
                EMBEDDING_TIMEOUT_SECONDS, CHAT_TIMEOUT_SECONDS, RERANK_TIMEOUT_SECONDS,
                MAX_RETRIEVAL_ROUNDS, MAX_SUBQUERIES, MAX_EVIDENCE_CHUNKS, HISTORY_TURNS,
                MAX_QUERY_CHARS, MAX_QUESTION_CHARS, MAX_ANSWER_CHARS, MAX_REVIEW_REASON_CHARS)
    if any(value <= 0 for value in positive):
        raise ValueError("setting.py 中的数量、长度、超时和预算必须为正数")
    if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
        raise ValueError("setting.py 要求 0 <= CHUNK_OVERLAP < CHUNK_SIZE")
    if max(RETRIEVAL_TOP_K, RETRIEVAL_RECALL) > MAX_RETRIEVAL_K:
        raise ValueError("召回数量不能超过 MAX_RETRIEVAL_K")
    if not 0 <= BM25_B <= 1 or BM25_DELTA < 0 or BM25_MIN_SCORE < 0:
        raise ValueError("BM25_B 须在 0–1 之间，DELTA 和 MIN_SCORE 不能为负")
    if min(EMBEDDING_MAX_RETRIES, CHAT_MAX_RETRIES) < 0 or not 0 <= CHAT_TEMPERATURE <= 2:
        raise ValueError("模型重试次数不能为负，温度须在 0–2 之间")
    if CHAT_MAX_TOKENS is not None and CHAT_MAX_TOKENS <= 0:
        raise ValueError("CHAT_MAX_TOKENS 须为正数或 None")
    if GRAPH_RECURSION_LIMIT < MAX_RETRIEVAL_ROUNDS * 3 + 5:
        raise ValueError("GRAPH_RECURSION_LIMIT 不足以支持配置的检索轮数")
    if not 0 < EVIDENCE_TOKEN_BUDGET < INPUT_TOKEN_BUDGET or not 0 <= MMR_LAMBDA <= 1:
        raise ValueError("上下文预算或 MMR 参数无效")


validate()
