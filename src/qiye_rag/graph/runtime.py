"""使用 .env 的 DeepSeek、v1 检索及 SQLite Checkpointer 组装运行时。"""
from qiye_rag import setting
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from qiye_rag.rag.catalog import Catalog
from qiye_rag.rag.config import load_config
from qiye_rag.rag.retrieval import search
from qiye_rag.rag.vectors import MilvusVectors, Reranker
from qiye_rag.graph.workflow import Conversation, build_graph


def create_chat(config: dict[str, str]) -> ChatOpenAI:
    """创建 DeepSeek 兼容聊天模型，不输出密钥。

    Args:
        config: .env 优先的配置。
    Returns:
        同步聊天模型。
    """
    key = config.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise ValueError("请在 .env 或环境变量中设置 DEEPSEEK_API_KEY")
    return ChatOpenAI(model=config.get("CHAT_MODEL", "deepseek-v4-flash"), api_key=key,
        base_url=config.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=setting.CHAT_TEMPERATURE, timeout=setting.CHAT_TIMEOUT_SECONDS,
        max_retries=setting.CHAT_MAX_RETRIES, max_tokens=setting.CHAT_MAX_TOKENS)


@contextmanager#也就是“进入时创建资源、退出时回收资源”的模式。
def open_conversation(catalog_path: Path, checkpoint_path: Path, kb: str,
                      levels: tuple[str, ...] = ("public", "internal")) -> Iterator[Conversation]:
    """组装固定授权范围的会话运行时，并在退出时释放本地资源。

    执行流程：
    1. 读取配置并校验检索参数。
    2. 校验清单文件存在且包含目标知识库。
    3. 初始化聊天模型、向量检索和重排器。
    4. 构造带权限约束的 `retrieve` 闭包。
    5. 打开 SQLite checkpoint，产出可复用的 Conversation。
    6. 无论成功失败，最终关闭 Catalog 句柄。

    Args:
        catalog_path: v1 成功索引清单（JSON）。
        checkpoint_path: SQLite 会话 checkpoint 文件路径。
        kb: 本次会话绑定的知识库编号。
        levels: 调用方授予的访问等级；属于可信输入，不应由模型改写。

    Yields:
        可重复问答的会话对象（图执行器 + 作用域信息）。

    Raises:
        ValueError: 配置非法、清单缺失、或清单中不存在目标知识库时抛出。
    """
    # 读取 .env/环境变量配置（供后续组件统一使用）
    config = load_config()

    # 校验 top_k，避免检索召回数量越界
    top_k = setting.RETRIEVAL_TOP_K
    if not 1 <= top_k <= setting.MAX_RETRIEVAL_K:
        raise ValueError(f"RETRIEVAL_TOP_K 须在 1–{setting.MAX_RETRIEVAL_K} 之间")

    # 清单文件是“激活版本视图”的来源；仅有 Milvus 集合不足以安全运行
    if not catalog_path.is_file():
        raise ValueError(f"本地索引清单不存在: {catalog_path.resolve()}。"
                         "已有向量集合仍需配套 JSON 清单；请指定 --catalog 和 --kb。")

    # 打开 Catalog，生命周期覆盖整个 context
    catalog = Catalog(catalog_path)
    try:
        #这里强调了一个关键设计：即使 Milvus 里有向量，也必须有配套 catalog 清单才能安全运行（因为激活版本与授权视图依赖清单）。
        # 目标知识库必须在当前清单中存在，避免出现“看起来可连库但实际无可用数据”的误判
        if not catalog.records(kb):
            available = ", ".join(catalog.state) or "无"
            raise ValueError(f"清单 {catalog_path.resolve()} 中没有知识库 {kb!r}；"
                             f"清单内知识库: {available}。请使用配套的 --kb 和 --catalog，"
                             "此错误不表示 Milvus 向量集合不存在。")

        # 构建模型与检索组件
        chat = create_chat(config)
        vectors = MilvusVectors(config.get("MILVUS_URI", "http://localhost:19530"),
            config.get("EMBEDDING_MODEL", "text-embedding-v3"),
            setting.MILVUS_COLLECTION, config.get("MILVUS_TOKEN", ""))
        reranker = Reranker(config.get("RERANK_MODEL", "qwen3-rerank"))
        #内层 retrieve(query) 闭包。它把 kb、levels、top_k、catalog、vectors、reranker 绑定在当前上下文里：
        def retrieve(query: str) -> list[dict]:
            """在固定授权范围内执行混合检索。

            Args:
                query: 模型生成的检索查询词。
            Returns:
                仅包含当前 KB、当前权限等级、当前激活版本的数据证据。
            """
            # 将 kb/levels/top_k 固定在闭包中，防止运行时被模型侧篡改
            return search(query, kb, catalog, vectors, reranker, levels=levels, top_k=top_k)

        # 确保 checkpoint 目录存在，再打开 SQLite 持久化器
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            # 作用域指纹：用于追踪本次会话绑定的数据与检索上下文
            scope = {"kb": kb, "levels": sorted(set(levels)), "status": "现行",
                     "catalog": str(catalog_path.resolve()), "vectors": vectors.signature}
            yield Conversation(build_graph(chat, retrieve, saver), scope)
    finally:
        # 即使中途异常，也保证关闭 Catalog 相关资源
        catalog.close()
