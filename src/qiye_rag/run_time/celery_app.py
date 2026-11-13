"""使用 REDIS_URI 作为 Celery broker 和任务结果后端。"""
from celery import Celery
from urllib.parse import urlsplit
from qiye_rag import setting
from qiye_rag.rag.config import load_config


def create_celery() -> Celery:
    """统一读取 Redis 地址，不在日志中输出连接凭据。

    Returns:
        可被 Web 和独立 worker 共用的 Celery 应用。
    """
    config = load_config()
    uri = config.get("REDIS_URI", "redis://localhost:6379/0")
    if urlsplit(uri).scheme not in {"redis", "rediss"}:
        raise ValueError("REDIS_URI 必须是 redis:// 或 rediss:// 地址")
    app = Celery("qiye_rag", broker=uri, backend=uri,
                 include=["qiye_rag.run_time.tasks"])#include 让 worker 能找到任务函数
    app.conf.update(task_track_started=True, #任务会有 STARTED 状态（不只是 PENDING/SUCCESS/FAILURE）
        task_serializer="json", result_serializer="json",accept_content=["json"],#并且 worker 只接受 JSON 内容，避免 pickle 这类更高风险的反序列化载荷
        worker_prefetch_multiplier=1,# 每个 worker 一次只取一个任务，避免任务堆积
        task_default_queue=setting.CELERY_QUEUE_NAME,
        result_expires=setting.CELERY_RESULT_EXPIRES_SECONDS,#它决定任务结果在 backend（这里是 Redis）里保留多久。时间太短会让前端轮询不到历史结果，太长会让 Redis 堆积无用状态。这个参数本质是“可追溯性 vs 存储成本”的平衡点。
        #设置了 Redis key 前缀隔离：以 qiye_rag:broker: 开头的 key 是 broker 队列，qiye_rag:result: 开头的 key 是任务结果。避免和其它 Celery 项目冲突。
        broker_transport_options={"global_keyprefix": "qiye_rag:broker:",
            #连接与读写超时
            "socket_connect_timeout": setting.REDIS_TIMEOUT_SECONDS,
            "socket_timeout": setting.REDIS_TIMEOUT_SECONDS},
        result_backend_transport_options={"global_keyprefix": "qiye_rag:result:"},
        redis_socket_connect_timeout=setting.REDIS_TIMEOUT_SECONDS,
        redis_socket_timeout=setting.REDIS_TIMEOUT_SECONDS)
    return app


celery_app = create_celery()
