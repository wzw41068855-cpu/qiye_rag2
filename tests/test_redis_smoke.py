"""显式启用的真实 Redis 队列测试，只发送合成字符串。"""
import os
from uuid import uuid4
from celery.contrib.testing.worker import start_worker
import pytest
from redis import Redis
from qiye_rag.run_time.celery_app import create_celery


@pytest.mark.skipif(os.getenv("RAG_REDIS_SMOKE") != "1", reason="设置 RAG_REDIS_SMOKE=1 验证本地 Redis")
def test_redis_worker_roundtrip() -> None:
    """连接 REDIS_URI，用独立队列验证真实 worker 和 Redis 结果后端。"""
    app = create_celery()
    prefix = "qiye_rag:test:" + uuid4().hex + ":"
    queue = "test_" + uuid4().hex
    app.conf.update(task_default_queue=queue,
                    broker_transport_options={"global_keyprefix": prefix},
                    result_backend_transport_options={"global_keyprefix": prefix})
    client = Redis.from_url(app.conf.broker_url, socket_connect_timeout=5, socket_timeout=5)

    @app.task(name="test.redis_echo_v3")
    def echo(value: str) -> str:
        """返回合成数据。

        Args:
            value: 测试值。
        Returns:
            原测试值。
        """
        return value

    try:
        assert client.ping()
        with start_worker(app, pool="solo", perform_ping_check=False, shutdown_timeout=15):
            result = echo.delay("redis-queue-ok")
            assert result.get(timeout=15, disable_sync_subtasks=False) == "redis-queue-ok"
            assert app.AsyncResult(result.id).state == "SUCCESS"
            result.forget()
    finally:
        # 仅清理本次随机前缀，不清空 Redis 数据库或其他应用队列。
        for key in client.scan_iter(match=prefix + "*"):
            client.delete(key)
        client.close()
        app.close()
