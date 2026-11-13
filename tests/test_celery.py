"""Celery 实际队列与任务入口测试，不调用 LLM/Milvus。"""
from pathlib import Path
from celery.contrib.testing.worker import start_worker
from qiye_rag.run_time import celery_app as module
from qiye_rag.run_time import tasks
import pytest


def test_memory_queue_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """默认单元测试隔离外部 Redis，仍验证真实 worker 的任务消费与结果回传。"""
    monkeypatch.setattr(module, "load_config", lambda: {})
    app = module.create_celery()
    app.conf.update(broker_url="memory://", result_backend="cache+memory://",
                    broker_transport_options={}, result_backend_transport_options={})

    @app.task(name="test.echo_v3")
    def echo(value: str) -> str:
        """返回合成输入，验证队列传递。

        Args:
            value: 测试字符串。
        Returns:
            原字符串。
        """
        return value

    try:
        with start_worker(app, pool="solo", perform_ping_check=False, shutdown_timeout=15):
            result = echo.delay("queue-ok")
            assert result.get(timeout=15, disable_sync_subtasks=False) == "queue-ok"
    finally:
        app.close()


def test_redis_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """broker 与结果后端使用同一个 REDIS_URI，忽略旧文件队列配置。"""
    uri = "redis://localhost:6379/12"
    monkeypatch.setattr(module, "load_config", lambda: {"REDIS_URI": uri,
        "CELERY_BROKER_URI": "filesystem://", "CELERY_RESULT_BACKEND_URI": "file:///ignored"})
    app = module.create_celery()
    try:
        assert app.conf.broker_url == uri and app.conf.result_backend == uri
        assert app.conf.broker_transport_options["global_keyprefix"] == "qiye_rag:broker:"
        assert "data_folder_in" not in app.conf.broker_transport_options
    finally:
        app.close()


def test_task_calls_ingestion(monkeypatch: pytest.MonkeyPatch) -> None:
    """正式任务直接复用索引服务，参数不含用户构造的磁盘路径。"""
    received = []

    def fake_ingest(*args: str) -> dict:
        """收集任务参数。

        Args:
            args: 知识库与文件信息。
        Returns:
            测试结果。
        """
        received.append(args)
        return {"chunks": 1}

    monkeypatch.setattr(tasks, "ingest", fake_ingest)
    assert tasks.ingest_document.run("demo", "a" * 32, "a.md", ".md") == {"chunks": 1}
    assert received == [("demo", "a" * 32, "a.md", ".md")]
