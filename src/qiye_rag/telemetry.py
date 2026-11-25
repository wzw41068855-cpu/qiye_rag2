"""按上下文隔离的轻量实验记录，不保存密钥与证据正文。"""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator
import json
import httpx

_events: ContextVar[list[dict] | None] = ContextVar("rag_events", default=None)


@contextmanager
def capture() -> Iterator[list[dict]]:
    """收集当前执行的阶段事件，退出后恢复外层上下文。

    Yields:
        本次执行的事件列表。
    """
    events: list[dict] = []
    token = _events.set(events)
    try:
        yield events
    finally:
        _events.reset(token)


@contextmanager
def stage(name: str, **details: Any) -> Iterator[dict]:
    """记录一个阶段的耗时、结果和异常类型。

    Args:
        name: 阶段名称。
        details: 不含密钥或正文的统计信息。
    Yields:
        可追加统计值的事件。
    """
    event = {"stage": name, "model_calls": 0, "input_tokens": None,
             "output_tokens": None, **details}
    started = perf_counter()
    try:
        yield event
    except Exception as exc:
        event["error"] = type(exc).__name__  # 不回显服务响应及凭据。
        raise
    finally:
        event["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
        events = _events.get()
        if events is not None:
            events.append(event)


def write_result(path: Path, result: dict) -> None:
    """追加一条实验 JSONL 结果。

    Args:
        path: 调用方指定的本地输出路径。
        result: 可序列化的单题结果。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")


def embedding_response(response: httpx.Response) -> None:
    """读取 Embedding SDK 原始 usage，不记录向量或请求凭据。

    Args:
        response: SDK 使用的 HTTP 响应。
    """
    response.read()
    with stage("embedding_http", model_calls=0, http_responses=1) as event:
        event["status_code"] = response.status_code
        if response.is_success:
            usage = response.json().get("usage", {})
            event.update(input_tokens=usage.get("prompt_tokens", usage.get("total_tokens")),
                         output_tokens=0, total_tokens=usage.get("total_tokens"), usage_source="api")
