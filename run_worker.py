"""启动 v3 Celery worker；Windows 演示固定使用 solo 单进程。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from qiye_rag.run_time.celery_app import celery_app
from qiye_rag import setting

if __name__ == "__main__":
    celery_app.worker_main(["worker", "--pool=solo", "--concurrency=1", f"--loglevel={setting.LOG_LEVEL}"])
