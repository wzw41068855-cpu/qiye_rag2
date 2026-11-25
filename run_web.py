"""v4 登录与授权上传问答入口，Celery worker 仍使用 run_worker_v3.py。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from qiye_rag.web.__main__ import main

if __name__ == "__main__":
    main()
