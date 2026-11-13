"""可在 PyCharm 直接运行的 v2 演示；依赖仍需安装。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qiye_rag.graph.__main__ import main

if __name__ == "__main__":
    main()
