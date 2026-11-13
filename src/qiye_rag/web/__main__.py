"""启动本地 v3 服务。"""
import uvicorn
from qiye_rag import setting


def main() -> None:
    """启动单进程 Web 演示。"""
    uvicorn.run("qiye_rag.web.app:app", host=setting.WEB_HOST, port=setting.WEB_PORT)


if __name__ == "__main__":
    main()
