"""v2 连续问答演示入口。"""
import argparse
from pathlib import Path
from qiye_rag.rag.config import PROJECT_ROOT
from qiye_rag.graph.runtime import open_conversation
from qiye_rag.graph.state import GraphState


def display(state: GraphState) -> None:
    """输出答案、引用和执行信息。

    Args:
        state: 最终状态。
    """
    print(f"\n分类：{state['kind']} | 检索轮次：{state['attempts']} | 校验通过：{state['approved']}")
    print(state["answer"])
    for citation in state["citations"]:
        print(f"[{citation['number']}] {citation.get('title', '')} | "
              f"{citation.get('source_path', '')} | {citation.get('location', '')}")
    if state.get("reason"):
        print(f"校验说明：{state['reason']}")


def main() -> None:
    """读取配套 v1 清单，开始问答或恢复中断执行。"""
    parser = argparse.ArgumentParser(description="v2 LangGraph 带引用问答")
    parser.add_argument("--kb", default="demo_v1")
    parser.add_argument("--catalog", type=Path, help="v1 清单，默认 .rag/<kb>/state.json")
    parser.add_argument("--checkpoints", type=Path, default=PROJECT_ROOT / ".rag/v2/sessions.sqlite")
    parser.add_argument("--session", default="default")
    parser.add_argument("--text", help="单次提问，否则进入连续问答")
    parser.add_argument("--resume", action="store_true", help="恢复同会话上次失败的执行")
    args = parser.parse_args()
    if args.catalog is None:
        args.catalog = PROJECT_ROOT / ".rag" / args.kb / "state.json"
    if args.resume and args.text is not None:
        parser.error("--resume 和 --text 不能同时使用")
    try:
        with open_conversation(args.catalog, args.checkpoints, args.kb) as conversation:
            if args.resume:
                display(conversation.resume(args.session))
                return
            if args.text is not None:
                display(conversation.ask(args.text, args.session))
                return
            print(f"会话：{args.session}；输入 exit 退出。当前仅使用授权的现行资料。")
            while True:
                text = input("\ntext> ").strip()
                if text.lower() in {"exit", "quit", "q"}:
                    break
                if text:
                    display(conversation.ask(text, args.session))
    except (EOFError, KeyboardInterrupt):
        print("\n已退出；执行中的节点可通过 --resume 继续。")


if __name__ == "__main__":
    main()
