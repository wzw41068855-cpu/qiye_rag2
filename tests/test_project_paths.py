"""模块移动后的项目根路径和 v2 清单选择回归。"""
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from qiye_rag.rag.config import PROJECT_ROOT
from qiye_rag.graph import __main__ as entry
from qiye_rag.graph.runtime import open_conversation


def test_project_root_after_module_move() -> None:
    """rag 子包中的配置仍指向项目目录，而不是 src。"""
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_kb_selects_matching_catalog() -> None:
    """仅指定 my_rag 时默认选择同名本地清单。"""
    with patch.object(entry, "open_conversation") as factory, patch.object(entry, "display"), \
         patch("sys.argv", ["run_demo_v2.py", "--kb", "my_rag", "--text", "测试问题"]):
        entry.main()
    assert factory.call_args.args[0] == PROJECT_ROOT / ".rag/my_rag/state.json"
    assert factory.call_args.args[2] == "my_rag"


def test_missing_catalog_message(tmp_path: Path) -> None:
    """缺少本地清单时准确提示，不误报远端向量集合不存在。"""
    path = tmp_path / "missing/state.json"
    with pytest.raises(ValueError, match="本地索引清单不存在"):
        with open_conversation(path, tmp_path / "checkpoint.sqlite", "my_rag"):
            pass
    assert not path.parent.exists()
