"""统一 JSON 元数据存储的回归测试。"""
from unittest.mock import MagicMock
import pytest
from qiye_rag.rag.schema import collection_layout
from qiye_rag.rag.vectors import MilvusVectors


def schema(dynamic: bool = False) -> dict:
    """构造固定 JSON schema。

    Args:
        dynamic: 是否启用动态字段。
    Returns:
        测试用 schema 字典。
    """
    return {"auto_id": False, "enable_dynamic_field": dynamic, "fields": [
        {"name": "pk", "type": 21, "is_primary": True}, {"name": "text", "type": 21},
        {"name": "vector", "type": 101}, {"name": "metadata", "type": 23}]}


def test_reject_dynamic_or_expanded_metadata() -> None:
    """即使含 metadata，也不能开启动态字段或额外展开字段。"""
    assert collection_layout(schema()) == ("pk", "metadata")
    with pytest.raises(ValueError, match="关闭动态字段"):
        collection_layout(schema(True))
    expanded = schema()
    expanded["fields"].append({"name": "knowledge_base_id", "type": 21})
    with pytest.raises(ValueError):
        collection_layout(expanded)
    missing = schema()
    missing["fields"].pop()
    with pytest.raises(ValueError):
        collection_layout(missing)


def test_serialize_all_metadata_without_double_encoding() -> None:
    """嵌套、列表、中文、空值及版本信息均保留，禁止二次字符串编码。"""
    vec = MilvusVectors.__new__(MilvusVectors)
    vec.store = MagicMock()
    metadata = {"knowledge_base_id": "my_rag", "document_id": "HR001", "page": 1,
                "tags": ["人事", "制度"], "nested": {"enabled": True, "value": None},
                "content_id": "content", "index_version": "version", "chunk_id": "chunk"}
    vec.upsert([{"id": "chunk", "text": "正文", "metadata": metadata}])
    saved = vec.store.add_texts.call_args.kwargs["metadatas"][0]
    assert isinstance(saved, dict) and saved == metadata and saved is not metadata
    metadata["nested"]["value"] = "改变输入"
    assert saved["nested"]["value"] is None
    vec.store.reset_mock()
    for invalid in ({"value": float("nan")}, {"value": object()}):
        with pytest.raises((ValueError, TypeError)):
            vec.upsert([{"id": "bad", "text": "正文", "metadata": invalid}])
    vec.store.add_texts.assert_not_called()
