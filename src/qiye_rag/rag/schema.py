"""仅允许 metadata JSON 的 Milvus 集合结构。"""


def collection_layout(schema: dict) -> tuple[str, str]:
    """验证固定 JSON 元数据结构，拒绝动态字段集合。

    Args:
        schema: Milvus CollectionSchema.to_dict() 的结果。
    Returns:
        字符串主键名和固定的 metadata 字段名。
    Raises:
        ValueError: schema 不满足存储约定。
    """
    fields = {f["name"]: f for f in schema["fields"]}
    primary = next((f for f in fields.values() if f.get("is_primary")), {})
    name = primary.get("name")
    # 保留已建 JSON 集合的 id/pk 主键命名；不再允许展开或动态元数据。
    valid = (name in {"id", "pk"} and int(primary.get("type", 0)) == 21
             and not schema.get("auto_id", False)
             and not schema.get("enable_dynamic_field", False)
             and set(fields) == {name, "text", "vector", "metadata"}
             and int(fields["text"].get("type", 0)) == 21
             and int(fields["vector"].get("type", 0)) == 101
             and int(fields["metadata"].get("type", 0)) == 23)
    if valid:
        return name, "metadata"
    raise ValueError("Milvus Collection schema 不兼容：仅支持字符串主键 pk/id、text、vector "
                     "和 metadata JSON，且必须关闭动态字段。请指定新集合重新同步；旧集合未修改。")
