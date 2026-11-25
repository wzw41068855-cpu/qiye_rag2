"""将 my_plan 原始结果整理为可读报告，不伪造答案准确率。"""
import json
from pathlib import Path
from statistics import mean, median


def read_run(path: Path) -> tuple[dict, dict, list[dict]]:
    """读取一个完整实验。

    Args:
        path: 实验目录。
    Returns:
        元数据、Gold 和逐题结果。
    """
    return (json.loads((path / "metadata.json").read_text(encoding="utf-8")),
            json.loads((path / "gold.json").read_text(encoding="utf-8")),
            [json.loads(line) for line in (path / "results.jsonl").read_text(encoding="utf-8").splitlines()])


def statistics(path: Path) -> dict:
    """汇总质量、成本及预算，不重复计算嵌套事件。

    Args:
        path: 实验目录。
    Returns:
        可比较统计。
    """
    meta, gold, rows = read_run(path)
    events = [e for row in rows for e in row["events"]]
    measured = [row for row in rows if "metrics" in row]
    setup = json.loads((path / "setup.jsonl").read_text(encoding="utf-8").splitlines()[0])["events"]
    def average(name: str) -> float | None:
        """计算有分母的平均指标。

        Args:
            name: 指标名称。
        Returns:
            均值或未测量。
        """
        values = [r["metrics"][name] for r in measured if r["metrics"].get(name) is not None]
        return mean(values) if values else None
    return {"questions": len(rows), "errors": len(rows) - len(measured),
        "chunks": gold["chunk_statistics"]["count"], "characters": gold["chunk_statistics"]["characters"],
        "document_recall": average("document_recall"), "chunk_recall": average("chunk_recall"),
        "fact_evidence_recall": average("fact_evidence_recall"), "decision_accuracy": average("refusal_correct"),
        "calls": sum(r["logical_model_calls"] for r in rows),
        "input_tokens": sum(r["known_input_tokens"] for r in rows),
        "output_tokens": sum(r["known_output_tokens"] for r in rows),
        "total_tokens": sum(r["known_total_tokens"] for r in rows),
        "usage_complete": all(r["token_usage_complete"] for r in rows),
        "setup_embedding_tokens": sum(e.get("total_tokens") or 0 for e in setup if e["stage"] == "embedding_http"),
        "median_ms": median(r["elapsed_ms"] for r in rows),
        "max_evidence_tokens": max((e["evidence_tokens"] for e in events if "evidence_tokens" in e), default=0),
        "unauthorized_hits": sum(r["unauthorized_stage_hits"] for r in rows),
        "source_sha256": meta["source_sha256"]}


def comparison(paths: list[Path], target: Path, title: str, conclusion: str) -> None:
    """生成逐题对照、答案摘录和限制说明。

    Args:
        paths: 待比较的完整实验目录。
        target: Markdown 报告路径。
        title: 报告标题。
        conclusion: 根据结果人工撰写的结论。
    """
    stats = [statistics(path) for path in paths]
    table = [f"# {title}", "", "本报告由实际 JSONL 结果生成；每组完整运行同一 30 题，单次实验，无统计显著性主张。", "",
        "## 实验设置", "",
        "固定 data、参考事实、角色、模型、提示词、Graph 路由、1000/200 字符切分参数、Top K=5、Recall=20、最多两轮检索。证据预算为6000离线Token，完整输入预算12000，输出上限2000。只改变本实验明确列出的切分/检索策略。", "",
        "离线文本数来自 DeepSeek v4 官方 tokenizer，实际消耗来自 API usage；两者不可混用。每组独立临时 Milvus 集合已清理。Gold 根据冻结事实锚点映射，原版 Gold 不覆盖。", "",
        "## 汇总", "", "| 指标 | " + " | ".join(p.name for p in paths) + " |", "| --- | " + " | ".join("---:" for _ in paths) + " |"]
    fields = [("questions", "执行题数"), ("errors", "执行错误"), ("chunks", "索引块数"),
        ("characters", "索引总字符"), ("document_recall", "Document Recall（可回答题均值）"),
        ("chunk_recall", "Chunk Recall（仅作各版本内部诊断）"), ("fact_evidence_recall", "原文事实锚点在证据中的覆盖"),
        ("decision_accuracy", "审核状态与可回答性一致率（代理值）"), ("calls", "逻辑模型调用"),
        ("total_tokens", "问答实际已知总Token（各API相加）"), ("setup_embedding_tokens", "索引Embedding实际Token（单列）"),
        ("usage_complete", "问答API usage全部可得"), ("median_ms", "端到端P50毫秒"),
        ("max_evidence_tokens", "最大已装入证据离线Token"), ("unauthorized_hits", "全阶段未授权候选数")]
    for key, label in fields:
        values = [f"{s[key]:.4f}" if isinstance(s[key], float) else str(s[key]) for s in stats]
        table.append(f"| {label} | " + " | ".join(values) + " |")
    table += ["", "## 结论", "", conclusion, "", "## 逐题对照", "",
        "下表每格依次为：approved与可回答性是否一致；原文事实证据覆盖比例；实际已知 Token。approved=true也可能是一段有依据的正确拒答，故该代理值不是语义拒答正确率，更不是答案准确率（Q27有实际误判，见复核说明）。不可回答题没有事实召回分母，显示 —。", "",
        "| 题号 | 类别 | " + " | ".join(p.name for p in paths) + " |", "| --- | --- | " + " | ".join("---" for _ in paths) + " |"]
    runs = [read_run(p) for p in paths]
    for i, case in enumerate(runs[0][1]["cases"]):
        values = []
        for _, _, rows in runs:
            row = rows[i]
            if "error" in row:
                values.append("执行失败：" + row["error"])
            else:
                m = row["metrics"]
                fact = "—" if m["fact_evidence_recall"] is None else f"{m['fact_evidence_recall']:.2f}"
                values.append(f"{'✓' if m['refusal_correct'] else '✗'} / {fact} / {row['known_total_tokens']}")
        table.append(f"| {case['id']} | {case['category']} | " + " | ".join(values) + " |")
    table += ["", "## 重点失败题与答案摘录", ""]
    for qid in ("Q06", "Q16", "Q18", "Q20", "Q22", "Q26", "Q29"):
        case = next(c for c in runs[0][1]["cases"] if c["id"] == qid)
        table += [f"### {qid}：{case['question']}", "", "参考事实：" + ("；".join(case["reference_facts"]) or "授权资料无法确认，应拒答。"), ""]
        for path, (_, _, rows) in zip(paths, runs):
            row = next(r for r in rows if r["id"] == qid)
            table.append(f"- **{path.name}**：" + row.get("answer", "执行错误：" + row.get("error", "未知")).replace("\n", " "))
        table.append("")
    table += ["## 复现材料与限制", ""]
    for path, stat in zip(paths, stats):
        relative = path.relative_to(target.parent).as_posix()
        table += [f"- [{path.name} 原始结果]({relative}/results.jsonl)，[配置]({relative}/metadata.json)，[本组Gold]({relative}/gold.json)，源码摘要 `{stat['source_sha256']}`。"]
    table += ["", "不同切块的 Chunk Recall 不作直接胜负依据，以冻结事实覆盖和答案复核为主。基于原文锚点的覆盖指标只表示证据含对应文字，不自动证明模型正确理解、引用或回答。没有把 approved 当作事实准确率。", "",
        "本次未优化历史版本过滤和 Graph 重试策略，因此历史题拒答或问题范围扩大仍可能存在。服务并行运行与云端波动会影响延迟，性能差异不作严格因果结论。初始化索引Token与问答Token分开，不把已知数伪称缺失usage时的完整消耗。"]
    target.write_text("\n".join(table) + "\n", encoding="utf-8")
