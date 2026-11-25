"""导出TPE搜索记录、Pareto点和保留题真实Token对照报告。"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from statistics import mean
import optuna
from optuna.trial import TrialState
from qiye_rag.evaluation.tuning_data import BASELINE, DEV_IDS, NAMES
from qiye_rag.evaluation.tuning_engine import api_tokens
from qiye_rag.rag.config import PROJECT_ROOT
from qiye_rag import setting


def load_rows(folder: str) -> list[dict]:
    """读取完整逐题结果。

    Args:
        folder: 测量目录。
    Returns:
        逐题记录。
    """
    return [json.loads(line) for line in (Path(folder) / "results.jsonl").read_text(encoding="utf-8").splitlines()]


def validation_statistics(rows: list[dict]) -> dict:
    """按独立保留题和开发题分别统计。

    Args:
        rows: 最终强制重新执行的结果。
    Returns:
        子集覆盖率及实际消耗。
    """
    groups = {"原30题": [r for r in rows if r["id"].startswith("Q")],
              "22道保留题": [r for r in rows if r["id"].startswith("Q") and r["id"] not in DEV_IDS],
              "8道原开发题": [r for r in rows if r["id"] in DEV_IDS],
              "2道保留长历史题": [r for r in rows if r["id"].startswith("HV")]}
    result = {}
    for label, cases in groups.items():
        coverage = [r["coverage"] for r in cases if r["coverage"] is not None]
        result[label] = {"questions": len(cases), "answerable": len(coverage),
            "coverage": mean(coverage) if coverage else None, "actual_api_tokens": sum(r["api_tokens"] for r in cases),
            "mean_evidence_tokens": mean(r["evidence_tokens"] for r in cases) if cases else None,
            "usage_complete": all(r["usage_complete"] for r in cases)}
    return result


def history_resolution(folder: str) -> dict:
    """检查第一轮检索查询是否保留历史中的目标型号。

    Args:
        folder: 测量目录。
    Returns:
        指代解析诊断，不替代答案正确性。
    """
    gold = json.loads((Path(folder) / "gold.json").read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in gold["cases"]}
    result = {}
    for row in load_rows(folder):
        if not row["id"].startswith("H"):
            continue
        subject = re.search(r"[A-Z]{2,3}-\d+", by_id[row["id"]]["history"][0]["question"])[0]
        first = next((e for e in row["events"] if e["stage"] == "retriever" and e["round"] == 1), {})
        result[row["id"]] = {"subject": subject, "first_query_resolved": subject in " ".join(first.get("queries", [])),
                             "approved": row.get("approved"), "answer": row.get("answer", "")}
    return result


def token_breakdown(rows: list[dict]) -> dict:
    """按模型拆分真实usage，防止只看合计掩盖消耗转移。

    Args:
        rows: 逐题实际结果。
    Returns:
        Embedding、Rerank及聊天输入输出。
    """
    events = [e for r in rows for e in r["events"]]
    chat = [e for e in events if e["stage"] in ("plan", "research", "verdict")]
    return {"Embedding": api_tokens([e for e in events if e["stage"] == "embedding_http"]),
            "Rerank": api_tokens([e for e in events if e["stage"] == "rerank_api"]),
            "聊天输入": sum(e.get("input_tokens") or 0 for e in chat),
            "聊天输出": sum(e.get("output_tokens") or 0 for e in chat)}


def balanced_report(root: Path, validation: dict, summaries: dict, lines: list[str]) -> None:
    """追加补充折中验证，保留原始覆盖优先候选的结果。

    Args:
        root: 原实验目录。
        validation: 原基线与候选记录。
        summaries: 各配置子集统计，原地追加折中配置。
        lines: 报告文本，原地追加。
    """
    path = root / "balanced_validation.json"
    if not path.exists():
        return
    balanced = json.loads(path.read_text(encoding="utf-8"))
    rows = load_rows(balanced["result"]["path"])
    stats = validation_statistics(rows)
    summaries["balanced"] = stats
    before = summaries["baseline"]
    passed = all(stats[k]["coverage"] >= before[k]["coverage"] for k in before) and (
        balanced["result"]["actual_api_tokens"] < validation["baseline"]["actual_api_tokens"])
    values = balanced["parameters"] if passed else BASELINE
    decision = {"selected_profile": "balanced" if passed else "baseline", "parameters": values,
                "validation_gate_passed": passed, "independent_final_test": False,
                "reason": "各子集证据覆盖不低于基线且32题真实Token更低" if passed else "折中配置未通过验证门槛，保留原值"}
    (root / "final_selection.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    lines += ["", "## 补充折中验证与最终选择", "",
        "覆盖优先的原TPE候选在真实验证中成本增加，因此从既有B组Pareto结果中取覆盖不低于A组输入水平的最低Token点，"
        "Graph沿用C组选择。没有新增搜索区间，也没有按保留题逐题调参。",
        f"来源B组trial {balanced['source_trial']}，开发覆盖门槛{balanced['development_coverage_floor']:.4f}。",
        "这次补充比较是在看到第一次验证成本后发起，验证结果参与了最终配置选择；22道题已不再是完全未接触的最终测试集。"
        "三套原始结果均保留，后续应在新增问题上复验。", "",
        "| 子集 | 基线覆盖 | 折中覆盖 | 基线实际Token | 折中实际Token |", "| --- | ---: | ---: | ---: | ---: |"]
    for label in stats:
        a, b = before[label], stats[label]
        lines.append(f"| {label} | {a['coverage']:.4f} | {b['coverage']:.4f} | {a['actual_api_tokens']} | {b['actual_api_tokens']} |")
    total = balanced["result"]
    base = validation["baseline"]
    lines += ["", f"32题：基线覆盖{base['coverage']:.4f}、Token {base['actual_api_tokens']}；折中覆盖{total['coverage']:.4f}、"
        f"Token {total['actual_api_tokens']}，Token变化{total['actual_api_tokens'] / base['actual_api_tokens'] - 1:+.2%}。",
        f"验证门槛通过：{passed}。最终选择：{decision['selected_profile']}。", "",
        "| 参数 | 原值 | 覆盖优先候选 | 最终选择 |", "| --- | ---: | ---: | ---: | ---: |"]
    for name in NAMES:
        lines.append(f"| {name} | {BASELINE[name]} | {validation['parameters'][name]} | {values[name]} |")
    breakdown = token_breakdown(rows)
    probes = history_resolution(balanced["result"]["path"])
    (root / "balanced_history_diagnostics.json").write_text(json.dumps(probes, ensure_ascii=False, indent=2), encoding="utf-8")
    lines += ["", "折中配置真实Token拆分：" + "；".join(f"{k}={v}" for k, v in breakdown.items()) + "。",
        "切块256/25只是在当前小语料上的有限收益，不代表通用最佳值。MAX_RETRIEVAL_K=100和证据上限30处于非活跃区间，"
        "保留原值以减少无收益变化；HISTORY_TURNS=6保留较长指代窗口，探针数量不足以证明普遍最优。",
        "单轮检索降低消耗，但会放弃部分二次检索补救机会；新增困难问题时应重新比较1/2轮。",
        "改动切块参数后需要重新同步应用索引并重启程序。当前历史规则题检索缺口、Researcher未接收完整消歧问题仍需单独优化。"]
    setup = [json.loads(s) for s in (root / "balanced_check/setup.jsonl").read_text(encoding="utf-8").splitlines()]
    lines += [f"补充验证另执行32次Agent问答、实际API Token {total['actual_api_tokens']}，索引Token {sum(s['tokens'] for s in setup)}。"]
    for qid, probe in probes.items():
        lines.append(f"- balanced/{qid}：首轮查询识别={probe['first_query_resolved']}；最终答复：{probe['answer']}")


def generate(root: Path) -> None:
    """生成完整参数实验报告和CSV。

    Args:
        root: 调参运行目录。
    """
    selected = json.loads((root / "selected.json").read_text(encoding="utf-8"))
    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    studies = {g: optuna.load_study(study_name=f"group_{g}", storage="sqlite:///" + (root / "studies.sqlite").resolve().as_posix())
               for g in ("A", "B", "C")}
    records = []
    lines = ["# 分组Optuna TPE参数选择实验报告", "", "## 协议与目标", "",
        "三组各30trials，TPESampler(seed=20261113,n_startup_trials=8)，双目标为最大事实证据覆盖、最小Token消耗。"
        "协议在运行前固定，见上级目录实验设计.md。原30题中的8题和两个HS长历史探针用于开发，其余22题及HV探针只在选完参数后验证。", "",
        "A/B目标Token是真实检索API消耗加两次离线证据Token的筛选代理，不是完整聊天计费；C及最终验证使用完整API usage。"
        "同一有效参数复用已测结果并记cache_hit，不把30个trials描述成30次独立真实模型测量。索引实际Token单列。", "",
        "方法依据：[Optuna 4.5 TPESampler](https://optuna.readthedocs.io/en/v4.5.0/reference/samplers/generated/optuna.samplers.TPESampler.html)。"
        "本次是有限范围的顺序分组搜索，不保证全局最优。", "", "## 三组搜索", ""]
    for group, study in studies.items():
        complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
        cached = sum(t.user_attrs.get("measurement", {}).get("cache_hit", False) for t in complete)
        unique = {t.user_attrs["measurement"]["path"] for t in complete}
        lines += [f"### 组{group}", "", f"完成{len(complete)} trials；本组非缓存测量{len(complete) - cached}次；引用独立测量目录{len(unique)}个。"
                  f"选择trial {selected[group]['trial']}，Pareto trials：{selected[group]['pareto_trials']}。", "",
                  "| Trial | 事实覆盖 | Token目标 | 复用成绩 | 参数 |", "| --- | ---: | ---: | --- | --- |"]
        for t in complete:
            measurement = t.user_attrs["measurement"]
            params = t.user_attrs["full_parameters"]
            lines.append(f"| {t.number} | {t.values[0]:.4f} | {t.values[1]:.0f} | {measurement['cache_hit']} | `{json.dumps(t.params)}` |")
            records.append({"group": group, "trial": t.number, "coverage": t.values[0], "objective_tokens": t.values[1],
                "cache_hit": measurement["cache_hit"], "actual_api_tokens": measurement["actual_api_tokens"],
                "token_kind": measurement["token_kind"], "measurement": measurement["path"],
                "history_resolved": sum(x["first_query_resolved"] for x in history_resolution(measurement["path"]).values()) if group == "C" else None,
                **params})
        lines += [""]
    with (root / "trials.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summaries = {key: validation_statistics(load_rows(validation[key]["path"])) for key in ("baseline", "recommended")}
    lines += ["## 最终候选参数", "", "下表为搜索选出的候选，是否写入默认值还需结合保留题及回归测试；MAX_RETRIEVAL_K在非活跃区间保留100是兼容策略，不是质量收益。", "",
              "| 参数 | 原值 | 候选值 |", "| --- | ---: | ---: |"]
    for name in NAMES:
        lines.append(f"| {name} | {BASELINE[name]} | {validation['parameters'][name]} |")
    lines += ["", "## 选择后的真实Agent验证", "", "原30题包含8道开发题，不能称为30道独立测试。下表单列22道保留题，"
              "历史规则题缺失不会从分母中删除。事实覆盖只表示正确证据进入上下文，不代表答案事实完全正确；不以approved替代准确率。", "",
              "| 子集 | 可回答题数 | 基线事实覆盖 | 候选事实覆盖 | 基线API Token | 候选API Token |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label in summaries["baseline"]:
        before, after = summaries["baseline"][label], summaries["recommended"][label]
        lines.append(f"| {label} | {before['answerable']} | {before['coverage']:.4f} | {after['coverage']:.4f} | {before['actual_api_tokens']} | {after['actual_api_tokens']} |")
    lines += ["", "## 逐题变化", "", "| 题号 | 基线覆盖 | 候选覆盖 | 基线Token | 候选Token |", "| --- | ---: | ---: | ---: | ---: |"]
    original = load_rows(validation["baseline"]["path"])
    recommendation = load_rows(validation["recommended"]["path"])
    for before, after in zip(original, recommendation):
        lines.append(f"| {before['id']} | {before['coverage']} | {after['coverage']} | {before['api_tokens']} | {after['api_tokens']} |")
    breakdown = {"baseline": token_breakdown(original), "recommended": token_breakdown(recommendation)}
    lines += ["", "## 模型实际Token拆分（各32题）", "", "| 模型或阶段 | 基线 | 候选 |", "| --- | ---: | ---: |"]
    for name in breakdown["baseline"]:
        lines.append(f"| {name} | {breakdown['baseline'][name]} | {breakdown['recommended'][name]} |")
    lines += ["", "不同API的Token合计用来比较数量，不等同费用；没有假设它们单价相同。"]
    diagnostics = {key: history_resolution(validation[key]["path"]) for key in ("baseline", "recommended")}
    lines += ["", "## 长历史信息传递诊断", "",
        "HISTORY_TURNS只控制Router历史窗口，现有Researcher接收原问题和证据、不接收消解后的完整问题。"
        "因此即使查询正确识别目标，最终仍可能拒答。下列查询实体匹配只是补充诊断，不是答案语义准确率；未将它事后悄悄混入原TPE目标。", ""]
    for label, probes in diagnostics.items():
        for qid, value in probes.items():
            lines.append(f"- {label}/{qid}：目标{value['subject']}，首轮查询识别={value['first_query_resolved']}；最终答复：{value['answer']}")
    (root / "history_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    setup = [json.loads(line) for line in (root / "setup.jsonl").read_text(encoding="utf-8").splitlines()]
    measurements = [json.loads(p.read_text(encoding="utf-8")) for p in (root / "measurements").glob("*/summary.json")]
    lines += ["", "## 实验总开销与边界", "",
        f"共建立{len(setup)}份内容不同的独立评测索引，索引API Token合计{sum(r['tokens'] for r in setup)}。重复trial不重新计费。"
        "搜索开销应对measurements目录的唯一实测结果求和，不能把被复用的objective值乘以trial数量。", "",
        f"本次唯一实测目录{len(measurements)}份，累计题目执行{sum(r['questions'] for r in measurements)}次（包含检索筛选和完整Agent，二者不混称问答次数）；"
        f"这些实测API Token总计{sum(r['actual_api_tokens'] for r in measurements)}，不含上面单列的索引开销。", "",
        "较大CHUNK_SIZE/不同overlap可能产生完全相同内容，这些平台不能被解释为每个数值都获得独立质量证据。"
        "MAX_RETRIEVAL_K只限制合法上限，MAX_EVIDENCE_CHUNKS超过实际Top K时同样不生效。长历史探针使HISTORY_TURNS具备可测区分，"
        "但仅两个探针不足以概括真实企业多轮会话。", "",
        "模型、依赖和执行时源码摘要见environment.json；原Gold及离线Tokenizer摘要见report_provenance.json；trial库见studies.sqlite，全部采样及实际测量映射见trials.csv。"
        "参数临时上下文在每次测量后恢复；实验索引只使用随机qiye_tune_*集合，正常结束清理，不修改生产知识库。"]
    balanced_report(root, validation, summaries, lines)
    provenance = {"gold_sha256": hashlib.sha256((PROJECT_ROOT / "eval/enterprise_30.json").read_bytes()).hexdigest(),
                  "tokenizer_path": setting.CHAT_TOKENIZER_PATH,
                  "tokenizer_sha256": hashlib.sha256((PROJECT_ROOT / setting.CHAT_TOKENIZER_PATH).read_bytes()).hexdigest()}
    (root / "report_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (root / "validation_statistics.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "参数选择实验报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    generate(parser.parse_args().root)
