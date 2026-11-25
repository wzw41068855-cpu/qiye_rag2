"""验证开发集Pareto上的折中点，不改变原始TPE研究结果。"""
import argparse
import json
from pathlib import Path
import optuna
from optuna.trial import TrialState
from qiye_rag.evaluation.tuning_data import BASELINE
from qiye_rag.evaluation.tuning_engine import TuningEngine


def main() -> None:
    """保留B组输入覆盖水平，选择其低Token点并重新验证。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    selected = json.loads((args.root / "selected.json").read_text(encoding="utf-8"))
    study = optuna.load_study(study_name="group_B", storage="sqlite:///" + (args.root / "studies.sqlite").resolve().as_posix())
    threshold = selected["A"]["values"][0]
    # 只读取既有开发成绩；不利用保留题逐题反推参数。
    trials = [t for t in study.trials if t.state == TrialState.COMPLETE and t.values[0] >= threshold - 1e-10]
    best = min(trials, key=lambda t: (t.values[1], sum(abs(t.user_attrs["full_parameters"][k] - v) / max(v, 1)
                                                     for k, v in BASELINE.items())))
    values = selected["C"]["parameters"].copy()
    for name in ("RETRIEVAL_TOP_K", "RETRIEVAL_RECALL", "MAX_EVIDENCE_CHUNKS"):
        values[name] = best.user_attrs["full_parameters"][name]
    values["MAX_RETRIEVAL_K"] = 100
    folder = args.root / "balanced_check"
    folder.mkdir(exist_ok=False)
    engine = TuningEngine(folder)
    try:
        result = engine.evaluate(values, "agent", "final_balanced", final=True)
        record = {"selection_rule": "B组开发覆盖不低于A组选择值，在这些trial中最小化Token代理；Graph取C组选择值",
                  "development_coverage_floor": threshold, "source_trial": best.number,
                  "parameters": values, "result": result}
        (args.root / "balanced_validation.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False, indent=2))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
