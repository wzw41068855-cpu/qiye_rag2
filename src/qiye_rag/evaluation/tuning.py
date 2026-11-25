"""分组多目标TPE：先检索筛选，再真实Agent与保留题验证。"""
import argparse
import json
from pathlib import Path
import optuna
from optuna.trial import TrialState
from qiye_rag.evaluation.tuning_data import BASELINE
from qiye_rag.evaluation.tuning_engine import TuningEngine


def suggest(trial: optuna.Trial, group: str, fixed: dict) -> dict:
    """在预先固定的合法范围采样。

    Args:
        trial: 当前试验。
        group: A、B或C。
        fixed: 其他组的固定值。
    Returns:
        完整生效参数。
    """
    result = fixed.copy()
    if group == "A":
        result["CHUNK_SIZE"] = trial.suggest_categorical("CHUNK_SIZE", [256, 384, 512, 768, 1000])
        ratio = trial.suggest_categorical("overlap_ratio", [0.0, 0.1, 0.2])
        result["CHUNK_OVERLAP"] = int(result["CHUNK_SIZE"] * ratio)
    elif group == "B":
        for name, choices in {"RETRIEVAL_TOP_K": [3, 5, 8, 10], "RETRIEVAL_RECALL": [10, 20, 40],
                              "MAX_RETRIEVAL_K": [40, 60, 100], "MAX_EVIDENCE_CHUNKS": [3, 5, 8, 12, 30]}.items():
            result[name] = trial.suggest_categorical(name, choices)
    else:
        for name, choices in {"MAX_RETRIEVAL_ROUNDS": [1, 2, 3], "MAX_SUBQUERIES": [1, 2, 3],
                              "HISTORY_TURNS": [1, 3, 6]}.items():
            result[name] = trial.suggest_categorical(name, choices)
    return result


def choose(study: optuna.Study) -> optuna.trial.FrozenTrial:
    """先最大覆盖，再最小Token，同值优先原始默认。

    Args:
        study: 已完成的双目标研究。
    Returns:
        推荐trial。
    """
    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    def key(trial: optuna.trial.FrozenTrial) -> tuple:
        """取得不使用验证题的排序键。

        Args:
            trial: 已完成试验。
        Returns:
            覆盖、消耗、默认距离。
        """
        values = trial.user_attrs["full_parameters"]
        distance = sum(abs(values[k] - v) / max(v, 1) for k, v in BASELINE.items())
        return -round(trial.values[0], 10), trial.values[1], distance
    return min(trials, key=key)


def main() -> None:
    """运行三组TPE及独立的最终参数对照。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    if args.trials < 30:
        parser.error("正式参数选择每组至少30trials")
    args.output.mkdir(parents=True, exist_ok=False)
    engine = TuningEngine(args.output)
    fixed = BASELINE.copy()
    selected = {}
    try:
        for group in ("A", "B", "C"):
            study = optuna.create_study(study_name=f"group_{group}", directions=["maximize", "minimize"],
                sampler=optuna.samplers.TPESampler(seed=20261113, n_startup_trials=8),
                storage="sqlite:///" + (args.output / "studies.sqlite").resolve().as_posix())
            initial = {"A": {"CHUNK_SIZE": 1000, "overlap_ratio": 0.2},
                       "B": {k: BASELINE[k] for k in ("RETRIEVAL_TOP_K", "RETRIEVAL_RECALL", "MAX_RETRIEVAL_K", "MAX_EVIDENCE_CHUNKS")},
                       "C": {k: BASELINE[k] for k in ("MAX_RETRIEVAL_ROUNDS", "MAX_SUBQUERIES", "HISTORY_TURNS")}}[group]
            study.enqueue_trial(initial)
            def objective(trial: optuna.Trial) -> tuple[float, int]:
                """测量本trial，写入可审查原始证据。

                Args:
                    trial: Optuna trial。
                Returns:
                    事实覆盖与明确口径的Token目标。
                """
                values = suggest(trial, group, fixed)
                trial.set_user_attr("full_parameters", values)
                measured = engine.evaluate(values, "agent" if group == "C" else "retrieval", f"{group}_{trial.number:02}")
                trial.set_user_attr("measurement", measured)
                print(json.dumps({"group": group, "trial": trial.number, "params": values, **measured}, ensure_ascii=False), flush=True)
                return measured["coverage"], measured["tokens"]
            study.optimize(objective, n_trials=args.trials)
            best = choose(study)
            fixed = best.user_attrs["full_parameters"].copy()
            # 上限的非活跃区间没有可识别性能收益，保持兼容已有CLI的100上限。
            fixed["MAX_RETRIEVAL_K"] = 100
            selected[group] = {"trial": best.number, "values": best.values, "parameters": fixed.copy(),
                               "pareto_trials": [t.number for t in study.best_trials]}
            (args.output / "selected.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
        # 选择完成之后才接触保留题；强制重新调用真实模型，不复用搜索得分。
        baseline = engine.evaluate(BASELINE, "agent", "final_baseline", final=True)
        recommended = engine.evaluate(fixed, "agent", "final_recommended", final=True)
        (args.output / "validation.json").write_text(json.dumps({"baseline": baseline,
            "recommended": recommended, "parameters": fixed}, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
