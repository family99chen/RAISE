#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_BUDGETS = [120, 300, 600]
DEFAULT_SEEDS = [11, 22, 33]
DEFAULT_DATASETS = ["triviaqa", "scienceqa", "longbench-qasper", "longbench-multifield"]
METRIC_KEYS = ["LLMAAJ", "BERTScore-F1", "ROUGE-L", "F1", "BLEU", "ExactMatch"]
BASELINES = ["greedy", "tpe", "mab_ts"]
ALGORITHMS = ["grpo", "greedy", "tpe", "mab_ts"]


@dataclass
class RunRow:
    dataset: str
    algorithm: str
    budget: int
    seed: int
    report_path: str
    calls_observed: int
    best_objective: float
    auc_best_so_far: float
    best_index: int
    best_score: float
    error_trials: int
    metrics: Dict[str, float]


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return default


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {}
    return data


def _extract_trials(report: Dict[str, object]) -> List[Dict[str, object]]:
    trials = report.get("trials")
    if not isinstance(trials, list):
        return []
    out: List[Dict[str, object]] = []
    for t in trials:
        if isinstance(t, dict):
            out.append(t)
    return out


def _objective_from_trial(trial: Dict[str, object]) -> float:
    if "score" in trial:
        return _safe_float(trial.get("score"), default=float("-inf"))
    return _safe_float(trial.get("reward"), default=float("-inf"))


def _best_so_far(series: List[float], budget: int) -> List[float]:
    if budget <= 0:
        return []
    if not series:
        return [0.0] * budget
    out: List[float] = []
    best = float("-inf")
    for value in series[:budget]:
        best = max(best, value)
        out.append(best)
    if len(out) < budget:
        pad = out[-1] if out else 0.0
        out.extend([pad] * (budget - len(out)))
    return out


def _auc_mean(series: List[float]) -> float:
    if not series:
        return 0.0
    return sum(series) / len(series)


def _extract_metrics_from_trial(trial: Dict[str, object]) -> Dict[str, float]:
    report = trial.get("report")
    if not isinstance(report, dict):
        return {k: 0.0 for k in METRIC_KEYS}
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return {k: 0.0 for k in METRIC_KEYS}
    return {k: _safe_float(metrics.get(k), 0.0) for k in METRIC_KEYS}


def _count_error_trials(trials: List[Dict[str, object]], upto: int) -> int:
    n = 0
    for t in trials[:upto]:
        err = t.get("error")
        errs = t.get("errors")
        if err:
            n += 1
            continue
        if isinstance(errs, list) and errs:
            n += 1
    return n


def _read_one_run(
    dataset: str,
    results_root: str,
    algorithm: str,
    budget: int,
    seed: int,
) -> Optional[RunRow]:
    if algorithm == "greedy":
        path = os.path.join(
            results_root,
            "greedy",
            "full",
            f"seed_{seed}",
            "report.json",
        )
    else:
        path = os.path.join(
            results_root,
            algorithm,
            f"budget_{budget}",
            f"seed_{seed}",
            "report.json",
        )
    if not os.path.isfile(path):
        return None

    payload = _load_json(path)
    trials = _extract_trials(payload)
    objectives = [_objective_from_trial(t) for t in trials]
    curve = _best_so_far(objectives, budget)
    calls = min(len(trials), budget)
    if calls == 0:
        best_idx = -1
        best_obj = 0.0
        best_score = 0.0
        best_metrics = {k: 0.0 for k in METRIC_KEYS}
    else:
        best_idx = max(range(calls), key=lambda i: objectives[i])
        best_obj = objectives[best_idx]
        best_score = _safe_float(trials[best_idx].get("score"), 0.0)
        best_metrics = _extract_metrics_from_trial(trials[best_idx])

    return RunRow(
        dataset=dataset,
        algorithm=algorithm,
        budget=budget,
        seed=seed,
        report_path=path,
        calls_observed=calls,
        best_objective=best_obj,
        auc_best_so_far=_auc_mean(curve),
        best_index=best_idx + 1 if best_idx >= 0 else 0,
        best_score=best_score,
        error_trials=_count_error_trials(trials, calls),
        metrics=best_metrics,
    )


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return mean(values), pstdev(values)


def _bootstrap_ci(
    values: Sequence[float],
    rng: random.Random,
    n_iter: int = 2000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    vals = list(values)
    n = len(vals)
    samples = []
    for _ in range(n_iter):
        draw = [vals[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(draw) / n)
    samples.sort()
    lo_i = int((alpha / 2.0) * (n_iter - 1))
    hi_i = int((1.0 - alpha / 2.0) * (n_iter - 1))
    return samples[lo_i], samples[hi_i]


def _paired_delta(a: Sequence[float], b: Sequence[float]) -> float:
    wins = 0
    losses = 0
    for x, y in zip(a, b):
        if x > y:
            wins += 1
        elif x < y:
            losses += 1
    n = max(1, len(a))
    return (wins - losses) / n


def _binom_two_sided(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    tail = 0.0
    for i in range(0, k + 1):
        tail += math.comb(n, i) * (0.5 ** n)
    p = min(1.0, 2.0 * tail)
    return p


def _sign_test(a: Sequence[float], b: Sequence[float]) -> float:
    diffs = [x - y for x, y in zip(a, b) if abs(x - y) > 1e-12]
    n = len(diffs)
    if n == 0:
        return 1.0
    positive = sum(1 for d in diffs if d > 0)
    k = min(positive, n - positive)
    return _binom_two_sided(k, n)


def _wilcoxon_pvalue(a: Sequence[float], b: Sequence[float]) -> Tuple[float, str]:
    try:
        from scipy.stats import wilcoxon  # type: ignore

        stat = wilcoxon(a, b, zero_method="wilcox", correction=False, alternative="two-sided")
        return float(stat.pvalue), "wilcoxon"
    except Exception:
        return _sign_test(a, b), "sign_test_fallback"


def _write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _try_plot_anytime(
    output_dir: str,
    budgets: Sequence[int],
    runs: List[RunRow],
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    for budget in budgets:
        curves: Dict[str, List[List[float]]] = {k: [] for k in ALGORITHMS}
        for row in runs:
            if row.budget != budget:
                continue
            payload = _load_json(row.report_path)
            trials = _extract_trials(payload)
            objectives = [_objective_from_trial(t) for t in trials]
            curve = _best_so_far(objectives, budget)
            curves[row.algorithm].append(curve)

        plt.figure(figsize=(8, 5))
        plotted = 0
        for alg in ALGORITHMS:
            alg_curves = curves.get(alg, [])
            if not alg_curves:
                continue
            avg = []
            for i in range(budget):
                avg.append(sum(c[i] for c in alg_curves) / len(alg_curves))
            xs = list(range(1, budget + 1))
            plt.plot(xs, avg, label=alg)
            plotted += 1
        if plotted == 0:
            plt.close()
            continue
        plt.xlabel("Evaluation Calls")
        plt.ylabel("Best-so-far Objective")
        plt.title(f"Anytime Curves @ budget={budget}")
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"anytime_budget_{budget}.png")
        plt.savefig(fig_path, dpi=160)
        plt.close()


def _analyze_dataset(
    dataset: str,
    results_root: str,
    output_dir: str,
    budgets: Sequence[int],
    seeds: Sequence[int],
    bootstrap_iter: int,
    bootstrap_seed: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    runs: List[RunRow] = []
    missing: List[Dict[str, object]] = []

    for budget in budgets:
        for seed in seeds:
            for alg in ALGORITHMS:
                row = _read_one_run(dataset, results_root, alg, budget, seed)
                if row is None:
                    missing.append(
                        {
                            "dataset": dataset,
                            "algorithm": alg,
                            "budget": budget,
                            "seed": seed,
                        }
                    )
                    continue
                runs.append(row)

    per_run_rows: List[Dict[str, object]] = []
    for r in runs:
        row = {
            "dataset": r.dataset,
            "algorithm": r.algorithm,
            "budget": r.budget,
            "seed": r.seed,
            "calls_observed": r.calls_observed,
            "best_score": r.best_objective,
            "auc_best_so_far": r.auc_best_so_far,
            "best_index": r.best_index,
            "trial_score": r.best_score,
            "error_trials": r.error_trials,
            "report_path": r.report_path,
        }
        for key in METRIC_KEYS:
            row[key] = r.metrics.get(key, 0.0)
        per_run_rows.append(row)
    _write_csv(
        os.path.join(output_dir, "per_run.csv"),
        per_run_rows,
        [
            "dataset",
            "algorithm",
            "budget",
            "seed",
            "calls_observed",
            "best_score",
            "auc_best_so_far",
            "best_index",
            "trial_score",
            "error_trials",
            *METRIC_KEYS,
            "report_path",
        ],
    )

    rng = random.Random(bootstrap_seed)
    objective_rows: List[Dict[str, object]] = []
    metrics_rows: List[Dict[str, object]] = []

    for budget in budgets:
        for alg in ALGORITHMS:
            sub = [r for r in runs if r.budget == budget and r.algorithm == alg]
            if not sub:
                continue
            best_objs = [r.best_objective for r in sub]
            aucs = [r.auc_best_so_far for r in sub]
            obj_mean, obj_std = _mean_std(best_objs)
            auc_mean, auc_std = _mean_std(aucs)
            ci_lo, ci_hi = _bootstrap_ci(best_objs, rng=rng, n_iter=bootstrap_iter)
            objective_rows.append(
                {
                    "dataset": dataset,
                    "budget": budget,
                    "algorithm": alg,
                    "n": len(sub),
                    "best_score_mean": obj_mean,
                    "best_score_std": obj_std,
                    "best_score_ci95_lo": ci_lo,
                    "best_score_ci95_hi": ci_hi,
                    "auc_mean": auc_mean,
                    "auc_std": auc_std,
                    "error_trials_mean": sum(r.error_trials for r in sub) / len(sub),
                }
            )

            metric_summary = {"dataset": dataset, "budget": budget, "algorithm": alg, "n": len(sub)}
            for key in METRIC_KEYS:
                vals = [r.metrics.get(key, 0.0) for r in sub]
                m, s = _mean_std(vals)
                metric_summary[f"{key}_mean"] = m
                metric_summary[f"{key}_std"] = s
            metrics_rows.append(metric_summary)

    _write_csv(
        os.path.join(output_dir, "table1_best_score.csv"),
        objective_rows,
        [
            "dataset",
            "budget",
            "algorithm",
            "n",
            "best_score_mean",
            "best_score_std",
            "best_score_ci95_lo",
            "best_score_ci95_hi",
            "auc_mean",
            "auc_std",
            "error_trials_mean",
        ],
    )

    _write_csv(
        os.path.join(output_dir, "table2_metrics.csv"),
        metrics_rows,
        [
            "dataset",
            "budget",
            "algorithm",
            "n",
            "LLMAAJ_mean",
            "LLMAAJ_std",
            "BERTScore-F1_mean",
            "BERTScore-F1_std",
            "ROUGE-L_mean",
            "ROUGE-L_std",
            "F1_mean",
            "F1_std",
            "BLEU_mean",
            "BLEU_std",
            "ExactMatch_mean",
            "ExactMatch_std",
        ],
    )

    significance_rows: List[Dict[str, object]] = []
    for budget in budgets:
        grpo_rows = [r for r in runs if r.budget == budget and r.algorithm == "grpo"]
        grpo_by_seed = {r.seed: r for r in grpo_rows}
        for base in BASELINES:
            base_rows = [r for r in runs if r.budget == budget and r.algorithm == base]
            base_by_seed = {r.seed: r for r in base_rows}
            shared = sorted(set(grpo_by_seed.keys()) & set(base_by_seed.keys()))
            if not shared:
                continue
            grpo_vals = [grpo_by_seed[s].best_objective for s in shared]
            base_vals = [base_by_seed[s].best_objective for s in shared]
            p_value, test_name = _wilcoxon_pvalue(grpo_vals, base_vals)
            wins = sum(1 for g, b in zip(grpo_vals, base_vals) if g > b)
            significance_rows.append(
                {
                    "dataset": dataset,
                    "budget": budget,
                    "baseline": base,
                    "n_pairs": len(shared),
                    "test": test_name,
                    "p_value": p_value,
                    "grpo_win_rate": wins / len(shared),
                    "paired_delta": _paired_delta(grpo_vals, base_vals),
                    "grpo_mean": _mean_std(grpo_vals)[0],
                    "baseline_mean": _mean_std(base_vals)[0],
                }
            )

    _write_csv(
        os.path.join(output_dir, "significance.csv"),
        significance_rows,
        [
            "dataset",
            "budget",
            "baseline",
            "n_pairs",
            "test",
            "p_value",
            "grpo_win_rate",
            "paired_delta",
            "grpo_mean",
            "baseline_mean",
        ],
    )

    _try_plot_anytime(output_dir=output_dir, budgets=budgets, runs=runs)

    summary = {
        "dataset": dataset,
        "results_root": results_root,
        "budgets": budgets,
        "seeds": seeds,
        "total_runs_loaded": len(runs),
        "missing_runs": missing,
        "outputs": {
            "per_run_csv": os.path.join(output_dir, "per_run.csv"),
            "table1_best_score_csv": os.path.join(output_dir, "table1_best_score.csv"),
            "table2_metrics_csv": os.path.join(output_dir, "table2_metrics.csv"),
            "significance_csv": os.path.join(output_dir, "significance.csv"),
        },
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"[{dataset}] Analysis complete. Summary: {os.path.join(output_dir, 'summary.json')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze five-algorithm benchmark outputs and produce tables/figures."
    )
    parser.add_argument(
        "--results_root",
        default="outputs-experiments",
        help="Root directory containing per-dataset results.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs-experiments/analysis",
        help="Directory to write aggregated outputs.",
    )
    parser.add_argument(
        "--datasets",
        default="triviaqa,scienceqa,longbench-qasper,longbench-multifield",
        help="Comma-separated dataset names to analyze.",
    )
    parser.add_argument(
        "--budgets",
        default="120,300,600",
        help="Comma-separated budgets to analyze.",
    )
    parser.add_argument(
        "--seeds",
        default="11,22,33",
        help="Comma-separated seeds to analyze.",
    )
    parser.add_argument(
        "--bootstrap_iter",
        type=int,
        default=2000,
        help="Bootstrap iterations for confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=2026,
        help="Seed for bootstrap resampling.",
    )
    args = parser.parse_args()

    results_root = os.path.abspath(args.results_root)
    output_dir = os.path.abspath(args.output_dir)
    budgets = _parse_int_list(args.budgets)
    seeds = _parse_int_list(args.seeds)
    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    if not dataset_names:
        dataset_names = list(DEFAULT_DATASETS)

    for dataset in dataset_names:
        dataset_results = os.path.join(results_root, dataset, "results")
        dataset_output = os.path.join(output_dir, dataset)
        _analyze_dataset(
            dataset=dataset,
            results_root=dataset_results,
            output_dir=dataset_output,
            budgets=budgets,
            seeds=seeds,
            bootstrap_iter=args.bootstrap_iter,
            bootstrap_seed=args.bootstrap_seed,
        )


if __name__ == "__main__":
    main()
