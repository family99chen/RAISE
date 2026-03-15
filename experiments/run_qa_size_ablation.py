#!/usr/bin/env python3
"""Run a QA-size ablation on one text dataset.

Default behavior:
- Uses local `longbench-qasper/qasper.jsonl`
- Builds nested QA subsets of size 20, 50, 100 from one fixed shuffle
- Runs the selected algorithms on each subset
- Reports per-size averages and cross-size variation

Example:

  python experiments/run_qa_size_ablation.py \
      --algorithms all \
      --budget 30 \
      --seeds 42,43,44
"""

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_QASPER_JSONL = os.path.join(
    PROJECT_ROOT, "data", "datasets", "longbench-qasper", "qasper.jsonl"
)
DEFAULT_CONFIG = os.path.join(
    PROJECT_ROOT, "configs", "experiments", "configforalgo_fullspace.yaml"
)

ALL_ALGORITHMS = [
    "randomalgo",
    "greedy",
    "coordinate_descent",
    "simulated_annealing",
    "iterative_local_search",
    "tpe",
    "cross_entropy",
    "regularized_evolution",
    "mab_ts",
    "mab_ucb",
    "successive_halving",
    "grpo",
    "doctor_grpo",
    "ppo",
    "reinforce_pp",
    "upperbound",
    "thupperbound",
]

NO_SEED_ALGORITHMS = {"greedy", "upperbound", "thupperbound"}


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _build_qasper_outputs(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    qa: List[Dict[str, Any]] = []
    corpus: List[Dict[str, Any]] = []
    for item in items:
        qid = item.get("_id") or item.get("id") or str(len(corpus))
        query = item.get("input") or item.get("question")
        context = item.get("context")
        answers = item.get("answers") or item.get("answer") or item.get("references")
        if not query or not context:
            continue

        corpus.append({"id": str(qid), "content": str(context)})
        if answers is None:
            references: List[str] = []
        elif isinstance(answers, list):
            references = [str(a) for a in answers]
        else:
            references = [str(answers)]
        qa.append({"query": str(query), "references": references})
    return qa, corpus


def _prepare_nested_subsets(
    raw_jsonl: str,
    sizes: Sequence[int],
    subset_seed: int,
    output_root: str,
    dataset_name: str,
) -> List[Dict[str, Any]]:
    items = _read_jsonl(raw_jsonl)
    max_size = max(sizes)
    if len(items) < max_size:
        raise ValueError(
            f"Raw dataset only has {len(items)} items, but max requested size is {max_size}."
        )

    rng = random.Random(subset_seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    generated_root = os.path.join(output_root, "generated_datasets")
    os.makedirs(generated_root, exist_ok=True)
    subsets: List[Dict[str, Any]] = []

    for size in sorted(sizes):
        subset_items = shuffled[:size]
        qa, corpus = _build_qasper_outputs(subset_items)
        subset_dir = os.path.join(generated_root, f"{dataset_name}_{size}")
        os.makedirs(subset_dir, exist_ok=True)
        qa_path = os.path.join(subset_dir, "qa.json")
        corpus_path = os.path.join(subset_dir, "corpus.json")
        with open(qa_path, "w", encoding="utf-8") as handle:
            json.dump(qa, handle, ensure_ascii=False, indent=2)
        with open(corpus_path, "w", encoding="utf-8") as handle:
            json.dump(corpus, handle, ensure_ascii=False, indent=2)
        subsets.append(
            {
                "dataset": dataset_name,
                "size": size,
                "qa_json": qa_path,
                "corpus_json": corpus_path,
                "qa_count": len(qa),
                "corpus_count": len(corpus),
            }
        )
    return subsets


def _build_cmd(
    algorithm: str,
    qa_json: str,
    corpus_json: str,
    config_yaml: str,
    eval_mode: str,
    report_path: str,
    score_weights: str,
    budget: Optional[int],
    seed: Optional[int],
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        f"raisex.search.algorithms.{algorithm}",
        "--qa_json",
        qa_json,
        "--corpus_json",
        corpus_json,
        "--config_yaml",
        config_yaml,
        "--eval_mode",
        eval_mode,
        "--report_path",
        report_path,
    ]
    if score_weights:
        cmd.extend(["--score_weights", score_weights])
    if budget is not None:
        cmd.extend(["--max_evals", str(budget)])
    if seed is not None and algorithm not in NO_SEED_ALGORITHMS:
        cmd.extend(["--seed", str(seed)])
    return cmd


def _load_report(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _extract_best_score(report: Dict[str, Any]) -> Optional[float]:
    if "best_score" in report:
        try:
            return float(report["best_score"])
        except Exception:
            pass
    metrics = (report.get("report") or report.get("best_report") or {}).get("metrics")
    if isinstance(metrics, dict):
        for key in ("LLMAAJ", "BERTScore-F1", "ROUGE-L", "METEOR", "F1", "BLEU"):
            if key in metrics:
                try:
                    return float(metrics[key])
                except Exception:
                    pass
    trials = report.get("trials")
    if isinstance(trials, list):
        scores: List[float] = []
        for item in trials:
            if not isinstance(item, dict):
                continue
            try:
                scores.append(float(item.get("score")))
            except Exception:
                continue
        if scores:
            return max(scores)
    return None


def _extract_metrics(report: Dict[str, Any]) -> Dict[str, float]:
    metrics = (report.get("report") or report.get("best_report") or {}).get("metrics")
    if isinstance(metrics, dict):
        return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}

    trials = report.get("trials")
    if isinstance(trials, list):
        best_trial = None
        best_score = float("-inf")
        for item in trials:
            if not isinstance(item, dict):
                continue
            try:
                score = float(item.get("score"))
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_trial = item
        if best_trial:
            trial_metrics = (best_trial.get("report") or {}).get("metrics")
            if isinstance(trial_metrics, dict):
                return {k: float(v) for k, v in trial_metrics.items() if isinstance(v, (int, float))}
    return {}


def _average_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {}
    keys = sorted({k for m in metrics_list for k in m})
    return {
        key: sum(m[key] for m in metrics_list if key in m) / len([m for m in metrics_list if key in m])
        for key in keys
    }


def _run_single(
    algorithm: str,
    qa_json: str,
    corpus_json: str,
    config_yaml: str,
    eval_mode: str,
    score_weights: str,
    budget: Optional[int],
    seed: Optional[int],
    report_path: str,
    log_path: str,
    overwrite: bool,
    tag: str,
) -> Dict[str, Any]:
    cmd = _build_cmd(
        algorithm=algorithm,
        qa_json=qa_json,
        corpus_json=corpus_json,
        config_yaml=config_yaml,
        eval_mode=eval_mode,
        report_path=report_path,
        score_weights=score_weights,
        budget=budget,
        seed=seed,
    )
    if (not overwrite) and os.path.isfile(report_path):
        report = _load_report(report_path)
        best = _extract_best_score(report)
        metrics = _extract_metrics(report)
        score_str = f"  best_score={best:.4f}" if best is not None else ""
        print(f"  {tag}  SKIPPED (exists){score_str}")
        return {"status": "skipped", "best_score": best, "metrics": metrics}

    print(f"  {tag}  running ...", flush=True)
    t0 = time.time()
    env = os.environ.copy()
    src_path = os.path.join(PROJECT_ROOT, "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing}" if existing else src_path
    env["PYTHONUNBUFFERED"] = "1"

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"[cmd] {' '.join(cmd)}\n\n")
        log_f.flush()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=log_f, text=True)

    elapsed = time.time() - t0
    report = _load_report(report_path)
    best = _extract_best_score(report)
    metrics = _extract_metrics(report)
    status = "ok" if proc.returncode == 0 else "failed"
    score_str = f"  best_score={best:.4f}" if best is not None else ""
    print(f"  {tag}  -> {status}  elapsed={elapsed:.0f}s{score_str}")
    return {
        "status": status,
        "returncode": proc.returncode,
        "elapsed_seconds": round(elapsed, 1),
        "best_score": best,
        "metrics": metrics,
    }


def run(
    algorithms: List[str],
    config_yaml: str,
    eval_mode: str,
    score_weights: str,
    budget: Optional[int],
    seeds: List[int],
    subset_seed: int,
    sizes: List[int],
    output_root: str,
    overwrite: bool,
    dry_run: bool,
    raw_jsonl: str,
    dataset_name: str,
) -> None:
    subsets = _prepare_nested_subsets(
        raw_jsonl=raw_jsonl,
        sizes=sizes,
        subset_seed=subset_seed,
        output_root=output_root,
        dataset_name=dataset_name,
    )

    n_seeds = len(seeds)
    total_runs = len(subsets) * len(algorithms) * n_seeds
    print(
        f"Plan: {len(subsets)} sizes x {len(algorithms)} algorithms x {n_seeds} seeds = {total_runs} runs"
    )
    print(f"Dataset: {dataset_name}")
    print(f"Sizes: {', '.join(str(s['size']) for s in subsets)}")
    print(f"Subset seed: {subset_seed}")
    print(f"Config: {config_yaml}")
    print(f"Output: {output_root}")
    print()

    if dry_run:
        for subset in subsets:
            for alg in algorithms:
                run_seeds = [seeds[0]] if alg in NO_SEED_ALGORITHMS else seeds
                for seed in run_seeds:
                    report_dir = os.path.join(output_root, f"size_{subset['size']}", alg)
                    seed_tag = f"seed_{seed}" if alg not in NO_SEED_ALGORITHMS else "default"
                    report_path = os.path.join(report_dir, f"{seed_tag}_report.json")
                    cmd = _build_cmd(
                        alg,
                        subset["qa_json"],
                        subset["corpus_json"],
                        config_yaml,
                        eval_mode,
                        report_path,
                        score_weights,
                        budget,
                        seed,
                    )
                    print(f"[size={subset['size']}/{alg}/seed={seed}] {' '.join(cmd)}")
            print()
        return

    all_results: List[Dict[str, Any]] = []
    combo_idx = 0
    total_combos = len(subsets) * len(algorithms)
    for subset in subsets:
        size = subset["size"]
        for alg in algorithms:
            combo_idx += 1
            print(f"\n[{combo_idx}/{total_combos}] size={size} / {alg}")
            run_seeds = [seeds[0]] if alg in NO_SEED_ALGORITHMS else seeds
            seed_results: List[Dict[str, Any]] = []
            for seed in run_seeds:
                report_dir = os.path.join(output_root, f"size_{size}", alg)
                os.makedirs(report_dir, exist_ok=True)
                seed_tag = f"seed_{seed}" if alg not in NO_SEED_ALGORITHMS else "default"
                report_path = os.path.join(report_dir, f"{seed_tag}_report.json")
                log_path = os.path.join(report_dir, f"{seed_tag}_run.log")
                result = _run_single(
                    algorithm=alg,
                    qa_json=subset["qa_json"],
                    corpus_json=subset["corpus_json"],
                    config_yaml=config_yaml,
                    eval_mode=eval_mode,
                    score_weights=score_weights,
                    budget=budget,
                    seed=seed,
                    report_path=report_path,
                    log_path=log_path,
                    overwrite=overwrite,
                    tag=f"seed={seed}",
                )
                seed_results.append(result)

            valid_scores = [r["best_score"] for r in seed_results if r.get("best_score") is not None]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
            valid_metrics = [r["metrics"] for r in seed_results if r.get("metrics")]
            avg_metrics = _average_metrics(valid_metrics)
            entry = {
                "dataset": dataset_name,
                "size": size,
                "algorithm": alg,
                "seeds": run_seeds,
                "n_seeds": len(run_seeds),
                "per_seed_scores": [r.get("best_score") for r in seed_results],
                "avg_score": avg_score,
                "avg_metrics": avg_metrics,
                "all_ok": all(r["status"] in ("ok", "skipped") for r in seed_results),
                "qa_json": subset["qa_json"],
                "corpus_json": subset["corpus_json"],
            }
            all_results.append(entry)
            if avg_score is not None:
                per = ", ".join(
                    f"{s:.4f}" if s is not None else "-" for s in entry["per_seed_scores"]
                )
                print(f"  => avg_score={avg_score:.4f}  (per_seed: {per})")

    size_summary_path = os.path.join(output_root, "size_summary.json")
    with open(size_summary_path, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, ensure_ascii=False, indent=2)

    variation_rows: List[Dict[str, Any]] = []
    for alg in algorithms:
        rows = [r for r in all_results if r["algorithm"] == alg and r.get("avg_score") is not None]
        rows.sort(key=lambda item: item["size"])
        scores = [float(r["avg_score"]) for r in rows]
        variation_rows.append(
            {
                "dataset": dataset_name,
                "algorithm": alg,
                "sizes": [r["size"] for r in rows],
                "avg_scores": scores,
                "score_range": (max(scores) - min(scores)) if scores else None,
                "score_std": statistics.pstdev(scores) if len(scores) >= 2 else 0.0,
                "all_ok": all(r["all_ok"] for r in rows) if rows else False,
            }
        )

    variation_json_path = os.path.join(output_root, "variation_summary.json")
    with open(variation_json_path, "w", encoding="utf-8") as handle:
        json.dump(variation_rows, handle, ensure_ascii=False, indent=2)

    metric_keys = sorted({k for r in all_results for k in r.get("avg_metrics", {})})
    csv_path = os.path.join(output_root, "size_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "size", "algorithm", "n_seeds", "avg_score"] + metric_keys)
        for row in all_results:
            writer.writerow(
                [
                    row["dataset"],
                    row["size"],
                    row["algorithm"],
                    row["n_seeds"],
                    "" if row["avg_score"] is None else f"{row['avg_score']:.6f}",
                    *[f"{row['avg_metrics'].get(k, 0.0):.6f}" for k in metric_keys],
                ]
            )

    variation_csv_path = os.path.join(output_root, "variation_summary.csv")
    with open(variation_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "algorithm", "sizes", "avg_scores", "score_range", "score_std", "all_ok"])
        for row in variation_rows:
            writer.writerow(
                [
                    row["dataset"],
                    row["algorithm"],
                    ",".join(str(s) for s in row["sizes"]),
                    ",".join(f"{s:.6f}" for s in row["avg_scores"]),
                    "" if row["score_range"] is None else f"{row['score_range']:.6f}",
                    "" if row["score_std"] is None else f"{row['score_std']:.6f}",
                    row["all_ok"],
                ]
            )

    print("\n===== Size Summary =====")
    header = f"{'Size':>6} {'Algorithm':<25} {'Seeds':>5} {'AvgScore':>10} {'Status':>8}"
    print(header)
    print("-" * len(header))
    for row in all_results:
        score_str = f"{row['avg_score']:.4f}" if row["avg_score"] is not None else "-"
        status_str = "OK" if row["all_ok"] else "FAIL"
        print(f"{row['size']:>6} {row['algorithm']:<25} {row['n_seeds']:>5} {score_str:>10} {status_str:>8}")

    print("\n===== Cross-Size Variation =====")
    vheader = f"{'Algorithm':<25} {'Range':>10} {'Std':>10} {'Sizes':>14}"
    print(vheader)
    print("-" * len(vheader))
    for row in variation_rows:
        range_str = "-" if row["score_range"] is None else f"{row['score_range']:.4f}"
        std_str = "-" if row["score_std"] is None else f"{row['score_std']:.4f}"
        sizes_str = ",".join(str(s) for s in row["sizes"])
        print(f"{row['algorithm']:<25} {range_str:>10} {std_str:>10} {sizes_str:>14}")

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset_name,
        "raw_jsonl": raw_jsonl,
        "sizes": sizes,
        "subset_seed": subset_seed,
        "algorithms": algorithms,
        "budget": budget,
        "seeds": seeds,
        "eval_mode": eval_mode,
        "score_weights": score_weights,
        "config_yaml": config_yaml,
        "generated_subsets": subsets,
        "size_summary_json": size_summary_path,
        "variation_summary_json": variation_json_path,
    }
    manifest_path = os.path.join(output_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"\nManifest: {manifest_path}")
    print(f"JSON report: {size_summary_path}")
    print(f"CSV report: {csv_path}")
    print(f"Variation JSON: {variation_json_path}")
    print(f"Variation CSV: {variation_csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run QA-size ablation on one text dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--algorithms",
        default="all",
        help='Comma-separated algorithm names or "all" for all 17.',
    )
    parser.add_argument("--config_yaml", default=DEFAULT_CONFIG, help="Algorithm config YAML.")
    parser.add_argument("--budget", type=int, default=None, help="Search budget per algorithm.")
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated random seeds.")
    parser.add_argument("--subset_seed", type=int, default=42, help="Seed for generating nested subsets.")
    parser.add_argument("--sizes", default="20,50,100", help="Comma-separated QA subset sizes.")
    parser.add_argument("--eval_mode", default="avg", choices=["avg", "per_item", "both"])
    parser.add_argument(
        "--score_weights",
        default="rougel1.0,meteor1.0,f11.0,bleu1.0",
        help="Weighted scoring string for optimization objective.",
    )
    parser.add_argument("--raw_jsonl", default=RAW_QASPER_JSONL, help="Raw JSONL file used to build subsets.")
    parser.add_argument("--dataset_name", default="longbench-qasper", help="Dataset label used in outputs.")
    parser.add_argument(
        "--output_root",
        default=os.path.join(PROJECT_ROOT, "outputs-qa-size-ablation"),
        help="Output directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if report exists.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only.")
    args = parser.parse_args()

    if args.algorithms.strip().lower() == "all":
        algorithms = list(ALL_ALGORITHMS)
    else:
        algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    unknown = [a for a in algorithms if a not in ALL_ALGORITHMS]
    if unknown:
        print(f"ERROR: Unknown algorithms: {', '.join(unknown)}")
        sys.exit(1)

    sizes = sorted({int(s.strip()) for s in args.sizes.split(",") if s.strip()})
    if not sizes or any(s <= 0 for s in sizes):
        print("ERROR: --sizes must contain positive integers.")
        sys.exit(1)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not os.path.isfile(args.raw_jsonl):
        print(f"ERROR: Raw JSONL not found: {args.raw_jsonl}")
        sys.exit(1)
    config_yaml = os.path.abspath(args.config_yaml)
    if not os.path.isfile(config_yaml):
        print(f"ERROR: Config not found: {config_yaml}")
        sys.exit(1)

    run(
        algorithms=algorithms,
        config_yaml=config_yaml,
        eval_mode=args.eval_mode,
        score_weights=args.score_weights,
        budget=args.budget,
        seeds=seeds,
        subset_seed=args.subset_seed,
        sizes=sizes,
        output_root=os.path.abspath(args.output_root),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        raw_jsonl=os.path.abspath(args.raw_jsonl),
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
