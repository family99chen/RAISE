#!/usr/bin/env python3
import argparse
import fcntl
import json
import math
import os
import re
import selectors
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_BUDGETS = [120, 300, 600]
DEFAULT_SEEDS = [11, 22, 33]
DEFAULT_SCORE_WEIGHTS = "rougel1.0,meteor1.0,f11.0,bleu1.0"
DEFAULT_DATASETS = [
    ("triviaqa", "data/datasets/triviaqa/qa.json", "data/datasets/triviaqa/corpus.json"),
    ("scienceqa", "data/datasets/scienceqa/qa.json", "data/datasets/scienceqa/corpus.json"),
    ("longbench-qasper", "data/datasets/longbench-qasper/qa.json", "data/datasets/longbench-qasper/corpus.json"),
    ("longbench-multifield", "data/datasets/longbench-multifield/qa.json", "data/datasets/longbench-multifield/corpus.json"),
]


@dataclass
class RunSpec:
    dataset: str
    algorithm: str
    budget: Optional[int]
    seed: int
    report_path: str
    log_path: str
    cmd: List[str]


def _parse_int_list(text: str) -> List[int]:
    items = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _budget_tag(budget: Optional[int]) -> str:
    if budget is None:
        return "full"
    return f"budget_{budget}"


def _build_grpo_cmd(
    python_exec: str,
    qa_json: str,
    corpus_json: str,
    config_yaml: str,
    eval_mode: str,
    score_weights: str,
    budget: int,
    seed: int,
    report_path: str,
    group_size: int = 8,
) -> List[str]:
    episodes = int(math.ceil(float(budget) / float(group_size)))
    return [
        python_exec,
        "-m", "raisex.cli.algo_cli", "--algorithm", "grpo",
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
        "--seed",
        str(seed),
        "--group_size",
        str(group_size),
        "--episodes",
        str(episodes),
        "--score_weights",
        score_weights,
    ]


def _build_ablation_overrides(name: str) -> List[str]:
    if name == "no_robust_adv":
        return ["--winsor_q", "0.0", "--adv_clip", "1000000"]
    if name == "no_ref_ema":
        return ["--ref_update_interval", "1000000", "--ref_ema_alpha", "0.0"]
    if name == "no_elite_replay":
        return [
            "--elite_k",
            "0",
            "--elite_mix_ratio",
            "0.0",
            "--elite_mix_ratio_end",
            "0.0",
        ]
    if name == "single_reward":
        return ["--reward_mode", "single"]
    raise ValueError(f"Unknown ablation: {name}")


def _build_run_specs(
    python_exec: str,
    dataset: str,
    qa_json: str,
    corpus_json: str,
    config_by_budget: Dict[int, str],
    eval_mode: str,
    score_weights: str,
    output_root: str,
    budgets: Sequence[int],
    seeds: Sequence[int],
    run_ablations: bool,
) -> List[RunSpec]:
    specs: List[RunSpec] = []
    base_algs = ["tpe", "mab_ts", "mab_ucb", "grpo"]

    for budget in budgets:
        config_yaml = config_by_budget.get(budget)
        if not config_yaml:
            raise ValueError(f"Missing config for budget {budget}")
        for seed in seeds:
            root = os.path.join(output_root, "results")
            for alg in base_algs:
                report_path = os.path.join(
                    root,
                    alg,
                    _budget_tag(budget),
                    f"seed_{seed}",
                    "report.json",
                )
                log_path = os.path.join(
                    root,
                    alg,
                    _budget_tag(budget),
                    f"seed_{seed}",
                    "run.log",
                )
                _ensure_dir(os.path.dirname(report_path))
                if alg == "tpe":
                    cmd = [
                        python_exec,
                        "-m", "raisex.cli.algo_cli", "--algorithm", "tpe",
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
                        "--samples",
                        str(budget),
                        "--seed",
                        str(seed),
                        "--startup_trials",
                        "20",
                        "--gamma",
                        "0.2",
                        "--candidate_pool_size",
                        "24",
                        "--score_weights",
                        score_weights,
                    ]
                elif alg == "mab_ts":
                    pool_size = max(1000, 5 * int(budget))
                    cmd = [
                        python_exec,
                        "-m", "raisex.cli.algo_cli", "--algorithm", "mab_ts",
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
                        "--budget",
                        str(budget),
                        "--pool_size",
                        str(pool_size),
                        "--seed",
                        str(seed),
                        "--score_weights",
                        score_weights,
                    ]
                elif alg == "mab_ucb":
                    pool_size = max(1000, 5 * int(budget))
                    cmd = [
                        python_exec,
                        "-m", "raisex.cli.algo_cli", "--algorithm", "mab_ucb",
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
                        "--budget",
                        str(budget),
                        "--pool_size",
                        str(pool_size),
                        "--seed",
                        str(seed),
                        "--score_weights",
                        score_weights,
                    ]
                else:
                    cmd = _build_grpo_cmd(
                        python_exec=python_exec,
                        qa_json=qa_json,
                        corpus_json=corpus_json,
                        config_yaml=config_yaml,
                        eval_mode=eval_mode,
                        score_weights=score_weights,
                        budget=budget,
                        seed=seed,
                        report_path=report_path,
                    )
                specs.append(
                    RunSpec(
                        dataset=dataset,
                        algorithm=alg,
                        budget=budget,
                        seed=seed,
                        report_path=report_path,
                        log_path=log_path,
                        cmd=cmd,
                    )
                )

            if run_ablations and budget == max(budgets):
                for ablation_name in (
                    "no_robust_adv",
                    "no_ref_ema",
                    "no_elite_replay",
                    "single_reward",
                ):
                    report_path = os.path.join(
                        root,
                        "grpo_ablation",
                        ablation_name,
                        _budget_tag(budget),
                        f"seed_{seed}",
                        "report.json",
                    )
                    log_path = os.path.join(
                        root,
                        "grpo_ablation",
                        ablation_name,
                        _budget_tag(budget),
                        f"seed_{seed}",
                        "run.log",
                    )
                    _ensure_dir(os.path.dirname(report_path))
                    cmd = _build_grpo_cmd(
                        python_exec=python_exec,
                        qa_json=qa_json,
                        corpus_json=corpus_json,
                        config_yaml=config_yaml,
                        eval_mode=eval_mode,
                        score_weights=score_weights,
                        budget=budget,
                        seed=seed,
                        report_path=report_path,
                    )
                    cmd.extend(_build_ablation_overrides(ablation_name))
                    specs.append(
                        RunSpec(
                            dataset=dataset,
                            algorithm=f"grpo_ablation:{ablation_name}",
                            budget=budget,
                            seed=seed,
                            report_path=report_path,
                            log_path=log_path,
                            cmd=cmd,
                        )
                    )

    return specs


def _run_one(spec: RunSpec, workdir: str, overwrite: bool) -> Dict[str, object]:
    def _mask_secrets(text: str) -> str:
        text = re.sub(r'("api_key"\s*:\s*")([^"]+)(")', r"\1***\3", text)
        text = re.sub(r"(api_key\s*:\s*\")([^\"]+)(\")", r"\1***\3", text)
        text = re.sub(r"(api_key\s*:\s*)(\S+)", r"\1***", text)
        return text

    start = time.time()
    if (not overwrite) and os.path.isfile(spec.report_path):
        return {
            "dataset": spec.dataset,
            "algorithm": spec.algorithm,
            "budget": spec.budget,
            "seed": spec.seed,
            "report_path": spec.report_path,
            "log_path": spec.log_path,
            "status": "skipped_exists",
            "elapsed_seconds": 0.0,
            "returncode": 0,
            "cmd": spec.cmd,
        }

    _ensure_dir(os.path.dirname(spec.log_path))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    with open(spec.log_path, "w", encoding="utf-8") as log_handle:
        log_handle.write(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_handle.write(f"[cmd] {' '.join(spec.cmd)}\n\n")
        log_handle.flush()
        process = subprocess.Popen(
            spec.cmd,
            cwd=workdir,
            text=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=0,
        )
        selector = selectors.DefaultSelector()
        returncode: Optional[int] = None
        last_heartbeat = 0.0
        heartbeat_interval = 20.0
        buffer = b""

        if process.stdout is not None:
            fd = process.stdout.fileno()
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            selector.register(process.stdout, selectors.EVENT_READ)

        while True:
            if returncode is None:
                returncode = process.poll()

            events = selector.select(timeout=1.0)
            for key, _mask in events:
                try:
                    chunk = os.read(key.fileobj.fileno(), 8192)
                except BlockingIOError:
                    chunk = b""
                if not chunk:
                    continue
                buffer += chunk
                try:
                    text = buffer.decode("utf-8", errors="replace")
                    buffer = b""
                except Exception:
                    text = ""
                if text:
                    text = _mask_secrets(text)
                    log_handle.write(text)
                    log_handle.flush()
                    print(text, end="")

            now = time.time()
            if now - last_heartbeat >= heartbeat_interval:
                last_heartbeat = now
                trial_progress = ""
                if os.path.isfile(spec.report_path):
                    try:
                        with open(spec.report_path, "r", encoding="utf-8") as handle:
                            report = json.load(handle) or {}
                        trials = report.get("trials")
                        if isinstance(trials, list):
                            trial_progress = f" trials={len(trials)}"
                            if trials:
                                last = trials[-1] if isinstance(trials[-1], dict) else {}
                                metric = last.get("metric")
                                score = last.get("score")
                                if metric is not None or score is not None:
                                    trial_progress += f" last_metric={metric} last_score={score}"
                    except Exception:
                        trial_progress = ""
                heartbeat_line = (
                    f"[heartbeat] {time.strftime('%Y-%m-%d %H:%M:%S')} elapsed_seconds={now - start:.1f}"
                    f"{trial_progress}\n"
                )
                log_handle.write(heartbeat_line)
                log_handle.flush()
                print(heartbeat_line, end="")

            if returncode is not None:
                if process.stdout is None:
                    break
                try:
                    remaining = os.read(process.stdout.fileno(), 8192)
                except Exception:
                    remaining = b""
                if remaining:
                    buffer += remaining
                    text = buffer.decode("utf-8", errors="replace")
                    buffer = b""
                    if text:
                        text = _mask_secrets(text)
                        log_handle.write(text)
                        log_handle.flush()
                        print(text, end="")
                    continue
                break

        if buffer:
            text = buffer.decode("utf-8", errors="replace")
            text = _mask_secrets(text)
            log_handle.write(text)
            log_handle.flush()
            print(text, end="")

        if returncode is None:
            returncode = process.wait()
    elapsed = time.time() - start
    with open(spec.log_path, "a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[end] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_handle.write(f"[returncode] {returncode}\n")
    return {
        "dataset": spec.dataset,
        "algorithm": spec.algorithm,
        "budget": spec.budget,
        "seed": spec.seed,
        "report_path": spec.report_path,
        "log_path": spec.log_path,
        "status": "ok" if returncode == 0 else "failed",
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "cmd": spec.cmd,
    }


def _write_manifest(path: str, payload: Dict[str, object]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fair-comparison experiments for GRPO/TPE/MAB-TS/MAB-UCB in raise."
    )
    parser.add_argument("--qa_json", help="Path to QA JSON/JSONL.")
    parser.add_argument("--corpus_json", help="Path to corpus JSON.")
    parser.add_argument("--config_yaml", help="Path to search config YAML.")
    parser.add_argument(
        "--output_root",
        default="outputs-experiments",
        help="Root directory to save run artifacts.",
    )
    parser.add_argument(
        "--budgets",
        default="120,300,600",
        help="Comma-separated evaluation budgets.",
    )
    parser.add_argument(
        "--seeds",
        default="11,22,33",
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--eval_mode",
        default="both",
        choices=["avg", "per_item", "both"],
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--score_weights",
        default=DEFAULT_SCORE_WEIGHTS,
        help="Unified objective weights string.",
    )
    parser.add_argument(
        "--python_exec",
        default=sys.executable,
        help="Python executable used to launch algorithms.",
    )
    parser.add_argument(
        "--datasets",
        default="triviaqa,scienceqa,longbench-qasper,longbench-multifield",
        help="Comma-separated dataset names to run when qa/corpus are not provided.",
    )
    parser.add_argument(
        "--run_ablations",
        action="store_true",
        help="Run GRPO ablations at max budget.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run even if report already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    workdir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_root = os.path.abspath(os.path.join(workdir, args.output_root))
    budgets = _parse_int_list(args.budgets)
    seeds = _parse_int_list(args.seeds)
    if args.config_yaml:
        config_yaml = args.config_yaml
        if not os.path.isabs(config_yaml):
            config_yaml = os.path.abspath(os.path.join(workdir, config_yaml))
        config_by_budget = {budget: config_yaml for budget in budgets}
    else:
        config_by_budget = {
            120: os.path.join(workdir, "configs", "algorithms", "default.yaml"),
            300: os.path.join(workdir, "configs", "experiments", "configforalgo_1k.yaml"),
            600: os.path.join(workdir, "configs", "experiments", "configforalgo_10k.yaml"),
        }
    config_by_budget = {k: os.path.abspath(v) for k, v in config_by_budget.items()}

    datasets = []
    if args.qa_json and args.corpus_json:
        datasets.append(("custom", args.qa_json, args.corpus_json))
    else:
        name_set = {name.strip() for name in args.datasets.split(",") if name.strip()}
        for name, qa_path, corpus_path in DEFAULT_DATASETS:
            if name in name_set:
                datasets.append((name, os.path.join(workdir, qa_path), os.path.join(workdir, corpus_path)))
        if not datasets:
            raise SystemExit("No datasets selected. Provide --qa_json/--corpus_json or set --datasets.")

    specs: List[RunSpec] = []
    for name, qa_path, corpus_path in datasets:
        dataset_root = os.path.join(output_root, name)
        specs.extend(
            _build_run_specs(
                python_exec=args.python_exec,
                dataset=name,
                qa_json=qa_path,
                corpus_json=corpus_path,
                config_by_budget=config_by_budget,
                eval_mode=args.eval_mode,
                score_weights=args.score_weights,
                output_root=dataset_root,
                budgets=budgets,
                seeds=seeds,
                run_ablations=args.run_ablations,
            )
        )

    manifest = {
        "created_at_epoch": time.time(),
        "workdir": workdir,
        "datasets": [
            {"name": name, "qa_json": qa_path, "corpus_json": corpus_path}
            for name, qa_path, corpus_path in datasets
        ],
        "config_by_budget": config_by_budget,
        "eval_mode": args.eval_mode,
        "score_weights": args.score_weights,
        "budgets": budgets,
        "seeds": seeds,
        "run_ablations": bool(args.run_ablations),
        "overwrite": bool(args.overwrite),
        "runs": [],
    }

    if args.dry_run:
        for spec in specs:
            print(" ".join(spec.cmd))
        return

    for idx, spec in enumerate(specs, start=1):
        btag = _budget_tag(spec.budget)
        print(
            f"[{idx}/{len(specs)}] {spec.algorithm} {btag} seed={spec.seed} log={spec.log_path}"
        )
        result = _run_one(spec=spec, workdir=workdir, overwrite=args.overwrite)
        manifest["runs"].append(result)
        if result["status"] == "failed":
            print("  -> failed")
        else:
            print(f"  -> {result['status']}")
        manifest_path = os.path.join(output_root, "run_manifest.json")
        _write_manifest(manifest_path, manifest)

    print(f"Manifest written to: {os.path.join(output_root, 'run_manifest.json')}")


if __name__ == "__main__":
    main()
