import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _project_src() -> Path:
    return _project_root() / "src"


def _parse_algorithms(single: str | None, multiple: str | None) -> List[str]:
    candidates: List[str] = []
    if single:
        candidates.append(single)
    if multiple:
        candidates.extend(part.strip() for part in multiple.split(","))

    normalized: List[str] = []
    seen = set()
    for item in candidates:
        if not item:
            continue
        name = item[:-3] if item.endswith(".py") else item
        if name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


def _strip_report_path_args(args: List[str]) -> List[str]:
    cleaned: List[str] = []
    skip_next = False
    for token in args:
        if skip_next:
            skip_next = False
            continue
        if token == "--report_path":
            skip_next = True
            continue
        if token.startswith("--report_path="):
            continue
        cleaned.append(token)
    return cleaned


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _extract_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("report", {}).get("metrics")
    if isinstance(metrics, dict):
        return metrics

    best_metrics = payload.get("best_report", {}).get("metrics")
    if isinstance(best_metrics, dict):
        return best_metrics

    top_metrics = payload.get("metrics")
    if isinstance(top_metrics, dict):
        return top_metrics

    trials = payload.get("trials")
    if isinstance(trials, list):
        best_trial: Dict[str, Any] | None = None
        best_score = float("-inf")
        for item in trials:
            if not isinstance(item, dict):
                continue
            report = item.get("report")
            trial_metrics = report.get("metrics") if isinstance(report, dict) else None
            if not isinstance(trial_metrics, dict):
                continue
            score = item.get("score")
            try:
                numeric_score = float(score)
            except Exception:
                numeric_score = float("-inf")
            if best_trial is None or numeric_score > best_score:
                best_trial = item
                best_score = numeric_score
        if isinstance(best_trial, dict):
            report = best_trial.get("report")
            if isinstance(report, dict) and isinstance(report.get("metrics"), dict):
                return report["metrics"]

    return {}


def _run_one_algorithm(algorithm: str, passthrough_args: List[str], verbose: bool) -> Tuple[Dict[str, Any], bool]:
    spec = importlib.util.find_spec(f"raisex.search.algorithms.{algorithm}")
    if spec is None:
        return (
            {
                "algorithm": algorithm,
                "status": "failed",
                "returncode": 127,
                "metrics": {},
                "error": "module_not_found",
            },
            True,
        )

    fd, report_path = tempfile.mkstemp(prefix=f"raise-{algorithm}-", suffix=".json")
    os.close(fd)

    cmd = [
        sys.executable,
        "-m",
        f"raisex.search.algorithms.{algorithm}",
        *_strip_report_path_args(passthrough_args),
        "--report_path",
        report_path,
    ]

    env = os.environ.copy()
    src_path = str(_project_src())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing}" if existing else src_path

    completed = subprocess.run(cmd, capture_output=True, text=True, env=env)
    report_payload = _load_json(report_path)
    metrics = _extract_metrics(report_payload)

    failed = completed.returncode != 0 or not metrics
    error = ""
    if completed.returncode != 0:
        error = "non_zero_exit"
    elif not metrics:
        error = "metrics_not_found"

    result: Dict[str, Any] = {
        "algorithm": algorithm,
        "status": "failed" if failed else "ok",
        "returncode": completed.returncode,
        "metrics": metrics,
        "error": error,
    }
    if verbose:
        result["report_path"] = report_path
        result["stdout"] = completed.stdout
        result["stderr"] = completed.stderr
    else:
        try:
            os.remove(report_path)
        except OSError:
            pass

    return result, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run raise algorithms and summarize metrics")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--algorithm", help="Single algorithm module name, e.g. randomalgo")
    group.add_argument(
        "--algorithms",
        help="Comma-separated algorithm module names, e.g. randomalgo,greedy",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include per-algorithm stdout/stderr and report path in output.",
    )
    args, unknown = parser.parse_known_args()

    algorithms = _parse_algorithms(args.algorithm, args.algorithms)
    if not algorithms:
        print(
            json.dumps(
                [{"algorithm": "", "status": "failed", "returncode": 2, "metrics": {}, "error": "no_algorithm"}],
                ensure_ascii=False,
                indent=2,
            )
        )
        sys.exit(2)

    any_failed = False
    results: List[Dict[str, Any]] = []
    for algorithm in algorithms:
        entry, failed = _run_one_algorithm(algorithm, unknown, args.verbose)
        results.append(entry)
        any_failed = any_failed or failed

    print(json.dumps(results, ensure_ascii=False, indent=2))
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
