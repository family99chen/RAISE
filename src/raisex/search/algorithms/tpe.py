import json
import math
import os
import random
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

if os.getenv("PIPELINE_DEBUG") == "1" or os.getenv("EVAL_DEBUG") == "1":
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

from raisex.core.evaluation_service import evaluate_rag, evaluate_rag_multimodal

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = handle.read().strip()
    if not data:
        return {}
    import yaml

    parsed = yaml.safe_load(data) or {}
    return parsed if isinstance(parsed, dict) else {}


def _dump_yaml(data: Dict[str, Any], path: str) -> None:
    import yaml

    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def _allowed_values(node: Any) -> List[Any]:
    if node is None:
        return []
    if isinstance(node, list):
        return node
    if not isinstance(node, dict):
        return [node]
    allowed = node.get("allowed")
    if not isinstance(allowed, list):
        return []
    return [v for v in allowed if v != "..."]


def _split_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    search_space = config.get("rag_search_space") or {}
    eval_metrics = config.get("eval_metrics")
    algo_cfg = {
        key: value
        for key, value in config.items()
        if key not in {"rag_search_space", "eval_metrics"}
    }
    return search_space, algo_cfg, eval_metrics


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict):
            current = merged.get(key)
            if not isinstance(current, dict):
                current = {}
            merged_child = _deep_update(current, value)
            if merged_child:
                merged[key] = merged_child
            else:
                merged.pop(key, None)
            continue
        if isinstance(value, list):
            # Lists in algo config are candidate pools, not fixed values.
            continue
        merged[key] = value
    return merged


def _override_choices(
    module: str, key: str, algo_cfg: Dict[str, Any]
) -> Optional[List[Any]]:
    if not isinstance(algo_cfg, dict):
        return None
    section = algo_cfg.get(module)
    if not isinstance(section, dict) or key not in section:
        return None
    value = section.get(key)
    if isinstance(value, list):
        return value
    if value is None:
        return None
    return [value]


def _parse_score_weights(text: str) -> Optional[Dict[str, float]]:
    if not text:
        return None
    name_map = {
        "llmaaj": "LLMAAJ",
        "bertf1": "BERTScore-F1",
        "bert": "BERTScore-F1",
        "rougel": "ROUGE-L",
        "meteor": "METEOR",
        "f1": "F1",
        "bleu": "BLEU",
        "exactmatch": "ExactMatch",
        "em": "ExactMatch",
    }
    weights: Dict[str, float] = {}
    for raw in text.split(","):
        part = raw.strip().lower()
        if not part:
            continue
        idx = len(part)
        while idx > 0 and (part[idx - 1].isdigit() or part[idx - 1] == "."):
            idx -= 1
        if idx == len(part):
            continue
        name = part[:idx]
        weight_str = part[idx:]
        metric_key = name_map.get(name)
        if not metric_key:
            continue
        try:
            weight = float(weight_str)
        except Exception:
            continue
        weights[metric_key] = weight
    return weights or None


def _score_from_report(
    report: Dict[str, Any],
    preferred: Optional[str],
    weights: Optional[Dict[str, float]],
) -> Tuple[str, float]:
    metrics = report.get("metrics") or {}
    if weights:
        total = 0.0
        denom = 0.0
        for key, weight in weights.items():
            if key not in metrics:
                continue
            try:
                total += float(metrics[key]) * weight
                denom += weight
            except Exception:
                continue
        return "weighted", (total / denom) if denom > 0 else 0.0
    if preferred and preferred in metrics:
        try:
            return preferred, float(metrics[preferred])
        except Exception:
            return preferred, 0.0
    for name in ("LLMAAJ", "BERTScore-F1", "ROUGE-L", "METEOR", "F1", "BLEU"):
        if name in metrics:
            try:
                return name, float(metrics[name])
            except Exception:
                return name, 0.0
    return "LLMAAJ", 0.0


def _sanitize_selection(selection: Dict[str, Any]) -> None:
    chunking = selection.get("chunking")
    if isinstance(chunking, dict):
        chunking.pop("model_url", None)
        chunking.pop("model_name", None)


def _sanitize_selection_for_log(selection: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = json.loads(json.dumps(selection))

    def _sanitize_node(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("api_key", None)
            node.pop("apikey", None)
            for value in node.values():
                _sanitize_node(value)
        elif isinstance(node, list):
            for value in node:
                _sanitize_node(value)

    chunking = sanitized.get("chunking")
    if isinstance(chunking, dict):
        chunking.pop("model_url", None)
        chunking.pop("model_name", None)
    _sanitize_node(sanitized)
    return sanitized


def _write_temp_selection(selection: Dict[str, Any]) -> str:
    fd, path = tempfile.mkstemp(prefix="tpe_selection_", suffix=".yaml")
    os.close(fd)
    _dump_yaml(selection, path)
    return path


def _is_multimodal(search_space: Dict[str, Any], algo_cfg: Dict[str, Any]) -> bool:
    if isinstance(search_space, dict) and "clip" in search_space:
        return True
    return isinstance(algo_cfg, dict) and "clip" in algo_cfg


def _set_eval_schema_env(config_path: str, use_multimodal: bool) -> None:
    if use_multimodal:
        os.environ["RAGSEARCH_CONFIG_MULTIMODAL"] = config_path
    else:
        os.environ["RAGSEARCH_CONFIG"] = config_path


def _paired_model_choices(
    params: Dict[str, Any], algo_cfg: Dict[str, Any], module: str
) -> Optional[List[Tuple[Any, Any]]]:
    if not isinstance(params, dict):
        return None
    url_override = _override_choices(module, "model_url", algo_cfg)
    name_override = _override_choices(module, "model_name", algo_cfg)
    url_choices = _allowed_values(params.get("model_url"))
    if url_override:
        url_choices = url_override
    name_choices = _allowed_values(params.get("model_name"))
    if name_override:
        name_choices = name_override
    if not url_choices or not name_choices:
        return None
    if len(url_choices) != len(name_choices):
        return None
    return list(zip(url_choices, name_choices))


def _evaluate_selection(
    qa_json_path: str,
    corpus_json_path: str,
    selection: Dict[str, Any],
    eval_mode: str,
    preferred_metric: Optional[str],
    score_weights: Optional[Dict[str, float]],
    eval_fn,
) -> Tuple[float, Dict[str, Any]]:
    selection_path = _write_temp_selection(selection)
    try:
        result = eval_fn(
            qa_json_path=qa_json_path,
            corpus_json_path=corpus_json_path,
            config_path=selection_path,
            eval_mode=eval_mode,
        )
    finally:
        os.remove(selection_path)
    report = result.get("eval_report") or {}
    metric_name, score = _score_from_report(report, preferred_metric, score_weights)
    return score, {
        "metric": metric_name,
        "score": score,
        "report": report,
        "pipeline_total_time_seconds": report.get("pipeline_total_time_seconds"),
        "outputs": result.get("outputs"),
        "chunking": result.get("chunking"),
        "error": result.get("error"),
        "errors": result.get("errors"),
    }


def _module_forced_on(algo_cfg: Dict[str, Any], module: str) -> bool:
    if not isinstance(algo_cfg, dict):
        return False
    section = algo_cfg.get(module)
    return isinstance(section, dict) and len(section) > 0


def _random_selection(
    search_space: Dict[str, Any],
    algo_cfg: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    selection: Dict[str, Any] = {}
    for module, params in search_space.items():
        if not isinstance(params, dict):
            continue
        is_optional = module in {"rewriter", "reranker", "pruner"}
        if is_optional and not _module_forced_on(algo_cfg, module) and rng.random() < 0.5:
            continue

        selection[module] = {}
        pair_choices = _paired_model_choices(params, algo_cfg, module)
        if pair_choices:
            choice = rng.choice(pair_choices)
            selection[module]["model_url"] = choice[0]
            selection[module]["model_name"] = choice[1]
        for key, value in params.items():
            if pair_choices and key in {"model_url", "model_name"}:
                continue
            choices = _allowed_values(value)
            override = _override_choices(module, key, algo_cfg)
            if override:
                choices = override
            if choices:
                selection[module][key] = rng.choice(choices)
        if not selection[module]:
            selection.pop(module, None)
    return selection


def _build_param_specs(
    search_space: Dict[str, Any],
    algo_cfg: Dict[str, Any],
    module_order: List[str],
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for module in module_order:
        params = search_space.get(module)
        if not isinstance(params, dict):
            continue
        is_optional = module in {"rewriter", "reranker", "pruner"}
        forced_on = _module_forced_on(algo_cfg, module)
        if is_optional and not forced_on:
            specs.append(
                {
                    "name": f"{module}.__enabled__",
                    "module": module,
                    "key": "__enabled__",
                    "choices": [True, False],
                    "is_enable": True,
                }
            )
        pair_choices = _paired_model_choices(params, algo_cfg, module)
        if pair_choices:
            specs.append(
                {
                    "name": f"{module}.__model_pair__",
                    "module": module,
                    "key": "__model_pair__",
                    "choices": pair_choices,
                    "is_enable": False,
                }
            )
        for key, value in params.items():
            if pair_choices and key in {"model_url", "model_name"}:
                continue
            choices = _allowed_values(value)
            override = _override_choices(module, key, algo_cfg)
            if override:
                choices = override
            if not choices:
                continue
            specs.append(
                {
                    "name": f"{module}.{key}",
                    "module": module,
                    "key": key,
                    "choices": choices,
                    "is_enable": False,
                }
            )
    return specs


def _extract_param_values(
    selection: Dict[str, Any], specs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for spec in specs:
        module = spec["module"]
        if spec["is_enable"]:
            values[spec["name"]] = module in selection
            continue
        if module in selection and isinstance(selection[module], dict):
            if spec["key"] == "__model_pair__":
                url = selection[module].get("model_url")
                name = selection[module].get("model_name")
                values[spec["name"]] = (url, name) if url is not None and name is not None else None
            else:
                values[spec["name"]] = selection[module].get(spec["key"])
        else:
            values[spec["name"]] = None
    return values


def _choice_weights(
    choices: List[Any],
    good_counts: Dict[Any, int],
    bad_counts: Dict[Any, int],
    n_good: int,
    n_bad: int,
    alpha: float,
) -> List[float]:
    weights: List[float] = []
    for choice in choices:
        p_good = (good_counts.get(choice, 0) + alpha) / (n_good + alpha * len(choices))
        if n_bad > 0:
            p_bad = (bad_counts.get(choice, 0) + alpha) / (n_bad + alpha * len(choices))
        else:
            p_bad = 1.0 / len(choices)
        weights.append(p_good / p_bad if p_bad > 0 else p_good)
    return weights


def _sample_choice(rng: random.Random, choices: List[Any], weights: List[float]) -> Any:
    total = sum(weights)
    if total <= 0:
        return rng.choice(choices)
    r = rng.random() * total
    upto = 0.0
    for choice, weight in zip(choices, weights):
        upto += weight
        if r <= upto:
            return choice
    return choices[-1]


def _build_tpe_counts(
    trials: List[Dict[str, Any]],
    specs: List[Dict[str, Any]],
    gamma: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[Any, int]], Dict[str, Dict[Any, int]]]:
    if not trials:
        return [], [], {}, {}
    sorted_trials = sorted(
        trials, key=lambda t: t.get("score", float("-inf")), reverse=True
    )
    n_good = max(1, int(len(sorted_trials) * gamma))
    good = sorted_trials[:n_good]
    bad = sorted_trials[n_good:]

    good_counts: Dict[str, Dict[Any, int]] = {}
    bad_counts: Dict[str, Dict[Any, int]] = {}
    for trial in good:
        values = _extract_param_values(trial.get("selection", {}), specs)
        for spec in specs:
            name = spec["name"]
            val = values.get(name)
            if val is None:
                continue
            good_counts.setdefault(name, {})
            good_counts[name][val] = good_counts[name].get(val, 0) + 1
    for trial in bad:
        values = _extract_param_values(trial.get("selection", {}), specs)
        for spec in specs:
            name = spec["name"]
            val = values.get(name)
            if val is None:
                continue
            bad_counts.setdefault(name, {})
            bad_counts[name][val] = bad_counts[name].get(val, 0) + 1

    return good, bad, good_counts, bad_counts


def _sample_values_from_counts(
    specs: List[Dict[str, Any]],
    rng: random.Random,
    good_counts: Dict[str, Dict[Any, int]],
    bad_counts: Dict[str, Dict[Any, int]],
    n_good: int,
    n_bad: int,
    alpha: float,
) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    disabled_modules: set[str] = set()
    for spec in specs:
        module = spec["module"]
        if module in disabled_modules and not spec["is_enable"]:
            continue
        choices = spec["choices"]
        weights = _choice_weights(
            choices,
            good_counts.get(spec["name"], {}),
            bad_counts.get(spec["name"], {}),
            n_good,
            n_bad,
            alpha,
        )
        picked = _sample_choice(rng, choices, weights)
        values[spec["name"]] = picked
        if spec["is_enable"] and picked is False:
            disabled_modules.add(module)
    return values


def _surrogate_score_from_counts(
    values: Dict[str, Any],
    specs: List[Dict[str, Any]],
    good_counts: Dict[str, Dict[Any, int]],
    bad_counts: Dict[str, Dict[Any, int]],
    n_good: int,
    n_bad: int,
    alpha: float,
) -> float:
    score = 0.0
    disabled_modules: set[str] = set()
    for spec in specs:
        module = spec["module"]
        if spec["is_enable"]:
            enabled = values.get(spec["name"])
            if enabled is False:
                disabled_modules.add(module)
            continue
        if module in disabled_modules:
            continue
        val = values.get(spec["name"])
        if val is None:
            continue
        choices = spec["choices"]
        p_good = (good_counts.get(spec["name"], {}).get(val, 0) + alpha) / (
            n_good + alpha * len(choices)
        )
        if n_bad > 0:
            p_bad = (bad_counts.get(spec["name"], {}).get(val, 0) + alpha) / (
                n_bad + alpha * len(choices)
            )
        else:
            p_bad = 1.0 / len(choices)
        ratio = p_good / p_bad if p_bad > 0 else p_good
        if ratio > 0:
            score += math.log(ratio)
    return score


def _sample_tpe_selection(
    trials: List[Dict[str, Any]],
    specs: List[Dict[str, Any]],
    rng: random.Random,
    gamma: float,
    alpha: float,
) -> Dict[str, Any]:
    good, bad, good_counts, bad_counts = _build_tpe_counts(trials, specs, gamma)
    if not good:
        return {}
    return _sample_values_from_counts(
        specs, rng, good_counts, bad_counts, len(good), len(bad), alpha
    )


def _build_selection_from_values(
    values: Dict[str, Any], specs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    selection: Dict[str, Any] = {}
    disabled_modules: set[str] = set()
    for spec in specs:
        module = spec["module"]
        if spec["is_enable"]:
            if values.get(spec["name"]) is False:
                disabled_modules.add(module)
            continue
        if module in disabled_modules:
            continue
        value = values.get(spec["name"])
        if value is None:
            continue
        selection.setdefault(module, {})
        if spec["key"] == "__model_pair__":
            selection[module]["model_url"] = value[0]
            selection[module]["model_name"] = value[1]
        else:
            selection[module][spec["key"]] = value
    if "chunking" not in selection:
        selection["chunking"] = {}
    return {k: v for k, v in selection.items() if v}


def tpe_search(
    qa_json_path: str,
    corpus_json_path: str,
    config_path: str,
    eval_mode: str,
    report_path: str,
    samples: int,
    seed: int,
    score_weights: Optional[Dict[str, float]] = None,
    startup_trials: int = 10,
    gamma: float = 0.2,
    alpha: float = 1.0,
    candidate_pool_size: int = 24,
) -> Dict[str, Any]:
    config = _load_yaml(config_path)
    search_space, algo_cfg, eval_metrics = _split_config(config)
    use_multimodal = _is_multimodal(search_space, algo_cfg)
    _set_eval_schema_env(config_path, use_multimodal)
    eval_fn = evaluate_rag_multimodal if use_multimodal else evaluate_rag
    preferred_metric = None
    if isinstance(algo_cfg, dict):
        preferred_metric = algo_cfg.get("score_metric") or algo_cfg.get("metric")

    rng = random.Random(seed)
    trials: List[Dict[str, Any]] = []
    best_score: float = float("-inf")
    best_config: Dict[str, Any] = {}
    seen: set[str] = set()

    module_order = ["rewriter", "chunking", "retrieve", "clip", "reranker", "pruner", "generator"]
    specs = _build_param_specs(search_space, algo_cfg, module_order)

    bar = tqdm(total=samples, desc="tpe", unit="trial") if tqdm else None

    def _write_report_snapshot() -> None:
        report_dir = os.path.dirname(report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        snapshot = {
            "best_score": best_score,
            "best_config": best_config,
            "trials": trials,
        }
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, ensure_ascii=False, indent=2)

    attempts = 0
    max_attempts = max(samples * 50, 100)
    while len(trials) < samples and attempts < max_attempts:
        attempts += 1
        if len(trials) < startup_trials:
            candidate = _random_selection(search_space, algo_cfg, rng)
        else:
            good, bad, good_counts, bad_counts = _build_tpe_counts(trials, specs, gamma)
            best_values: Optional[Dict[str, Any]] = None
            best_surrogate = float("-inf")
            pool = max(1, int(candidate_pool_size))
            if good:
                for _ in range(pool):
                    values = _sample_values_from_counts(
                        specs,
                        rng,
                        good_counts,
                        bad_counts,
                        len(good),
                        len(bad),
                        alpha,
                    )
                    surrogate = _surrogate_score_from_counts(
                        values,
                        specs,
                        good_counts,
                        bad_counts,
                        len(good),
                        len(bad),
                        alpha,
                    )
                    if surrogate > best_surrogate or best_values is None:
                        best_surrogate = surrogate
                        best_values = values
            if best_values:
                candidate = _build_selection_from_values(best_values, specs)
            else:
                candidate = {}
            if not candidate:
                candidate = _random_selection(search_space, algo_cfg, rng)
        if eval_metrics:
            candidate["eval_metrics"] = eval_metrics
        if algo_cfg:
            candidate = _deep_update(candidate, algo_cfg)
        key = json.dumps(candidate, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)

        _sanitize_selection(candidate)
        trial_index = len(trials) + 1
        safe_candidate = _sanitize_selection_for_log(candidate)
        print(
            f"\n[tpe] trial={trial_index}/{samples} start selection={json.dumps(safe_candidate, ensure_ascii=False)}",
            flush=True,
        )
        trial_start = time.time()
        score, payload = _evaluate_selection(
            qa_json_path,
            corpus_json_path,
            candidate,
            eval_mode,
            preferred_metric,
            score_weights,
            eval_fn,
        )
        trial_elapsed = time.time() - trial_start
        metric_name = payload.get("metric")
        metric_score = payload.get("score")
        print(
            f"[tpe] trial={trial_index}/{samples} end metric={metric_name} score={metric_score} elapsed_seconds={trial_elapsed:.2f}",
            flush=True,
        )
        record = {
            "index": len(trials) + 1,
            "score": payload.get("score"),
            "metric": payload.get("metric"),
            "selection": safe_candidate,
            "report": payload.get("report"),
            "pipeline_total_time_seconds": payload.get("pipeline_total_time_seconds"),
            "outputs": payload.get("outputs"),
            "chunking": payload.get("chunking"),
            "error": payload.get("error"),
            "errors": payload.get("errors"),
        }
        trials.append(record)
        if score >= best_score:
            best_score = score
            best_config = json.loads(json.dumps(safe_candidate))
        _write_report_snapshot()
        if bar:
            bar.update(1)

    if bar:
        bar.close()

    result = {
        "best_score": best_score,
        "best_config": best_config,
        "trials": trials,
    }
    _write_report_snapshot()
    return result


def main() -> None:
    import argparse

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_algo_config = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../configs/algorithms/default.yaml")
    )
    default_report = os.path.join(base_dir, "outputs", "tpe_report.json")

    parser = argparse.ArgumentParser(description="TPE hyperparameter search for RAG.")
    parser.add_argument("--qa_json", required=True, help="Path to QA JSON/JSONL.")
    parser.add_argument("--corpus_json", required=True, help="Path to corpus JSON.")
    parser.add_argument(
        "--config_yaml",
        default=default_algo_config,
        help="Path to algo config with search space.",
    )
    parser.add_argument(
        "--eval_mode",
        default="both",
        choices=["avg", "per_item", "both"],
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--report_path",
        default=default_report,
        help="Path to write TPE report JSON.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of configurations to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--startup_trials",
        type=int,
        default=10,
        help="Number of random trials before TPE sampling.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.2,
        help="Fraction of good trials to model.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Smoothing for categorical probabilities.",
    )
    parser.add_argument(
        "--candidate_pool_size",
        type=int,
        default=24,
        help="Number of TPE candidates to sample before choosing best.",
    )
    parser.add_argument(
        "--score_weights",
        default="",
        help="Weighted metrics, e.g. 'bertf11,llmaaj2'.",
    )
    parser.add_argument(
        "--max_evals", type=int, default=None,
        help="Unified max evaluations (overrides native budget param).",
    )
    args = parser.parse_args()
    if args.max_evals is not None:
        args.samples = args.max_evals

    score_weights = _parse_score_weights(args.score_weights)
    tpe_search(
        qa_json_path=args.qa_json,
        corpus_json_path=args.corpus_json,
        config_path=args.config_yaml,
        eval_mode=args.eval_mode,
        report_path=args.report_path,
        samples=args.samples,
        seed=args.seed,
        score_weights=score_weights,
        startup_trials=args.startup_trials,
        gamma=args.gamma,
        alpha=args.alpha,
        candidate_pool_size=args.candidate_pool_size,
    )


if __name__ == "__main__":
    main()
