import json
import os
import random
import sys
import tempfile
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


def _split_config(
    config: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
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


def _write_temp_selection(selection: Dict[str, Any]) -> str:
    fd, path = tempfile.mkstemp(prefix="cem_selection_", suffix=".yaml")
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


def _module_forced_on(algo_cfg: Dict[str, Any], module: str) -> bool:
    if not isinstance(algo_cfg, dict):
        return False
    section = algo_cfg.get(module)
    return isinstance(section, dict) and len(section) > 0


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


def _build_param_specs(
    search_space: Dict[str, Any],
    algo_cfg: Dict[str, Any],
    module_order: List[str],
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    optional_modules = {"rewriter", "reranker", "pruner"}
    for module in module_order:
        params = search_space.get(module)
        if not isinstance(params, dict):
            continue
        is_optional = module in optional_modules
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
            override = _override_choices(module, key, algo_cfg)
            choices = override if override else _allowed_values(value)
            if len(choices) == 0:
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


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: Dict[str, Any] = {}
        for k, v in value.items():
            if str(k).lower() in {"api_key", "apikey"}:
                redacted[k] = "***"
            else:
                redacted[k] = _redact_secrets(v)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(v) for v in value]
    return value


def _sample_choice(rng: random.Random, probs: List[float]) -> int:
    r = rng.random()
    upto = 0.0
    for idx, p in enumerate(probs):
        upto += p
        if r <= upto:
            return idx
    return len(probs) - 1


def _normalize(probs: List[float]) -> List[float]:
    total = sum(probs)
    if total <= 0:
        return [1.0 / len(probs)] * len(probs)
    return [p / total for p in probs]


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


def _prepare_selection(
    selection: Dict[str, Any],
    algo_cfg: Dict[str, Any],
    eval_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    candidate = json.loads(json.dumps(selection))
    if eval_metrics:
        candidate["eval_metrics"] = eval_metrics
    if algo_cfg:
        candidate = _deep_update(candidate, algo_cfg)
    return candidate


def _evaluate_selection(
    qa_json_path: str,
    corpus_json_path: str,
    selection: Dict[str, Any],
    eval_mode: str,
    preferred_metric: Optional[str],
    score_weights: Optional[Dict[str, float]],
    eval_fn,
) -> Tuple[float, Dict[str, Any]]:
    _sanitize_selection(selection)
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


def cross_entropy_search(
    qa_json_path: str,
    corpus_json_path: str,
    config_path: str,
    eval_mode: str,
    report_path: str,
    iterations: int,
    samples_per_iter: int,
    elite_fraction: float,
    seed: int,
    alpha: float,
    score_weights: Optional[Dict[str, float]] = None,
    max_evals: Optional[int] = None,
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
    module_order = ["rewriter", "chunking", "retrieve", "clip", "reranker", "pruner", "generator"]
    specs = _build_param_specs(search_space, algo_cfg, module_order)

    dist: Dict[str, List[float]] = {}
    for spec in specs:
        count = len(spec["choices"])
        dist[spec["name"]] = [1.0 / count] * count

    trials: List[Dict[str, Any]] = []
    best_score: float = float("-inf")
    best_config: Dict[str, Any] = {}

    def _write_report_snapshot() -> None:
        report_dir = os.path.dirname(report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        snapshot = {
            "best_score": best_score if best_score != float("-inf") else 0.0,
            "best_config": best_config,
            "trials": trials,
        }
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, ensure_ascii=False, indent=2)

    alpha = max(0.0, min(1.0, float(alpha)))
    for it in range(iterations):
        batch: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        bar = tqdm(total=samples_per_iter, desc=f"cem-{it+1}", unit="trial") if tqdm else None
        for idx in range(samples_per_iter):
            values: Dict[str, Any] = {}
            disabled_modules: set[str] = set()
            for spec in specs:
                module = spec["module"]
                if module in disabled_modules and not spec["is_enable"]:
                    continue
                probs = dist[spec["name"]]
                choice_idx = _sample_choice(rng, probs)
                picked = spec["choices"][choice_idx]
                values[spec["name"]] = picked
                if spec["is_enable"] and picked is False:
                    disabled_modules.add(module)
            selection = _build_selection_from_values(values, specs)
            candidate = _prepare_selection(selection, algo_cfg, eval_metrics)
            candidate_for_log = _redact_secrets(candidate)
            print(
                f"\n[cem] iter={it+1} index={idx+1} selection={json.dumps(candidate_for_log, ensure_ascii=False)}"
            )
            score, payload = _evaluate_selection(
                qa_json_path,
                corpus_json_path,
                candidate,
                eval_mode,
                preferred_metric,
                score_weights,
                eval_fn,
            )
            if payload.get("error"):
                score = -1.0
            record = {
                "iteration": it + 1,
                "index": idx + 1,
                "score": score,
                "metric": payload.get("metric"),
                "selection": candidate_for_log,
                "report": payload.get("report"),
                "pipeline_total_time_seconds": payload.get("pipeline_total_time_seconds"),
                "outputs": payload.get("outputs"),
                "chunking": payload.get("chunking"),
                "error": payload.get("error"),
                "errors": payload.get("errors"),
            }
            trials.append(record)
            if max_evals is not None and len(trials) >= max_evals:
                break
            if score >= best_score:
                best_score = score
                best_config = json.loads(json.dumps(candidate_for_log))
            batch.append((score, values, candidate))
            _write_report_snapshot()
            if bar:
                bar.update(1)
        if bar:
            bar.close()
        if max_evals is not None and len(trials) >= max_evals:
            break
        if not batch:
            break
        batch.sort(key=lambda x: x[0], reverse=True)
        elite_count = max(1, int(len(batch) * elite_fraction))
        elite = batch[:elite_count]

        for spec in specs:
            choices = spec["choices"]
            counts = {choice: 0 for choice in choices}
            total = 0
            for _, values, _ in elite:
                val = values.get(spec["name"])
                if val is None:
                    continue
                counts[val] = counts.get(val, 0) + 1
                total += 1
            if total == 0:
                continue
            empirical = [counts.get(choice, 0) / total for choice in choices]
            empirical = _normalize(empirical)
            current = dist[spec["name"]]
            updated = [
                (1.0 - alpha) * cur + alpha * emp
                for cur, emp in zip(current, empirical)
            ]
            dist[spec["name"]] = _normalize(updated)

    result = {
        "best_score": best_score if best_score != float("-inf") else 0.0,
        "best_config": best_config,
        "trials": trials,
    }
    _write_report_snapshot()
    return result


def main() -> None:
    import argparse

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_algo_config = os.path.join(os.path.dirname(__file__), "configforalgo.yaml")
    default_report = os.path.join(base_dir, "outputs", "cem_report.json")

    parser = argparse.ArgumentParser(description="Cross-entropy method search for RAG.")
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
        help="Path to write report JSON.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=6,
        help="Number of CEM iterations.",
    )
    parser.add_argument(
        "--samples_per_iter",
        type=int,
        default=12,
        help="Samples per iteration.",
    )
    parser.add_argument(
        "--elite_fraction",
        type=float,
        default=0.25,
        help="Elite fraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Smoothing for probabilities.",
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

    score_weights = _parse_score_weights(args.score_weights)
    cross_entropy_search(
        qa_json_path=args.qa_json,
        corpus_json_path=args.corpus_json,
        config_path=args.config_yaml,
        eval_mode=args.eval_mode,
        report_path=args.report_path,
        iterations=args.iterations,
        samples_per_iter=args.samples_per_iter,
        elite_fraction=args.elite_fraction,
        seed=args.seed,
        alpha=args.alpha,
        score_weights=score_weights,
        max_evals=args.max_evals,
    )


if __name__ == "__main__":
    main()
