import asyncio
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


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        elif isinstance(value, list):
            # Lists in algo config are candidate pools, not fixed values.
            continue
        else:
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
        if chunking.get("model_url") is None:
            chunking.pop("model_url", None)
        if chunking.get("model_name") is None:
            chunking.pop("model_name", None)


def _split_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    search_space = config.get("rag_search_space") or {}
    eval_metrics = config.get("eval_metrics")
    algo_cfg = {
        key: value
        for key, value in config.items()
        if key not in {"rag_search_space", "eval_metrics"}
    }
    return search_space, algo_cfg, eval_metrics


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


def _param_choices(value: Any, override: Optional[List[Any]]) -> List[Any]:
    if override:
        return override
    allowed = _allowed_values(value)
    if allowed:
        return allowed
    if value is None:
        return []
    if isinstance(value, dict):
        return []
    return [value]


def _paired_model_choices(
    params: Dict[str, Any], algo_cfg: Dict[str, Any], module: str
) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(params, dict):
        return None
    url_override = _override_choices(module, "model_url", algo_cfg)
    name_override = _override_choices(module, "model_name", algo_cfg)
    url_choices = _param_choices(params.get("model_url"), url_override)
    name_choices = _param_choices(params.get("model_name"), name_override)
    if not url_choices or not name_choices:
        return None
    if any(choice is None for choice in url_choices) or any(
        choice is None for choice in name_choices
    ):
        return None
    if len(url_choices) != len(name_choices):
        return None
    return [
        {"model_url": url, "model_name": name}
        for url, name in zip(url_choices, name_choices)
    ]


def _build_module_variants(
    module: str,
    params: Dict[str, Any],
    algo_cfg: Dict[str, Any],
    is_optional: bool,
) -> List[Optional[Dict[str, Any]]]:
    forced_on = _module_forced_on(algo_cfg, module)
    pair_choices = _paired_model_choices(params, algo_cfg, module)
    keys: List[str] = []
    choices_list: List[List[Any]] = []
    for key, value in params.items():
        if pair_choices and key in {"model_url", "model_name"}:
            continue
        override = _override_choices(module, key, algo_cfg)
        choices = _param_choices(value, override)
        if not choices:
            continue
        keys.append(key)
        choices_list.append(choices)

    variants: List[Dict[str, Any]] = []
    pair_list = pair_choices or [None]
    for pair in pair_list:
        current: Dict[str, Any] = dict(pair) if pair else {}
        if not keys:
            variants.append(dict(current))
        else:
            def _build(idx: int, current_map: Dict[str, Any]) -> None:
                if idx == len(keys):
                    variants.append(dict(current_map))
                    return
                key = keys[idx]
                for choice in choices_list[idx]:
                    current_map[key] = choice
                    _build(idx + 1, current_map)
                current_map.pop(key, None)

            _build(0, current)

    if is_optional and not forced_on:
        return [None] + variants
    return variants


def _filter_algo_cfg(
    algo_cfg: Dict[str, Any],
    selection: Dict[str, Any],
    optional_modules: set[str],
) -> Dict[str, Any]:
    if not isinstance(algo_cfg, dict):
        return {}
    filtered: Dict[str, Any] = {}
    for key, value in algo_cfg.items():
        if key in optional_modules and key not in selection:
            continue
        filtered[key] = value
    return filtered


def _write_temp_selection(selection: Dict[str, Any]) -> str:
    fd, path = tempfile.mkstemp(prefix="random_selection_", suffix=".yaml")
    os.close(fd)
    _dump_yaml(selection, path)
    return path


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
        for key, value in params.items():
            choices = _allowed_values(value)
            override = _override_choices(module, key, algo_cfg)
            if override:
                choices = override
            if choices:
                selection[module][key] = rng.choice(choices)
        if not selection[module]:
            selection.pop(module, None)
    return selection


def _selection_key(selection: Dict[str, Any]) -> str:
    return json.dumps(selection, sort_keys=True, ensure_ascii=False)


def random_search(
    qa_json_path: str,
    corpus_json_path: str,
    config_path: str,
    eval_mode: str,
    report_path: str,
    samples: int,
    seed: int,
    score_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    config = _load_yaml(config_path)
    search_space, algo_cfg, eval_metrics = _split_config(config)
    preferred_metric = None
    if isinstance(algo_cfg, dict):
        preferred_metric = algo_cfg.get("score_metric") or algo_cfg.get("metric")
    use_multimodal = _is_multimodal(search_space, algo_cfg)
    _set_eval_schema_env(config_path, use_multimodal)
    eval_fn = evaluate_rag_multimodal if use_multimodal else evaluate_rag
    rng = random.Random(seed)
    trials: List[Dict[str, Any]] = []
    best_score: float = float("-inf")
    best_config: Dict[str, Any] = {}

    optional_modules = {"rewriter", "reranker", "pruner"}
    all_configs: List[Dict[str, Any]] = [{}]
    for module, params in search_space.items():
        if not isinstance(params, dict):
            continue
        is_optional = module in optional_modules
        variants = _build_module_variants(module, params, algo_cfg, is_optional)
        next_configs: List[Dict[str, Any]] = []
        for base in all_configs:
            for variant in variants:
                if variant is None:
                    next_configs.append(dict(base))
                else:
                    merged = json.loads(json.dumps(base))
                    merged[module] = variant
                    next_configs.append(merged)
        all_configs = next_configs

    if eval_metrics:
        for config in all_configs:
            config["eval_metrics"] = eval_metrics
    if algo_cfg:
        for idx, config in enumerate(all_configs):
            filtered_cfg = _filter_algo_cfg(algo_cfg, config, optional_modules)
            all_configs[idx] = _deep_update(config, filtered_cfg)

    report_total = len(all_configs)
    print(f"[random] total_configurations={report_total}")

    rng.shuffle(all_configs)
    if samples > 0:
        all_configs = all_configs[: min(samples, len(all_configs))]

    bar = tqdm(total=len(all_configs), desc="random", unit="trial") if tqdm else None

    def _write_report_snapshot() -> None:
        report_dir = os.path.dirname(report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        snapshot = {
            "best_score": best_score,
            "best_config": best_config,
            "trials": trials,
            "total_configurations": report_total,
            "all_configurations": all_configs,
        }
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, ensure_ascii=False, indent=2)

    for idx, candidate in enumerate(all_configs):
        _sanitize_selection(candidate)
        print(f"\n[random] trial={idx+1} selection={json.dumps(candidate, ensure_ascii=False)}")
        score, payload = _evaluate_selection(
            qa_json_path,
            corpus_json_path,
            candidate,
            eval_mode,
            preferred_metric,
            score_weights,
            eval_fn,
        )
        record = {
            "index": idx + 1,
            "score": payload.get("score"),
            "metric": payload.get("metric"),
            "selection": candidate,
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
            best_config = json.loads(json.dumps(candidate))
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
    default_algo_config = os.path.join(os.path.dirname(__file__), "configforalgo.yaml")
    default_report = os.path.join(base_dir, "outputs", "random_report.json")

    parser = argparse.ArgumentParser(description="Random search for RAG hyperparameters.")
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
        help="Path to write random search report JSON.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of random configurations to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
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
    random_search(
        qa_json_path=args.qa_json,
        corpus_json_path=args.corpus_json,
        config_path=args.config_yaml,
        eval_mode=args.eval_mode,
        report_path=args.report_path,
        samples=args.samples,
        seed=args.seed,
        score_weights=score_weights,
    )


if __name__ == "__main__":
    main()
