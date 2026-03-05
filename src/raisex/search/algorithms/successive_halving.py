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


def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    if path.endswith(".jsonl"):
        items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("QA JSON must be a list of objects.")
    return data


def _write_temp_json(items: List[Dict[str, Any]]) -> str:
    fd, path = tempfile.mkstemp(prefix="sh_qa_", suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(items, handle, ensure_ascii=False)
    return path


def _subset_qa_path(
    qa_items: List[Dict[str, Any]],
    rng: random.Random,
    size: int,
    fallback_path: str,
) -> Tuple[str, Optional[str]]:
    total = len(qa_items)
    if size >= total:
        return fallback_path, None
    indices = rng.sample(range(total), size)
    subset = [qa_items[i] for i in indices]
    path = _write_temp_json(subset)
    return path, path


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
    fd, path = tempfile.mkstemp(prefix="sh_selection_", suffix=".yaml")
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


def successive_halving_search(
    qa_json_path: str,
    corpus_json_path: str,
    config_path: str,
    eval_mode: str,
    report_path: str,
    num_configs: int,
    eta: int,
    seed: int,
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
    eta = max(2, int(eta))
    num_configs = max(1, int(num_configs))
    qa_items = _load_json_or_jsonl(qa_json_path)
    total_items = len(qa_items)
    if total_items <= 0:
        raise ValueError("QA JSON is empty.")

    configs: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = max(num_configs * 50, 100)
    seen: set[str] = set()
    while len(configs) < num_configs and attempts < max_attempts:
        attempts += 1
        selection = _random_selection(search_space, algo_cfg, rng)
        candidate = _prepare_selection(selection, algo_cfg, eval_metrics)
        key = json.dumps(candidate, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        configs.append(candidate)

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

    rounds = 0
    remaining = num_configs
    while True:
        rounds += 1
        if remaining <= 1:
            break
        remaining = max(1, remaining // eta)
    min_resource = max(1, total_items // (eta ** (rounds - 1)))

    round_idx = 0
    while len(configs) > 0:
        round_idx += 1
        resource = min(total_items, int(min_resource * (eta ** (round_idx - 1))))
        qa_round_path, cleanup_path = _subset_qa_path(
            qa_items, rng, resource, qa_json_path
        )
        round_scores: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        bar = tqdm(total=len(configs), desc=f"sh-round-{round_idx}", unit="trial") if tqdm else None
        for idx, candidate in enumerate(configs):
            print(f"\n[sh] round={round_idx} index={idx+1} selection={json.dumps(candidate, ensure_ascii=False)}")
            score, payload = _evaluate_selection(
                qa_round_path,
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
                "round": round_idx,
                "index": idx + 1,
                "score": score,
                "metric": payload.get("metric"),
                "selection": candidate,
                "report": payload.get("report"),
                "pipeline_total_time_seconds": payload.get("pipeline_total_time_seconds"),
                "outputs": payload.get("outputs"),
                "chunking": payload.get("chunking"),
                "error": payload.get("error"),
                "errors": payload.get("errors"),
                "resource": resource,
            }
            trials.append(record)
            if max_evals is not None and len(trials) >= max_evals:
                break
            if score >= best_score:
                best_score = score
                best_config = json.loads(json.dumps(candidate))
            round_scores.append((score, candidate, record))
            _write_report_snapshot()
            if bar:
                bar.update(1)
        if bar:
            bar.close()
        if cleanup_path:
            os.remove(cleanup_path)
        if max_evals is not None and len(trials) >= max_evals:
            break
        if len(configs) <= 1:
            break
        round_scores.sort(key=lambda x: x[0], reverse=True)
        keep = max(1, len(round_scores) // eta)
        configs = [item[1] for item in round_scores[:keep]]

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
    default_report = os.path.join(base_dir, "outputs", "successive_halving_report.json")

    parser = argparse.ArgumentParser(description="Successive halving search for RAG.")
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
        "--num_configs",
        type=int,
        default=24,
        help="Number of initial configurations.",
    )
    parser.add_argument(
        "--eta",
        type=int,
        default=3,
        help="Reduction factor.",
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

    score_weights = _parse_score_weights(args.score_weights)
    successive_halving_search(
        qa_json_path=args.qa_json,
        corpus_json_path=args.corpus_json,
        config_path=args.config_yaml,
        eval_mode=args.eval_mode,
        report_path=args.report_path,
        num_configs=args.num_configs,
        eta=args.eta,
        seed=args.seed,
        score_weights=score_weights,
        max_evals=args.max_evals,
    )


if __name__ == "__main__":
    main()
