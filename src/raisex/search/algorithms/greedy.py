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


def _extract_qa(qa_items: List[Dict[str, Any]]) -> Tuple[List[str], List[List[str]]]:
    queries: List[str] = []
    references: List[List[str]] = []
    for item in qa_items:
        query = item.get("query") or item.get("question")
        if not query:
            raise ValueError("Each QA item must include 'query' or 'question'.")
        queries.append(str(query))
        refs = item.get("references") or item.get("answers") or item.get("reference")
        if refs is None:
            references.append([])
        elif isinstance(refs, list):
            references.append([str(r) for r in refs])
        else:
            references.append([str(refs)])
    return queries, references


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

def _random_selection(
    search_space: Dict[str, Any],
    algo_cfg: Dict[str, Any],
    force_all_on: bool = False,
) -> Dict[str, Any]:
    selection: Dict[str, Any] = {}
    for section, params in search_space.items():
        if not isinstance(params, dict):
            continue
        if force_all_on is False and section in {"rewriter", "reranker", "pruner"}:
            selection[section] = {}
        else:
            selection[section] = {}
        pair_choices = _paired_model_choices(params, algo_cfg, section)
        if pair_choices:
            choice = random.choice(pair_choices)
            selection[section]["model_url"] = choice[0]
            selection[section]["model_name"] = choice[1]
        for key, value in params.items():
            if pair_choices and key in {"model_url", "model_name"}:
                continue
            choices = _allowed_values(value)
            override = _override_choices(section, key, algo_cfg)
            if override:
                choices = override
            if choices:
                selection[section][key] = random.choice(choices)
            else:
                selection[section][key] = value
        if not selection[section]:
            selection.pop(section, None)
    return selection


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
            # Lists in algo config are treated as candidate pools, not fixed values.
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
    fd, path = tempfile.mkstemp(prefix="greedy_selection_", suffix=".yaml")
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


def _count_greedy_steps(search_space: Dict[str, Any], algo_cfg: Dict[str, Any]) -> int:
    total = 0
    for section, params in search_space.items():
        if section in {"generator", "eval_metrics"}:
            continue
        if not isinstance(params, dict):
            continue
        # optional modules get an on/off decision step
        if section in {"rewriter", "reranker", "pruner"}:
            total += 1
        pair_choices = _paired_model_choices(params, algo_cfg, section)
        if pair_choices:
            total += len(pair_choices)
        for key, value in params.items():
            if pair_choices and key in {"model_url", "model_name"}:
                continue
            choices = _allowed_values(value)
            override = _override_choices(section, key, algo_cfg)
            if override:
                choices = override
            if choices:
                total += len(choices)
    return total


def greedy_search(
    qa_json_path: str,
    corpus_json_path: str,
    config_path: str,
    eval_mode: str,
    report_path: str,
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

    current = _random_selection(search_space, algo_cfg, force_all_on=True)
    if eval_metrics:
        current["eval_metrics"] = eval_metrics
    if algo_cfg:
        current = _deep_update(current, algo_cfg)

    trials: List[Dict[str, Any]] = []
    best_score: float = float("-inf")
    best_config: Dict[str, Any] = current
    budget_exhausted = False

    total_steps = _count_greedy_steps(search_space, algo_cfg)
    if max_evals is not None:
        total_steps = min(total_steps, max_evals)
    bar = tqdm(total=total_steps, desc="greedy", unit="trial") if tqdm else None

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

    def run_trial(stage: str, selection: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        nonlocal best_score, best_config, budget_exhausted
        if max_evals is not None and len(trials) >= max_evals:
            budget_exhausted = True
            return float("-inf"), {}
        _sanitize_selection(selection)
        print(f"\n[greedy] trial={stage} selection={json.dumps(selection, ensure_ascii=False)}")
        score, payload = _evaluate_selection(
            qa_json_path,
            corpus_json_path,
            selection,
            eval_mode,
            preferred_metric,
            score_weights,
            eval_fn,
        )
        record = {
            "stage": stage,
            "score": payload.get("score"),
            "metric": payload.get("metric"),
            "selection": selection,
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
            best_config = json.loads(json.dumps(selection))
        _write_report_snapshot()
        if bar:
            bar.update(1)
        return score, record

    module_order = ["rewriter", "chunking", "retrieve", "clip", "reranker", "pruner", "generator"]

    for module in module_order:
        if budget_exhausted:
            break
        params = search_space.get(module)
        if not isinstance(params, dict):
            continue

        is_optional = module in {"rewriter", "reranker", "pruner"}
        score_off = float("-inf")
        if is_optional:
            candidate = dict(current)
            candidate.pop(module, None)
            score_off, _ = run_trial(f"{module}:off", candidate)

        # Greedy per-parameter (module on)
        best_on_score = float("-inf")
        best_on_config = None
        if module in current or not is_optional:
            pair_choices = _paired_model_choices(params, algo_cfg, module)
            if pair_choices:
                best_pair = None
                best_pair_score = float("-inf")
                for pair in pair_choices:
                    if budget_exhausted:
                        break
                    candidate = json.loads(json.dumps(current))
                    candidate.setdefault(module, {})
                    candidate[module]["model_url"] = pair[0]
                    candidate[module]["model_name"] = pair[1]
                    score, _ = run_trial(
                        f"{module}.model_pair:{pair}", candidate
                    )
                    if score >= best_pair_score:
                        best_pair_score = score
                        best_pair = pair
                if best_pair is not None:
                    current.setdefault(module, {})
                    current[module]["model_url"] = best_pair[0]
                    current[module]["model_name"] = best_pair[1]
                    best_on_score = max(best_on_score, best_pair_score)
            for key, value in params.items():
                if budget_exhausted:
                    break
                if pair_choices and key in {"model_url", "model_name"}:
                    continue
                choices = _allowed_values(value)
                override = _override_choices(module, key, algo_cfg)
                if override:
                    choices = override
                if not choices:
                    continue
                best_val = None
                best_val_score = float("-inf")
                for choice in choices:
                    if budget_exhausted:
                        break
                    candidate = json.loads(json.dumps(current))
                    candidate.setdefault(module, {})
                    candidate[module][key] = choice
                    score, _ = run_trial(f"{module}.{key}:{choice}", candidate)
                    if score >= best_val_score:
                        best_val_score = score
                        best_val = choice
                if best_val is not None:
                    current.setdefault(module, {})
                    current[module][key] = best_val
                    best_on_score = max(best_on_score, best_val_score)
            
            # Evaluate final "on" configuration
            final_on_candidate = json.loads(json.dumps(current))
            final_on_score, _ = run_trial(f"{module}:final_on", final_on_candidate)
            best_on_score = final_on_score
            best_on_config = final_on_candidate

        # Compare off vs best on configuration
        if is_optional:
            if score_off >= best_on_score:
                current.pop(module, None)
            elif best_on_config is not None:
                current = best_on_config

    if bar:
        if bar.total != len(trials):
            bar.total = len(trials)
            bar.n = len(trials)
            bar.refresh()
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
    default_report = os.path.join(base_dir, "outputs", "greedy_report.json")

    parser = argparse.ArgumentParser(description="Greedy hyperparameter search for RAG.")
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
        help="Path to write greedy report JSON.",
    )
    parser.add_argument(
        "--score_weights",
        default="",
        help="Weighted metrics, e.g. 'bertf11,llmaaj2'.",
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=None,
        help="Maximum number of evaluations before stopping. Default: unlimited.",
    )
    args = parser.parse_args()

    qa_json_path = args.qa_json
    corpus_json_path = args.corpus_json
    config_path = args.config_yaml
    eval_mode = args.eval_mode
    report_path = args.report_path
    score_weights = _parse_score_weights(args.score_weights)
    greedy_search(
        qa_json_path,
        corpus_json_path,
        config_path,
        eval_mode,
        report_path,
        score_weights=score_weights,
        max_evals=args.max_evals,
    )


if __name__ == "__main__":
    main()
