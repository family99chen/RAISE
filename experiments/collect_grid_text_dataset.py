#!/usr/bin/env python3
"""Enumerate text-only RAG configurations and collect critic-training data."""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from raisex.core.evaluation_service import evaluate_rag
from raisex.search.algorithms.grpo import (
    PolicyNetwork,
    _deep_update,
    _evaluate_selection,
    _is_multimodal,
    _load_yaml,
    _parse_score_weights,
    _set_eval_schema_env,
    _split_config,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(_json_safe(value), ensure_ascii=False))


def _dataset_name_from_path(path: str) -> str:
    return os.path.basename(os.path.dirname(os.path.abspath(path)))


def _module_order(policy: PolicyNetwork) -> List[str]:
    order: List[str] = []
    for param in policy.params:
        module = str(param["module"])
        if module not in order:
            order.append(module)
    for module in policy.fixed_params:
        if module not in order:
            order.append(module)
    return order


def _base_enabled_modules(policy: PolicyNetwork, module_order: List[str]) -> Set[str]:
    return {
        module
        for module in module_order
        if module not in policy.optional_modules or module in policy.forced_modules
    }


def _apply_fixed_params(
    selection: Dict[str, Any],
    policy: PolicyNetwork,
    enabled_modules: Set[str],
) -> None:
    for module, fixed in policy.fixed_params.items():
        if module not in enabled_modules:
            continue
        selection.setdefault(module, {})
        for key, value in fixed.items():
            selection[module].setdefault(key, _json_safe(value))


def _build_schema(policy: PolicyNetwork) -> Dict[str, Any]:
    params: List[Dict[str, Any]] = []
    for idx, param in enumerate(policy.params):
        params.append(
            {
                "param_idx": idx,
                "name": param["name"],
                "module": param["module"],
                "key": param["key"],
                "choices": _json_safe(param["choices"]),
            }
        )
    return {
        "optional_modules": sorted(policy.optional_modules),
        "forced_modules": sorted(policy.forced_modules),
        "fixed_params": _json_safe(policy.fixed_params),
        "params": params,
    }


def _iter_decision_sequences(policy: PolicyNetwork) -> Iterable[List[Dict[str, Any]]]:
    params = policy.params
    enabled_optional: Dict[str, bool] = {
        module: (module in policy.forced_modules) for module in policy.optional_modules
    }
    decisions: List[Dict[str, Any]] = []

    def rec(idx: int) -> Iterable[List[Dict[str, Any]]]:
        if idx >= len(params):
            yield list(decisions)
            return

        param = params[idx]
        module = str(param["module"])
        key = str(param["key"])

        if (
            key != "__enabled__"
            and module in policy.optional_modules
            and module not in policy.forced_modules
            and not enabled_optional.get(module, False)
        ):
            yield from rec(idx + 1)
            return

        for choice_idx, choice_value in enumerate(param["choices"]):
            previous_enabled = enabled_optional.get(module)
            if key == "__enabled__":
                enabled_optional[module] = bool(choice_value)
            decisions.append(
                {
                    "param_idx": idx,
                    "param_name": param["name"],
                    "module": module,
                    "key": key,
                    "choice_idx": choice_idx,
                    "choice_value": _json_safe(choice_value),
                }
            )
            yield from rec(idx + 1)
            decisions.pop()
            if key == "__enabled__":
                if previous_enabled is None:
                    enabled_optional.pop(module, None)
                else:
                    enabled_optional[module] = previous_enabled

    yield from rec(0)


def _count_decision_sequences(policy: PolicyNetwork) -> int:
    params = policy.params
    enabled_optional: Dict[str, bool] = {
        module: (module in policy.forced_modules) for module in policy.optional_modules
    }

    def rec(idx: int) -> int:
        if idx >= len(params):
            return 1

        param = params[idx]
        module = str(param["module"])
        key = str(param["key"])

        if (
            key != "__enabled__"
            and module in policy.optional_modules
            and module not in policy.forced_modules
            and not enabled_optional.get(module, False)
        ):
            return rec(idx + 1)

        total = 0
        for choice_value in param["choices"]:
            previous_enabled = enabled_optional.get(module)
            if key == "__enabled__":
                enabled_optional[module] = bool(choice_value)
            total += rec(idx + 1)
            if key == "__enabled__":
                if previous_enabled is None:
                    enabled_optional.pop(module, None)
                else:
                    enabled_optional[module] = previous_enabled
        return total

    return rec(0)


def _selection_from_decisions(
    policy: PolicyNetwork,
    decisions: List[Dict[str, Any]],
    module_order: List[str],
) -> Dict[str, Any]:
    enabled_modules = _base_enabled_modules(policy, module_order)
    selection: Dict[str, Any] = {}
    _apply_fixed_params(selection, policy, enabled_modules)

    for decision in decisions:
        module = str(decision["module"])
        key = str(decision["key"])
        value = _json_safe(decision["choice_value"])

        if key == "__enabled__":
            if value:
                enabled_modules.add(module)
                selection.setdefault(module, {})
                _apply_fixed_params(selection, policy, enabled_modules)
            else:
                enabled_modules.discard(module)
                selection.pop(module, None)
            continue

        if module not in enabled_modules:
            continue

        selection.setdefault(module, {})
        if key == "__model_pair__":
            selection[module]["model_url"] = value[0]
            selection[module]["model_name"] = value[1]
        else:
            selection[module][key] = value

    if "chunking" not in selection:
        selection["chunking"] = {}
    return selection


def _state_snapshot(
    selection: Dict[str, Any],
    enabled_modules: Set[str],
    step_cursor: int,
    dataset_name: str,
) -> Dict[str, Any]:
    return {
        "step_cursor": step_cursor,
        "dataset_name": dataset_name,
        "enabled_modules": sorted(enabled_modules),
        "partial_selection": _json_clone(selection),
    }


def _transitions_from_decisions(
    policy: PolicyNetwork,
    decisions: List[Dict[str, Any]],
    module_order: List[str],
    dataset_name: str,
    episode_id: str,
    final_score: float,
    metric_name: str,
) -> List[Dict[str, Any]]:
    enabled_modules = _base_enabled_modules(policy, module_order)
    selection: Dict[str, Any] = {}
    _apply_fixed_params(selection, policy, enabled_modules)
    transitions: List[Dict[str, Any]] = []

    total_steps = len(decisions)
    for idx, decision in enumerate(decisions):
        state = _state_snapshot(selection, enabled_modules, idx, dataset_name)

        module = str(decision["module"])
        key = str(decision["key"])
        value = _json_safe(decision["choice_value"])

        if key == "__enabled__":
            if value:
                enabled_modules.add(module)
                selection.setdefault(module, {})
                _apply_fixed_params(selection, policy, enabled_modules)
            else:
                enabled_modules.discard(module)
                selection.pop(module, None)
        else:
            selection.setdefault(module, {})
            if key == "__model_pair__":
                selection[module]["model_url"] = value[0]
                selection[module]["model_name"] = value[1]
            else:
                selection[module][key] = value

        done = idx == total_steps - 1
        transitions.append(
            {
                "episode_id": episode_id,
                "step_id": idx,
                "state": state,
                "action": {
                    "param_idx": int(decision["param_idx"]),
                    "param_name": decision["param_name"],
                    "module": module,
                    "key": key,
                    "choice_idx": int(decision["choice_idx"]),
                    "choice_value": value,
                    "behavior_policy": "grid",
                    "logp": None,
                },
                "next_state": _state_snapshot(selection, enabled_modules, idx + 1, dataset_name),
                "reward": final_score if done else 0.0,
                "done": done,
                "return": final_score,
                "metric_name": metric_name,
                "final_score": final_score,
            }
        )

    return transitions


def _selection_hash(selection: Dict[str, Any]) -> str:
    payload = json.dumps(selection, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_completed_hashes(path: str) -> Set[str]:
    completed: Set[str] = set()
    if not os.path.isfile(path):
        return completed
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            selection_hash = row.get("selection_hash")
            if isinstance(selection_hash, str):
                completed.add(selection_hash)
    return completed


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enumerate text-only RAG configurations and collect critic-training data."
    )
    parser.add_argument("--qa_json", required=True, help="Path to QA JSON.")
    parser.add_argument("--corpus_json", required=True, help="Path to corpus JSON.")
    parser.add_argument(
        "--config_yaml",
        default=os.path.join(PROJECT_ROOT, "configs", "configforalgo_fullspace_text.yaml"),
        help="Text-only search-space config.",
    )
    parser.add_argument(
        "--output_root",
        default=os.path.join(PROJECT_ROOT, "outputs-grid-collector"),
        help="Root directory for collected data.",
    )
    parser.add_argument(
        "--eval_mode",
        default="avg",
        choices=["avg", "per_item", "both"],
        help="Evaluation mode forwarded to RAISE.",
    )
    parser.add_argument(
        "--score_weights",
        default="",
        help="Optional weighted metrics string, e.g. 'f11.0,rougel1.0'.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max new configs to evaluate.")
    parser.add_argument("--resume", action="store_true", help="Skip selections already in episodes.jsonl.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle enumeration order before evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument("--save_outputs", action="store_true", help="Persist full pipeline outputs per episode.")
    parser.add_argument("--dry_run", action="store_true", help="Only print combination count and exit.")
    args = parser.parse_args()

    qa_json = os.path.abspath(args.qa_json)
    corpus_json = os.path.abspath(args.corpus_json)
    config_yaml = os.path.abspath(args.config_yaml)
    output_root = os.path.abspath(args.output_root)
    dataset_name = _dataset_name_from_path(qa_json)
    dataset_root = os.path.join(output_root, dataset_name)
    os.makedirs(dataset_root, exist_ok=True)

    config = _load_yaml(config_yaml)
    search_space, algo_cfg, eval_metrics = _split_config(config)
    if _is_multimodal(search_space, algo_cfg):
        raise SystemExit("Config is multimodal. Remove clip and use a text-only config.")

    _set_eval_schema_env(config_yaml, use_multimodal=False)
    policy = PolicyNetwork(search_space, algo_cfg)
    module_order = _module_order(policy)
    total_candidates = _count_decision_sequences(policy)

    schema_path = os.path.join(dataset_root, "policy_schema.json")
    episodes_path = os.path.join(dataset_root, "episodes.jsonl")
    transitions_path = os.path.join(dataset_root, "transitions.jsonl")
    manifest_path = os.path.join(dataset_root, "manifest.json")

    print(f"Dataset: {dataset_name}")
    print(f"Config: {config_yaml}")
    print(f"Total candidate configurations: {total_candidates}")

    if args.dry_run:
        _write_json(schema_path, _build_schema(policy))
        _write_json(
            manifest_path,
            {
                "dataset_name": dataset_name,
                "qa_json": qa_json,
                "corpus_json": corpus_json,
                "config_yaml": config_yaml,
                "eval_mode": args.eval_mode,
                "score_weights": args.score_weights,
                "total_candidates": total_candidates,
                "dry_run": True,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        return

    decision_sequences = list(_iter_decision_sequences(policy))
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(decision_sequences)

    completed_hashes = _load_completed_hashes(episodes_path) if args.resume else set()
    completed_before = len(completed_hashes)
    new_count = 0
    score_weights = _parse_score_weights(args.score_weights)

    _write_json(schema_path, _build_schema(policy))

    with open(episodes_path, "a", encoding="utf-8") as episodes_handle, open(
        transitions_path, "a", encoding="utf-8"
    ) as transitions_handle:
        for seq_idx, decisions in enumerate(decision_sequences, start=1):
            base_selection = _selection_from_decisions(policy, decisions, module_order)
            selection: Dict[str, Any] = _json_clone(base_selection)
            if eval_metrics:
                selection["eval_metrics"] = _json_clone(eval_metrics)
            if algo_cfg:
                selection = _deep_update(selection, algo_cfg)

            selection_hash = _selection_hash(selection)
            if selection_hash in completed_hashes:
                continue

            score, payload = _evaluate_selection(
                qa_json_path=qa_json,
                corpus_json_path=corpus_json,
                selection=selection,
                eval_mode=args.eval_mode,
                preferred_metric=(algo_cfg.get("score_metric") or algo_cfg.get("metric")) if isinstance(algo_cfg, dict) else None,
                score_weights=score_weights,
                eval_fn=evaluate_rag,
            )

            metric_name = str(payload.get("metric") or "")
            report = payload.get("report") or {}
            if payload.get("error"):
                score = -1.0

            episode_id = f"{dataset_name}-{selection_hash[:12]}"
            episode = {
                "episode_id": episode_id,
                "selection_hash": selection_hash,
                "dataset_name": dataset_name,
                "qa_json": qa_json,
                "corpus_json": corpus_json,
                "config_yaml": config_yaml,
                "eval_mode": args.eval_mode,
                "selection": selection,
                "metric_name": metric_name,
                "final_score": score,
                "metrics": _json_clone(report.get("metrics") or {}),
                "pipeline_total_time_seconds": payload.get("pipeline_total_time_seconds"),
                "error": payload.get("error"),
                "errors": payload.get("errors"),
            }
            if args.save_outputs:
                episode["outputs"] = _json_clone(payload.get("outputs"))
                episode["chunking"] = _json_clone(payload.get("chunking"))

            episodes_handle.write(json.dumps(episode, ensure_ascii=False) + "\n")

            transitions = _transitions_from_decisions(
                policy=policy,
                decisions=decisions,
                module_order=module_order,
                dataset_name=dataset_name,
                episode_id=episode_id,
                final_score=score,
                metric_name=metric_name,
            )
            for row in transitions:
                transitions_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            episodes_handle.flush()
            transitions_handle.flush()
            completed_hashes.add(selection_hash)
            new_count += 1

            print(
                f"[{new_count}] seq={seq_idx}/{len(decision_sequences)} "
                f"score={score:.4f} metric={metric_name} episode_id={episode_id}",
                flush=True,
            )

            if args.limit is not None and new_count >= args.limit:
                break

    _write_json(
        manifest_path,
        {
            "dataset_name": dataset_name,
            "qa_json": qa_json,
            "corpus_json": corpus_json,
            "config_yaml": config_yaml,
            "eval_mode": args.eval_mode,
            "score_weights": args.score_weights,
            "total_candidates": total_candidates,
            "completed_before": completed_before,
            "completed_after": len(completed_hashes),
            "newly_collected": new_count,
            "resume": bool(args.resume),
            "shuffle": bool(args.shuffle),
            "seed": args.seed,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


if __name__ == "__main__":
    main()
