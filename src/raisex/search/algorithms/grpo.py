import json
import math
import os
import random
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

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
    fd, path = tempfile.mkstemp(prefix="rl_selection_", suffix=".yaml")
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


class PolicyNetwork:
    def __init__(self, search_space: Dict[str, Any], algo_cfg: Dict[str, Any]):
        self.params: List[Dict[str, Any]] = []
        self.fixed_params: Dict[str, Dict[str, Any]] = {}
        self.optional_modules = {"rewriter", "reranker", "pruner"}
        # Each param: { "name": str, "module": str, "key": str, "choices": list, "logits": list[float] }
        
        module_order = ["rewriter", "chunking", "retrieve", "clip", "reranker", "pruner", "generator"]
        self.forced_modules = {
            module for module in module_order if _module_forced_on(algo_cfg, module)
        }
        
        for module in module_order:
            params = search_space.get(module)
            if not isinstance(params, dict):
                continue
            
            is_optional = module in self.optional_modules
            forced_on = _module_forced_on(algo_cfg, module)
            
            # 1. Enable/Disable decision for optional modules
            if is_optional and not forced_on:
                self.params.append({
                    "name": f"{module}.__enabled__",
                    "module": module,
                    "key": "__enabled__",
                    "choices": [True, False],
                    "logits": [0.0, 0.0]
                })

            # 2. Paired model choices
            pair_choices = _paired_model_choices(params, algo_cfg, module)
            if pair_choices:
                self.params.append({
                    "name": f"{module}.__model_pair__",
                    "module": module,
                    "key": "__model_pair__",
                    "choices": pair_choices,
                    "logits": [0.0] * len(pair_choices)
                })

            # 3. Individual parameters
            for key, value in params.items():
                if pair_choices and key in {"model_url", "model_name"}:
                    continue
                
                override = _override_choices(module, key, algo_cfg)
                choices = override if override else _allowed_values(value)
                if not choices:
                    continue
                
                # If only 1 choice, no need to learn
                if len(choices) > 1:
                    self.params.append({
                        "name": f"{module}.{key}",
                        "module": module,
                        "key": key,
                        "choices": choices,
                        "logits": [0.0] * len(choices)
                    })
                else:
                    self.fixed_params.setdefault(module, {})[key] = choices[0]

    def softmax(self, logits: List[float]) -> List[float]:
        if not logits:
            return []
        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        sum_exps = sum(exps)
        return [e / sum_exps for e in exps]

    def sample(self, rng: random.Random) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        selection: Dict[str, Any] = {}
        trajectory: List[Dict[str, Any]] = []
        
        # Modules enabled status
        enabled_modules = set()
        
        # First pass: Determine enabled modules
        for idx, param in enumerate(self.params):
            if param["key"] == "__enabled__":
                probs = self.softmax(param["logits"])
                choice_idx = self._choice_index(rng, probs)
                is_enabled = param["choices"][choice_idx]
                logp = math.log(max(probs[choice_idx], 1e-12))
                trajectory.append({"param_idx": idx, "choice_idx": choice_idx, "logp": logp})
                if is_enabled:
                    enabled_modules.add(param["module"])
            elif param["module"] not in self.optional_modules:
                 enabled_modules.add(param["module"])

        for module in self.fixed_params:
            if (
                module in self.optional_modules
                and module not in enabled_modules
                and module not in self.forced_modules
            ):
                continue
            enabled_modules.add(module)

        # Second pass: Select values for enabled modules
        for module in enabled_modules:
            fixed = self.fixed_params.get(module)
            if fixed:
                selection.setdefault(module, {})
                for key, value in fixed.items():
                    selection[module].setdefault(key, value)

        for idx, param in enumerate(self.params):
            module = param["module"]
            if param["key"] == "__enabled__":
                continue
            
            if module not in enabled_modules:
                continue

            probs = self.softmax(param["logits"])
            choice_idx = self._choice_index(rng, probs)
            choice_val = param["choices"][choice_idx]
            logp = math.log(max(probs[choice_idx], 1e-12))
            trajectory.append({"param_idx": idx, "choice_idx": choice_idx, "logp": logp})

            selection.setdefault(module, {})
            
            if param["key"] == "__model_pair__":
                selection[module]["model_url"] = choice_val[0]
                selection[module]["model_name"] = choice_val[1]
            else:
                selection[module][param["key"]] = choice_val

        # Ensure chunking exists
        if "chunking" not in selection:
             selection["chunking"] = {}
             
        return selection, trajectory

    def _choice_index(self, rng: random.Random, probs: List[float]) -> int:
        r = rng.random()
        upto = 0.0
        for i, p in enumerate(probs):
            upto += p
            if r <= upto:
                return i
        return len(probs) - 1

    def clone_logits(self) -> List[List[float]]:
        return [list(param["logits"]) for param in self.params]

    def apply_reward_prior(
        self,
        sums: List[List[float]],
        counts: List[List[int]],
        mix: float,
        min_count: int,
        scale: float,
    ) -> None:
        if mix <= 0:
            return
        for i, param in enumerate(self.params):
            if i >= len(counts):
                continue
            available: List[int] = []
            means: List[Optional[float]] = [None] * len(param["choices"])
            for j in range(len(param["choices"])):
                if j >= len(counts[i]):
                    continue
                c = counts[i][j]
                if c >= min_count:
                    means[j] = sums[i][j] / c
                    available.append(j)
            if len(available) < 2:
                continue
            mu = sum(means[j] for j in available if means[j] is not None) / len(available)
            var = sum((means[j] - mu) ** 2 for j in available if means[j] is not None) / len(
                available
            )
            std = math.sqrt(var) if var > 0 else 0.0
            if std <= 0:
                continue
            for j in available:
                prior = (means[j] - mu) / std * scale
                param["logits"][j] = (1.0 - mix) * param["logits"][j] + mix * prior

    def apply_ucb_bonus(
        self,
        counts: List[List[int]],
        coef: float,
        min_total: int,
    ) -> None:
        if coef <= 0:
            return
        for i, param in enumerate(self.params):
            if i >= len(counts):
                continue
            total = sum(counts[i])
            if total < min_total:
                continue
            log_term = math.log(total + 1.0)
            for j in range(len(param["choices"])):
                c = counts[i][j] if j < len(counts[i]) else 0
                bonus = coef * math.sqrt(log_term / (c + 1.0))
                param["logits"][j] += bonus

    def update_grpo(
        self,
        trajectories: List[List[Dict[str, Any]]],
        rewards: List[float],
        ref_logits: List[List[float]],
        learning_rate: float,
        kl_coeff: float,
        clip_ratio: float,
        update_epochs: int,
        weights: Optional[List[float]] = None,
    ) -> None:
        if not rewards:
            return
        if weights is None or len(weights) != len(rewards):
            weights = [1.0] * len(rewards)
        weight_total = sum(weights)
        if weight_total <= 0:
            return
        mean = sum(r * w for r, w in zip(rewards, weights)) / weight_total
        variance = sum(
            w * (r - mean) ** 2 for r, w in zip(rewards, weights)
        ) / weight_total
        std = math.sqrt(variance) if variance > 0 else 0.0
        denom = std if std > 0 else 1.0
        advantages = [(r - mean) / denom for r in rewards]

        epochs = max(1, int(update_epochs))
        clip_ratio = max(0.0, float(clip_ratio))
        for _ in range(epochs):
            logits_updates = [[0.0] * len(p["logits"]) for p in self.params]
            for traj, adv, weight in zip(trajectories, advantages, weights):
                scaled_adv = adv * weight
                for step in traj:
                    param_idx = step["param_idx"]
                    choice_idx = step["choice_idx"]
                    old_logp = float(step.get("logp", 0.0))
                    param = self.params[param_idx]
                    probs = self.softmax(param["logits"])
                    ref_probs = self.softmax(ref_logits[param_idx])
                    new_logp = math.log(max(probs[choice_idx], 1e-12))
                    ratio = math.exp(new_logp - old_logp)
                    if scaled_adv >= 0:
                        ratio_used = min(ratio, 1.0 + clip_ratio)
                    else:
                        ratio_used = max(ratio, 1.0 - clip_ratio)
                    for j in range(len(param["logits"])):
                        pg_grad = (1.0 if j == choice_idx else 0.0) - probs[j]
                        kl_grad = probs[j] - ref_probs[j]
                        logits_updates[param_idx][j] += learning_rate * (
                            ratio_used * scaled_adv * pg_grad - kl_coeff * kl_grad
                        )
            for i, param in enumerate(self.params):
                for j in range(len(param["logits"])):
                    param["logits"][j] += logits_updates[i][j]


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


def rl_search(
    qa_json_path: str,
    corpus_json_path: str,
    config_path: str,
    eval_mode: str,
    report_path: str,
    episodes: int,
    seed: int,
    learning_rate: float = 0.1,
    group_size: int = 4,
    kl_coeff: float = 0.02,
    clip_ratio: float = 0.2,
    update_epochs: int = 2,
    score_weights: Optional[Dict[str, float]] = None,
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
    policy = PolicyNetwork(search_space, algo_cfg)
    choice_sums: List[List[float]] = [
        [0.0] * len(param["choices"]) for param in policy.params
    ]
    choice_counts: List[List[int]] = [
        [0] * len(param["choices"]) for param in policy.params
    ]
    selection_cache: Dict[str, Dict[str, Any]] = {}
    elite_buffer: List[Dict[str, Any]] = []
    
    trials: List[Dict[str, Any]] = []
    best_score: float = float("-inf")
    best_config: Dict[str, Any] = {}
    group_size = max(1, int(group_size))
    bar = tqdm(total=episodes, desc="rl-grpo", unit="ep") if tqdm else None
    
    # Initialize reference logits (fixed reference policy)
    ref_logits = policy.clone_logits()
    prior_mix = 0.2
    prior_min_count = 2
    prior_scale = 1.0
    ucb_coef = 0.0
    ucb_min_total = 4
    elite_size = 0
    elite_weight = 0.5
    if isinstance(algo_cfg, dict):
        prior_mix = float(algo_cfg.get("prior_mix", prior_mix))
        prior_min_count = int(algo_cfg.get("prior_min_count", prior_min_count))
        prior_scale = float(algo_cfg.get("prior_scale", prior_scale))
        ucb_coef = float(algo_cfg.get("ucb_coef", ucb_coef))
        ucb_min_total = int(algo_cfg.get("ucb_min_total", ucb_min_total))
        elite_size = int(algo_cfg.get("elite_size", elite_size))
        elite_weight = float(algo_cfg.get("elite_weight", elite_weight))

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

    for ep in range(episodes):
        trajectories: List[List[Dict[str, Any]]] = []
        rewards: List[float] = []
        policy.apply_reward_prior(
            sums=choice_sums,
            counts=choice_counts,
            mix=prior_mix,
            min_count=prior_min_count,
            scale=prior_scale,
        )
        policy.apply_ucb_bonus(
            counts=choice_counts,
            coef=ucb_coef,
            min_total=ucb_min_total,
        )

        for idx in range(group_size):
            selection, trajectory = policy.sample(rng)
            if eval_metrics:
                selection["eval_metrics"] = eval_metrics
            if algo_cfg:
                selection = _deep_update(selection, algo_cfg)

            print(
                f"\n[grpo] episode={ep+1} group={idx+1}/{group_size} selection={json.dumps(selection, ensure_ascii=False)}"
            )

            cache_key = json.dumps(selection, sort_keys=True, ensure_ascii=False)
            cached = selection_cache.get(cache_key)
            if cached:
                score = cached["score"]
                payload = cached["payload"]
            else:
                score, payload = _evaluate_selection(
                    qa_json_path,
                    corpus_json_path,
                    selection,
                    eval_mode,
                    preferred_metric,
                    score_weights,
                    eval_fn,
                )
                selection_cache[cache_key] = {
                    "score": score,
                    "payload": payload,
                }

            if payload.get("error"):
                score = -1.0

            trajectories.append(trajectory)
            rewards.append(score)
            for step in trajectory:
                param_idx = step["param_idx"]
                choice_idx = step["choice_idx"]
                if param_idx < len(choice_sums) and choice_idx < len(choice_sums[param_idx]):
                    choice_sums[param_idx][choice_idx] += score
                    choice_counts[param_idx][choice_idx] += 1

            record = {
                "episode": ep + 1,
                "group": idx + 1,
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

        if elite_size > 0:
            for traj, reward in zip(trajectories, rewards):
                elite_buffer.append({"reward": reward, "trajectory": traj})
            elite_buffer.sort(key=lambda item: item["reward"], reverse=True)
            elite_buffer = elite_buffer[:elite_size]

        combined_trajectories = list(trajectories)
        combined_rewards = list(rewards)
        combined_weights = [1.0] * len(rewards)
        if elite_size > 0 and elite_buffer:
            for item in elite_buffer:
                combined_trajectories.append(item["trajectory"])
                combined_rewards.append(item["reward"])
                combined_weights.append(elite_weight)

        policy.update_grpo(
            trajectories=combined_trajectories,
            rewards=combined_rewards,
            ref_logits=ref_logits,
            learning_rate=learning_rate,
            kl_coeff=kl_coeff,
            clip_ratio=clip_ratio,
            update_epochs=update_epochs,
            weights=combined_weights,
        )
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
    default_report = os.path.join(base_dir, "outputs", "rl_report.json")

    parser = argparse.ArgumentParser(description="RL (GRPO) search for RAG.")
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
        help="Path to write RL report JSON.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of RL episodes.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=4,
        help="Samples per GRPO update.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate.",
    )
    parser.add_argument(
        "--kl_coeff",
        type=float,
        default=0.02,
        help="KL penalty coefficient.",
    )
    parser.add_argument(
        "--clip_ratio",
        type=float,
        default=0.2,
        help="PPO clip ratio.",
    )
    parser.add_argument(
        "--update_epochs",
        type=int,
        default=2,
        help="Policy update epochs per group.",
    )
    parser.add_argument(
        "--score_weights",
        default="",
        help="Weighted metrics, e.g. 'bertf11,llmaaj2'.",
    )
    args = parser.parse_args()

    score_weights = _parse_score_weights(args.score_weights)
    rl_search(
        qa_json_path=args.qa_json,
        corpus_json_path=args.corpus_json,
        config_path=args.config_yaml,
        eval_mode=args.eval_mode,
        report_path=args.report_path,
        episodes=args.episodes,
        seed=args.seed,
        learning_rate=args.lr,
        group_size=args.group_size,
        kl_coeff=args.kl_coeff,
        clip_ratio=args.clip_ratio,
        update_epochs=args.update_epochs,
        score_weights=score_weights,
    )


if __name__ == "__main__":
    main()
