import asyncio
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml

from raisex.core.config_validator import check_config, check_config_multimodal
from raisex.pipelines.multimodal.pipeline import (
    getupperbound_external as getupperbound_external_multimodal,
)
from raisex.pipelines.multimodal.pipeline import run_batch_async as run_batch_async_multimodal
from raisex.pipelines.text.pipeline import (
    getupperbound_external as getupperbound_external_pipeline,
)
from raisex.pipelines.text.pipeline import run_batch_async

_DEFAULT_ALGORITHMS = [
    "cross_entropy",
    "doctor_grpo",
    "randomalgo",
    "greedy",
    "grpo",
    "iterative_local_search",
    "mab_ts",
    "mab_ucb",
    "ppo",
    "regularized_evolution",
    "reinforce_pp",
    "simulated_annealing",
    "successive_halving",
    "tpe",
    "upperbound",
    "thupperbound",
]


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _project_src() -> str:
    return os.path.join(_project_root(), "src")


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


def _eval_cache_root() -> str:
    return os.environ.get(
        "RAISEX_EVAL_CACHE_DIR",
        os.path.join(_project_root(), ".eval_cache"),
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return _sha256_bytes(handle.read())


def _normalize_for_cache(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize_for_cache(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_normalize_for_cache(item) for item in value]
    return value


def _sanitize_for_cache_preview(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: Dict[str, Any] = {}
        for key, item in value.items():
            if key.lower() == "api_key":
                sanitized[key] = "***"
            else:
                sanitized[key] = _sanitize_for_cache_preview(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_for_cache_preview(item) for item in value]
    return value


def _config_payload_for_cache(config_path: str) -> Any:
    with open(config_path, "r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle) or {}
    return _normalize_for_cache(parsed)


def _dataset_name_from_path(path: str) -> str:
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if parent:
        return parent
    return os.path.splitext(os.path.basename(path))[0]


def _eval_cache_paths(
    qa_json_path: str,
    corpus_json_path: str,
    config_path: str,
    eval_mode: str,
    multimodal: bool,
) -> Tuple[str, str, Dict[str, Any]]:
    config_payload = _config_payload_for_cache(config_path)
    cache_meta = {
        "schema": "raisex-eval-cache-v1",
        "modality": "multimodal" if multimodal else "text",
        "eval_mode": eval_mode,
        "qa_sha256": _sha256_file(qa_json_path),
        "corpus_sha256": _sha256_file(corpus_json_path),
        "config": _sanitize_for_cache_preview(config_payload),
    }
    key_payload = {
        "schema": cache_meta["schema"],
        "modality": cache_meta["modality"],
        "eval_mode": cache_meta["eval_mode"],
        "qa_sha256": cache_meta["qa_sha256"],
        "corpus_sha256": cache_meta["corpus_sha256"],
        "config": config_payload,
    }
    key = _sha256_bytes(
        json.dumps(key_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    dataset = _dataset_name_from_path(corpus_json_path)
    cache_dir = os.path.join(
        _eval_cache_root(),
        "v1",
        cache_meta["modality"],
        eval_mode,
        dataset,
        key[:2],
    )
    cache_path = os.path.join(cache_dir, f"{key}.json")
    lock_path = os.path.join(cache_dir, f"{key}.lock")
    return cache_path, lock_path, cache_meta


def _read_eval_cache(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        result = payload.get("result")
        return result if isinstance(result, dict) else None
    except Exception:
        return None


def _write_eval_cache(path: str, meta: Dict[str, Any], result: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    payload = {
        "schema": meta.get("schema", "raisex-eval-cache-v1"),
        "created_at": time.time(),
        "meta": meta,
        "result": result,
    }
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _evaluate_with_cache(
    qa_json_path: str,
    corpus_json_path: str,
    config_path: str,
    eval_mode: str,
    multimodal: bool,
) -> Dict[str, Any]:
    qa_items = _load_json_or_jsonl(qa_json_path)
    queries, references_list = _extract_qa(qa_items)
    cache_path, lock_path, cache_meta = _eval_cache_paths(
        qa_json_path=qa_json_path,
        corpus_json_path=corpus_json_path,
        config_path=config_path,
        eval_mode=eval_mode,
        multimodal=multimodal,
    )
    cached = _read_eval_cache(cache_path)
    if cached is not None:
        return cached

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        cached = _read_eval_cache(cache_path)
        if cached is not None:
            return cached

        if multimodal:
            result = asyncio.run(
                run_batch_async_multimodal(
                    queries=queries,
                    selection_path=config_path,
                    data_json_path=corpus_json_path,
                    references_list=references_list,
                    answers_list=None,
                    eval_mode=eval_mode,
                    debug_dump=False,
                )
            )
        else:
            result = asyncio.run(
                run_batch_async(
                    queries=queries,
                    selection_path=config_path,
                    data_json_path=corpus_json_path,
                    references_list=references_list,
                    answers_list=None,
                    eval_mode=eval_mode,
                    debug_dump=False,
                )
            )

        public_result = {
            "eval_report": result.get("report"),
            "outputs": result.get("outputs"),
            "chunking": result.get("chunking"),
        }
        _write_eval_cache(cache_path, cache_meta, public_result)
        return public_result


def evaluate_rag(
    qa_json_path: str,
    corpus_json_path: str,
    config_path: str,
    eval_mode: str = "both",
) -> Dict[str, Any]:
    check = check_config(config_path)
    if not check.get("is_valid", False):
        return {"error": "invalid_config", "errors": check.get("errors", [])}
    return _evaluate_with_cache(
        qa_json_path=qa_json_path,
        corpus_json_path=corpus_json_path,
        config_path=config_path,
        eval_mode=eval_mode,
        multimodal=False,
    )


def evaluate_rag_multimodal(
    qa_json_path: str,
    corpus_json_path: str,
    config_path: str,
    eval_mode: str = "both",
) -> Dict[str, Any]:
    check = check_config_multimodal(config_path)
    if not check.get("is_valid", False):
        return {"error": "invalid_config", "errors": check.get("errors", [])}
    return _evaluate_with_cache(
        qa_json_path=qa_json_path,
        corpus_json_path=corpus_json_path,
        config_path=config_path,
        eval_mode=eval_mode,
        multimodal=True,
    )


# you can use if you want, but may exceed the actual upperbound that config really can access
def theoretical_getupperbound(
    qa_json_path: str,
    corpus_json_path: str,
    config_path: str,
    eval_mode: str = "both",
) -> Dict[str, Any]:
    result = getupperbound_external_pipeline(
        qa_json_path=qa_json_path,
        corpus_json_path=corpus_json_path,
        config_path=config_path,
        eval_mode=eval_mode,
    )
    return {
        "eval_report": result.get("report"),
        "outputs": result.get("outputs"),
    }


def theoretical_getupperbound_multimodal(
    qa_json_path: str,
    corpus_json_path: str,
    config_path: str,
    eval_mode: str = "both",
) -> Dict[str, Any]:
    result = getupperbound_external_multimodal(
        qa_json_path=qa_json_path,
        corpus_json_path=corpus_json_path,
        config_path=config_path,
        eval_mode=eval_mode,
    )
    return {
        "eval_report": result.get("report"),
        "outputs": result.get("outputs"),
    }


def run_algorithms(
    qa_json_path: str,
    corpus_json_path: str,
    config_path: str,
    algorithms: Optional[List[str]] = None,
    eval_mode: str = "both",
    score_weights: str = "",
    seed: Optional[int] = None,
    max_evals: Optional[int] = None,
    extra_args: Optional[Dict[str, List[str]]] = None,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    algo_names = algorithms or list(_DEFAULT_ALGORITHMS)
    project_root = _project_root()
    results: List[Dict[str, Any]] = []
    for name in algo_names:
        module_name = name[:-3] if name.endswith(".py") else name
        spec = importlib.util.find_spec(f"raisex.search.algorithms.{module_name}")
        if spec is None:
            results.append(
                {
                    "algorithm": name,
                    "error": "module_not_found",
                    "module_name": module_name,
                }
            )
            continue
        cmd = [
            sys.executable,
            "-m",
            "raisex.cli.algo_cli",
            "--algorithm",
            module_name,
            "--qa_json",
            qa_json_path,
            "--corpus_json",
            corpus_json_path,
            "--config_yaml",
            config_path,
            "--eval_mode",
            eval_mode,
        ]
        if score_weights:
            cmd.extend(["--score_weights", score_weights])
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        if max_evals is not None:
            cmd.extend(["--max_evals", str(max_evals)])
        if extra_args and name in extra_args:
            cmd.extend(extra_args[name])
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        src_path = _project_src()
        env["PYTHONPATH"] = (
            f"{src_path}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else src_path
        )
        completed = subprocess.run(
            cmd,
            cwd=cwd or project_root,
            capture_output=True,
            text=True,
            env=env,
        )
        results.append(
            {
                "algorithm": name,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "cmd": cmd,
            }
        )
    return {"results": results}


def main() -> None:
    if len(sys.argv) < 4:
        print(
            "Usage:\n"
            "  python -m raisex.cli.eval_cli <qa_json> <corpus_json> <config_yaml> [eval_mode]\n"
            "  python -m raisex.cli.eval_cli multimodal <qa_json> <corpus_json> <config_yaml> [eval_mode]"
        )
        sys.exit(1)
    if sys.argv[1] == "multimodal":
        if len(sys.argv) < 5:
            print(
                "Usage: python -m raisex.cli.eval_cli multimodal <qa_json> <corpus_json> <config_yaml> [eval_mode]"
            )
            sys.exit(1)
        qa_json_path = sys.argv[2]
        corpus_json_path = sys.argv[3]
        config_path = sys.argv[4]
        eval_mode = sys.argv[5] if len(sys.argv) > 5 else "both"
        result = evaluate_rag_multimodal(
            qa_json_path, corpus_json_path, config_path, eval_mode=eval_mode
        )
        if result.get("error"):
            print(json.dumps(result, ensure_ascii=False, indent=2))
            sys.exit(2)
        outputs = result.get("outputs") or []
        report = result.get("eval_report") or {}
        per_item = report.get("per_item") or []
        item_count = min(len(outputs), len(per_item)) if per_item else len(outputs)
        item_summaries: List[Dict[str, Any]] = []
        for idx in range(item_count):
            output = outputs[idx] if idx < len(outputs) else {}
            scores = per_item[idx] if idx < len(per_item) else {}
            image_count = 0
            image_retrieval = output.get("image_retrieval")
            if isinstance(image_retrieval, list):
                image_count = len(image_retrieval)
            item_summaries.append(
                {
                    "index": idx,
                    "image_count": image_count,
                    "answer": output.get("answer", ""),
                    "references": scores.get("references") or [],
                    "llmaaj_reason": scores.get("LLMAAJ_reason") or "",
                    "score": scores,
                }
            )
        print(json.dumps(item_summaries, ensure_ascii=False, indent=2))
    else:
        qa_json_path = sys.argv[1]
        corpus_json_path = sys.argv[2]
        config_path = sys.argv[3]
        eval_mode = sys.argv[4] if len(sys.argv) > 4 else "both"
        result = evaluate_rag(qa_json_path, corpus_json_path, config_path, eval_mode=eval_mode)
        if result.get("error"):
            print(json.dumps(result, ensure_ascii=False, indent=2))
            sys.exit(2)
        outputs = result.get("outputs") or []
        report = result.get("eval_report") or {}
        per_item = report.get("per_item") or []
        item_count = min(len(outputs), len(per_item)) if per_item else len(outputs)
        item_summaries: List[Dict[str, Any]] = []
        for idx in range(item_count):
            output = outputs[idx] if idx < len(outputs) else {}
            scores = per_item[idx] if idx < len(per_item) else {}
            item_summaries.append(
                {
                    "index": idx,
                    "answer": output.get("answer", ""),
                    "references": scores.get("references") or [],
                    "llmaaj_reason": scores.get("LLMAAJ_reason") or "",
                    "score": scores,
                }
            )
        if item_summaries:
            print(json.dumps(item_summaries, ensure_ascii=False, indent=2))
    metrics = (result.get("eval_report") or {}).get("metrics") or {}
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
