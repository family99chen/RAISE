import os
import re
import sys
import time
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple

_BERT_SCORER_CACHE: Dict[Tuple[str, Optional[int], str], Any] = {}


def clear_eval_cache() -> None:
    _BERT_SCORER_CACHE.clear()

try:
    from tqdm import tqdm
except Exception:
    tqdm = None
from raisex.llmfactory.llmfactory import create_llm

try:
    import torch
except Exception:
    torch = None


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


def _load_pipeline_config() -> Dict[str, Any]:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(base_dir, "pipelineconfig.yaml")
    config = _load_yaml(config_path)
    if os.getenv("EVAL_DEBUG") == "1":
        print(f"[eval][llmaaj] pipelineconfig={config_path} keys={list(config.keys())}")
    return config


def _get_eval_max_workers() -> int:
    config = _load_pipeline_config().get("eval", {})
    try:
        value = int(config.get("max_workers", 8))
        return value if value > 0 else 8
    except Exception:
        return 8


def _get_eval_item_timeout() -> float:
    config = _load_pipeline_config().get("eval", {})
    try:
        value = float(config.get("per_item_timeout_seconds", 120))
        return value if value > 0 else 0.0
    except Exception:
        return 120.0


def _tokenize(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if " " in text:
        return [t for t in text.split() if t]
    if re.search(r"[\u4e00-\u9fff]", text):
        return list(text)
    return [text]


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _clean_text(text: str) -> str:
    return _normalize_text(text)


def exact_match(pred: str, refs: List[str]) -> float:
    pred_norm = _clean_text(pred)
    for ref in refs:
        if pred_norm == _clean_text(ref):
            return 1.0
    return 0.0


def f1_score(pred: str, refs: List[str]) -> float:
    pred_tokens = _tokenize(_clean_text(pred))
    if not pred_tokens:
        return 0.0
    best = 0.0
    for ref in refs:
        ref_tokens = _tokenize(_clean_text(ref))
        if not ref_tokens:
            continue
        common: Dict[str, int] = {}
        for t in pred_tokens:
            common[t] = common.get(t, 0) + 1
        match = 0
        for t in ref_tokens:
            if common.get(t, 0) > 0:
                common[t] -= 1
                match += 1
        precision = match / len(pred_tokens)
        recall = match / len(ref_tokens)
        if precision + recall == 0:
            score = 0.0
        else:
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def rouge_l(pred: str, refs: List[str]) -> float:
    def lcs(a: List[str], b: List[str]) -> int:
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]

    pred_tokens = _tokenize(_clean_text(pred))
    if not pred_tokens:
        return 0.0
    best = 0.0
    for ref in refs:
        ref_tokens = _tokenize(_clean_text(ref))
        if not ref_tokens:
            continue
        lcs_len = lcs(pred_tokens, ref_tokens)
        precision = lcs_len / len(pred_tokens)
        recall = lcs_len / len(ref_tokens)
        if precision + recall == 0:
            score = 0.0
        else:
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def bleu_score(pred: str, refs: List[str]) -> Optional[float]:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except Exception as exc:
        if os.getenv("EVAL_DEBUG") == "1":
            return {"score": None, "reason": "", "raw": f"ERROR: {type(exc).__name__}: {exc}"}
        return {"score": None, "reason": "", "raw": f"ERROR: {type(exc).__name__}"}
    try:
        pred_tokens = _tokenize(_clean_text(pred))
        ref_tokens = [_tokenize(_clean_text(r)) for r in refs]
        if not pred_tokens or not ref_tokens:
            return 0.0
        smoothie = SmoothingFunction().method1
        return float(sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie))
    except Exception:
        return None


def meteor_score_value(pred: str, refs: List[str]) -> Optional[float]:
    try:
        from nltk.translate.meteor_score import meteor_score
    except Exception:
        return None
    try:
        pred_tokens = _tokenize(_clean_text(pred))
        refs_tokens = [_tokenize(_clean_text(r)) for r in refs if r]
        if not pred_tokens or not refs_tokens:
            return 0.0
        return float(meteor_score(refs_tokens, pred_tokens))
    except Exception:
        return None


def _get_bert_model_path(eval_cfg: Optional[Dict[str, Any]]) -> Optional[str]:
    if not eval_cfg:
        return None
    bert_cfg = eval_cfg.get("bert", {})
    path = bert_cfg.get("model_url")
    if not path:
        return None
    return path if os.path.exists(path) else None


def _get_bert_num_layers(eval_cfg: Optional[Dict[str, Any]], model_path: Optional[str]) -> Optional[int]:
    if not eval_cfg:
        return None
    bert_cfg = eval_cfg.get("bert", {})
    num_layers = bert_cfg.get("num_layers")
    if isinstance(num_layers, int):
        return num_layers
    if model_path and "bert-base-uncased" in model_path:
        return 12
    return None


def _debug_bert(message: str) -> None:
    if os.getenv("EVAL_DEBUG") == "1":
        print(f"[eval][bert] {message}")


def _normalize_metric_names(tokens: List[str]) -> Optional[set[str]]:
    mapping = {
        "llmaaj": "LLMAAJ",
        "bertf1": "BERTScore-F1",
        "bert": "BERTScore-F1",
        "bertscore-f1": "BERTScore-F1",
        "bertscore": "BERTScore-F1",
        "rougel": "ROUGE-L",
        "rouge-l": "ROUGE-L",
        "meteor": "METEOR",
        "f1": "F1",
        "bleu": "BLEU",
        "exactmatch": "ExactMatch",
        "em": "ExactMatch",
    }
    enabled: set[str] = set()
    for token in tokens:
        if not token:
            continue
        key = token.strip()
        mapped = mapping.get(key.lower())
        enabled.add(mapped or key)
    return enabled or None


def _parse_enabled_metrics_from_cfg(
    eval_cfg: Optional[Dict[str, Any]],
) -> Optional[set[str]]:
    if not eval_cfg:
        return None
    raw = eval_cfg.get("enabled")
    if raw is None:
        return None
    if isinstance(raw, str):
        tokens = [t for t in re.split(r"[,\s]+", raw.strip()) if t]
        return _normalize_metric_names(tokens)
    if isinstance(raw, list):
        tokens = [str(t) for t in raw if t is not None]
        return _normalize_metric_names(tokens)
    return None


def _is_metric_enabled(metric_name: str, eval_cfg: Optional[Dict[str, Any]]) -> bool:
    enabled = _parse_enabled_metrics_from_cfg(eval_cfg)
    if enabled is None:
        return True
    return metric_name in enabled


def _is_bert_enabled(eval_cfg: Optional[Dict[str, Any]]) -> bool:
    if os.getenv("EVAL_DISABLE_BERT") == "1":
        return False
    if not eval_cfg:
        return False
    bert_cfg = eval_cfg.get("bert")
    if not isinstance(bert_cfg, dict):
        return False
    if bert_cfg.get("enabled") is False:
        return False
    return True


def _get_llmaaj_prompt() -> str:
    config = _load_pipeline_config().get("llmaaj", {})
    prompt = config.get(
        "prompt",
        "You are a judge. Determine if the answer is correct based on the reference list.\n"
        "Rules:\n- Be a careful judge. Irrelevant extra words are allowed.\n"
        "- The answer must match a reference item exactly or be a very close paraphrase with the same meaning.\n"
        "- If only partially related, overly general, or only loosely similar, it is incorrect.\n"
        "Return JSON only: {{\"score\": 0 or 1, \"reason\": \"short explanation\"}}.\n"
        "Query: {query}\nReference list: {reference}\nAnswer: {answer}",
    )
    if os.getenv("EVAL_DEBUG") == "1":
        print(f"[eval][llmaaj] prompt={prompt!r}")
    return prompt


def _bert_f1_pair(
    preds: List[str],
    refs: List[str],
    eval_cfg: Optional[Dict[str, Any]],
) -> Optional[List[float]]:
    try:
        from bert_score import BERTScorer
    except Exception:
        return None
    try:
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        refs = [_clean_text(r) for r in refs]
        preds = [_clean_text(p) for p in preds]
        model_path = _get_bert_model_path(eval_cfg)
        num_layers = _get_bert_num_layers(eval_cfg, model_path)
        _debug_bert(f"device={device} model_path={model_path or 'auto'}")
        cache_key = (model_path or "auto", num_layers, device)
        scorer = _BERT_SCORER_CACHE.get(cache_key)
        if scorer is None:
            if model_path:
                scorer = BERTScorer(
                    model_type=model_path,
                    lang=None,
                    device=device,
                    rescale_with_baseline=False,
                    num_layers=num_layers,
                )
            else:
                scorer = BERTScorer(
                    lang="en",
                    device=device,
                    rescale_with_baseline=False,
                )
            _BERT_SCORER_CACHE[cache_key] = scorer
        _, _, f1 = scorer.score(preds, refs)
        return [float(x) for x in f1.tolist()]
    except Exception as exc:
        _debug_bert(f"error={type(exc).__name__}: {exc}")
        return None


def bert_f1(
    preds: List[str], refs_list: List[List[str]], eval_cfg: Optional[Dict[str, Any]]
) -> Optional[float]:
    per_item = bert_f1_per_item(preds, refs_list, eval_cfg)
    if per_item is None:
        return None
    return sum(per_item) / len(per_item) if per_item else 0.0


def bert_f1_per_item(
    preds: List[str],
    refs_list: List[List[str]],
    eval_cfg: Optional[Dict[str, Any]],
) -> Optional[List[float]]:
    if not preds:
        return None
    max_scores = [0.0 for _ in preds]
    for i, refs in enumerate(refs_list):
        if not refs:
            continue
        best = 0.0
        for ref in refs:
            scores = _bert_f1_pair(preds=[preds[i]], refs=[ref], eval_cfg=eval_cfg)
            if scores:
                best = max(best, scores[0])
        max_scores[i] = best
    return max_scores


def evaluate_metrics(
    preds: List[str],
    refs_list: List[List[str]],
    eval_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not preds:
        return {}

    metrics: Dict[str, Any] = {}

    try:
        if _is_metric_enabled("ExactMatch", eval_cfg):
            exact = [exact_match(p, r) for p, r in zip(preds, refs_list)]
            metrics["ExactMatch"] = sum(exact) / len(exact) if exact else 0.0
    except Exception:
        metrics["ExactMatch"] = 0.0

    try:
        if _is_metric_enabled("F1", eval_cfg):
            f1s = [f1_score(p, r) for p, r in zip(preds, refs_list)]
            metrics["F1"] = sum(f1s) / len(f1s) if f1s else 0.0
    except Exception:
        metrics["F1"] = 0.0

    try:
        if _is_metric_enabled("ROUGE-L", eval_cfg):
            rouges = [rouge_l(p, r) for p, r in zip(preds, refs_list)]
            metrics["ROUGE-L"] = sum(rouges) / len(rouges) if rouges else 0.0
    except Exception:
        metrics["ROUGE-L"] = 0.0

    try:
        if _is_metric_enabled("BLEU", eval_cfg):
            bleus = [bleu_score(p, r) for p, r in zip(preds, refs_list)]
            bleu_vals = [b for b in bleus if b is not None]
            metrics["BLEU"] = (sum(bleu_vals) / len(bleu_vals)) if bleu_vals else 0.0
    except Exception:
        metrics["BLEU"] = 0.0

    try:
        if _is_metric_enabled("METEOR", eval_cfg):
            meteors = [meteor_score_value(p, r) for p, r in zip(preds, refs_list)]
            meteor_vals = [m for m in meteors if m is not None]
            metrics["METEOR"] = (sum(meteor_vals) / len(meteor_vals)) if meteor_vals else 0.0
    except Exception:
        metrics["METEOR"] = 0.0

    try:
        if _is_metric_enabled("BERTScore-F1", eval_cfg) and _is_bert_enabled(eval_cfg):
            bert = bert_f1(preds, refs_list, eval_cfg)
            metrics["BERTScore-F1"] = 0.0 if bert is None else bert
        else:
            metrics["BERTScore-F1"] = 0.0
    except Exception:
        metrics["BERTScore-F1"] = 0.0

    return metrics


def _clean_llmaaj_output(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def _parse_llmaaj_output(text: str) -> Tuple[Optional[int], str, str]:
    cleaned = _clean_llmaaj_output(text)
    if not cleaned:
        return None, "", cleaned
    json_match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if json_match:
        json_text = json_match.group(0)
        try:
            import json

            payload = json.loads(json_text)
            if isinstance(payload, dict):
                raw_score = payload.get("score")
                if raw_score is None:
                    raw_score = payload.get("label") or payload.get("result") or payload.get("correct")
                score = None
                if isinstance(raw_score, bool):
                    score = 1 if raw_score else 0
                elif isinstance(raw_score, (int, float)):
                    score = int(raw_score)
                elif isinstance(raw_score, str):
                    match = re.search(r"\b([01])\b", raw_score.strip())
                    if match:
                        score = int(match.group(1))
                reason = payload.get("reason") or payload.get("explanation") or payload.get("rationale") or ""
                return score, str(reason).strip(), cleaned
        except Exception:
            pass
    first_line = cleaned.splitlines()[0].strip()
    match = re.match(r"^\s*([01])\b[:\-]?\s*(.*)$", first_line)
    if match:
        reason = match.group(2).strip()
        return int(match.group(1)), reason, cleaned
    match = re.search(r"\b([01])\b", cleaned)
    if match:
        return int(match.group(1)), cleaned, cleaned
    return None, cleaned, cleaned


def llmaaj_judge(
    query: str, answer: str, reference: str, llmaaj_cfg: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    url = llmaaj_cfg.get("model_url")
    if not url:
        return None
    prompt = _get_llmaaj_prompt()
    model_name = llmaaj_cfg.get("model_name") or None
    api_key = llmaaj_cfg.get("api_key") or None

    try:
        llm = create_llm(url=url, api_key=api_key, model_name=model_name)
        user_prompt = prompt.format(query=query, answer=answer, reference=reference)
        output = llm.generate(user_prompt)
        score, reason, raw = _parse_llmaaj_output(output)
        if os.getenv("EVAL_DEBUG") == "1":
            print(f"[eval][llmaaj] raw={raw!r}")
            print(f"[eval][llmaaj] parsed_score={score} reason={reason!r}")
        return {"score": score, "reason": reason, "raw": raw}
    except Exception as exc:
        if os.getenv("EVAL_DEBUG") == "1":
            print(f"[eval][llmaaj] error={type(exc).__name__}: {exc}")
            return {"score": None, "reason": "", "raw": f"ERROR: {type(exc).__name__}: {exc}"}
        return {"score": None, "reason": "", "raw": f"ERROR: {type(exc).__name__}"}


def _zero_metrics() -> Dict[str, float]:
    return {
        "ExactMatch": 0.0,
        "F1": 0.0,
        "ROUGE-L": 0.0,
        "BLEU": 0.0,
        "METEOR": 0.0,
        "BERTScore-F1": 0.0,
    }


def _compute_per_item(
    query: str,
    pred: str,
    refs: List[str],
    llmaaj_cfg: Optional[Dict[str, Any]],
    eval_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    enable_exact = _is_metric_enabled("ExactMatch", eval_cfg)
    enable_f1 = _is_metric_enabled("F1", eval_cfg)
    enable_rouge = _is_metric_enabled("ROUGE-L", eval_cfg)
    enable_bleu = _is_metric_enabled("BLEU", eval_cfg)
    enable_meteor = _is_metric_enabled("METEOR", eval_cfg)
    enable_llmaaj = _is_metric_enabled("LLMAAJ", eval_cfg)
    item = {
        "ExactMatch": exact_match(pred, refs) if enable_exact else 0.0,
        "F1": f1_score(pred, refs) if enable_f1 else 0.0,
        "ROUGE-L": rouge_l(pred, refs) if enable_rouge else 0.0,
        "BLEU": (bleu_score(pred, refs) or 0.0) if enable_bleu else 0.0,
        "METEOR": (meteor_score_value(pred, refs) or 0.0) if enable_meteor else 0.0,
        "LLMAAJ": 0.0,
        "LLMAAJ_reason": "",
        "LLMAAJ_raw": "",
    }
    if enable_llmaaj and llmaaj_cfg:
        reference_list = [r for r in refs if r]
        if reference_list:
            reference_text = "\n".join(f"- {r}" for r in reference_list)
            judged = llmaaj_judge(query, pred, reference_text, llmaaj_cfg)
            if judged:
                score = judged.get("score")
                item["LLMAAJ"] = float(score) if score in (0, 1) else 0.0
                item["LLMAAJ_reason"] = str(judged.get("reason") or "").strip()
                item["LLMAAJ_raw"] = str(judged.get("raw") or "").strip()
        else:
            item["LLMAAJ"] = 0.0
    return item


def evaluate_report(
    preds: List[str],
    refs_list: List[List[str]],
    queries: Optional[List[str]] = None,
    mode: str = "both",
    eval_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not preds or not refs_list or len(preds) != len(refs_list):
        return {"metrics": _zero_metrics(), "per_item": []}

    llmaaj_cfg = (eval_cfg or {}).get("llmaaj") if eval_cfg else None
    max_workers = _get_eval_max_workers()

    per_item: List[Dict[str, Any]] = []
    need_per_item = mode in {"per_item", "both"}
    need_llmaaj = _is_metric_enabled("LLMAAJ", eval_cfg) and llmaaj_cfg
    if need_per_item or need_llmaaj:
        indexed = list(enumerate(zip(queries or ["" for _ in preds], preds, refs_list)))
        results: List[Optional[Dict[str, Any]]] = [None] * len(indexed)
        try:
            per_item_timeout = _get_eval_item_timeout()
            default_item = {
                "ExactMatch": 0.0,
                "F1": 0.0,
                "ROUGE-L": 0.0,
                "BLEU": 0.0,
                "METEOR": 0.0,
                "LLMAAJ": 0.0,
            }
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_compute_per_item, q, p, r, llmaaj_cfg, eval_cfg): idx
                    for idx, (q, p, r) in indexed
                }
                pending = set(futures.keys())
                start_times = {f: time.monotonic() for f in pending}
                bar = (
                    tqdm(total=len(futures), desc="eval", unit="qa", file=sys.stdout)
                    if tqdm is not None
                    else None
                )
                while pending:
                    done, not_done = concurrent.futures.wait(
                        pending,
                        timeout=0.1,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        idx = futures[future]
                        try:
                            results[idx] = future.result()
                        except Exception:
                            results[idx] = dict(default_item)
                        if bar:
                            bar.update(1)
                    pending = not_done
                    if per_item_timeout > 0:
                        now = time.monotonic()
                        timed_out = [
                            f for f in pending if (now - start_times.get(f, now)) > per_item_timeout
                        ]
                        for future in timed_out:
                            future.cancel()
                            idx = futures[future]
                            results[idx] = dict(default_item)
                            pending.remove(future)
                            if bar:
                                bar.update(1)
                if bar:
                    bar.close()
        except Exception:
            results = [dict(default_item) for _ in preds]

        per_item = [r if r is not None else {
            "ExactMatch": 0.0,
            "F1": 0.0,
            "ROUGE-L": 0.0,
            "BLEU": 0.0,
            "METEOR": 0.0,
            "LLMAAJ": 0.0,
            "LLMAAJ_reason": "",
            "LLMAAJ_raw": "",
        } for r in results]

        if _is_metric_enabled("BERTScore-F1", eval_cfg) and _is_bert_enabled(eval_cfg):
            bert_items = bert_f1_per_item(preds, refs_list, eval_cfg)
            if bert_items is None:
                for item in per_item:
                    item["BERTScore-F1"] = 0.0
            else:
                for item, score in zip(per_item, bert_items):
                    item["BERTScore-F1"] = score
        else:
            for item in per_item:
                item["BERTScore-F1"] = 0.0
        for idx, item in enumerate(per_item):
            item["answer"] = preds[idx] if idx < len(preds) else ""
            item["references"] = refs_list[idx] if idx < len(refs_list) else []

    metrics = evaluate_metrics(preds, refs_list, eval_cfg)
    if not metrics:
        metrics = _zero_metrics()

    report: Dict[str, Any] = {}
    if mode in {"avg", "both"}:
        if per_item:
            metrics["LLMAAJ"] = sum(i["LLMAAJ"] for i in per_item) / len(per_item)
        else:
            metrics["LLMAAJ"] = 0.0
        report["metrics"] = metrics
    if mode in {"per_item", "both"}:
        report["per_item"] = per_item
    return report
