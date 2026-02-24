import argparse
import json
import os
import random
from typing import Any, Dict, Iterable, List, Tuple


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _iter_hf(
    config: str,
    split: str,
    cache_dir: str | None,
    streaming: bool,
    seed: int,
    sample_size: int | None,
) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("Please install datasets: pip install datasets") from exc

    dataset = load_dataset(
        "mandarjoshi/trivia_qa",
        config,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
    )
    if sample_size is None or sample_size <= 0:
        return list(dataset)
    if streaming:
        buffer_size = min(max(sample_size * 10, 1000), 100000)
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
        items: List[Dict[str, Any]] = []
        for item in dataset:
            items.append(item)
            if len(items) >= sample_size:
                break
        return items
    return list(dataset.shuffle(seed=seed).select(range(sample_size)))


def _normalize_contexts(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(v) for v in raw if v is not None and str(v).strip() != ""]
    return [str(raw)] if str(raw).strip() else []


def _extract_contexts(item: Dict[str, Any]) -> List[str]:
    search_results = item.get("search_results") or {}
    if isinstance(search_results, dict):
        contexts = _normalize_contexts(search_results.get("search_context"))
        if contexts:
            return contexts
    return []


def _is_valid_item(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    query = item.get("question") or item.get("query")
    if not query:
        return False
    contexts = _extract_contexts(item)
    return bool(contexts)


def _iter_hf_valid(
    config: str,
    split: str,
    cache_dir: str | None,
    seed: int,
    target_count: int,
) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("Please install datasets: pip install datasets") from exc

    dataset = load_dataset(
        "mandarjoshi/trivia_qa",
        config,
        split=split,
        cache_dir=cache_dir,
        streaming=True,
    )
    buffer_size = min(max(target_count * 10, 1000), 100000)
    dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    items: List[Dict[str, Any]] = []
    for item in dataset:
        if not _is_valid_item(item):
            continue
        items.append(item)
        if len(items) >= target_count:
            break
    return items


def _extract_answers(item: Dict[str, Any]) -> List[str]:
    answer = item.get("answer") or {}
    if isinstance(answer, dict):
        aliases = answer.get("aliases") or []
        if isinstance(aliases, list):
            return [str(a) for a in aliases if a is not None]
        if aliases:
            return [str(aliases)]
    return []


def _build_outputs(
    qa_items: List[Dict[str, Any]],
    corpus_items: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    qa: List[Dict[str, Any]] = []
    corpus: List[Dict[str, Any]] = []

    for idx, item in enumerate(qa_items):
        if not isinstance(item, dict):
            continue
        qid = item.get("question_id") or item.get("qid") or str(idx)
        query = item.get("question") or item.get("query")
        if not query:
            continue
        contexts = _extract_contexts(item)
        if not contexts:
            continue
        refs = _extract_answers(item)

        qa.append({"id": str(qid), "query": str(query), "references": refs})

    for idx, item in enumerate(corpus_items):
        if not isinstance(item, dict):
            continue
        qid = item.get("question_id") or item.get("qid") or str(idx)
        contexts = _extract_contexts(item)
        if not contexts:
            continue
        for cidx, ctx in enumerate(contexts):
            corpus.append({"id": f"{qid}_{cidx}", "content": ctx})

    return qa, corpus


def _pick_extra_items(
    all_items: List[Dict[str, Any]],
    qa_items: List[Dict[str, Any]],
    extra_count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if extra_count <= 0:
        return []
    qa_ids = {
        str(item.get("question_id") or item.get("qid") or "")
        for item in qa_items
        if isinstance(item, dict)
    }
    candidates = [
        item
        for item in all_items
        if isinstance(item, dict)
        and str(item.get("question_id") or item.get("qid") or "") not in qa_ids
        and _is_valid_item(item)
    ]
    if not candidates:
        return []
    if extra_count >= len(candidates):
        return candidates
    random.seed(seed)
    return random.sample(candidates, extra_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build QA/corpus from TriviaQA.")
    parser.add_argument("--qa_out", required=True, help="Path to QA output JSON.")
    parser.add_argument("--corpus_out", required=True, help="Path to corpus output JSON.")
    parser.add_argument(
        "--input_jsonl",
        default="",
        help="Optional local JSONL (if provided, HF download is skipped).",
    )
    parser.add_argument("--sample_size", type=int, default=0, help="Sample size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--config",
        default="rc",
        help="TriviaQA config (e.g. rc, unfiltered).",
    )
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument("--cache_dir", default="", help="HF cache dir.")
    parser.add_argument(
        "--corpus_scope",
        choices=["sample", "all"],
        default="sample",
        help="Use sampled QA corpus or full corpus.",
    )
    parser.add_argument(
        "--corpus_extra",
        type=int,
        default=0,
        help="Add extra corpus from random other questions.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming to avoid full download.",
    )
    args = parser.parse_args()

    sample_size = args.sample_size if args.sample_size > 0 else None
    if sample_size is None and args.corpus_scope == "sample" and args.corpus_extra > 0:
        sample_size = args.corpus_extra
        print(f"sample_size not set; default to corpus_extra={sample_size}")
    if args.input_jsonl:
        all_items = _read_jsonl(args.input_jsonl)
    else:
        cache_dir = args.cache_dir or None
        if args.corpus_scope == "sample" and sample_size is not None:
            target = sample_size + max(args.corpus_extra, 0)
            print("enable streaming: avoid full download for sampling")
            all_items = _iter_hf_valid(
                config=args.config,
                split=args.split,
                cache_dir=cache_dir,
                seed=args.seed,
                target_count=target,
            )
        else:
            all_items = _iter_hf(
                config=args.config,
                split=args.split,
                cache_dir=cache_dir,
                streaming=args.streaming,
                seed=args.seed,
                sample_size=None,
            )
    valid_items = [item for item in all_items if _is_valid_item(item)]
    qa_items = valid_items
    if sample_size is not None and sample_size > 0:
        if len(valid_items) < sample_size:
            print(f"warning: valid qa items {len(valid_items)} < sample_size {sample_size}")
        else:
            random.seed(args.seed)
            qa_items = random.sample(valid_items, sample_size)
    corpus_items = all_items if args.corpus_scope == "all" else qa_items
    extra_items = _pick_extra_items(all_items, qa_items, args.corpus_extra, args.seed)
    if extra_items:
        corpus_items = list(corpus_items) + extra_items
    qa, corpus = _build_outputs(qa_items, corpus_items)

    with open(args.qa_out, "w", encoding="utf-8") as handle:
        json.dump(qa, handle, ensure_ascii=False, indent=2)
    with open(args.corpus_out, "w", encoding="utf-8") as handle:
        json.dump(corpus, handle, ensure_ascii=False, indent=2)

    print(f"qa items: {len(qa)}")
    print(f"corpus items: {len(corpus)}")


if __name__ == "__main__":
    main()
