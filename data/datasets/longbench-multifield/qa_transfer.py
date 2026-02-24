import json
import os
import random
import sys
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


def _normalize_contexts(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(v) for v in raw if v is not None]
    if isinstance(raw, dict):
        contexts: List[str] = []
        for value in raw.values():
            if isinstance(value, list):
                contexts.extend([str(v) for v in value if v is not None])
            elif value is not None:
                contexts.append(str(value))
        return contexts
    return [str(raw)]


def _build_outputs(
    items: List[Dict[str, Any]],
    sample_size: int | None,
    corpus_size: int | None,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    qa: List[Dict[str, Any]] = []
    corpus: List[Dict[str, Any]] = []

    rng = random.Random(seed)
    qa_items = items
    if sample_size is not None and sample_size > 0 and sample_size < len(items):
        qa_items = rng.sample(items, sample_size)

    if corpus_size is None:
        corpus_items = list(qa_items)
    else:
        if corpus_size < len(qa_items):
            corpus_size = len(qa_items)
        if corpus_size > len(items):
            corpus_size = len(items)
        remaining = [item for item in items if item not in qa_items]
        extra_needed = max(corpus_size - len(qa_items), 0)
        extra_items = rng.sample(remaining, min(extra_needed, len(remaining)))
        corpus_items = list(qa_items) + extra_items

    for idx, item in enumerate(qa_items):
        qid = item.get("_id") or item.get("id") or item.get("qid") or str(idx)
        query = item.get("input") or item.get("question")
        contexts = _normalize_contexts(item.get("context"))
        answers = item.get("answers") or item.get("answer") or item.get("references")

        if not query or not contexts:
            continue

        if answers is None:
            references: List[str] = []
        elif isinstance(answers, list):
            references = [str(a) for a in answers]
        else:
            references = [str(answers)]

        qa.append({"id": str(qid), "query": str(query), "references": references})

    for idx, item in enumerate(corpus_items):
        qid = item.get("_id") or item.get("id") or item.get("qid") or str(idx)
        contexts = _normalize_contexts(item.get("context"))
        if not contexts:
            continue
        if len(contexts) == 1:
            corpus.append({"id": str(qid), "content": contexts[0]})
        else:
            for cidx, ctx in enumerate(contexts):
                corpus.append({"id": f"{qid}_{cidx}", "content": ctx})

    return qa, corpus


def main() -> None:
    if len(sys.argv) < 4:
        print(
            "Usage: python qa_transfer.py <input.jsonl> <qa_out.json> <corpus_out.json> "
            "[sample_size] [corpus_size] [seed]"
        )
        sys.exit(1)

    input_path = sys.argv[1]
    qa_out = sys.argv[2]
    corpus_out = sys.argv[3]
    sample_size = int(sys.argv[4]) if len(sys.argv) > 4 else None
    corpus_size = None
    seed = 42
    if len(sys.argv) > 5:
        if len(sys.argv) == 6:
            seed = int(sys.argv[5])
        else:
            corpus_size = int(sys.argv[5])
            seed = int(sys.argv[6]) if len(sys.argv) > 6 else 42

    items = _read_jsonl(input_path)
    qa, corpus = _build_outputs(
        items, sample_size=sample_size, corpus_size=corpus_size, seed=seed
    )

    with open(qa_out, "w", encoding="utf-8") as handle:
        json.dump(qa, handle, ensure_ascii=False, indent=2)
    with open(corpus_out, "w", encoding="utf-8") as handle:
        json.dump(corpus, handle, ensure_ascii=False, indent=2)

    print(f"qa items: {len(qa)}")
    print(f"corpus items: {len(corpus)}")


if __name__ == "__main__":
    main()
