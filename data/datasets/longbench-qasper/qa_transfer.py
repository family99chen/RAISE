import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple


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


def _build_outputs(
    items: List[Dict[str, Any]],
    sample_size: int | None,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    qa: List[Dict[str, Any]] = []
    corpus: List[Dict[str, Any]] = []

    if sample_size is not None and sample_size > 0 and sample_size < len(items):
        random.seed(seed)
        items = random.sample(items, sample_size)

    for item in items:
        qid = item.get("_id") or item.get("id") or str(len(corpus))
        query = item.get("input") or item.get("question")
        context = item.get("context")
        answers = item.get("answers") or item.get("answer") or item.get("references")

        if not query or not context:
            continue

        corpus.append({"id": str(qid), "content": str(context)})

        if answers is None:
            references = []
        elif isinstance(answers, list):
            references = [str(a) for a in answers]
        else:
            references = [str(answers)]

        qa.append({"query": str(query), "references": references})
    return qa, corpus


def main() -> None:
    if len(sys.argv) < 4:
        print(
            "Usage: python qa_transfer.py <input.jsonl> <qa_out.json> <corpus_out.json> [sample_size] [seed]"
        )
        sys.exit(1)

    input_path = sys.argv[1]
    qa_out = sys.argv[2]
    corpus_out = sys.argv[3]
    sample_size = int(sys.argv[4]) if len(sys.argv) > 4 else None
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42

    items = _read_jsonl(input_path)
    qa, corpus = _build_outputs(items, sample_size=sample_size, seed=seed)

    with open(qa_out, "w", encoding="utf-8") as handle:
        json.dump(qa, handle, ensure_ascii=False, indent=2)
    with open(corpus_out, "w", encoding="utf-8") as handle:
        json.dump(corpus, handle, ensure_ascii=False, indent=2)

    print(f"qa items: {len(qa)}")
    print(f"corpus items: {len(corpus)}")


if __name__ == "__main__":
    main()
