import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple


def _load_problems(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("problems.json must be a dict keyed by id.")
    return data


def _build_question(question: str, choices: List[Any]) -> str:
    lines = [str(question).strip(), "Choices:"]
    for idx, choice in enumerate(choices):
        lines.append(f"{idx}. {choice}")
    return "\n".join(lines)


def _resolve_image_path(data_root: str, split: str, qid: str, image_name: str) -> str:
    return os.path.join(data_root, split, str(qid), image_name)


def _has_image(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _build_outputs(
    problems: Dict[str, Any],
    data_root: str,
    sample_size: int | None,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    for qid, item in problems.items():
        if not isinstance(item, dict):
            continue
        if item.get("split") != "train":
            continue
        hint = item.get("hint")
        if hint is None or str(hint).strip() == "":
            continue
        image_name = item.get("image") or ""
        image_path = _resolve_image_path(data_root, "train", qid, str(image_name))
        if not _has_image(image_path):
            continue
        candidates.append((str(qid), item))

    if sample_size is not None and sample_size > 0 and sample_size < len(candidates):
        random.seed(seed)
        candidates = random.sample(candidates, sample_size)

    qa: List[Dict[str, Any]] = []
    corpus: List[Dict[str, Any]] = []
    for qid, item in candidates:
        question = item.get("question") or ""
        choices = item.get("choices") or []
        answer_idx = item.get("answer")
        hint = str(item.get("hint") or "").strip()
        image_name = str(item.get("image") or "")
        image_path = _resolve_image_path(data_root, "train", qid, image_name)

        if not question or not choices or hint == "":
            continue
        if not isinstance(answer_idx, int) or answer_idx < 0 or answer_idx >= len(choices):
            continue
        if not _has_image(image_path):
            continue

        query = _build_question(str(question), [str(c) for c in choices])
        answer_text = str(choices[answer_idx])

        qa.append({"id": qid, "query": query, "references": [answer_text]})
        corpus.append({"id": qid, "content": hint, "image_path": image_path})

    return qa, corpus


def main() -> None:
    if len(sys.argv) < 4:
        print(
            "Usage: python qa_transfer.py <problems.json> <qa_out.json> <corpus_out.json> [sample_size] [seed] [data_root]"
        )
        sys.exit(1)

    problems_path = sys.argv[1]
    qa_out = sys.argv[2]
    corpus_out = sys.argv[3]
    sample_size = int(sys.argv[4]) if len(sys.argv) > 4 else None
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42
    data_root = (
        sys.argv[6]
        if len(sys.argv) > 6
        else "/home/cz/ragsearch-update/datasets/ScienceQA/data"
    )

    problems = _load_problems(problems_path)
    qa, corpus = _build_outputs(
        problems, data_root=data_root, sample_size=sample_size, seed=seed
    )

    with open(qa_out, "w", encoding="utf-8") as handle:
        json.dump(qa, handle, ensure_ascii=False, indent=2)
    with open(corpus_out, "w", encoding="utf-8") as handle:
        json.dump(corpus, handle, ensure_ascii=False, indent=2)

    print(f"qa items: {len(qa)}")
    print(f"corpus items: {len(corpus)}")


if __name__ == "__main__":
    main()
