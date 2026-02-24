import argparse
import json
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from datasets import load_dataset


def _iter_dataset(
    subset: str,
    split: str,
    streaming: bool,
) -> Iterable[Dict[str, Any]]:
    return load_dataset("hotpotqa/hotpot_qa", subset, split=split, streaming=streaming)


def _flatten_contexts(
    titles: Sequence[str],
    sentences_by_title: Sequence[Sequence[str]],
    scope: str,
    supporting_titles: Set[str],
    supporting_sent_ids: Set[tuple[str, int]],
) -> List[str]:
    contexts: List[str] = []
    for title, sentences in zip(titles, sentences_by_title):
        if scope == "matched":
            if title not in supporting_titles:
                continue
            for sent_idx, sentence in enumerate(sentences):
                if (title, sent_idx) in supporting_sent_ids:
                    contexts.append(sentence)
        else:
            for sentence in sentences:
                contexts.append(sentence)
    return contexts


def _dedup_contexts(contexts: List[str]) -> List[str]:
    seen: Set[str] = set()
    deduped: List[str] = []
    for ctx in contexts:
        if ctx in seen:
            continue
        seen.add(ctx)
        deduped.append(ctx)
    return deduped


def _extract_contexts(
    item: Dict[str, Any],
    scope: str,
    dedup: bool,
) -> List[str]:
    context = item.get("context") or {}
    titles = context.get("title") or []
    sentences_by_title = context.get("sentences") or []
    supporting = item.get("supporting_facts") or {}
    supporting_titles = set(supporting.get("title") or [])
    supporting_pairs = set(
        (str(t), int(s))
        for t, s in zip(
            supporting.get("title") or [],
            supporting.get("sent_id") or [],
        )
        if t is not None and s is not None
    )

    contexts = _flatten_contexts(
        titles, sentences_by_title, scope, supporting_titles, supporting_pairs
    )
    if dedup:
        contexts = _dedup_contexts(contexts)
    return contexts


def _sample_dataset(
    dataset: Iterable[Dict[str, Any]],
    sample_size: int,
    seed: int,
    streaming: bool,
) -> Iterable[Dict[str, Any]]:
    if sample_size <= 0:
        return dataset
    if streaming:
        return dataset.shuffle(seed=seed, buffer_size=10_000).take(sample_size)
    return dataset.shuffle(seed=seed).select(range(sample_size))


def _build_corpus(
    dataset: Iterable[Dict[str, Any]],
    scope: str,
    dedup: bool,
) -> List[Dict[str, Any]]:
    corpus: List[Dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        qid = item.get("id") or item.get("qid") or str(idx)
        contexts = _extract_contexts(item, scope=scope, dedup=dedup)
        for cidx, ctx in enumerate(contexts):
            corpus.append({"id": f"{qid}_{cidx}", "content": ctx})
    return corpus


def _build_qa(
    dataset: Iterable[Dict[str, Any]],
    scope: str,
    dedup: bool,
) -> List[Dict[str, Any]]:
    qa: List[Dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        qid = item.get("id") or item.get("qid") or str(idx)
        query = item.get("question")
        answers = item.get("answer")
        contexts = _extract_contexts(item, scope=scope, dedup=dedup)
        if not query or not contexts:
            continue
        if answers is None:
            references: List[str] = []
        elif isinstance(answers, list):
            references = [str(a) for a in answers]
        else:
            references = [str(answers)]
        qa.append({"id": str(qid), "query": str(query), "references": references})
    return qa


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HotpotQA and sample QA pairs.")
    parser.add_argument("--subset", default="distractor", help="Dataset subset.")
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument("--qa_output", required=True, help="Output QA JSON path.")
    parser.add_argument("--corpus_output", required=True, help="Output corpus JSON path.")
    parser.add_argument("--sample_size", type=int, default=0, help="Number of QA samples (0=all).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--context_scope",
        choices=["matched", "all"],
        default="matched",
        help="Use matched contexts or all contexts for QA/corpus.",
    )
    parser.add_argument(
        "--dedup_contexts",
        action="store_true",
        help="Deduplicate contexts per question.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for dataset loading.",
    )
    parser.add_argument(
        "--corpus_scope",
        choices=["sampled", "all"],
        default="sampled",
        help="Build corpus from sampled QA or full dataset.",
    )
    args = parser.parse_args()

    qa_dataset = _iter_dataset(args.subset, args.split, args.streaming)
    qa_dataset = _sample_dataset(
        dataset=qa_dataset,
        sample_size=args.sample_size,
        seed=args.seed,
        streaming=args.streaming,
    )
    qa_items = list(qa_dataset)
    qa = _build_qa(qa_items, scope=args.context_scope, dedup=args.dedup_contexts)
    if args.corpus_scope == "all":
        corpus_dataset = _iter_dataset(args.subset, args.split, args.streaming)
        corpus = _build_corpus(
            dataset=corpus_dataset,
            scope=args.context_scope,
            dedup=args.dedup_contexts,
        )
    else:
        corpus = _build_corpus(
            dataset=qa_items,
            scope=args.context_scope,
            dedup=args.dedup_contexts,
        )
    with open(args.corpus_output, "w", encoding="utf-8") as handle:
        json.dump(corpus, handle, ensure_ascii=False, indent=2)
    with open(args.qa_output, "w", encoding="utf-8") as handle:
        json.dump(qa, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
