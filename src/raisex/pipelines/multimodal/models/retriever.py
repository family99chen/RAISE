import asyncio
import math
from typing import Any, Dict, List, Tuple


def _tokenize(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if " " in text:
        return [t for t in text.split() if t]
    return list(text)


def _bm25_scores(docs: List[str], query: str, k1: float = 1.5, b: float = 0.75) -> List[float]:
    tokens_list = [_tokenize(d) for d in docs]
    avgdl = sum(len(t) for t in tokens_list) / (len(tokens_list) or 1)
    doc_freq: Dict[str, int] = {}
    for tokens in tokens_list:
        for t in set(tokens):
            doc_freq[t] = doc_freq.get(t, 0) + 1

    query_tokens = _tokenize(query)
    scores = [0.0 for _ in docs]
    for i, tokens in enumerate(tokens_list):
        doc_len = len(tokens)
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for q in query_tokens:
            df = doc_freq.get(q, 0)
            if df == 0:
                continue
            idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
            freq = tf.get(q, 0)
            denom = freq + k1 * (1 - b + b * (doc_len / (avgdl or 1)))
            score += idf * (freq * (k1 + 1)) / (denom or 1)
        scores[i] = score
    return scores


def _normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [0.0 for _ in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]


def retrieve(
    query: str,
    collection: Any,
    topk: int,
    bm25_weight: float,
) -> List[Dict[str, Any]]:
    try:
        if topk <= 0 or collection is None:
            return []
        bm25_weight = min(max(bm25_weight, 0.0), 1.0)

        all_data = collection.get(include=["documents", "metadatas"])
        docs = all_data.get("documents", []) or []
        metas = all_data.get("metadatas", []) or []
        ids = all_data.get("ids", []) or []

        if not docs:
            return []

        results: List[Tuple[str, str, Dict[str, Any], float]] = []

        if bm25_weight == 1.0:
            bm25_scores = _bm25_scores(docs, query)
            bm25_norm = _normalize(bm25_scores)
            ranked = sorted(
                zip(ids, docs, metas, bm25_norm),
                key=lambda x: x[3],
                reverse=True,
            )
            for item in ranked[:topk]:
                results.append(item)
        elif bm25_weight == 0.0:
            dense = collection.query(query_texts=[query], n_results=topk)
            dense_ids = dense.get("ids", [[]])[0]
            dense_docs = dense.get("documents", [[]])[0]
            dense_metas = dense.get("metadatas", [[]])[0]
            distances = dense.get("distances", [[]])[0]
            sims = [1.0 / (1.0 + d) for d in distances]
            for idx, doc, meta, sim in zip(dense_ids, dense_docs, dense_metas, sims):
                results.append((idx, doc, meta, sim))
        else:
            bm25_scores = _bm25_scores(docs, query)
            bm25_norm = _normalize(bm25_scores)

            dense = collection.query(query_texts=[query], n_results=len(docs))
            dense_ids = dense.get("ids", [[]])[0]
            distances = dense.get("distances", [[]])[0]
            dense_sims = [1.0 / (1.0 + d) for d in distances]
            dense_norm = _normalize(dense_sims)

            dense_map = {doc_id: dense_norm[i] for i, doc_id in enumerate(dense_ids)}

            hybrid_scores = []
            for i, doc_id in enumerate(ids):
                dense_score = dense_map.get(doc_id, 0.0)
                score = (1.0 - bm25_weight) * dense_score + bm25_weight * bm25_norm[i]
                hybrid_scores.append(score)

            ranked = sorted(
                zip(ids, docs, metas, hybrid_scores),
                key=lambda x: x[3],
                reverse=True,
            )
            for item in ranked[:topk]:
                results.append(item)

        return [
            {
                "id": item[0],
                "document": item[1],
                "metadata": item[2],
                "score": item[3],
            }
            for item in results
        ]
    except Exception:
        return []


async def retrieve_async(
    query: str,
    collection: Any,
    topk: int,
    bm25_weight: float,
) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(
        retrieve,
        query=query,
        collection=collection,
        topk=topk,
        bm25_weight=bm25_weight,
    )