import asyncio
from typing import Any, Dict, List

_RERANKER_CACHE: Dict[str, Any] = {}


def clear_reranker_cache() -> None:
    _RERANKER_CACHE.clear()


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    model_url: str,
    topk: int,
) -> List[Dict[str, Any]]:
    try:
        if not model_url:
            return candidates
        if topk <= 0:
            return []

        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            raise RuntimeError("sentence-transformers is required for reranker.") from exc

        model = _RERANKER_CACHE.get(model_url)
        if model is None:
            model = CrossEncoder(model_url)
            _RERANKER_CACHE[model_url] = model
        pairs = [(query, c["document"]) for c in candidates]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(candidates, scores), key=lambda x: float(x[1]), reverse=True
        )
        return [item[0] for item in ranked[:topk]]
    except Exception:
        return candidates


async def rerank_async(
    query: str,
    candidates: List[Dict[str, Any]],
    model_url: str,
    topk: int,
) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(
        rerank,
        query=query,
        candidates=candidates,
        model_url=model_url,
        topk=topk,
    )
