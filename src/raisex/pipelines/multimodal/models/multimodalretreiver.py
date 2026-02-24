import asyncio
import threading
from typing import Any, Dict, List

try:
    import torch
except Exception:
    torch = None


_MODEL_CACHE: Dict[tuple[str, str], Any] = {}
_CLIP_LOCK = threading.Lock()


def _get_clip_text_model(model_url: str) -> Any:
    cache_key = (model_url, _prefer_device())
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError("sentence-transformers is required for CLIP text encoder.") from exc
    model = SentenceTransformer(model_url, device=cache_key[1])
    _MODEL_CACHE[cache_key] = model
    return model


def _prefer_device() -> str:
    if torch and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def retrieve_images(
    query: str,
    collection: Any,
    model_url: str,
    topk: int,
) -> List[Dict[str, Any]]:
    try:
        if topk <= 0 or collection is None or not model_url:
            return []
        with _CLIP_LOCK:
            model = _get_clip_text_model(model_url)
            query_vec = model.encode([query], convert_to_numpy=True)[0]
            result = collection.query(query_embeddings=[query_vec], n_results=topk)
        ids = result.get("ids", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        sims = [1.0 / (1.0 + d) for d in distances]
        return [
            {"id": idx, "metadata": meta, "score": sim}
            for idx, meta, sim in zip(ids, metas, sims)
        ]
    except Exception:
        return []


async def retrieve_images_async(
    query: str,
    collection: Any,
    model_url: str,
    topk: int,
) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(
        retrieve_images,
        query=query,
        collection=collection,
        model_url=model_url,
        topk=topk,
    )