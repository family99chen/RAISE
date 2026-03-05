import asyncio
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import torch
except Exception:
    torch = None

_EMBEDDING_FN_CACHE: Dict[Tuple[str, str], Any] = {}


def _prefer_device() -> str:
    if torch and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_embedding_fn(model_path: str, device: str) -> Any:
    key = (model_path, device)
    cached = _EMBEDDING_FN_CACHE.get(key)
    if cached is not None:
        return cached
    from chromadb.utils import embedding_functions

    fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_path,
        device=device,
    )
    _EMBEDDING_FN_CACHE[key] = fn
    return fn


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size.")
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _validate_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("Each record must be a dict with 'id' and 'content'.")
        if "id" not in item or "content" not in item:
            raise ValueError("Each record must include 'id' and 'content'.")
        normalized.append({"id": str(item["id"]), "content": str(item["content"])})
    return normalized


_CHROMA_MAX_BATCH = 5000


def build_chroma_db(
    records: Iterable[Dict[str, Any]],
    embedding_model_path: str,
    chunk_size: int | None,
    chunk_overlap: int = 0,
    collection_name: str = "ragsearch",
    debug_dump: bool = False,
    persist_dir: Optional[str] = None,
):
    try:
        import chromadb

        device = _prefer_device()
        embedding_fn = _get_embedding_fn(embedding_model_path, device)

        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            done_marker = os.path.join(persist_dir, "_build_complete")
            client = chromadb.PersistentClient(path=persist_dir)
            collection = client.get_or_create_collection(
                name="chunks",
                embedding_function=embedding_fn,
            )
            existing = collection.count()
            if existing > 0 and os.path.isfile(done_marker):
                return {
                    "client": client,
                    "collection": collection,
                    "collection_name": "chunks",
                    "num_chunks": existing,
                    "debug_dump": None,
                    "from_cache": True,
                }
            if existing > 0:
                client.delete_collection(name="chunks")
                collection = client.get_or_create_collection(
                    name="chunks",
                    embedding_function=embedding_fn,
                )
        else:
            client = chromadb.Client()
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass
            collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_fn,
            )

        records = _validate_records(records)
        ids: List[str] = []
        docs: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        for record in records:
            if chunk_size is None:
                chunks = [record["content"]]
            else:
                chunks = _chunk_text(record["content"], chunk_size, chunk_overlap)
            for idx, chunk in enumerate(chunks):
                ids.append(f"{record['id']}__{idx}")
                docs.append(chunk)
                metadatas.append({"source_id": record["id"], "chunk_index": idx})

        if ids:
            for start in range(0, len(ids), _CHROMA_MAX_BATCH):
                end = start + _CHROMA_MAX_BATCH
                collection.add(
                    ids=ids[start:end],
                    documents=docs[start:end],
                    metadatas=metadatas[start:end],
                )

        if persist_dir:
            with open(done_marker, "w") as _f:
                _f.write(str(len(ids)))

        debug_dump_data = None
        if debug_dump:
            debug_dump_data = collection.get(
                include=["embeddings", "documents", "metadatas"]
            )

        col_name = "chunks" if persist_dir else collection_name
        return {
            "client": client,
            "collection": collection,
            "collection_name": col_name,
            "num_chunks": len(ids),
            "debug_dump": debug_dump_data,
            "from_cache": False,
        }
    except Exception:
        return {
            "client": None,
            "collection": None,
            "collection_name": None,
            "num_chunks": 0,
            "debug_dump": None,
        }


async def build_chroma_db_async(
    records: Iterable[Dict[str, Any]],
    embedding_model_path: str,
    chunk_size: int | None,
    chunk_overlap: int = 0,
    collection_name: str = "ragsearch",
    debug_dump: bool = False,
    persist_dir: Optional[str] = None,
):
    return await asyncio.to_thread(
        build_chroma_db,
        records=records,
        embedding_model_path=embedding_model_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        collection_name=collection_name,
        debug_dump=debug_dump,
        persist_dir=persist_dir,
    )
