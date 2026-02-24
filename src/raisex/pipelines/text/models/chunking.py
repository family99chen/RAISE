import asyncio
import json
from typing import Any, Dict, Iterable, List, Tuple

try:
    import torch
except Exception:
    torch = None


def _prefer_device() -> str:
    if torch and torch.cuda.is_available():
        return "cuda"
    return "cpu"


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


def build_chroma_db(
    records: Iterable[Dict[str, Any]],
    embedding_model_path: str,
    chunk_size: int | None,
    chunk_overlap: int = 0,
    collection_name: str = "ragsearch",
    debug_dump: bool = False,
):
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        records = _validate_records(records)
        device = _prefer_device()
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_path,
            device=device,
        )

        client = chromadb.Client()
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
        )

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
            collection.add(ids=ids, documents=docs, metadatas=metadatas)

        debug_dump_data = None
        if debug_dump:
            debug_dump_data = collection.get(
                include=["embeddings", "documents", "metadatas"]
            )

        return {
            "client": client,
            "collection": collection,
            "collection_name": collection_name,
            "num_chunks": len(ids),
            "debug_dump": debug_dump_data,
            "error": None,
            "error_type": None,
        }
    except Exception as exc:
        return {
            "client": None,
            "collection": None,
            "collection_name": None,
            "num_chunks": 0,
            "debug_dump": None,
            "error": repr(exc),
            "error_type": type(exc).__name__,
        }


async def build_chroma_db_async(
    records: Iterable[Dict[str, Any]],
    embedding_model_path: str,
    chunk_size: int | None,
    chunk_overlap: int = 0,
    collection_name: str = "ragsearch",
    debug_dump: bool = False,
):
    return await asyncio.to_thread(
        build_chroma_db,
        records=records,
        embedding_model_path=embedding_model_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        collection_name=collection_name,
        debug_dump=debug_dump,
    )
