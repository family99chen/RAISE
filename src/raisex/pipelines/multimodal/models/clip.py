import os
import uuid
from typing import Any, Dict, Iterable, List

try:
    import torch
except Exception:
    torch = None


def _prefer_device() -> str:
    if torch and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_image_path(record: Dict[str, Any]) -> str | None:
    raw = (
        record.get("image_path")
        or record.get("imagepath")
        or record.get("path")
        or record.get("image")
        or record.get("file")
    )
    if not raw:
        return None
    return str(raw)


def build_clip_embeddings(
    records: Iterable[Dict[str, Any]],
    model_url: str,
    model_name: str | None = None,
    api_key: str | None = None,
) -> Dict[str, Any]:
    try:
        if not model_url:
            return {"ids": [], "paths": [], "embeddings": []}

        try:
            from PIL import Image
        except Exception as exc:
            raise RuntimeError("Pillow is required for image loading.") from exc

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError("sentence-transformers is required for CLIP embeddings.") from exc

        device = _prefer_device()
        model = SentenceTransformer(model_url, device=device)

        ids: List[str] = []
        paths: List[str] = []
        images: List[Any] = []
        for item in records:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id") or item.get("image_id") or item.get("uid")
            image_path = _resolve_image_path(item)
            if item_id is None or not image_path or not os.path.isfile(image_path):
                continue
            try:
                img = Image.open(image_path).convert("RGB")
            except Exception:
                continue
            ids.append(str(item_id))
            paths.append(image_path)
            images.append(img)

        if os.getenv("MM_DEBUG") == "1":
            print(
                f"[clip] prepared images={len(images)} ids={len(ids)} paths={len(paths)}"
            )

        if not images:
            return {"ids": [], "paths": [], "embeddings": []}

        try:
            embeddings = model.encode(images=images, batch_size=8, convert_to_numpy=True)
        except TypeError:
            embeddings = model.encode(images, batch_size=8, convert_to_numpy=True)
        if os.getenv("MM_DEBUG") == "1":
            try:
                shape = getattr(embeddings, "shape", None)
                print(f"[clip] embeddings shape={shape}")
            except Exception:
                print("[clip] embeddings shape=unknown")
        return {"ids": ids, "paths": paths, "embeddings": embeddings}
    except Exception:
        return {"ids": [], "paths": [], "embeddings": []}


def build_clip_chroma_db(
    records: Iterable[Dict[str, Any]],
    model_url: str,
    model_name: str | None = None,
    api_key: str | None = None,
    collection_name: str | None = None,
):
    try:
        import chromadb

        result = build_clip_embeddings(
            records=records, model_url=model_url, model_name=model_name, api_key=api_key
        )
        ids = result.get("ids") or []
        paths = result.get("paths") or []
        embeddings = result.get("embeddings")
        if embeddings is None:
            embeddings = []
        if not ids or len(ids) != len(embeddings):
            return {"client": None, "collection": None, "collection_name": None}

        name = collection_name or f"ragsearch_images_{uuid.uuid4().hex}"
        client = chromadb.Client()
        try:
            client.delete_collection(name=name)
        except Exception:
            pass
        collection = client.get_or_create_collection(name=name)
        metadatas = [{"path": p} for p in paths]
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        return {
            "client": client,
            "collection": collection,
            "collection_name": name,
            "num_images": len(ids),
        }
    except Exception:
        return {"client": None, "collection": None, "collection_name": None}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Test CLIP image embeddings.")
    parser.add_argument("--model_url", required=True, help="Path or name of CLIP model.")
    parser.add_argument("--image_path", required=True, help="Path to a test image.")
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    records = [{"id": "demo-image-1", "image_path": args.image_path}]
    result = build_clip_embeddings(records=records, model_url=args.model_url)
    ids = result.get("ids") or []
    embeddings = result.get("embeddings")
    if embeddings is None:
        embeddings = []
    print(f"ids={ids}")
    print(f"num_embeddings={len(embeddings)}")
    if embeddings is not None and len(embeddings) > 0:
        try:
            print(f"embedding_dim={len(embeddings[0])}")
        except Exception:
            print("embedding_dim=unknown")


if __name__ == "__main__":
    main()