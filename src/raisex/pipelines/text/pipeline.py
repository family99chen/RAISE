import asyncio
import json
import os
import uuid
import gc
import time
import tempfile
from typing import Any, Dict, List, Optional

import yaml

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from raisex.pipelines.text.models.chunking import build_chroma_db, build_chroma_db_async
from raisex.pipelines.text.models.eval import clear_eval_cache, evaluate_report
from raisex.pipelines.text.models.generator import generate_answer, generate_answer_async
from raisex.pipelines.text.models.pruner import prune_chunks, prune_chunks_async
from raisex.pipelines.text.models.reranker import (
    clear_reranker_cache,
    rerank,
    rerank_async,
)
from raisex.pipelines.text.models.retriever import retrieve, retrieve_async
from raisex.pipelines.text.models.rewriter import rewrite_query, rewrite_query_async

try:
    import torch
except Exception:
    torch = None


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping (dict).")
    return data


def _load_pipeline_config() -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(__file__), "pipelineconfig.yaml")
    if not os.path.isfile(config_path):
        return {}
    return _load_yaml(config_path)


def _load_json(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of records.")
    return data


def _pick_first(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    if isinstance(value, dict) and "allowed" in value:
        allowed = value.get("allowed")
        if isinstance(allowed, list):
            for item in allowed:
                if item != "...":
                    return item
            return allowed[0] if allowed else None
    return value


def _build_upperbound_selection_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    selection: Dict[str, Any] = {}
    generator = config.get("generator")
    if not isinstance(generator, dict):
        generator = (config.get("rag_search_space") or {}).get("generator")
    if isinstance(generator, dict):
        model_url = _pick_first(generator.get("model_url"))
        model_name = _pick_first(generator.get("model_name"))
        api_key = _pick_first(generator.get("api_key"))
        if model_url:
            selection["generator"] = {"model_url": str(model_url)}
            if model_name:
                selection["generator"]["model_name"] = str(model_name)
            if api_key:
                selection["generator"]["api_key"] = str(api_key)

    eval_metrics = config.get("eval_metrics")
    if isinstance(eval_metrics, dict):
        selection["eval_metrics"] = eval_metrics
    return selection


def _extract_upperbound_contexts(
    qa_items: List[Dict[str, Any]], corpus_items: List[Dict[str, Any]]
) -> List[str]:
    corpus_by_id: Dict[str, str] = {}
    corpus_ids: List[str] = []
    for item in corpus_items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id") or item.get("qid") or item.get("doc_id")
        content = item.get("content")
        if item_id is None or content is None:
            continue
        item_id_str = str(item_id)
        corpus_by_id[item_id_str] = str(content)
        corpus_ids.append(item_id_str)

    contexts: List[str] = []
    for idx, qa in enumerate(qa_items):
        if not isinstance(qa, dict):
            contexts.append("")
            continue
        qa_id = qa.get("id") or qa.get("qid") or qa.get("doc_id")
        if qa_id is not None:
            qa_id_str = str(qa_id)
            exact = corpus_by_id.get(qa_id_str)
            if exact is not None:
                contexts.append(exact)
                continue
            matched = [
                corpus_by_id[cid]
                for cid in corpus_ids
                if cid == qa_id_str or cid.startswith(f"{qa_id_str}_")
            ]
            if matched:
                contexts.append("\n".join(matched))
                continue
        if idx < len(corpus_items):
            contexts.append(str(corpus_items[idx].get("content", "")))
        else:
            contexts.append("")
    return contexts


def run_chunking_stage(
    data_json_path: str, selection_path: str, debug_dump: bool = False
) -> Dict[str, Any]:
    try:
        selection = _load_yaml(selection_path)
        chunking_cfg = selection.get("chunking")
        retrieve_cfg = selection.get("retrieve", {})

        model_url = retrieve_cfg.get("model_url")
        if not model_url:
            raise ValueError("retrieve.model_url is required for embedding model.")

        if chunking_cfg is None:
            chunk_size = None
            chunk_overlap = 0
        else:
            chunk_size = int(chunking_cfg.get("chunk_size", 512))
            chunk_overlap = int(chunking_cfg.get("chunk_overlap", 64))

        records = _load_json(data_json_path)
        collection_name = f"ragsearch_{uuid.uuid4().hex}"
        return build_chroma_db(
            records=records,
            embedding_model_path=model_url,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            collection_name=collection_name,
            debug_dump=debug_dump,
        )
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


async def run_chunking_stage_async(
    data_json_path: str, selection_path: str, debug_dump: bool = False
) -> Dict[str, Any]:
    try:
        selection = _load_yaml(selection_path)
        chunking_cfg = selection.get("chunking")
        retrieve_cfg = selection.get("retrieve", {})

        model_url = retrieve_cfg.get("model_url")
        if not model_url:
            raise ValueError("retrieve.model_url is required for embedding model.")

        if chunking_cfg is None:
            chunk_size = None
            chunk_overlap = 0
        else:
            chunk_size = int(chunking_cfg.get("chunk_size", 512))
            chunk_overlap = int(chunking_cfg.get("chunk_overlap", 64))

        records = _load_json(data_json_path)
        collection_name = f"ragsearch_{uuid.uuid4().hex}"
        return await build_chroma_db_async(
            records=records,
            embedding_model_path=model_url,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            collection_name=collection_name,
            debug_dump=debug_dump,
        )
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


def run_rewriter_stage(query: str, selection_path: str) -> str:
    try:
        selection = _load_yaml(selection_path)
        if "rewriter" not in selection:
            return query
        rewriter_cfg = selection.get("rewriter", {})
        return rewrite_query(query, rewriter_cfg)
    except Exception:
        return ""


async def run_rewriter_stage_async(query: str, selection_path: str) -> str:
    try:
        selection = _load_yaml(selection_path)
        if "rewriter" not in selection:
            return query
        rewriter_cfg = selection.get("rewriter", {})
        return await rewrite_query_async(query, rewriter_cfg)
    except Exception:
        return ""


def run_retriever_stage(query: str, selection_path: str, collection: Any) -> list[Dict[str, Any]]:
    try:
        selection = _load_yaml(selection_path)
        retrieve_cfg = selection.get("retrieve", {})
        topk = int(retrieve_cfg.get("topk", 5))
        bm25_weight = float(retrieve_cfg.get("bm25_weight", 0.0))
        return retrieve(query=query, collection=collection, topk=topk, bm25_weight=bm25_weight)
    except Exception:
        return []


async def run_retriever_stage_async(
    query: str, selection_path: str, collection: Any
) -> list[Dict[str, Any]]:
    try:
        selection = _load_yaml(selection_path)
        retrieve_cfg = selection.get("retrieve", {})
        topk = int(retrieve_cfg.get("topk", 5))
        bm25_weight = float(retrieve_cfg.get("bm25_weight", 0.0))
        return await retrieve_async(
            query=query, collection=collection, topk=topk, bm25_weight=bm25_weight
        )
    except Exception:
        return []


def run_pruner_stage(
    query: str,
    selection_path: str,
    candidates: list[Dict[str, Any]],
) -> str:
    try:
        selection = _load_yaml(selection_path)
        if "pruner" not in selection:
            return "\n".join([c["document"] for c in candidates])
        pruner_cfg = selection.get("pruner", {})
        return prune_chunks(query=query, candidates=candidates, pruner_cfg=pruner_cfg)
    except Exception:
        return ""


async def run_pruner_stage_async(
    query: str,
    selection_path: str,
    candidates: list[Dict[str, Any]],
) -> str:
    try:
        selection = _load_yaml(selection_path)
        if "pruner" not in selection:
            return "\n".join([c["document"] for c in candidates])
        pruner_cfg = selection.get("pruner", {})
        return await prune_chunks_async(
            query=query, candidates=candidates, pruner_cfg=pruner_cfg
        )
    except Exception:
        return ""


def run_reranker_stage(
    query: str,
    selection_path: str,
    candidates: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    try:
        selection = _load_yaml(selection_path)
        if "reranker" not in selection:
            return candidates
        reranker_cfg = selection.get("reranker", {})
        model_url = reranker_cfg.get("model_url")
        topk = int(reranker_cfg.get("topk", len(candidates)))
        if not model_url:
            return candidates
        return rerank(query=query, candidates=candidates, model_url=model_url, topk=topk)
    except Exception:
        return candidates


async def run_reranker_stage_async(
    query: str,
    selection_path: str,
    candidates: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    try:
        selection = _load_yaml(selection_path)
        if "reranker" not in selection:
            return candidates
        reranker_cfg = selection.get("reranker", {})
        model_url = reranker_cfg.get("model_url")
        topk = int(reranker_cfg.get("topk", len(candidates)))
        if not model_url:
            return candidates
        return await rerank_async(
            query=query, candidates=candidates, model_url=model_url, topk=topk
        )
    except Exception:
        return candidates


def run_generator_stage(query: str, selection_path: str, context: str) -> str:
    try:
        selection = _load_yaml(selection_path)
        generator_cfg = selection.get("generator", {})
        model_url = generator_cfg.get("model_url")
        if not model_url:
            return ""
        model_name = generator_cfg.get("model_name")
        api_key = generator_cfg.get("api_key")
        return generate_answer(
            query=query,
            context=context,
            model_url=model_url,
            api_key=api_key,
            model_name=model_name,
        )
    except Exception:
        return ""


async def run_generator_stage_async(
    query: str, selection_path: str, context: str
) -> str:
    try:
        selection = _load_yaml(selection_path)
        generator_cfg = selection.get("generator", {})
        model_url = generator_cfg.get("model_url")
        if not model_url:
            return ""
        model_name = generator_cfg.get("model_name")
        api_key = generator_cfg.get("api_key")
        return await generate_answer_async(
            query=query,
            context=context,
            model_url=model_url,
            api_key=api_key,
            model_name=model_name,
        )
    except Exception:
        return ""


async def run_pipeline_async(
    query: str,
    selection_path: str,
    collection: Any,
    idx: Optional[int] = None,
) -> Dict[str, Any]:
    debug = os.getenv("PIPELINE_DEBUG") == "1"
    qtag = f"qid{idx}" if idx is not None else "qid-"
    t0 = time.perf_counter()
    rewritten = await run_rewriter_stage_async(query, selection_path)
    if debug:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[pipeline][debug] {qtag} rewrite done ({elapsed_ms:.1f} ms)", flush=True)
    t0 = time.perf_counter()
    retrieval = await run_retriever_stage_async(rewritten, selection_path, collection)
    if debug:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[pipeline][debug] {qtag} retrieve done ({elapsed_ms:.1f} ms)", flush=True)
    t0 = time.perf_counter()
    reranked = await run_reranker_stage_async(rewritten, selection_path, retrieval)
    if debug:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[pipeline][debug] {qtag} rerank done ({elapsed_ms:.1f} ms)", flush=True)
    t0 = time.perf_counter()
    pruned_text = await run_pruner_stage_async(rewritten, selection_path, reranked)
    if debug:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[pipeline][debug] {qtag} prune done ({elapsed_ms:.1f} ms)", flush=True)
    t0 = time.perf_counter()
    answer = await run_generator_stage_async(query, selection_path, pruned_text)
    if debug:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[pipeline][debug] {qtag} generate done ({elapsed_ms:.1f} ms)", flush=True)
    return {
        "query": query,
        "rewritten": rewritten,
        "retrieval": retrieval,
        "reranked": reranked,
        "pruned_text": pruned_text,
        "answer": answer,
    }


async def run_batch_async(
    queries: List[str],
    selection_path: str,
    data_json_path: str,
    references_list: List[List[str]] | None = None,
    answers_list: List[str] | None = None,
    eval_mode: str = "both",
    debug_dump: bool = False,
) -> Dict[str, Any]:
    batch_t0 = time.perf_counter()
    result = await run_chunking_stage_async(
        data_json_path, selection_path, debug_dump=debug_dump
    )
    collection = result.get("collection")
    client = result.get("client")
    collection_name = result.get("collection_name")
    chunking = {
        "num_chunks": result.get("num_chunks", 0),
        "error": result.get("error"),
        "error_type": result.get("error_type"),
    }

    config = _load_pipeline_config()
    max_tasks = int(config.get("concurrency", {}).get("max_tasks", 8))
    semaphore = asyncio.Semaphore(max_tasks)

    async def _run_one(idx: int, q: str) -> Dict[str, Any]:
        async with semaphore:
            result_item = await run_pipeline_async(q, selection_path, collection, idx=idx)
            return {"_index": idx, **result_item}

    selection = _load_yaml(selection_path)
    eval_cfg = selection.get("eval_metrics")
    total = len(queries)
    tasks = [asyncio.create_task(_run_one(idx, q)) for idx, q in enumerate(queries)]
    outputs: List[Optional[Dict[str, Any]]] = [None] * total
    if tqdm is not None:
        with tqdm(total=total, desc="progress", unit="qa") as bar:
            for coro in asyncio.as_completed(tasks):
                result_item = await coro
                idx = result_item.get("_index")
                if isinstance(idx, int) and 0 <= idx < total:
                    outputs[idx] = {k: v for k, v in result_item.items() if k != "_index"}
                else:
                    outputs.append({k: v for k, v in result_item.items() if k != "_index"})
                bar.update(1)
    else:
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result_item = await coro
            idx = result_item.get("_index")
            if isinstance(idx, int) and 0 <= idx < total:
                outputs[idx] = {k: v for k, v in result_item.items() if k != "_index"}
            else:
                outputs.append({k: v for k, v in result_item.items() if k != "_index"})
            completed += 1
            print(f"\rprogress: {completed}/{total}", end="", flush=True)
        if total > 0:
            print()
    try:
        outputs_clean = [o or {} for o in outputs]
        preds = answers_list if answers_list is not None else [o.get("answer", "") for o in outputs_clean]
        report = evaluate_report(
            preds=preds,
            refs_list=references_list or [],
            queries=queries,
            mode=eval_mode,
            eval_cfg=eval_cfg,
        )
        batch_elapsed_seconds = time.perf_counter() - batch_t0
        if isinstance(report, dict):
            report["pipeline_total_time_seconds"] = batch_elapsed_seconds
        return {
            "debug_dump": result.get("debug_dump"),
            "outputs": outputs_clean,
            "report": report,
            "pipeline_total_time_seconds": batch_elapsed_seconds,
            "chunking": chunking,
        }
    finally:
        if client is not None and collection_name:
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass
        clear_eval_cache()
        clear_reranker_cache()
        if torch and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        gc.collect()


async def getupperbound_external_async(
    qa_json_path: str,
    corpus_json_path: str,
    config_path: str,
    eval_mode: str = "both",
) -> Dict[str, Any]:
    config = _load_yaml(config_path)
    selection = _build_upperbound_selection_from_config(config)
    fd, selection_path = tempfile.mkstemp(prefix="upperbound_ext_", suffix=".yaml")
    os.close(fd)
    with open(selection_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(selection, handle, sort_keys=False, allow_unicode=True)
    try:
        qa_items = _load_json(qa_json_path)
        corpus_items = _load_json(corpus_json_path)
        contexts = _extract_upperbound_contexts(qa_items, corpus_items)

        queries: List[str] = []
        references_list: List[List[str]] = []
        for item in qa_items:
            query = item.get("query") or item.get("question")
            queries.append("" if query is None else str(query))
            refs = item.get("references") or item.get("answers") or item.get("reference")
            if refs is None:
                references_list.append([])
            elif isinstance(refs, list):
                references_list.append([str(r) for r in refs])
            else:
                references_list.append([str(refs)])

        config_pipeline = _load_pipeline_config()
        max_tasks = int(config_pipeline.get("concurrency", {}).get("max_tasks", 8))
        semaphore = asyncio.Semaphore(max_tasks)

        async def _run_one(idx: int, q: str, ctx: str) -> Dict[str, Any]:
            async with semaphore:
                answer = await run_generator_stage_async(q, selection_path, ctx)
                return {"_index": idx, "query": q, "context": ctx, "answer": answer}

        tasks = [
            asyncio.create_task(_run_one(idx, q, ctx))
            for idx, (q, ctx) in enumerate(zip(queries, contexts))
        ]
        outputs: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
        total = len(tasks)
        if tqdm is not None:
            with tqdm(total=total, desc="upperbound", unit="qa") as bar:
                for coro in asyncio.as_completed(tasks):
                    result_item = await coro
                    idx = result_item.get("_index")
                    if isinstance(idx, int) and 0 <= idx < total:
                        outputs[idx] = {k: v for k, v in result_item.items() if k != "_index"}
                    else:
                        outputs.append({k: v for k, v in result_item.items() if k != "_index"})
                    bar.update(1)
        else:
            completed = 0
            for coro in asyncio.as_completed(tasks):
                result_item = await coro
                idx = result_item.get("_index")
                if isinstance(idx, int) and 0 <= idx < total:
                    outputs[idx] = {k: v for k, v in result_item.items() if k != "_index"}
                else:
                    outputs.append({k: v for k, v in result_item.items() if k != "_index"})
                completed += 1
                print(f"\rprogress: {completed}/{total}", end="", flush=True)
            if total > 0:
                print()

        eval_cfg = selection.get("eval_metrics")
        outputs_clean = [o or {} for o in outputs]
        report = evaluate_report(
            preds=[o.get("answer", "") for o in outputs_clean],
            refs_list=references_list,
            queries=queries,
            mode=eval_mode,
            eval_cfg=eval_cfg,
        )
        return {"outputs": outputs_clean, "report": report}
    finally:
        os.remove(selection_path)


def getupperbound_external(
    qa_json_path: str,
    corpus_json_path: str,
    config_path: str,
    eval_mode: str = "both",
) -> Dict[str, Any]:
    return asyncio.run(
        getupperbound_external_async(
            qa_json_path=qa_json_path,
            corpus_json_path=corpus_json_path,
            config_path=config_path,
            eval_mode=eval_mode,
        )
    )


def main() -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    selection_path = os.path.join(base_dir, "configs", "demo2.yaml")
    data_json_path = '/home/cz/ragsearch-jane/data/datasets/triviaqa/corpus.json'
    queries = [
        "I am a businessperson. What are OpenAI's main products?",
        "Who is Justin's mother?",
        "What is David's profession?",
    ]
    references_list = [
        ["OpenAI's main products include ChatGPT, DALL-E, Codex, and the API."],
        ["Justin's mother is David."],
        ["David is a doctor."],
    ]
    result = asyncio.run(
        run_batch_async(
            queries=queries,
            selection_path=selection_path,
            data_json_path=data_json_path,
            answers_list=None,
            references_list=references_list,
            eval_mode="both",
            debug_dump=False,
        )
    )
    if result.get("debug_dump") is not None:
        print("chroma_debug_dump:")
        print(result["debug_dump"])
    print("batch_outputs:")
    #print(result["outputs"])
    print("eval_report:")
    print(result["report"])


if __name__ == "__main__":
    main()
