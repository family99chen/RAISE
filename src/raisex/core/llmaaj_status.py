"""Distinguish judged-0 from judge failures and pipeline failures in reports."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional


def summarize_llmaaj(per_item: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(per_item or [])
    counts = Counter(str(item.get("LLMAAJ_status") or "unknown") for item in items)
    judged_0 = int(counts.get("judged_0", 0))
    judged_1 = int(counts.get("judged_1", 0))
    judged_n = judged_0 + judged_1
    timeout_n = int(counts.get("timeout", 0))
    error_n = int(counts.get("error", 0))
    parse_n = int(counts.get("parse_error", 0))
    skipped_n = int(counts.get("skipped_no_ref", 0) + counts.get("skipped", 0))
    n = len(items)
    score = (judged_1 / judged_n) if judged_n else None
    if n == 0:
        status = "unavailable"
    elif timeout_n == n:
        status = "all_timeout"
    elif error_n == n:
        status = "all_error"
    elif parse_n == n:
        status = "all_parse_error"
    elif judged_n == 0:
        status = "unavailable"
    elif timeout_n or error_n or parse_n:
        status = "partial"
    else:
        status = "ok"
    return {
        "score": score,
        "status": status,
        "n": n,
        "judged_n": judged_n,
        "judged_0": judged_0,
        "judged_1": judged_1,
        "timeout_n": timeout_n,
        "error_n": error_n,
        "parse_error_n": parse_n,
        "skipped_n": skipped_n,
    }


def summarize_pipeline(
    outputs: Optional[Iterable[Dict[str, Any]]],
    chunking: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = [o for o in (outputs or []) if isinstance(o, dict)]
    error_n = 0
    empty_n = 0
    for row in rows:
        status = str(row.get("pipeline_status") or "")
        if row.get("error") or row.get("error_type") or status == "error":
            error_n += 1
        if not str(row.get("answer") or "").strip():
            empty_n += 1
    chunk_err = None
    if isinstance(chunking, dict) and chunking.get("error"):
        chunk_err = chunking.get("error_type") or "chunking_error"
    n = len(rows)
    if chunk_err or (n and error_n == n):
        status = "error"
    elif error_n or empty_n:
        status = "partial"
    else:
        status = "ok"
    return {
        "status": status,
        "n": n,
        "error_n": error_n,
        "empty_n": empty_n,
        "chunking_error": chunk_err,
    }


def attach_status_metrics(
    metrics: Dict[str, Any],
    llmaaj: Dict[str, Any],
    pipeline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics["LLMAAJ"] = llmaaj.get("score")
    metrics["LLMAAJ_status"] = llmaaj.get("status")
    metrics["LLMAAJ_judged_n"] = llmaaj.get("judged_n", 0)
    metrics["LLMAAJ_judged_0"] = llmaaj.get("judged_0", 0)
    metrics["LLMAAJ_judged_1"] = llmaaj.get("judged_1", 0)
    metrics["LLMAAJ_timeout_n"] = llmaaj.get("timeout_n", 0)
    metrics["LLMAAJ_error_n"] = llmaaj.get("error_n", 0)
    metrics["LLMAAJ_parse_error_n"] = llmaaj.get("parse_error_n", 0)
    if pipeline:
        metrics["pipeline_status"] = pipeline.get("status")
        metrics["pipeline_error_n"] = pipeline.get("error_n", 0)
        metrics["pipeline_empty_n"] = pipeline.get("empty_n", 0)
    return metrics
