"""Thin wrappers around core evaluation and search-space utilities."""

from typing import Any, Dict

from raisex.core.config_validator import check_config, check_config_multimodal
from raisex.core.evaluation_service import (
    evaluate_rag,
    evaluate_rag_multimodal,
    run_algorithms,
)
from raisex.core.search_space_service import get_search_space, get_search_space_multimodal


def check_config_valid(config_path: str, multimodal: bool = False) -> Dict[str, Any]:
    if multimodal:
        return check_config_multimodal(config_path)
    return check_config(config_path)


def find_search_space(config_path: str | None = None, multimodal: bool = False) -> Dict[str, Any]:
    if multimodal:
        return get_search_space_multimodal(config_path)
    return get_search_space(config_path)


__all__ = [
    "evaluate_rag",
    "evaluate_rag_multimodal",
    "check_config_valid",
    "find_search_space",
    "run_algorithms",
]
