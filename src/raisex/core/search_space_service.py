import os
from typing import Any, Dict, List

import yaml

from raisex.core.config_loader import (
    resolve_multimodal_schema_path,
    resolve_text_schema_path,
)


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping (dict).")
    return data


def _pick_first_allowed(allowed: List[Any]) -> Any:
    for item in allowed:
        if item != "...":
            return item
    return allowed[0] if allowed else None


def _build_selection_template(node: Any, key: str = "") -> Any:
    if isinstance(node, dict):
        if "allowed" in node:
            allowed = node.get("allowed", [])
            if isinstance(allowed, list):
                return _pick_first_allowed(allowed)
            return allowed
        if isinstance(node, str):
            return node
        return {k: _build_selection_template(v, k) for k, v in node.items()}
    return node


def _format_scalar(value: Any) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _render_template(data: Dict[str, Any], indent: int = 0) -> str:
    lines: List[str] = []
    pad = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.append(_render_template(value, indent + 1))
        else:
            lines.append(f"{pad}{key}: {_format_scalar(value)}")
    return "\n".join(lines)


def _build_description(is_multimodal: bool) -> List[str]:
    base = [
        "This endpoint lists all searchable RAG hyperparameters.",
        "All selectable values are constrained to the configured search-space schema.",
        "Model URL can be provided at runtime.",
        "Model name and API key are optional and only required when using an API. Change to local mode automatically if api key and model name are not provided.",
        "Prompt templates must be chosen by ID and map to the fixed templates below.",
        "Evaluation resources can be provided under eval_metrics.llmaaj and eval_metrics.bert.",
        "Use eval_metrics.enabled to control which metrics are computed.",
        "Use the template below to create a selection YAML file.",
    ]
    if is_multimodal:
        base.extend(
            [
                "Required: retrieve.model_url, retrieve.topk, retrieve.bm25_weight, chunking.chunk_size, generator.model_url, clip.model_url.",
                "Optional sections: rewriter, reranker (can be omitted or partial).",
            ]
        )
    else:
        base.extend(
            [
                "Required: retrieve.model_url, retrieve.topk, retrieve.bm25_weight, chunking.chunk_size, generator.model_url.",
                "Optional sections: rewriter, reranker, pruner (can be omitted or partial).",
            ]
        )
    base.append("To run a search, the user should submit a YAML file path with chosen values.")
    return base


def get_search_space(config_path: str | None = None) -> Dict[str, Any]:
    config = _load_yaml(resolve_text_schema_path(config_path))
    search_space = config.get("rag_search_space", {})
    prompt_templates = config.get("prompt_templates", {})
    eval_metrics = config.get("eval_metrics", {})
    selection_template = _build_selection_template(search_space)
    if eval_metrics:
        selection_template["eval_metrics"] = _build_selection_template(eval_metrics)

    return {
        "description": _build_description(is_multimodal=False),
        "search_space": search_space,
        "prompt_templates": prompt_templates,
        "selection_template_text": _render_template(selection_template),
        "response_format": {"selection_yaml_path": "<absolute-or-relative-path-to-selection.yaml>"},
    }


def get_search_space_multimodal(config_path: str | None = None) -> Dict[str, Any]:
    config = _load_yaml(resolve_multimodal_schema_path(config_path))
    search_space = config.get("rag_search_space", {})
    prompt_templates = config.get("prompt_templates", {})
    eval_metrics = config.get("eval_metrics", {})
    selection_template = _build_selection_template(search_space)
    if eval_metrics:
        selection_template["eval_metrics"] = _build_selection_template(eval_metrics)

    return {
        "description": _build_description(is_multimodal=True),
        "search_space": search_space,
        "prompt_templates": prompt_templates,
        "selection_template_text": _render_template(selection_template),
        "response_format": {"selection_yaml_path": "<absolute-or-relative-path-to-selection.yaml>"},
    }
