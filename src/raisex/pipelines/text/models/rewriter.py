import asyncio
import os
from typing import Any, Dict, Optional

import yaml

from raisex.core.config_loader import resolve_text_schema_path
from raisex.llmfactory.llmfactory import create_llm


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping (dict).")
    return data


def _find_base_config_path() -> str:
    return resolve_text_schema_path(None)


def _resolve_prompt_template(template_id: str, config_path: Optional[str] = None) -> str:
    base_path = config_path or _find_base_config_path()
    config = _load_yaml(base_path)
    templates = config.get("prompt_templates", {}).get("rewriter", {})
    return templates.get(str(template_id), templates.get("1", "Rewrite the query."))


def rewrite_query(
    query: str,
    rewriter_cfg: Dict[str, Any],
    config_path: Optional[str] = None,
) -> str:
    try:
        if not rewriter_cfg:
            return query

        url = rewriter_cfg.get("model_url")
        if not url:
            return query

        model_name = rewriter_cfg.get("model_name")
        api_key = rewriter_cfg.get("api_key")
        template_id = rewriter_cfg.get("prompt_template_id", "1")
        system_prompt = _resolve_prompt_template(template_id, config_path=config_path)

        llm = create_llm(url=url, api_key=api_key, model_name=model_name)
        return llm.generate(query, system=system_prompt)
    except Exception:
        return ""


async def rewrite_query_async(
    query: str,
    rewriter_cfg: Dict[str, Any],
    config_path: Optional[str] = None,
) -> str:
    try:
        if not rewriter_cfg:
            return query

        url = rewriter_cfg.get("model_url")
        if not url:
            return query

        model_name = rewriter_cfg.get("model_name")
        api_key = rewriter_cfg.get("api_key")
        template_id = rewriter_cfg.get("prompt_template_id", "1")
        system_prompt = _resolve_prompt_template(template_id, config_path=config_path)

        llm = create_llm(url=url, api_key=api_key, model_name=model_name)
        if hasattr(llm, "generate_async"):
            return await llm.generate_async(query, system=system_prompt)
        return await asyncio.to_thread(llm.generate, query, system=system_prompt)
    except Exception:
        return ""
