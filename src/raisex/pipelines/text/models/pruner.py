import asyncio
import os
from typing import Any, Dict, List, Optional

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
    templates = config.get("prompt_templates", {}).get("pruner", {})
    return templates.get(str(template_id), templates.get("1", "Prune irrelevant chunks."))


def prune_chunks(
    query: str,
    candidates: List[Dict[str, Any]],
    pruner_cfg: Dict[str, Any],
    config_path: Optional[str] = None,
) -> str:
    try:
        combined = "\n".join([c["document"] for c in candidates])
        if not pruner_cfg:
            return combined

        url = pruner_cfg.get("model_url")
        if not url:
            return combined

        model_name = pruner_cfg.get("model_name")
        api_key = pruner_cfg.get("api_key")
        template_id = pruner_cfg.get("prompt_template_id", "1")
        system_prompt = _resolve_prompt_template(template_id, config_path=config_path)

        user_prompt = (
            f"Query: {query}\n"
            "Context:\n"
            + combined
            + "\nReturn pruned context only."
        )

        llm = create_llm(url=url, api_key=api_key, model_name=model_name)
        output = llm.generate(user_prompt, system=system_prompt)
        return output.strip() or combined
    except Exception:
        return ""


async def prune_chunks_async(
    query: str,
    candidates: List[Dict[str, Any]],
    pruner_cfg: Dict[str, Any],
    config_path: Optional[str] = None,
) -> str:
    try:
        combined = "\n".join([c["document"] for c in candidates])
        if not pruner_cfg:
            return combined

        url = pruner_cfg.get("model_url")
        if not url:
            return combined

        model_name = pruner_cfg.get("model_name")
        api_key = pruner_cfg.get("api_key")
        template_id = pruner_cfg.get("prompt_template_id", "1")
        system_prompt = _resolve_prompt_template(template_id, config_path=config_path)

        user_prompt = (
            f"Query: {query}\n"
            "Context:\n"
            + combined
            + "\nReturn pruned context only."
        )

        llm = create_llm(url=url, api_key=api_key, model_name=model_name)
        if hasattr(llm, "generate_async"):
            output = await llm.generate_async(user_prompt, system=system_prompt)
        else:
            output = await asyncio.to_thread(llm.generate, user_prompt, system=system_prompt)
        return output.strip() or combined
    except Exception:
        return ""
