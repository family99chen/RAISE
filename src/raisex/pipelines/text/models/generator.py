import asyncio
import os
import sys
from typing import Any, Dict, Optional

from raisex.llmfactory.llmfactory import create_llm


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = handle.read().strip()
    if not data:
        return {}
    import yaml

    parsed = yaml.safe_load(data) or {}
    return parsed if isinstance(parsed, dict) else {}


def _load_pipeline_config() -> Dict[str, Any]:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(base_dir, "pipelineconfig.yaml")
    return _load_yaml(config_path)


def _get_system_prompt() -> str:
    config = _load_pipeline_config().get("generator", {})
    return config.get(
        "prompt",
        "You are a helpful assistant. Answer the question using only the provided context. "
        "If the answer is not in the context, say you do not know.",
    )


def generate_answer(
    query: str,
    context: str,
    model_url: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    try:
        llm = create_llm(url=model_url, api_key=api_key, model_name=model_name)
        user_prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"
        return llm.generate(user_prompt, system=_get_system_prompt())
    except Exception:
        return ""


async def generate_answer_async(
    query: str,
    context: str,
    model_url: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    try:
        llm = create_llm(url=model_url, api_key=api_key, model_name=model_name)
        user_prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"
        system_prompt = _get_system_prompt()
        if hasattr(llm, "generate_async"):
            return await llm.generate_async(user_prompt, system=system_prompt)
        return await asyncio.to_thread(llm.generate, user_prompt, system=system_prompt)
    except Exception:
        return ""
