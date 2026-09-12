"""Load RAISE/.env and resolve CityU LLMAAJ settings."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

_LOADED = False

DEFAULT_CITYU_URL = "http://127.0.0.1:8888/v1"
DEFAULT_CITYU_MODEL = "your-judge-model"


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def load_project_env() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    path = os.path.join(project_root(), ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


DEFAULT_QWEN_URL = "http://127.0.0.1:8000/v1"
DEFAULT_QWEN_MODEL = "qwen36-rag"
_STALE_QWEN_MARKERS = (
    "localhost:9000",
    "127.0.0.1:9000",
    "127.0.0.1:8088",
    "10.37.1.182:8000",
    "10.37.1.9:8001",
    "144.214.37.81",
    "10.37.1.1",
)


def resolve_qwen_cfg(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    load_project_env()
    src = dict(cfg or {})
    url = str(src.get("model_url") or "").strip()
    env_url = (os.environ.get("QWEN_BASE_URL") or "").strip()
    if env_url or (not url) or any(marker in url for marker in _STALE_QWEN_MARKERS):
        url = env_url or DEFAULT_QWEN_URL
    key = str(src.get("api_key") or "").strip() or (os.environ.get("QWEN_API_KEY") or "").strip()
    name = (
        str(src.get("model_name") or "").strip()
        or os.environ.get("QWEN_MODEL")
        or DEFAULT_QWEN_MODEL
    )
    return {"model_url": url, "api_key": key, "model_name": name}


def resolve_llmaaj_cfg(llmaaj_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    load_project_env()
    cfg = dict(llmaaj_cfg or {})
    env_url = (os.environ.get("CITYU_LLM_URL") or "").strip()
    env_name = (os.environ.get("CITYU_LLM_MODEL") or "").strip()
    cfg["model_url"] = env_url or str(cfg.get("model_url") or "").strip() or DEFAULT_CITYU_URL
    cfg["model_name"] = env_name or str(cfg.get("model_name") or "").strip() or DEFAULT_CITYU_MODEL
    key = str(cfg.get("api_key") or "").strip() or (os.environ.get("CITYU_LLM_KEY") or "").strip()
    cfg["api_key"] = key
    return cfg
