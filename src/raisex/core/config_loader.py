import os
from pathlib import Path
from typing import Optional


def project_root_from_module() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_text_schema_path(explicit_path: Optional[str] = None) -> str:
    if explicit_path:
        return str(Path(explicit_path).resolve())

    env_path = os.getenv("RAGSEARCH_CONFIG")
    if env_path:
        return str(Path(env_path).resolve())

    root = project_root_from_module()
    new_default = root / "configs" / "search_space" / "text.yaml"
    if new_default.is_file():
        return str(new_default)

    raise FileNotFoundError(
        "No text search-space config found. Checked RAGSEARCH_CONFIG and configs/search_space/text.yaml"
    )


def resolve_multimodal_schema_path(explicit_path: Optional[str] = None) -> str:
    if explicit_path:
        return str(Path(explicit_path).resolve())

    env_path = os.getenv("RAGSEARCH_CONFIG_MULTIMODAL")
    if env_path:
        return str(Path(env_path).resolve())

    root = project_root_from_module()
    new_default = root / "configs" / "search_space" / "multimodal.yaml"
    if new_default.is_file():
        return str(new_default)

    raise FileNotFoundError(
        "No multimodal search-space config found. Checked RAGSEARCH_CONFIG_MULTIMODAL and configs/search_space/multimodal.yaml"
    )
