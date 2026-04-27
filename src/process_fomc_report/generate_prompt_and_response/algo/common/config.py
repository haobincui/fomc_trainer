from __future__ import annotations

from pathlib import Path

import yaml

from .paths import REPO_ROOT
from .response_templates import LEGACY_XML_TEMPLATE, normalize_response_template


DEFAULT_CONFIG = REPO_ROOT / "configs" / "main" / "prompt_pipeline.yaml"


def load_pipeline_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    payload["response_template"] = normalize_response_template(
        payload.get("response_template", LEGACY_XML_TEMPLATE)
    )
    payload["_config_path"] = str(config_path)
    return payload


def get_response_template(payload: dict | None) -> str:
    config_payload = payload or {}
    return normalize_response_template(
        config_payload.get("response_template", LEGACY_XML_TEMPLATE)
    )


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def resolve_paths(values: list[str] | tuple[str, ...]) -> list[Path]:
    return [resolve_path(value) for value in values]
