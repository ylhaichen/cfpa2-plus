from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_scalar_override(value: str) -> Any:
    parsed = yaml.safe_load(value)
    return parsed


def set_deep_value(base: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    if not dotted_key.strip():
        raise ValueError("Override key must be non-empty")

    out = dict(base)
    cursor = out
    parts = [part.strip() for part in dotted_key.split(".") if part.strip()]
    if not parts:
        raise ValueError(f"Invalid override key: {dotted_key!r}")

    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
        next_value = dict(next_value)
        cursor[part] = next_value
        cursor = next_value

    cursor[parts[-1]] = value
    return out


def build_override_from_pairs(pairs: list[str] | None) -> dict[str, Any]:
    override: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Invalid override pair {pair!r}; expected KEY=VALUE")
        key, raw = pair.split("=", 1)
        override = set_deep_value(override, key.strip(), parse_scalar_override(raw.strip()))
    return override


def load_override_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = load_yaml(path)
    return payload or {}


def combine_overrides(*overrides: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for override in overrides:
        if override:
            merged = deep_merge(merged, override)
    return merged


def load_experiment_config(
    base_cfg_path: str | Path,
    planner_cfg_path: str | Path | None = None,
    env_cfg_path: str | Path | None = None,
    extra_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(base_cfg_path)
    if planner_cfg_path is not None:
        cfg = deep_merge(cfg, load_yaml(planner_cfg_path))
    if env_cfg_path is not None:
        cfg = deep_merge(cfg, load_yaml(env_cfg_path))
    if extra_override:
        cfg = deep_merge(cfg, extra_override)
    return cfg


def write_config_snapshot(path: str | Path, cfg: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
