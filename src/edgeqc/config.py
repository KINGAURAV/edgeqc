"""YAML config loading with dotted-key CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: list[str] | None = None,
                device: str | Path | None = None) -> dict:
    """Load default config, merge an optional user/product config, then an optional device config
    (configs/devices/*.yaml: threads, core split, frame rate), then apply `a.b.c=value` overrides."""
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    for extra in (path, device):
        if extra and Path(extra).resolve() != DEFAULT_CONFIG:
            with open(extra) as f:
                cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    # A product profile can carry per-device settings (e.g. smaller models for the Pi 4) under
    # device_overrides.<name>; they apply when the device config sets device.name to that name.
    name = (cfg.get("device") or {}).get("name")
    per_device = cfg.pop("device_overrides", None) or {}
    if name and name in per_device:
        cfg = _deep_merge(cfg, per_device[name] or {})
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
