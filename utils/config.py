"""Loads YAML configs (with `_base_` inheritance) and applies dotted CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


class Config(dict):
    """A dict that also supports attribute access, so `cfg.train.batch_size` works."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key, value):
        self[key] = value


def _deep_update(base: dict, extra: dict) -> dict:
    """Recursively merges `extra` into `base` so partial override files stay small."""
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _apply_override(cfg: dict, dotted: str) -> None:
    """Applies one `a.b.c=value` string, parsing the value as YAML so types stay correct."""
    if "=" not in dotted:
        raise ValueError(f"Override '{dotted}' must look like key.subkey=value")
    path, raw = dotted.split("=", 1)
    keys = path.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = yaml.safe_load(raw)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Reads a YAML file, resolves its `_base_` parent chain, then applies CLI overrides."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if "_base_" in raw:
        parent = load_config(path.parent / raw.pop("_base_"))
        raw = _deep_update(dict(parent), raw)

    for dotted in overrides or []:
        _apply_override(raw, dotted)

    return Config(raw)


def save_config(cfg: Config, path: str | Path) -> None:
    """Dumps the fully-resolved config next to the checkpoints for reproducibility."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(cfg), fh, sort_keys=False)
