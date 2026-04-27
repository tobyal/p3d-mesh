from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


def _update_dataclass_from_dict(dc_obj: Any, raw: dict[str, Any]) -> Any:
    """
    用 dict 递归更新 dataclass 实例
    """
    if not is_dataclass(dc_obj):
        raise TypeError(f"Expected dataclass instance, got: {type(dc_obj)}")

    valid_fields = {f.name for f in fields(dc_obj)}

    for key, value in raw.items():
        if key not in valid_fields:
            raise KeyError(
                f"Unknown config field '{key}' for {dc_obj.__class__.__name__}"
            )

        current_value = getattr(dc_obj, key)

        if is_dataclass(current_value) and isinstance(value, dict):
            _update_dataclass_from_dict(current_value, value)
        else:
            setattr(dc_obj, key, value)

    return dc_obj


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Top-level config must be a dict, got: {type(data)}")

    return data


def load_config(path: str | Path, cfg_obj: Any) -> Any:
    """
    通用配置加载器

    参数:
        path: yaml 路径
        cfg_obj: dataclass 实例，比如 Stage1Config()

    返回:
        更新后的 cfg_obj
    """
    raw = load_yaml_file(path)
    _update_dataclass_from_dict(cfg_obj, raw)
    return cfg_obj


def config_to_dict(cfg: Any) -> dict[str, Any]:
    if not is_dataclass(cfg):
        raise TypeError(f"Expected dataclass instance, got: {type(cfg)}")
    return asdict(cfg)