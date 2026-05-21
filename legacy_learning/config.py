from __future__ import annotations

import argparse
import ast
import copy
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "run": {
        "stage": "legacy",
        "seed": 123,
        "device": "auto",
        "output_dir": "outputs/run",
        "max_steps": 40000,
        "log_interval": 100,
        "ckpt_interval": 2500,
        "dry_run": False,
    },
    "model": {
        "name": "fontdiffuser",
        "repo_path": None,
        "image_size": 96,
    },
    "checkpoint": {
        "base_checkpoint": None,
    },
    "data": {
        "manifest": "data/manifests/legacy_train.csv",
        "image_size": 96,
        "batch_size": 4,
        "num_workers": 0,
        "shuffle": True,
    },
    "optimization": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
    },
    "legacy_learning": {
        "enabled": True,
        "preservation_weight": 1.0,
        "alpha": 1.0,
        "beta_schedule": "cosine",
        "fixed_beta": None,
        "content_preservation_content_encoder_only": True,
        "use_distance_term": True,
        "use_direction_term": True,
    },
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def set_if_not_none(cfg: Dict[str, Any], section: str, key: str, value: Any) -> None:
    if value is not None:
        cfg.setdefault(section, {})[key] = value


def load_yaml_config(config_path: str | None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return cfg

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) if yaml is not None else minimal_yaml_load(f.read())
        loaded = loaded or {}
    return deep_update(cfg, loaded)


def minimal_yaml_load(text: str) -> Dict[str, Any]:
    """Tiny YAML subset parser for the simple config files in this repo.

    It supports nested dictionaries by indentation and scalar values such as
    strings, numbers, booleans, null, and Python-like inline lists.
    Install PyYAML for full YAML support.
    """

    root: Dict[str, Any] = {}
    stack = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if value == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)

    return root


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def load_runtime_config() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Legacy learning runner for FFG models.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stage", type=str, choices=["pretrain", "legacy"], default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_repo", type=str, default=None)
    parser.add_argument("--base_checkpoint", type=str, default=None)
    parser.add_argument("--data_manifest", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    set_if_not_none(cfg, "run", "stage", args.stage)
    set_if_not_none(cfg, "run", "output_dir", args.output_dir)
    set_if_not_none(cfg, "run", "max_steps", args.max_steps)
    set_if_not_none(cfg, "run", "device", args.device)
    set_if_not_none(cfg, "model", "name", args.model)
    set_if_not_none(cfg, "model", "repo_path", args.model_repo)
    set_if_not_none(cfg, "checkpoint", "base_checkpoint", args.base_checkpoint)
    set_if_not_none(cfg, "data", "manifest", args.data_manifest)
    set_if_not_none(cfg, "data", "batch_size", args.batch_size)
    set_if_not_none(cfg, "optimization", "learning_rate", args.learning_rate)

    if args.dry_run:
        cfg["run"]["dry_run"] = True

    return cfg
