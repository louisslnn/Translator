from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "batch_size": 64,
    "num_epochs": 30,
    "lr": 3e-4,
    "seq_len": 350,
    "d_model": 512,
    "lang_src": "en",
    "lang_tgt": "fr",
    "model_folder": "weights",
    "model_basename": "tmodel_",
    "preload": None,
    "tokenizer_file": "tokenizer_{0}.json",
    "experiment_name": "runs/tmodel",
    "save_every": 1,
    "validate_every_steps": 500,
    "validation_examples": 2,
    "warmup_steps": 4000,
    "grad_accumulation_steps": 1,
    "mixed_precision": "bf16",
    "device": "cuda",
    "allow_tf32": True,
    "enable_compile": False,
    "compile_fullgraph": False,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
    "drop_last": True,
    "seed": 42,
    "dataset_cache_dir": None,
    "log_every_steps": 50,
}


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path} must define a mapping.")
    return data


def _merge_configs(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not override:
        return base
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _merge_configs(base[key], value)
        else:
            base[key] = value
    return base


def get_config(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration values respecting (in order of precedence):
      1. defaults from this module
      2. values defined in an optional YAML file
      3. inline overrides supplied by the caller
    """

    config: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)

    if config_path:
        yaml_path = Path(config_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file {config_path} was not found.")
        config = _merge_configs(config, _read_yaml(yaml_path))

    if overrides:
        config = _merge_configs(config, overrides)

    # Ensure folders exist and are strings
    config["model_folder"] = str(Path(config["model_folder"]))
    config["experiment_name"] = str(config["experiment_name"])
    config["tokenizer_file"] = str(config["tokenizer_file"])

    dataset_cache = config.get("dataset_cache_dir")
    if dataset_cache:
        config["dataset_cache_dir"] = str(Path(dataset_cache))

    return config


def get_weights_file_path(config: Dict[str, Any], epoch: str) -> str:
    model_folder = Path(config["model_folder"])
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(model_folder / model_filename)