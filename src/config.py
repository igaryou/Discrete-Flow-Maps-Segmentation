from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "",
        "seed": 42,
        "output_dir": "",
        "stage": "diagonal_pretrain",
    },
    "runtime": {
        "device": "auto",
        "amp": True,
        "amp_dtype": "bf16",
        "compile": False,
        "deterministic": False,
        "config_path": None,
    },
    "dataset": {
        "name": "cityscapes",
        "root": "",
        "num_classes": 20,
        "eval_num_classes": 19,
        "void_class_index": 19,
        "image_size": [256, 512],
        "crop_size": None,
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": False,
    },
    "augmentation": {
        "enabled": True,
        "horizontal_flip": {"enabled": True, "probability": 0.5},
        "color_jitter": {
            "enabled": True,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
        },
        "imagenet_normalize": False,
    },
    "model": {
        "backbone": "unet",
        "num_classes": 20,
        "fusion_channels": 128,
        "rrdb_blocks": 5,
        "rrdb_growth_channels": 32,
        "unet": {
            "base_channels": 64,
            "channel_mults": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attention_levels": [3],
            "num_heads": 4,
            "dropout": 0.0,
            "time_embedding_dim": 256,
        },
    },
    "source": {
        "prior_type": "image_gaussian",
        "prior_noise_std": 1.0,
        "backbone": "segformer",
        "segformer_variant": "b0",
        "pretrained": True,
        "checkpoint": None,
        "freeze": False,
        "freeze_encoder": False,
        "decoder_channels": 128,
        "learned_logvar": False,
        "fixed_std": None,
        "mu_tanh_scale": 0.0,
        "use_loss_align": True,
        "align_weight": 0.15,
        "var_weight": 0.0,
        "align_eps": 1.0e-8,
    },
    "flow": {
        "time_eps": 1.0e-5,
        "probability_eps": 1.0e-8,
        "start_time": 0.0,
    },
    "time_sampling": {
        "distribution": "uniform_sorted",
        "min_time": 0.0,
        "max_time": 1.0,
        "min_gap": 1.0e-5,
    },
    "training": {
        "epochs": 150,
        "max_iterations": None,
        "batch_size": 4,
        "grad_accum_steps": 1,
        "optimizer": {
            "name": "adamw",
            "lr": 5.0e-5,
            "weight_decay": 1.0e-4,
            "betas": [0.9, 0.999],
            "parameter_groups": {
                "model": {"lr": None},
                "source": {"lr": None},
            },
        },
        "scheduler": {
            "name": "cosine",
            "warmup_epochs": 20,
            "eta_min": 1.0e-6,
        },
        "grad_clip": 1.0,
        "label_smoothing": 0.1,
        "log_interval": 50,
        "checkpoint_interval_epochs": 10,
        "validation_epochs": [50, 100, 150],
    },
    "loss": {
        "primary": {"type": "diagonal_ce", "weight": 1.0},
        "consistency": {
            "enabled": False,
            "type": "esd",
            "weight": 0.0,
            "start_epoch": 0,
            "warmup_epochs": 0,
            "schedule": "linear",
            "max_weight": 1.0,
            "adaptive_kl": {
                "enabled": False,
                "c": 1.0e-6,
                "r": 0.5,
                "normalize_mean": True,
                "max_weight": 100.0,
            },
            "invalid_teacher": {
                "strategy": "mask_pixel",
                "log_eps": 1.0e-6,
                "skip_batch_threshold": None,
            },
        },
    },
    "checkpoint": {
        "resume": None,
        "init_from": None,
        "load_optimizer": True,
        "load_scheduler": True,
        "strict_model": True,
    },
    "evaluation": {
        "split": "val",
        "batch_size": 4,
        "num_steps": 15,
        "sampler": "flow_map",
        "save_predictions": True,
        "max_visualizations": 16,
        "max_batches": None,
        "checkpoint": None,
    },
    "wandb": {
        "enabled": True,
        "project": "DFM",
        "name": None,
        "entity": None,
        "mode": "online",
        "tags": [],
    },
}

REQUIRED_SECTIONS = tuple(DEFAULT_CONFIG)
REQUIRED_KEYS = (
    "experiment.name",
    "experiment.output_dir",
    "experiment.stage",
    "dataset.root",
    "dataset.num_classes",
    "model.num_classes",
    "training.epochs",
    "loss.primary.type",
)


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    return value


def _merge(defaults: dict[str, Any], supplied: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(defaults)
    for key, value in supplied.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _check_unknown(supplied: dict[str, Any], schema: dict[str, Any], prefix: str = "") -> None:
    for key, value in supplied.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if key not in schema:
            raise ValueError(f"Unknown config key: {dotted}")
        if isinstance(value, dict):
            if not isinstance(schema[key], dict):
                raise ValueError(f"Config key must not be a mapping: {dotted}")
            _check_unknown(value, schema[key], dotted)


def select(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _validate_required(raw: dict[str, Any]) -> None:
    for section in REQUIRED_SECTIONS:
        if section not in raw:
            raise ValueError(f"Missing required config section: {section}")
    for dotted in REQUIRED_KEYS:
        if select(raw, dotted, None) in (None, ""):
            raise ValueError(f"Missing required config key: {dotted}")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    stage = select(config, "experiment.stage")
    if stage not in {"diagonal_pretrain", "esd_distillation"}:
        raise ValueError("experiment.stage must be diagonal_pretrain or esd_distillation")
    if config["dataset"]["name"] != "cityscapes":
        raise ValueError("Only dataset.name=cityscapes is supported")
    if config["dataset"]["num_classes"] != 20 or config["model"]["num_classes"] != 20:
        raise ValueError("DFM Cityscapes training requires exactly 20 classes")
    if config["dataset"]["eval_num_classes"] != 19:
        raise ValueError("dataset.eval_num_classes must be 19")
    if config["dataset"]["void_class_index"] != 19:
        raise ValueError("dataset.void_class_index must be 19")
    if len(config["dataset"]["image_size"]) != 2:
        raise ValueError("dataset.image_size must be [height, width]")
    if config["dataset"]["crop_size"] is not None and len(config["dataset"]["crop_size"]) != 2:
        raise ValueError("dataset.crop_size must be null or [height, width]")
    if config["runtime"]["amp_dtype"] not in {"bf16", "fp16"}:
        raise ValueError("runtime.amp_dtype must be bf16 or fp16")
    if config["model"]["backbone"] != "unet":
        raise ValueError("This DFM implementation currently supports model.backbone=unet")
    if config["source"]["prior_type"] not in {"gaussian", "dirichlet", "image_gaussian"}:
        raise ValueError("source.prior_type must be gaussian, dirichlet, or image_gaussian")
    if config["source"]["backbone"] not in {"segformer", "unet"}:
        raise ValueError("source.backbone must be segformer or unet")
    if config["flow"]["time_eps"] <= 0:
        raise ValueError("flow.time_eps must be positive")
    ts = config["time_sampling"]
    if ts["distribution"] != "uniform_sorted":
        raise ValueError("time_sampling.distribution must be uniform_sorted")
    if not (0.0 <= ts["min_time"] < ts["max_time"] <= 1.0):
        raise ValueError("time range must satisfy 0 <= min_time < max_time <= 1")
    if not (0.0 < ts["min_gap"] <= ts["max_time"] - ts["min_time"]):
        raise ValueError("time_sampling.min_gap is outside the configured range")
    invalid = config["loss"]["consistency"]["invalid_teacher"]
    if invalid["strategy"] not in {"clamp", "mask_pixel", "skip_batch"}:
        raise ValueError("invalid_teacher.strategy must be clamp, mask_pixel, or skip_batch")
    if config["checkpoint"]["init_from"] and config["checkpoint"]["resume"]:
        raise ValueError("checkpoint.init_from and checkpoint.resume are mutually exclusive")
    consistency = config["loss"]["consistency"]
    if consistency["schedule"] != "linear":
        raise ValueError("loss.consistency.schedule currently supports only linear")
    if stage == "diagonal_pretrain" and consistency["enabled"]:
        raise ValueError("Stage 1 must not enable a consistency loss")
    if stage == "esd_distillation":
        if not consistency["enabled"] or consistency["type"] != "esd":
            raise ValueError("Stage 2 requires loss.consistency.enabled=true and type=esd")
        if not config["checkpoint"]["init_from"] and not config["checkpoint"]["resume"]:
            raise ValueError("Stage 2 requires checkpoint.init_from or checkpoint.resume")
    return config


def apply_overrides(config: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must have KEY=VALUE form: {override}")
        dotted, raw_value = override.split("=", 1)
        parts = dotted.split(".")
        cursor: Any = result
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise ValueError(f"Unknown override key: {dotted}")
            cursor = cursor[part]
        if not isinstance(cursor, dict) or parts[-1] not in cursor:
            raise ValueError(f"Unknown override key: {dotted}")
        cursor[parts[-1]] = yaml.safe_load(raw_value)
    return validate_config(result)


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {config_path}")
    _validate_required(raw)
    _check_unknown(raw, DEFAULT_CONFIG)
    config = _merge(DEFAULT_CONFIG, _expand(raw))
    config["runtime"]["config_path"] = str(config_path)
    return apply_overrides(validate_config(config), overrides)


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
