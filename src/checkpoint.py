from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel


SCHEDULER_STEP_UNIT = "epoch"
SCHEDULER_VERSION = 2


@dataclass
class TrainingState:
    start_epoch: int = 0
    global_step: int = 0
    best_miou: float = float("-inf")


def model_signature(config: dict) -> dict[str, Any]:
    return {
        "num_classes": config["dataset"]["num_classes"],
        "model": copy.deepcopy(config["model"]),
        "source": {
            key: copy.deepcopy(config["source"][key])
            for key in (
                "prior_type", "prior_noise_std", "backbone", "segformer_variant",
                "pretrained", "freeze_encoder", "decoder_channels",
                "learned_logvar", "fixed_std", "mu_tanh_scale",
            )
        },
    }


def checkpoint_payload(
    *,
    config: dict,
    epoch: int,
    global_step: int,
    model,
    source_model,
    optimizer,
    scheduler,
    scaler,
    metrics: dict,
    distributed: dict | None = None,
) -> dict:
    raw_model = model
    while isinstance(raw_model, DistributedDataParallel):
        raw_model = raw_model.module
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    raw_source = source_model
    while isinstance(raw_source, DistributedDataParallel):
        raw_source = raw_source.module
    raw_source = getattr(raw_source, "_orig_mod", raw_source)
    return {
        "stage": config["experiment"]["stage"],
        "epoch": epoch,
        "global_step": global_step,
        "model": raw_model.state_dict(),
        "source_model": raw_source.state_dict() if raw_source is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scheduler_step_unit": SCHEDULER_STEP_UNIT,
        "scheduler_version": SCHEDULER_VERSION,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": copy.deepcopy(config),
        "model_signature": model_signature(config),
        "metrics": copy.deepcopy(metrics),
        "distributed": copy.deepcopy(distributed or {
            "world_size": 1,
            "global_batch_size": config["training"]["batch_size"],
            "local_batch_size": config["training"]["batch_size"],
        }),
    }


def save_checkpoint(payload: dict, output_dir: str | Path, filename: str) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / filename
    temporary = output / f".{filename}.tmp"
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def _validate_stage1_checkpoint(checkpoint: dict, config: dict, path: str | Path) -> None:
    if checkpoint.get("stage") != "diagonal_pretrain":
        raise RuntimeError(
            f"Stage 2 init_from requires a diagonal_pretrain checkpoint: {path}"
        )
    saved_signature = checkpoint.get("model_signature")
    current_signature = model_signature(config)
    if saved_signature is None:
        saved_config = checkpoint.get("config", {})
        saved_signature = model_signature(saved_config)
    if saved_signature != current_signature:
        raise RuntimeError(
            "Stage 1 checkpoint is incompatible with Stage 2 model/source configuration.\n"
            f"saved={saved_signature}\ncurrent={current_signature}"
        )


def _without_module_prefix(state_dict: dict) -> dict:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def _resume_stage_compatible(checkpoint: dict, config: dict) -> bool:
    saved_stage = checkpoint.get("stage")
    current_stage = config["experiment"]["stage"]
    if saved_stage == current_stage:
        return True
    consistency_type = config["loss"]["consistency"]["type"]
    stage2_names = {"consistency_distillation", "esd_distillation"}
    return (
        saved_stage in stage2_names
        and current_stage in stage2_names
        and consistency_type == "esd"
    )


def _validate_resume_scheduler(checkpoint: dict, path: str | Path) -> None:
    step_unit = checkpoint.get("scheduler_step_unit")
    version = checkpoint.get("scheduler_version")
    if step_unit != SCHEDULER_STEP_UNIT or version != SCHEDULER_VERSION:
        raise RuntimeError(
            "Resume checkpoint uses an incompatible scheduler format: "
            f"{path} has scheduler_step_unit={step_unit!r}, "
            f"scheduler_version={version!r}; expected "
            f"{SCHEDULER_STEP_UNIT!r}, version {SCHEDULER_VERSION}. "
            "Legacy optimizer-step scheduler checkpoints cannot be resumed "
            "with the epoch scheduler."
        )


def initialize_or_resume(
    config: dict,
    model,
    source_model,
    optimizer,
    scheduler,
    scaler,
    logger=None,
) -> TrainingState:
    checkpoint_config = config["checkpoint"]
    init_from = checkpoint_config["init_from"]
    resume = checkpoint_config["resume"]
    if init_from and resume:
        raise ValueError("checkpoint.init_from and checkpoint.resume are mutually exclusive")
    strict = checkpoint_config["strict_model"]
    if init_from:
        checkpoint = torch.load(init_from, map_location="cpu", weights_only=False)
        _validate_stage1_checkpoint(checkpoint, config, init_from)
        model.load_state_dict(_without_module_prefix(checkpoint["model"]), strict=strict)
        saved_source = checkpoint.get("source_model")
        if source_model is not None:
            if saved_source is None:
                raise RuntimeError("Stage 1 checkpoint has no source_model state")
            source_model.load_state_dict(
                _without_module_prefix(saved_source), strict=strict
            )
        metrics = checkpoint.get("metrics", {})
        best = metrics.get("best_mIoU", metrics.get("mIoU", float("-inf")))
        lines = (
            f"Loaded Stage 1 checkpoint: {init_from}",
            f"Stage 1 completed epoch: {checkpoint.get('epoch', 'unknown')}",
            f"Stage 1 best mIoU: {best}",
            "Starting Stage 2 from epoch 1",
            "Optimizer state: newly initialized",
            "Scheduler state: newly initialized",
            f"Consistency loss: {config['loss']['consistency']['type']}",
            f"ESD enabled: {str(config['loss']['consistency']['type'] == 'esd').lower()}",
        )
        if logger is not None:
            for line in lines:
                logger.info(line)
        return TrainingState()
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if not _resume_stage_compatible(checkpoint, config):
            raise RuntimeError(
                f"Resume stage mismatch: checkpoint={checkpoint.get('stage')} "
                f"config={config['experiment']['stage']}"
            )
        _validate_resume_scheduler(checkpoint, resume)
        model.load_state_dict(_without_module_prefix(checkpoint["model"]), strict=strict)
        if source_model is not None:
            if checkpoint.get("source_model") is None:
                raise RuntimeError("Resume checkpoint has no source_model state")
            source_model.load_state_dict(
                _without_module_prefix(checkpoint["source_model"]), strict=strict
            )
        # A resume is deliberately a complete continuation. The load_* fields are
        # relevant to legacy/import workflows, but may not weaken resume semantics.
        if checkpoint.get("optimizer") is None or checkpoint.get("scheduler") is None:
            raise RuntimeError("Resume checkpoint lacks optimizer or scheduler state")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        saved_distributed = checkpoint.get("distributed", {})
        saved_global_batch = saved_distributed.get("global_batch_size")
        current_global_batch = config["training"]["batch_size"]
        if (
            saved_global_batch is not None
            and saved_global_batch != current_global_batch
        ):
            warnings.warn(
                "Resuming with a changed global batch size: "
                f"checkpoint={saved_global_batch}, current={current_global_batch}",
                RuntimeWarning,
            )
        metrics = checkpoint.get("metrics", {})
        if logger is not None:
            logger.info(
                "Resumed %s at completed epoch %s, global_step=%s",
                resume, checkpoint["epoch"], checkpoint["global_step"],
            )
        return TrainingState(
            start_epoch=int(checkpoint["epoch"]),
            global_step=int(checkpoint["global_step"]),
            best_miou=float(metrics.get("best_mIoU", metrics.get("mIoU", float("-inf")))),
        )
    return TrainingState()
