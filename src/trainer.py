from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from checkpoint import (
    checkpoint_payload,
    initialize_or_resume,
    save_checkpoint,
)
from config import save_resolved_config
from dataset import build_dataset
from distributed import (
    DistributedContext,
    DistributedEvalSampler,
    EpochMetricMeter,
    all_reduce_confusion_matrix,
    assert_config_equal_across_ranks,
    barrier,
    cleanup_distributed,
    parameter_checksum,
    reduce_metric_dict,
    reduce_scalar,
    seed_data_loader_worker,
    setup_distributed,
    unwrap_model,
    validate_global_batch_size,
    wrap_ddp,
)
from inference import sample_segmentation
from metrics import SegmentationMetrics
from model_factory import build_models
from training_objectives import (
    DDPCompatibleTrainingModel,
    run_model_training_objectives,
)
from utils import (
    append_jsonl,
    autocast_context,
    build_grad_scaler,
    init_wandb,
    seed_everything,
    setup_logger,
)
from visualization import save_prediction


MAX_REDUCTION_KEYS = {
    "esd_delta_abs_max",
    "esd_adaptive_weight_max",
    "esd_max_sample_invalid_ratio",
    "esd_jvp_output_abs_max",
    "csd_jvp_output_abs_max",
    "ecld_jvp_output_abs_max",
    "esd_teacher_max",
    "s_max",
    "t_max",
    "runtime_iteration_time",
}

MIN_REDUCTION_KEYS = {
    "esd_log_arg_min",
    "esd_teacher_min",
    "s_min",
    "t_min",
}


def _wandb_iteration_payload(
    reduced: dict[str, torch.Tensor],
) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key, value in reduced.items():
        if key == "runtime_iteration_time":
            name = "runtime/iteration_time"
        elif key == "runtime_samples_per_second":
            name = "runtime/samples_per_second"
        else:
            name = f"train/{key}"
        payload[name] = float(value)
    return payload


class NullLogger:
    def info(self, *args, **kwargs) -> None:
        del args, kwargs

    def warning(self, *args, **kwargs) -> None:
        del args, kwargs


def log_esd_experiment_metadata(config: dict, logger) -> None:
    consistency = config["loss"]["consistency"]
    esd = consistency["esd"]
    precision = consistency["precision"]
    invalid = consistency["invalid_teacher"]
    logger.info("ESD formulation: %s", esd["formulation"])
    logger.info("ESD source: %s", esd["source"])
    logger.info(
        "ESD additional numerical safeguards: %s",
        str(esd["additional_numerical_safeguards"]).lower(),
    )
    logger.info("ESD invalid teacher strategy: %s", invalid["strategy"])
    logger.info("ESD JVP dtype: %s", precision["jvp_dtype"])
    logger.info("ESD numerical dtype: %s", precision["numerical_dtype"])


def build_optimizer(config: dict, adapter: DDPCompatibleTrainingModel):
    optimizer_config = config["training"]["optimizer"]
    model_lr = (
        optimizer_config["parameter_groups"]["model"]["lr"]
        or optimizer_config["lr"]
    )
    groups = [{
        "params": [
            parameter
            for parameter in adapter.endpoint_model.parameters()
            if parameter.requires_grad
        ],
        "lr": model_lr,
        "name": "model",
    }]
    if adapter.source_model is not None and not config["source"]["freeze"]:
        source_parameters = [
            parameter
            for parameter in adapter.source_model.parameters()
            if parameter.requires_grad
        ]
        if source_parameters:
            source_lr = (
                optimizer_config["parameter_groups"]["source"]["lr"]
                or optimizer_config["lr"]
            )
            groups.append({
                "params": source_parameters,
                "lr": source_lr,
                "name": "source",
            })
    optimizer_class = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }.get(optimizer_config["name"])
    if optimizer_class is None:
        raise ValueError(f"Unknown optimizer: {optimizer_config['name']}")
    return optimizer_class(
        groups,
        lr=optimizer_config["lr"],
        weight_decay=optimizer_config["weight_decay"],
        betas=tuple(optimizer_config["betas"]),
    )


def build_scheduler(
    config: dict,
    optimizer,
    *,
    micro_steps_per_epoch: int = 1,
):
    scheduler_config = config["training"]["scheduler"]
    accumulation = config["training"]["grad_accum_steps"]
    optimizer_steps_per_epoch = math.ceil(micro_steps_per_epoch / accumulation)
    total_steps = max(config["training"]["epochs"] * optimizer_steps_per_epoch, 1)
    warmup_steps = scheduler_config["warmup_epochs"] * optimizer_steps_per_epoch
    base_lr = config["training"]["optimizer"]["lr"]
    eta_ratio = scheduler_config["eta_min"] / base_lr

    if scheduler_config["name"] == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if scheduler_config["name"] != "cosine":
        raise ValueError(f"Unknown scheduler: {scheduler_config['name']}")

    def multiplier(step_index: int) -> float:
        if warmup_steps > 0 and step_index < warmup_steps:
            return max((step_index + 1) / warmup_steps, 1.0 / warmup_steps)
        progress = (step_index - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return eta_ratio + (1.0 - eta_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _build_loaders(
    config: dict,
    context: DistributedContext,
    local_batch_size: int,
):
    train_dataset = build_dataset(config, "train", augment=True)
    val_dataset = build_dataset(config, config["evaluation"]["split"], augment=False)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            drop_last=True,
        )
        if context.distributed else None
    )
    val_sampler = (
        DistributedEvalSampler(
            val_dataset, rank=context.rank, world_size=context.world_size
        )
        if context.distributed else None
    )
    workers = config["dataset"]["num_workers"]
    generator = torch.Generator()
    generator.manual_seed(config["experiment"]["seed"] + context.rank)
    common = {
        "num_workers": workers,
        "pin_memory": config["dataset"]["pin_memory"],
        "persistent_workers": (
            config["dataset"]["persistent_workers"] and workers > 0
        ),
        "worker_init_fn": seed_data_loader_worker,
        "generator": generator,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        **common,
    )
    evaluation_global_batch = config["evaluation"]["batch_size"]
    evaluation_local_batch = validate_global_batch_size(
        evaluation_global_batch, context.world_size
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=evaluation_local_batch,
        sampler=val_sampler,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, train_sampler


@torch.no_grad()
def validate(
    config: dict,
    training_model,
    loader,
    context: DistributedContext,
    output_dir: Path,
) -> dict:
    adapter = unwrap_model(training_model)
    endpoint = adapter.endpoint_model
    source = adapter.source_model
    endpoint.eval()
    if source is not None:
        source.eval()
    metrics = SegmentationMetrics(
        config["dataset"]["num_classes"],
        config["dataset"]["void_class_index"],
        device=context.device,
    )
    visualized = 0
    maximum_batches = config["evaluation"]["max_batches"]
    for batch_index, (image, _, target) in enumerate(loader):
        if maximum_batches is not None and batch_index >= maximum_batches:
            break
        image = image.to(context.device, non_blocking=True)
        target = target.to(context.device, non_blocking=True)
        with autocast_context(config, context.device):
            prediction = sample_segmentation(endpoint, source, image, config)
        metrics.update(prediction, target)
        if context.is_main_process:
            remaining = config["evaluation"]["max_visualizations"] - visualized
            for sample_index in range(min(image.shape[0], max(remaining, 0))):
                save_prediction(
                    image[sample_index], target[sample_index], prediction[sample_index],
                    output_dir / "visualizations"
                    / f"val_{batch_index:04d}_{sample_index:02d}.png",
                    config["augmentation"]["imagenet_normalize"],
                )
                visualized += 1
    metrics.confusion_matrix = all_reduce_confusion_matrix(
        metrics.confusion_matrix, context
    )
    result = metrics.compute()
    endpoint.train()
    if source is not None:
        source.train(not config["source"]["freeze"])
    return result


def _operation_for_stage(stage: str) -> str:
    if stage == "diagonal_pretrain":
        return "stage1_objectives"
    if stage in {"consistency_distillation", "esd_distillation"}:
        return "stage2_objectives"
    if stage == "joint_training":
        return "joint_objectives"
    raise ValueError(f"Unknown training stage: {stage}")


def _distributed_metadata(
    context: DistributedContext,
    global_batch_size: int,
    local_batch_size: int,
) -> dict:
    return {
        "world_size": context.world_size,
        "global_batch_size": global_batch_size,
        "local_batch_size": local_batch_size,
    }


def _save_training_checkpoint(
    *,
    config: dict,
    training_model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    metrics: dict,
    context: DistributedContext,
    output_dir: Path,
    filenames: list[str],
    global_batch_size: int,
    local_batch_size: int,
) -> None:
    barrier(context)
    if context.is_main_process:
        adapter = unwrap_model(training_model)
        payload = checkpoint_payload(
            config=config,
            epoch=epoch,
            global_step=global_step,
            model=adapter.endpoint_model,
            source_model=adapter.source_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metrics=metrics,
            distributed=_distributed_metadata(
                context, global_batch_size, local_batch_size
            ),
        )
        for filename in filenames:
            save_checkpoint(payload, output_dir, filename)
    barrier(context)


def run_training(config: dict, *, joint_entrypoint: bool = False) -> dict:
    stage = config["experiment"]["stage"]
    if joint_entrypoint and stage != "joint_training":
        raise ValueError("train_joint.py requires experiment.stage=joint_training")
    if not joint_entrypoint and stage == "joint_training":
        raise ValueError("joint_training must use src/train_joint.py")
    if config["runtime"]["compile"]:
        raise ValueError("runtime.compile is not supported by the DDP/JVP composite trainer")

    context = setup_distributed(config)
    output_dir = Path(config["experiment"]["output_dir"])
    logger = NullLogger()
    wandb_run = None
    try:
        assert_config_equal_across_ranks(config, context)
        global_batch_size = config["training"]["batch_size"]
        local_batch_size = validate_global_batch_size(
            global_batch_size, context.world_size
        )
        effective_global_batch_size = (
            global_batch_size * config["training"]["grad_accum_steps"]
        )
        if (
            config["loss"]["consistency"]["enabled"]
            and config["loss"]["consistency"]["precision"]["jvp_dtype"] == "bf16"
            and context.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("bf16 JVP was requested but this CUDA device lacks bf16")

        if context.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_resolved_config(config, output_dir / "config_resolved.yaml")
            logger = setup_logger(output_dir)
        barrier(context)
        logger.info(
            "world_size=%d rank=%d local_rank=%d global_batch_size=%d "
            "local_batch_size=%d grad_accum_steps=%d effective_global_batch_size=%d",
            context.world_size, context.rank, context.local_rank,
            global_batch_size, local_batch_size,
            config["training"]["grad_accum_steps"], effective_global_batch_size,
        )
        if (
            context.is_main_process
            and stage in {
                "consistency_distillation",
                "esd_distillation",
                "joint_training",
            }
            and config["loss"]["consistency"]["type"] == "esd"
        ):
            log_esd_experiment_metadata(config, logger)

        seed_everything(
            config["experiment"]["seed"], config["runtime"]["deterministic"]
        )
        train_loader, val_loader, train_sampler = _build_loaders(
            config, context, local_batch_size
        )
        endpoint, source = build_models(config, context.device)
        adapter = DDPCompatibleTrainingModel(endpoint, source, config).to(context.device)
        optimizer = build_optimizer(config, adapter)
        max_iterations = config["training"]["max_iterations"]
        micro_steps = min(len(train_loader), max_iterations) if max_iterations else len(train_loader)
        scheduler = build_scheduler(
            config, optimizer, micro_steps_per_epoch=max(micro_steps, 1)
        )
        scaler = build_grad_scaler(config, context.device)
        state = initialize_or_resume(
            config, endpoint, source, optimizer, scheduler, scaler,
            logger if context.is_main_process else None,
        )
        training_model = wrap_ddp(adapter, context, config)
        # Model initialization/checkpoint loading used the same seed. From here on,
        # rank-local stochastic paths intentionally differ.
        torch.manual_seed(config["experiment"]["seed"] + context.rank)
        if context.device.type == "cuda":
            torch.cuda.manual_seed_all(config["experiment"]["seed"] + context.rank)
            torch.cuda.reset_peak_memory_stats(context.device)
        if context.is_main_process:
            wandb_run = init_wandb(config)

        training = config["training"]
        operation = _operation_for_stage(stage)
        metrics_path = output_dir / "metrics.jsonl"
        total_iterations = 0
        last_metrics: dict = {"best_mIoU": state.best_miou}
        optimizer.zero_grad(set_to_none=True)

        for epoch_index in range(state.start_epoch, training["epochs"]):
            training_model.train()
            adapter = unwrap_model(training_model)
            if config["source"]["freeze"] and adapter.source_model is not None:
                adapter.source_model.eval()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch_index)
            epoch_meter = EpochMetricMeter(
                min_keys=MIN_REDUCTION_KEYS,
                max_keys=MAX_REDUCTION_KEYS,
            )
            for batch_index, (image, one_hot, target) in enumerate(train_loader):
                if max_iterations is not None and total_iterations >= max_iterations:
                    break
                iteration_start = time.perf_counter()
                image = image.to(context.device, non_blocking=True)
                one_hot = one_hot.to(context.device, non_blocking=True)
                target = target.to(context.device, non_blocking=True)
                reaches_limit = (
                    max_iterations is not None
                    and total_iterations + 1 >= max_iterations
                )
                should_step = (
                    (batch_index + 1) % training["grad_accum_steps"] == 0
                    or batch_index + 1 == len(train_loader)
                    or reaches_limit
                )
                sync_context = (
                    training_model.no_sync()
                    if context.distributed and not should_step
                    else nullcontext()
                )
                with sync_context:
                    with autocast_context(config, context.device):
                        objectives = run_model_training_objectives(
                            training_model,
                            operation=operation,
                            image=image,
                            one_hot=one_hot,
                            target=target,
                            epoch_index=epoch_index,
                            progress_in_epoch=batch_index / max(len(train_loader), 1),
                        )
                        scaled_loss = (
                            objectives["loss"] / training["grad_accum_steps"]
                        )
                    scaler.scale(scaled_loss).backward()

                grad_norm = torch.zeros((), device=context.device)
                if should_step:
                    scaler.unscale_(optimizer)
                    if training["grad_clip"] is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            adapter.parameters(), training["grad_clip"]
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    state.global_step += 1
                if context.device.type == "cuda":
                    torch.cuda.synchronize(context.device)
                local_iteration_time = time.perf_counter() - iteration_start
                global_iteration_time = reduce_scalar(
                    local_iteration_time, context, "max"
                )
                samples_per_second = (
                    global_batch_size / global_iteration_time.clamp_min(1.0e-9)
                )
                total_iterations += 1
                batch_stats = dict(objectives["stats"])
                batch_stats.update({
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": grad_norm.detach(),
                    "epoch": epoch_index + 1,
                    "global_step": state.global_step,
                    "runtime_iteration_time": global_iteration_time,
                    "runtime_samples_per_second": samples_per_second,
                })
                epoch_meter.update(batch_stats)

                should_log = (
                    total_iterations % training["log_interval"] == 0
                    or total_iterations == 1
                )
                if should_log:
                    scalar_stats = {
                        key: value for key, value in batch_stats.items()
                        if (
                            torch.is_tensor(value) and value.numel() == 1
                        ) or isinstance(value, (int, float))
                    }
                    reduced = reduce_metric_dict(
                        scalar_stats,
                        context,
                        min_keys=MIN_REDUCTION_KEYS,
                        max_keys=MAX_REDUCTION_KEYS,
                    )
                    if context.is_main_process:
                        record = {
                            "scope": "iteration",
                            "stage": stage,
                            "consistency_type": objectives["consistency_type"],
                            "iteration": total_iterations,
                            **reduced,
                        }
                        append_jsonl(metrics_path, record)
                        logger.info(
                            "epoch=%d iter=%d step=%d loss=%.6g diag=%.6g "
                            "consistency=%.6g type=%s",
                            epoch_index + 1, total_iterations, state.global_step,
                            float(reduced["loss_total"]),
                            float(reduced["loss_diagonal"]),
                            float(reduced["loss_consistency"]),
                            objectives["consistency_type"],
                        )
                        if wandb_run is not None:
                            wandb_run.log(
                                _wandb_iteration_payload(reduced),
                                step=state.global_step,
                            )

            epoch_values = epoch_meter.compute()
            epoch_tensors = {
                key: torch.tensor(value, device=context.device)
                for key, value in epoch_values.items()
            }
            reduced_epoch = reduce_metric_dict(
                epoch_tensors,
                context,
                min_keys=MIN_REDUCTION_KEYS,
                max_keys=MAX_REDUCTION_KEYS,
            )
            if context.is_main_process:
                append_jsonl(metrics_path, {
                    "scope": "epoch", "stage": stage,
                    "consistency_type": config["loss"]["consistency"]["type"],
                    "epoch": epoch_index + 1, **reduced_epoch,
                })
                logger.info(
                    "epoch=%d average_loss=%.6g average_consistency=%.6g",
                    epoch_index + 1,
                    float(reduced_epoch.get("loss_total", torch.tensor(float("nan")))),
                    float(reduced_epoch.get("loss_consistency", torch.tensor(0.0))),
                )
                if wandb_run is not None:
                    wandb_run.log({
                        f"epoch/train/{key}": float(value)
                        for key, value in reduced_epoch.items()
                    } | {"epoch/train/epoch": epoch_index + 1}, step=state.global_step)

            validation_metrics = None
            if epoch_index + 1 in set(training["validation_epochs"]):
                validation_metrics = validate(
                    config, training_model, val_loader, context, output_dir
                )
                if context.is_main_process:
                    append_jsonl(metrics_path, {
                        "scope": "validation", "epoch": epoch_index + 1,
                        **validation_metrics,
                    })
                    logger.info(
                        "validation epoch=%d mIoU=%.6g pixel_acc=%.6g mAcc=%.6g",
                        epoch_index + 1, validation_metrics["mIoU"],
                        validation_metrics["pixel_acc"], validation_metrics["mAcc"],
                    )
                    if wandb_run is not None:
                        wandb_run.log({
                            "validation/mIoU": validation_metrics["mIoU"],
                            "validation/pixel_acc": validation_metrics["pixel_acc"],
                            "validation/mAcc": validation_metrics["mAcc"],
                        }, step=state.global_step)
                if validation_metrics["mIoU"] > state.best_miou:
                    state.best_miou = validation_metrics["mIoU"]

            last_metrics = {
                **{key: float(value) for key, value in reduced_epoch.items()},
                **(validation_metrics or {}),
                "best_mIoU": state.best_miou,
            }
            filenames = ["latest.pt"]
            if (epoch_index + 1) % training["checkpoint_interval_epochs"] == 0:
                filenames.append(f"epoch_{epoch_index + 1:04d}.pt")
            if (
                validation_metrics is not None
                and validation_metrics["mIoU"] >= state.best_miou
            ):
                filenames.append("best.pt")
            _save_training_checkpoint(
                config=config,
                training_model=training_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch_index + 1,
                global_step=state.global_step,
                metrics=last_metrics,
                context=context,
                output_dir=output_dir,
                filenames=filenames,
                global_batch_size=global_batch_size,
                local_batch_size=local_batch_size,
            )
            if max_iterations is not None and total_iterations >= max_iterations:
                break

        checksum_stats = parameter_checksum(unwrap_model(training_model), context)
        local_peak = (
            torch.cuda.max_memory_allocated(context.device) / 1024**2
            if context.device.type == "cuda" else 0.0
        )
        mean_peak = float(reduce_scalar(local_peak, context, "mean").cpu())
        max_peak = float(reduce_scalar(local_peak, context, "max").cpu())
        rank_peaks: list[float] = [local_peak]
        if context.distributed:
            gathered = [None] * context.world_size
            dist.all_gather_object(gathered, local_peak)
            rank_peaks = [float(value) for value in gathered]
        runtime_stats = {
            "runtime/world_size": context.world_size,
            "runtime/global_batch_size": global_batch_size,
            "runtime/local_batch_size": local_batch_size,
            "runtime/effective_global_batch_size": effective_global_batch_size,
            "runtime/iteration_time": float(
                reduced_epoch.get("runtime_iteration_time", torch.tensor(0.0))
            ),
            "runtime/samples_per_second": float(
                reduced_epoch.get("runtime_samples_per_second", torch.tensor(0.0))
            ),
            "runtime/peak_gpu_memory_mb": mean_peak,
            "runtime/max_peak_gpu_memory_mb_across_ranks": max_peak,
            "runtime/peak_gpu_memory_mb_by_rank": rank_peaks,
            **checksum_stats,
        }
        if context.is_main_process:
            append_jsonl(metrics_path, {"scope": "runtime", **runtime_stats})
            logger.info(
                "Peak GPU allocated memory by rank=%s; max=%.2f MiB; checksum_diff=%.3g",
                rank_peaks, max_peak, checksum_stats["checksum_max_diff"],
            )
            if wandb_run is not None:
                wandb_run.log({
                    key: value for key, value in runtime_stats.items()
                    if isinstance(value, (int, float))
                }, step=state.global_step)
        last_metrics.update(runtime_stats)
        return last_metrics
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        cleanup_distributed(context)
