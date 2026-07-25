from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from checkpoint import (
    checkpoint_payload,
    initialize_or_resume,
    save_checkpoint,
)
from config import load_config, save_resolved_config
from dataset import build_dataset
from discrete_flow_maps import (
    linear_path,
    sample_prior,
    sample_sorted_times,
    sample_stage1_times,
)
from inference import sample_segmentation
from losses import compute_training_losses, esd_schedule_weight
from metrics import SegmentationMetrics
from model_factory import build_models
from utils import (
    AverageMeter,
    append_jsonl,
    autocast_context,
    build_grad_scaler,
    init_wandb,
    resolve_device,
    seed_everything,
    setup_logger,
)
from visualization import save_prediction


def build_optimizer(config: dict, model, source_model):
    optimizer_config = config["training"]["optimizer"]
    model_lr = (
        optimizer_config["parameter_groups"]["model"]["lr"]
        or optimizer_config["lr"]
    )
    groups = [{
        "params": [parameter for parameter in model.parameters() if parameter.requires_grad],
        "lr": model_lr,
        "name": "model",
    }]
    if source_model is not None and not config["source"]["freeze"]:
        source_parameters = [
            parameter for parameter in source_model.parameters() if parameter.requires_grad
        ]
        if source_parameters:
            source_lr = (
                optimizer_config["parameter_groups"]["source"]["lr"]
                or optimizer_config["lr"]
            )
            groups.append({"params": source_parameters, "lr": source_lr, "name": "source"})
    optimizer_class = (
        torch.optim.AdamW if optimizer_config["name"] == "adamw" else torch.optim.Adam
    )
    if optimizer_config["name"] not in {"adam", "adamw"}:
        raise ValueError(f"Unknown optimizer: {optimizer_config['name']}")
    return optimizer_class(
        groups,
        lr=optimizer_config["lr"],
        weight_decay=optimizer_config["weight_decay"],
        betas=tuple(optimizer_config["betas"]),
    )


def build_scheduler(config: dict, optimizer):
    scheduler_config = config["training"]["scheduler"]
    if scheduler_config["name"] == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if scheduler_config["name"] != "cosine":
        raise ValueError(f"Unknown scheduler: {scheduler_config['name']}")
    epochs = config["training"]["epochs"]
    warmup = scheduler_config["warmup_epochs"]
    base_lr = config["training"]["optimizer"]["lr"]
    eta_ratio = scheduler_config["eta_min"] / base_lr

    def multiplier(epoch_index: int) -> float:
        if warmup > 0 and epoch_index < warmup:
            return max((epoch_index + 1) / warmup, 1.0 / warmup)
        progress = (epoch_index - warmup) / max(epochs - warmup, 1)
        return eta_ratio + (1.0 - eta_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def build_loader(config: dict, split: str, shuffle: bool) -> DataLoader:
    dataset = build_dataset(config, split, augment=(split == "train"))
    workers = config["dataset"]["num_workers"]
    return DataLoader(
        dataset,
        batch_size=(
            config["training"]["batch_size"]
            if split == "train" else config["evaluation"]["batch_size"]
        ),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=config["dataset"]["pin_memory"],
        persistent_workers=config["dataset"]["persistent_workers"] and workers > 0,
        drop_last=split == "train",
    )


def _esd_arguments(config: dict, s: torch.Tensor, t: torch.Tensor) -> dict:
    consistency = config["loss"]["consistency"]
    invalid = consistency["invalid_teacher"]
    adaptive = consistency["adaptive_kl"]
    return {
        "s": s,
        "t": t,
        "time_eps": config["flow"]["time_eps"],
        "log_eps": invalid["log_eps"],
        "invalid_strategy": invalid["strategy"],
        "skip_batch_threshold": invalid["skip_batch_threshold"],
        "adaptive_enabled": adaptive["enabled"],
        "adaptive_c": adaptive["c"],
        "adaptive_r": adaptive["r"],
        "adaptive_normalize_mean": adaptive["normalize_mean"],
        "adaptive_max_weight": adaptive["max_weight"],
    }


@torch.no_grad()
def validate(config: dict, model, source_model, loader, device, output_dir: Path):
    metrics = SegmentationMetrics(
        config["dataset"]["num_classes"], config["dataset"]["void_class_index"]
    )
    visualized = 0
    maximum_batches = config["evaluation"]["max_batches"]
    for batch_index, (image, _, target) in enumerate(loader):
        if maximum_batches is not None and batch_index >= maximum_batches:
            break
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast_context(config, device):
            prediction = sample_segmentation(model, source_model, image, config)
        metrics.update(prediction, target)
        remaining = config["evaluation"]["max_visualizations"] - visualized
        for sample_index in range(min(image.shape[0], max(remaining, 0))):
            save_prediction(
                image[sample_index], target[sample_index], prediction[sample_index],
                output_dir / "visualizations" / f"val_{batch_index:04d}_{sample_index:02d}.png",
                config["augmentation"]["imagenet_normalize"],
            )
            visualized += 1
    return metrics.compute()


def train(config: dict) -> dict:
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_dir / "config_resolved.yaml")
    logger = setup_logger(output_dir)
    seed_everything(
        config["experiment"]["seed"], config["runtime"]["deterministic"]
    )
    device = resolve_device(config["runtime"]["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    logger.info("Experiment=%s stage=%s device=%s", config["experiment"]["name"],
                config["experiment"]["stage"], device)

    train_loader = build_loader(config, "train", shuffle=True)
    val_loader = build_loader(config, config["evaluation"]["split"], shuffle=False)
    model, source_model = build_models(config, device)
    optimizer = build_optimizer(config, model, source_model)
    scheduler = build_scheduler(config, optimizer)
    scaler = build_grad_scaler(config, device)
    state = initialize_or_resume(
        config, model, source_model, optimizer, scheduler, scaler, logger
    )
    if config["runtime"]["compile"]:
        model = torch.compile(model)
    wandb_run = init_wandb(config)

    stage = config["experiment"]["stage"]
    training = config["training"]
    time_config = config["time_sampling"]
    consistency = config["loss"]["consistency"]
    metrics_path = output_dir / "metrics.jsonl"
    total_iterations = 0
    last_metrics: dict = {"best_mIoU": state.best_miou}
    optimizer.zero_grad(set_to_none=True)

    try:
        for epoch_index in range(state.start_epoch, training["epochs"]):
            model.train()
            if source_model is not None:
                source_model.train(not config["source"]["freeze"])
            epoch_meter = AverageMeter()
            for batch_index, (image, one_hot, target) in enumerate(train_loader):
                if (
                    training["max_iterations"] is not None
                    and total_iterations >= training["max_iterations"]
                ):
                    break
                image = image.to(device, non_blocking=True)
                one_hot = one_hot.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                with autocast_context(config, device):
                    x0, source_stats = sample_prior(
                        config, image, one_hot, source_model
                    )
                    if stage == "diagonal_pretrain":
                        diagonal_time = sample_stage1_times(
                            image.shape[0], device, time_config["min_time"],
                            time_config["max_time"],
                        )
                        x_path = linear_path(x0, one_hot, diagonal_time)
                        schedule_weight = 0.0
                        effective_weight = 0.0
                        esd_kwargs = None
                    else:
                        s, t = sample_sorted_times(
                            image.shape[0], device, time_config["min_time"],
                            time_config["max_time"], time_config["min_gap"],
                        )
                        diagonal_time = s
                        x_path = linear_path(x0, one_hot, s)
                        progress = batch_index / max(len(train_loader), 1)
                        schedule_weight = esd_schedule_weight(
                            epoch_index, progress, consistency["start_epoch"],
                            consistency["warmup_epochs"],
                        )
                        effective_weight = (
                            consistency["weight"]
                            * consistency["max_weight"]
                            * schedule_weight
                        )
                        esd_kwargs = _esd_arguments(config, s, t)

                    total_loss, batch_stats, _ = compute_training_losses(
                        stage=stage,
                        model=model,
                        x_path=x_path,
                        image=image,
                        target=target,
                        diagonal_time=diagonal_time,
                        label_smoothing=training["label_smoothing"],
                        primary_weight=config["loss"]["primary"]["weight"],
                        source_stats=source_stats,
                        esd_kwargs=esd_kwargs,
                        esd_effective_weight=effective_weight,
                    )
                    scaled_loss = total_loss / training["grad_accum_steps"]

                scaler.scale(scaled_loss).backward()
                update_now = (
                    (batch_index + 1) % training["grad_accum_steps"] == 0
                    or batch_index + 1 == len(train_loader)
                )
                grad_norm = torch.zeros((), device=device)
                if update_now:
                    scaler.unscale_(optimizer)
                    if training["grad_clip"] is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), training["grad_clip"]
                        )
                    else:
                        norms = [
                            parameter.grad.detach().norm()
                            for parameter in model.parameters()
                            if parameter.grad is not None
                        ]
                        grad_norm = torch.stack(norms).norm() if norms else grad_norm
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    state.global_step += 1

                total_iterations += 1
                batch_stats.update({
                    "esd_base_weight": consistency["weight"] if stage == "esd_distillation" else 0.0,
                    "esd_schedule_weight": schedule_weight,
                    "esd_effective_weight": effective_weight,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": grad_norm.detach(),
                    "epoch": epoch_index + 1,
                    "global_step": state.global_step,
                })
                epoch_meter.update(batch_stats)
                if total_iterations % training["log_interval"] == 0 or total_iterations == 1:
                    record = {
                        "scope": "iteration", "stage": stage,
                        "iteration": total_iterations, **batch_stats,
                    }
                    append_jsonl(metrics_path, record)
                    rendered = " ".join(
                        f"{key}={float(value):.6g}"
                        for key, value in batch_stats.items()
                        if key in {
                            "loss_total", "loss_diagonal", "loss_esd",
                            "esd_clamp_ratio", "esd_valid_pixel_ratio",
                            "esd_effective_weight",
                        }
                    )
                    logger.info(
                        "epoch=%d iter=%d step=%d %s",
                        epoch_index + 1, total_iterations, state.global_step, rendered,
                    )
                    if wandb_run is not None:
                        wandb_run.log({
                            f"train/{key}": (
                                float(value.detach()) if torch.is_tensor(value) else value
                            )
                            for key, value in batch_stats.items()
                        }, step=state.global_step)

            epoch_values = epoch_meter.compute()
            epoch_record = {
                "scope": "epoch", "stage": stage, "epoch": epoch_index + 1,
                **epoch_values,
            }
            append_jsonl(metrics_path, epoch_record)
            if wandb_run is not None:
                wandb_run.log({
                    f"epoch/train/{key}": value
                    for key, value in epoch_values.items()
                } | {"epoch/train/epoch": epoch_index + 1}, step=state.global_step)
            logger.info(
                "epoch=%d average_loss=%.6g esd_clamp=%.6g valid_pixels=%.6g",
                epoch_index + 1,
                epoch_values.get("loss_total", float("nan")),
                epoch_values.get("esd_clamp_ratio", 0.0),
                epoch_values.get("esd_valid_pixel_ratio", 1.0),
            )

            validation_metrics = None
            if epoch_index + 1 in set(training["validation_epochs"]):
                validation_metrics = validate(
                    config, model, source_model, val_loader, device, output_dir
                )
                logger.info(
                    "validation epoch=%d mIoU=%.6g pixel_acc=%.6g mAcc=%.6g",
                    epoch_index + 1, validation_metrics["mIoU"],
                    validation_metrics["pixel_acc"], validation_metrics["mAcc"],
                )
                append_jsonl(metrics_path, {
                    "scope": "validation", "epoch": epoch_index + 1,
                    **validation_metrics,
                })
                if wandb_run is not None:
                    wandb_run.log({
                        "validation/mIoU": validation_metrics["mIoU"],
                        "validation/pixel_acc": validation_metrics["pixel_acc"],
                        "validation/mAcc": validation_metrics["mAcc"],
                    }, step=state.global_step)
                if validation_metrics["mIoU"] > state.best_miou:
                    state.best_miou = validation_metrics["mIoU"]

            last_metrics = {
                **epoch_values,
                **(validation_metrics or {}),
                "best_mIoU": state.best_miou,
            }
            # Persist the scheduler after the completed epoch so resume reproduces
            # the exact learning rate that the next epoch would have used.
            scheduler.step()
            payload = checkpoint_payload(
                config=config, epoch=epoch_index + 1, global_step=state.global_step,
                model=model, source_model=source_model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, metrics=last_metrics,
            )
            save_checkpoint(payload, output_dir, "latest.pt")
            if (epoch_index + 1) % training["checkpoint_interval_epochs"] == 0:
                save_checkpoint(payload, output_dir, f"epoch_{epoch_index + 1:04d}.pt")
            if validation_metrics is not None and validation_metrics["mIoU"] >= state.best_miou:
                save_checkpoint(payload, output_dir, "best.pt")

            if (
                training["max_iterations"] is not None
                and total_iterations >= training["max_iterations"]
            ):
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        logger.info("Peak GPU allocated memory: %.2f MiB", peak_mb)
        last_metrics["peak_gpu_memory_mb"] = peak_mb
        append_jsonl(metrics_path, {"scope": "runtime", "peak_gpu_memory_mb": peak_mb})
    return last_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Discrete Flow Maps segmentation")
    parser.add_argument("--config", required=True, help="YAML configuration path")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Limited dotted YAML override; may be repeated",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    train(load_config(arguments.config, arguments.set))


if __name__ == "__main__":
    main()
