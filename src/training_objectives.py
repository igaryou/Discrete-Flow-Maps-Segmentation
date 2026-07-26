from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

import losses
from discrete_flow_maps import (
    linear_path,
    sample_consistency_times,
    sample_prior,
    sample_stage1_times,
)


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.float().sum() * 0.0


def compute_model_training_objectives(
    adapter: "DDPCompatibleTrainingModel",
    *,
    operation: str,
    image: torch.Tensor,
    one_hot: torch.Tensor,
    target: torch.Tensor,
    epoch_index: int,
    progress_in_epoch: float,
) -> dict[str, Any]:
    """Build the complete endpoint/source graph inside one composite forward."""
    if operation not in {"stage1_objectives", "stage2_objectives", "joint_objectives"}:
        raise ValueError(f"Unknown training operation: {operation}")
    config = adapter.config
    endpoint = adapter.endpoint_model
    source = adapter.source_model
    training = config["training"]
    consistency_config = config["loss"]["consistency"]
    time_config = config["time_sampling"]
    batch_size = image.shape[0]

    x0, source_stats = sample_prior(config, image, one_hot, source)
    zero = _zero(image)
    consistency_result = None
    u = None
    consistency_s = None
    consistency_t = None

    if operation == "stage1_objectives":
        diagonal_time = sample_stage1_times(
            batch_size, image.device,
            time_config["min_time"], time_config["max_time"],
        )
        diagonal_state = linear_path(x0, one_hot, diagonal_time)
        schedule_weight = 0.0
        effective_weight = 0.0
    else:
        consistency_s, u, consistency_t = sample_consistency_times(
            consistency_config["type"], batch_size, image.device,
            time_config["min_time"], time_config["max_time"],
            time_config["min_gap"],
        )
        consistency_state = linear_path(x0, one_hot, consistency_s)
        if operation == "joint_objectives":
            # Joint training intentionally samples an independent diagonal time.
            diagonal_time = sample_stage1_times(
                batch_size, image.device,
                time_config["min_time"], time_config["max_time"],
            )
            diagonal_state = linear_path(x0, one_hot, diagonal_time)
        else:
            # Preserve the original Stage 2 diagonal-at-s behavior.
            diagonal_time = consistency_s
            diagonal_state = consistency_state
        schedule_weight = losses.esd_schedule_weight(
            epoch_index, progress_in_epoch,
            consistency_config["start_epoch"],
            consistency_config["warmup_epochs"],
        )
        effective_weight = (
            consistency_config["weight"]
            * consistency_config["max_weight"]
            * schedule_weight
        )

    diagonal_logits = endpoint.forward_logits(
        diagonal_state, image, diagonal_time, diagonal_time
    )
    diagonal_loss = losses.diagonal_cross_entropy(
        diagonal_logits, target, training["label_smoothing"]
    ).float()

    if operation != "stage1_objectives":
        consistency_result = losses.compute_consistency_loss(
            consistency_config["type"],
            model=endpoint,
            x_s=consistency_state,
            image=image,
            s=consistency_s,
            u=u,
            t=consistency_t,
            precision=consistency_config["precision"],
            config=config,
        )
        consistency_loss = consistency_result.loss
    else:
        consistency_loss = zero

    total = (
        config["loss"]["primary"]["weight"] * diagonal_loss
        + effective_weight * consistency_loss
        + source_stats["weighted_var"]
        + source_stats["weighted_align"]
    ).float()
    stats = {
        "loss_total": total.detach(),
        "loss_diagonal": diagonal_loss.detach(),
        "loss_consistency": consistency_loss.detach(),
        "loss_source_var": source_stats["loss_source_var"].detach(),
        "loss_source_align": source_stats["loss_source_align"].detach(),
        "consistency_base_weight": total.new_tensor(
            consistency_config["weight"] if operation != "stage1_objectives" else 0.0
        ),
        "consistency_schedule_weight": total.new_tensor(schedule_weight),
        "consistency_effective_weight": total.new_tensor(effective_weight),
        # Legacy ESD log names are retained for existing dashboards.
        "esd_base_weight": total.new_tensor(
            consistency_config["weight"]
            if consistency_config["type"] == "esd" and operation != "stage1_objectives"
            else 0.0
        ),
        "esd_schedule_weight": total.new_tensor(
            schedule_weight if consistency_config["type"] == "esd" else 0.0
        ),
        "esd_effective_weight": total.new_tensor(
            effective_weight if consistency_config["type"] == "esd" else 0.0
        ),
        "diagonal_time_mean": diagonal_time.detach().float().mean(),
    }
    for key, value in source_stats.items():
        if key not in stats and torch.is_tensor(value) and value.numel() == 1:
            stats[key] = value.detach()
    if consistency_result is not None:
        stats.update(consistency_result.stats)
        stats["consistency_s_mean"] = consistency_s.detach().float().mean()
        stats["consistency_t_mean"] = consistency_t.detach().float().mean()
        if u is not None:
            stats["consistency_u_mean"] = u.detach().float().mean()
    return {
        "loss": total,
        "stats": stats,
        "operation": operation,
        "consistency_type": (
            consistency_config["type"] if operation != "stage1_objectives" else "none"
        ),
    }


class DDPCompatibleTrainingModel(nn.Module):
    """Composite endpoint/source adapter whose complete graph is one DDP forward."""

    _is_dfm_ddp_adapter = True

    def __init__(
        self,
        endpoint_model: nn.Module,
        source_model: nn.Module | None,
        config: dict,
    ) -> None:
        super().__init__()
        self.endpoint_model = endpoint_model
        self.source_model = source_model
        self.config = config

    def forward(self, *, operation: str, **kwargs):
        return compute_model_training_objectives(
            self, operation=operation, **kwargs
        )


def run_model_training_objectives(model: nn.Module, *, operation: str, **kwargs):
    """Never unwrap DDP here: the training graph must enter through DDP.forward."""
    return model(operation=operation, **kwargs)

