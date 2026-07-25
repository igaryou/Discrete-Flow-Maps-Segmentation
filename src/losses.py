from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.func import jvp


@dataclass
class ESDResult:
    loss: torch.Tensor
    stats: dict[str, torch.Tensor]
    teacher_prob: torch.Tensor
    student_prob: torch.Tensor
    valid_pixel: torch.Tensor
    adaptive_weight: torch.Tensor
    directional_logits: torch.Tensor


def diagonal_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, label_smoothing: float = 0.0
) -> torch.Tensor:
    return F.cross_entropy(logits, target, label_smoothing=label_smoothing)


def esd_schedule_weight(
    epoch: int, progress_in_epoch: float, start_epoch: int, warmup_epochs: int
) -> float:
    if epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, (epoch - start_epoch + progress_in_epoch) / warmup_epochs)


def _bucket_ratios(
    invalid_class: torch.Tensor, t: torch.Tensor
) -> dict[str, torch.Tensor]:
    buckets = (
        ("esd_clamp_ratio_t_0_0_5", t < 0.5),
        ("esd_clamp_ratio_t_0_5_0_9", (t >= 0.5) & (t < 0.9)),
        ("esd_clamp_ratio_t_0_9_0_99", (t >= 0.9) & (t < 0.99)),
        ("esd_clamp_ratio_t_0_99_1_0", t >= 0.99),
    )
    zero = invalid_class.float().sum() * 0.0
    return {
        name: invalid_class[mask].float().mean() if mask.any() else zero
        for name, mask in buckets
    }


def esd_loss(
    model,
    x_s: torch.Tensor,
    image: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    *,
    time_eps: float = 1.0e-5,
    log_eps: float = 1.0e-6,
    invalid_strategy: str = "mask_pixel",
    skip_batch_threshold: float | None = None,
    adaptive_enabled: bool = False,
    adaptive_c: float = 1.0e-6,
    adaptive_r: float = 0.5,
    adaptive_normalize_mean: bool = True,
    adaptive_max_weight: float | None = 100.0,
) -> ESDResult:
    """Logit-space ESD using a joint JVP over (x_s, s), entirely in float32."""
    if invalid_strategy not in {"clamp", "mask_pixel", "skip_batch"}:
        raise ValueError(f"Unknown invalid teacher strategy: {invalid_strategy}")
    device_type = x_s.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        x32 = x_s.float()
        image32 = image.float()
        s32 = s.float()
        t32 = t.float()

        # The diagonal drift is a teacher-side tangent. It intentionally carries no
        # reverse-mode gradient, while the JVP primal logits remain trainable.
        with torch.no_grad():
            logits_ss_teacher = model.forward_logits(x32, image32, s32, s32).float()
            prob_ss = torch.softmax(logits_ss_teacher, dim=1)
            denominator_s = (1.0 - s32).clamp_min(time_eps)
            b_s = (prob_ss - x32) / denominator_s[:, None, None, None]

        def logits_along_flow(x_in: torch.Tensor, s_in: torch.Tensor) -> torch.Tensor:
            return model.forward_logits(x_in, image32, s_in, t32).float()

        logits_st, directional_logits = jvp(
            logits_along_flow,
            (x32, s32),
            (b_s, torch.ones_like(s32)),
        )
        student_log_prob = F.log_softmax(logits_st.float(), dim=1)
        student_prob = student_log_prob.exp()
        directional_logits = directional_logits.float()

        center = (student_prob * directional_logits).sum(dim=1, keepdim=True)
        delta = directional_logits - center
        one_minus_s = 1.0 - s32
        one_minus_t = 1.0 - t32
        delta_time = t32 - s32
        log_arg_raw = (
            one_minus_t[:, None, None, None]
            - (one_minus_s * delta_time)[:, None, None, None] * delta
        )

        invalid_class = ~torch.isfinite(log_arg_raw) | (log_arg_raw <= log_eps)
        valid_pixel = torch.isfinite(log_arg_raw).all(dim=1)
        valid_pixel &= (log_arg_raw > log_eps).all(dim=1)
        invalid_pixel = invalid_class.any(dim=1)
        invalid_sample = invalid_pixel.flatten(1).any(dim=1)
        per_sample_ratio = invalid_class.float().flatten(1).mean(dim=1)

        finite_safe = torch.nan_to_num(
            log_arg_raw,
            nan=log_eps,
            neginf=log_eps,
            posinf=torch.finfo(torch.float32).max,
        )
        log_arg_safe = finite_safe.clamp_min(log_eps)
        teacher_logits = logits_ss_teacher.detach() - torch.log(log_arg_safe)
        teacher_prob = torch.softmax(teacher_logits, dim=1).detach()
        kl_per_pixel = F.kl_div(
            student_log_prob, teacher_prob, reduction="none"
        ).sum(dim=1)

        mismatch_sq = (teacher_prob - student_prob).square().sum(dim=1)
        if adaptive_enabled:
            adaptive_weight = (mismatch_sq.detach() + adaptive_c).pow(-adaptive_r)
            if adaptive_max_weight is not None:
                adaptive_weight = adaptive_weight.clamp_max(adaptive_max_weight)
            normalization_mask = valid_pixel if invalid_strategy == "mask_pixel" else torch.ones_like(valid_pixel)
            if adaptive_normalize_mean and normalization_mask.any():
                adaptive_weight = adaptive_weight / adaptive_weight[
                    normalization_mask
                ].mean().clamp_min(1.0e-8)
        else:
            adaptive_weight = torch.ones_like(kl_per_pixel)
        adaptive_weight = adaptive_weight.detach()
        loss_map = adaptive_weight * kl_per_pixel
        zero_with_graph = student_log_prob.sum() * 0.0

        invalid_ratio = invalid_class.float().mean()
        skip_for_threshold = (
            skip_batch_threshold is not None
            and float(invalid_ratio.detach()) > skip_batch_threshold
        )
        skip_for_strategy = (
            invalid_strategy == "skip_batch"
            and float(invalid_ratio.detach()) > (
                skip_batch_threshold if skip_batch_threshold is not None else 0.0
            )
        )
        if skip_for_threshold or skip_for_strategy:
            loss = zero_with_graph
        elif invalid_strategy == "mask_pixel":
            loss = loss_map[valid_pixel].mean() if valid_pixel.any() else zero_with_graph
        else:
            loss = loss_map.mean()

        entropy = -(teacher_prob * teacher_prob.clamp_min(1.0e-12).log()).sum(dim=1)
        stats = {
            "esd_log_arg_min": torch.nan_to_num(
                log_arg_raw.detach(), nan=0.0,
                posinf=torch.finfo(torch.float32).max,
                neginf=torch.finfo(torch.float32).min,
            ).min(),
            "esd_log_arg_mean": torch.nan_to_num(
                log_arg_raw.detach(), nan=0.0, posinf=0.0, neginf=0.0
            ).mean(),
            "esd_clamp_ratio": invalid_ratio.detach(),
            "esd_pixel_invalid_ratio": invalid_pixel.float().mean(),
            "esd_sample_invalid_ratio": invalid_sample.float().mean(),
            "esd_max_sample_invalid_ratio": per_sample_ratio.max(),
            "esd_nonfinite_ratio": (~torch.isfinite(log_arg_raw)).float().mean(),
            "esd_teacher_entropy": entropy.mean(),
            "esd_teacher_min": teacher_prob.min(),
            "esd_teacher_max": teacher_prob.max(),
            "esd_delta_abs_mean": delta.detach().abs().mean(),
            "esd_delta_abs_max": delta.detach().abs().max(),
            "esd_valid_pixel_ratio": valid_pixel.float().mean(),
            "esd_adaptive_weight_mean": adaptive_weight.mean(),
            "esd_adaptive_weight_max": adaptive_weight.max(),
            "esd_skipped_batch": torch.as_tensor(
                float(skip_for_threshold or skip_for_strategy), device=x_s.device
            ),
            "s_mean": s32.mean(),
            "s_min": s32.min(),
            "s_max": s32.max(),
            "t_mean": t32.mean(),
            "t_min": t32.min(),
            "t_max": t32.max(),
        }
        stats.update(_bucket_ratios(invalid_class.detach(), t32.detach()))

    return ESDResult(
        loss=loss,
        stats={key: value.detach() for key, value in stats.items()},
        teacher_prob=teacher_prob,
        student_prob=student_prob,
        valid_pixel=valid_pixel.detach(),
        adaptive_weight=adaptive_weight,
        directional_logits=directional_logits,
    )


def compute_training_losses(
    *,
    stage: str,
    model,
    x_path: torch.Tensor,
    image: torch.Tensor,
    target: torch.Tensor,
    diagonal_time: torch.Tensor,
    label_smoothing: float,
    primary_weight: float,
    source_stats: dict[str, torch.Tensor],
    esd_kwargs: dict[str, Any] | None = None,
    esd_effective_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], ESDResult | None]:
    """Shared stage switch; Stage 1 cannot call ESD or JVP."""
    diagonal_logits = model.forward_logits(
        x_path, image, diagonal_time, diagonal_time
    )
    diagonal = diagonal_cross_entropy(diagonal_logits, target, label_smoothing)
    total = (
        primary_weight * diagonal
        + source_stats["weighted_var"]
        + source_stats["weighted_align"]
    )
    stats = {
        "loss_diagonal": diagonal.detach(),
        "loss_source_var": source_stats["loss_source_var"].detach(),
        "loss_source_align": source_stats["loss_source_align"].detach(),
    }
    result = None
    if stage == "esd_distillation":
        if esd_kwargs is None:
            raise ValueError("Stage 2 requires esd_kwargs")
        result = esd_loss(model, x_path, image, **esd_kwargs)
        total = total + esd_effective_weight * result.loss
        stats["loss_esd"] = result.loss.detach()
        stats.update(result.stats)
    elif stage != "diagonal_pretrain":
        raise ValueError(f"Unknown training stage: {stage}")
    stats["loss_total"] = total.detach()
    return total, stats, result
