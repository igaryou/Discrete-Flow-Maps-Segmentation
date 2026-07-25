from __future__ import annotations

import torch


def _time_view(time: torch.Tensor, ndim: int = 4) -> torch.Tensor:
    if time.ndim != 1:
        raise ValueError(f"time must have shape [B], got {tuple(time.shape)}")
    return time.reshape(time.shape[0], *([1] * (ndim - 1)))


def linear_path(x0: torch.Tensor, x1: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    if x0.shape != x1.shape:
        raise ValueError("x0 and x1 must have the same shape")
    time_view = _time_view(time, x0.ndim).to(dtype=x0.dtype)
    return (1.0 - time_view) * x0 + time_view * x1


def flow_map(
    x_s: torch.Tensor,
    mean_denoiser: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    time_eps: float = 1.0e-5,
) -> torch.Tensor:
    """X_{s,t}=x_s+((t-s)/(1-s))(psi_{s,t}-x_s)."""
    if x_s.shape != mean_denoiser.shape:
        raise ValueError("x_s and mean_denoiser must have identical shapes")
    if s.shape != t.shape or s.ndim != 1:
        raise ValueError("s and t must both have shape [B]")
    denominator = (1.0 - s).clamp_min(time_eps)
    gamma = _time_view((t - s) / denominator, x_s.ndim).to(dtype=x_s.dtype)
    return x_s + gamma * (mean_denoiser - x_s)


def sample_stage1_times(
    batch_size: int,
    device: torch.device,
    min_time: float = 0.0,
    max_time: float = 1.0,
) -> torch.Tensor:
    return min_time + (max_time - min_time) * torch.rand(batch_size, device=device)


def sample_sorted_times(
    batch_size: int,
    device: torch.device,
    min_time: float = 0.0,
    max_time: float = 1.0,
    min_gap: float = 1.0e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample sorted uniform times and enforce a positive minimum gap safely."""
    times = min_time + (max_time - min_time) * torch.rand(batch_size, 2, device=device)
    times, _ = torch.sort(times, dim=1)
    s, t = times.unbind(dim=1)
    too_close = (t - s) < min_gap
    if too_close.any():
        # Preserve t when possible, otherwise move s left from max_time.
        proposed_t = s + min_gap
        t = torch.where(too_close, proposed_t.clamp_max(max_time), t)
        s = torch.where(too_close & (t - s < min_gap), (t - min_gap).clamp_min(min_time), s)
    if not torch.all(t - s >= min_gap * (1.0 - 1.0e-4)):
        raise RuntimeError("Failed to enforce time_sampling.min_gap")
    return s, t


def sample_prior(
    config: dict,
    image: torch.Tensor,
    target_one_hot: torch.Tensor | None,
    source_model,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sample x0 and return the CFM-compatible source losses/statistics."""
    batch, _, height, width = target_one_hot.shape if target_one_hot is not None else (
        image.shape[0], config["dataset"]["num_classes"], image.shape[2], image.shape[3]
    )
    classes = config["dataset"]["num_classes"]
    source = config["source"]
    dtype = image.dtype
    zero = image.sum() * 0.0
    if source["prior_type"] == "gaussian":
        x0 = torch.randn(
            batch, classes, height, width, device=image.device, dtype=dtype
        ) * source["prior_noise_std"]
        return x0, {
            "loss_source_var": zero, "loss_source_align": zero,
            "weighted_var": zero, "weighted_align": zero,
        }
    if source["prior_type"] == "dirichlet":
        concentration = torch.ones(classes, device=image.device, dtype=torch.float32)
        x0 = torch.distributions.Dirichlet(concentration).sample(
            (batch, height, width)
        ).permute(0, 3, 1, 2).to(dtype=dtype)
        return x0, {
            "loss_source_var": zero, "loss_source_align": zero,
            "weighted_var": zero, "weighted_align": zero,
        }
    if source_model is None:
        raise RuntimeError("source.prior_type=image_gaussian requires a source model")

    x0, mu, logvar = source_model(image)
    if x0.shape[-2:] != (height, width):
        x0 = torch.nn.functional.interpolate(
            x0, (height, width), mode="bilinear", align_corners=False
        )
        mu = torch.nn.functional.interpolate(
            mu, (height, width), mode="bilinear", align_corners=False
        )
        logvar = torch.nn.functional.interpolate(
            logvar, (height, width), mode="bilinear", align_corners=False
        )
    fixed_std = getattr(source_model, "fixed_std", None)
    loss_var = (
        zero if fixed_std is not None
        else 0.5 * torch.mean(torch.exp(logvar) - logvar - 1.0)
    )
    if source["use_loss_align"] and target_one_hot is not None:
        mu_norm = torch.nn.functional.normalize(mu, dim=1, eps=source["align_eps"])
        target_norm = torch.nn.functional.normalize(
            target_one_hot.to(mu), dim=1, eps=source["align_eps"]
        )
        loss_align = torch.nn.functional.mse_loss(mu_norm, target_norm)
    else:
        loss_align = zero
    return x0, {
        "loss_source_var": loss_var,
        "loss_source_align": loss_align,
        "weighted_var": source["var_weight"] * loss_var,
        "weighted_align": (
            source["align_weight"] * loss_align if source["use_loss_align"] else zero
        ),
        "source_mu_abs": mu.detach().abs().mean(),
        "source_logvar_mean": logvar.detach().mean(),
    }


def make_time_grid(num_steps: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if num_steps <= 0:
        raise ValueError("evaluation.num_steps must be positive")
    grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    return [(grid[index], grid[index + 1]) for index in range(num_steps)]

