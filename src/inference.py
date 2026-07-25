from __future__ import annotations

import torch

from discrete_flow_maps import flow_map, make_time_grid, sample_prior


@torch.no_grad()
def sample_segmentation(
    model,
    source_model,
    image: torch.Tensor,
    config: dict,
    num_steps: int | None = None,
    return_trajectory: bool = False,
):
    model.eval()
    if source_model is not None:
        source_model.eval()
    steps = num_steps or config["evaluation"]["num_steps"]
    x, _ = sample_prior(config, image, None, source_model)
    trajectory = [x.argmax(dim=1)]
    for scalar_s, scalar_t in make_time_grid(steps, image.device):
        batch = image.shape[0]
        s = scalar_s.expand(batch)
        t = scalar_t.expand(batch)
        logits = model.forward_logits(x, image, s, t)
        probability = torch.softmax(logits.float(), dim=1).to(x.dtype)
        x = flow_map(x, probability, s, t, config["flow"]["time_eps"])
        trajectory.append(x.argmax(dim=1))
    prediction = x.argmax(dim=1)
    if return_trajectory:
        return prediction, torch.stack(trajectory, dim=1)
    return prediction

