from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Iterator, Mapping, Sized

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    is_main_process: bool
    backend: str | None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def setup_distributed(
    config: dict | None = None,
    *,
    allow_cpu_distributed: bool = False,
    backend: str | None = None,
) -> DistributedContext:
    """Initialize env:// DDP when requested and WORLD_SIZE is greater than one."""
    distributed_config = (config or {}).get("distributed", {})
    enabled = distributed_config.get("enabled", "auto")
    init_method = distributed_config.get("init_method", "env://")
    requested_backend = backend or distributed_config.get("backend")
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)

    if enabled not in {"auto", True, False}:
        raise ValueError("distributed.enabled must be auto, true, or false")
    if world_size < 1:
        raise RuntimeError(f"WORLD_SIZE must be >= 1, got {world_size}")
    if not (0 <= rank < world_size):
        raise RuntimeError(f"RANK={rank} is invalid for WORLD_SIZE={world_size}")
    if local_rank < 0:
        raise RuntimeError(f"LOCAL_RANK must be >= 0, got {local_rank}")
    if enabled is False and world_size > 1:
        raise RuntimeError("WORLD_SIZE > 1 conflicts with distributed.enabled=false")
    use_distributed = world_size > 1 and enabled in {"auto", True}

    runtime_device = (config or {}).get("runtime", {}).get("device", "auto")
    if not use_distributed:
        if runtime_device == "auto":
            device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        else:
            device = torch.device(runtime_device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            torch.cuda.set_device(device)
        return DistributedContext(False, 0, 0, 1, device, True, None)

    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    selected_backend = requested_backend or (
        "nccl" if torch.cuda.is_available() else "gloo"
    )
    use_cpu_gloo = allow_cpu_distributed and selected_backend == "gloo"
    if torch.cuda.is_available() and not use_cpu_gloo:
        if selected_backend == "nccl" and not dist.is_nccl_available():
            raise RuntimeError("NCCL is unavailable")
        device_count = torch.cuda.device_count()
        if world_size > device_count:
            raise RuntimeError(
                f"WORLD_SIZE={world_size} exceeds visible CUDA devices={device_count}"
            )
        if local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is invalid for {device_count} CUDA devices"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if not allow_cpu_distributed:
            raise RuntimeError("Distributed training requires CUDA/NCCL")
        if selected_backend != "gloo":
            raise RuntimeError("CPU distributed tests require backend=gloo")
        device = torch.device("cpu")

    init_kwargs = {}
    if device.type == "cuda" and selected_backend == "nccl":
        init_kwargs["device_id"] = device
    try:
        dist.init_process_group(
            backend=selected_backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            **init_kwargs,
        )
    except Exception:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    return DistributedContext(
        True, rank, local_rank, world_size, device, rank == 0, selected_backend
    )


def cleanup_distributed(context: DistributedContext) -> None:
    if context.distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    if not context.distributed:
        return
    if not dist.is_initialized():
        raise RuntimeError("barrier requested without an initialized process group")
    if context.backend == "nccl":
        dist.barrier(device_ids=[context.local_rank])
    else:
        dist.barrier()


def validate_global_batch_size(global_batch_size: int, world_size: int) -> int:
    if global_batch_size <= 0:
        raise ValueError("training.batch_size must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if global_batch_size % world_size:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"world size {world_size}"
        )
    return global_batch_size // world_size


def wrap_ddp(module: torch.nn.Module, context: DistributedContext, config: dict):
    if not context.distributed:
        return module
    options = config["distributed"]
    cuda_kwargs = {}
    if context.device.type == "cuda":
        cuda_kwargs = {
            "device_ids": [context.local_rank],
            "output_device": context.local_rank,
        }
    return DistributedDataParallel(
        module,
        broadcast_buffers=options["broadcast_buffers"],
        find_unused_parameters=options["find_unused_parameters"],
        gradient_as_bucket_view=options["gradient_as_bucket_view"],
        **cuda_kwargs,
    )


def unwrap_model(module):
    current = module
    while isinstance(current, DistributedDataParallel):
        current = current.module
    return current


def reduce_scalar(
    value: torch.Tensor | float,
    context: DistributedContext,
    reduction: str = "mean",
) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.detach().clone()
        if tensor.dtype != torch.float64:
            tensor = tensor.float()
    else:
        tensor = torch.tensor(float(value), device=context.device)
    if not context.distributed:
        return tensor
    if reduction in {"sum", "mean"}:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if reduction == "mean":
            tensor /= context.world_size
    elif reduction == "max":
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    elif reduction == "min":
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    else:
        raise ValueError(f"Unknown distributed reduction: {reduction}")
    return tensor


def reduce_metric_dict(
    metrics: Mapping[str, torch.Tensor | float],
    context: DistributedContext,
    *,
    max_keys: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    max_keys = max_keys or set()
    return {
        key: reduce_scalar(value, context, "max" if key in max_keys else "mean")
        for key, value in metrics.items()
    }


def all_reduce_confusion_matrix(
    confusion_matrix: torch.Tensor, context: DistributedContext
) -> torch.Tensor:
    result = confusion_matrix.clone()
    if context.distributed:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def assert_config_equal_across_ranks(config: dict, context: DistributedContext) -> None:
    if not context.distributed:
        return
    sanitized = json.loads(json.dumps(config, sort_keys=True, default=str))
    sanitized.get("runtime", {}).pop("config_path", None)
    digest = hashlib.sha256(
        json.dumps(sanitized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    gathered: list[str | None] = [None] * context.world_size
    dist.all_gather_object(gathered, digest)
    if len(set(gathered)) != 1:
        raise RuntimeError(f"Resolved config differs across ranks: {gathered}")


def parameter_checksum(module: torch.nn.Module, context: DistributedContext) -> dict[str, float]:
    checksum = torch.zeros((), device=context.device, dtype=torch.float64)
    for parameter in module.parameters():
        checksum += parameter.detach().double().sum()
    minimum = reduce_scalar(checksum, context, "min")
    maximum = reduce_scalar(checksum, context, "max")
    difference = float((maximum - minimum).abs().cpu())
    if difference > 1.0e-7:
        raise RuntimeError(f"DDP parameter checksum mismatch: max-min={difference}")
    return {"parameter_checksum": float(checksum.cpu()), "checksum_max_diff": difference}


def seed_data_loader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class DistributedEvalSampler(Sampler[int]):
    """Non-padding sampler: each evaluation index occurs on exactly one rank."""

    def __init__(self, dataset: Sized, *, rank: int, world_size: int) -> None:
        if not (0 <= rank < world_size):
            raise ValueError(f"Invalid rank={rank} for world_size={world_size}")
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        return math.ceil(max(len(self.dataset) - self.rank, 0) / self.world_size)
