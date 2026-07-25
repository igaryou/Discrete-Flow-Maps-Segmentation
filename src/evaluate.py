from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import load_config
from dataset import build_dataset
from inference import sample_segmentation
from metrics import SegmentationMetrics
from model_factory import build_models
from utils import autocast_context, resolve_device, seed_everything
from visualization import save_prediction


@torch.no_grad()
def evaluate(config: dict, checkpoint_path: str | Path) -> dict:
    device = resolve_device(config["runtime"]["device"])
    seed_everything(config["experiment"]["seed"], config["runtime"]["deterministic"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, source_model = build_models(config, device)
    model.load_state_dict(checkpoint["model"], strict=config["checkpoint"]["strict_model"])
    if source_model is not None:
        if checkpoint.get("source_model") is None:
            raise RuntimeError("Checkpoint has no source_model state")
        source_model.load_state_dict(
            checkpoint["source_model"], strict=config["checkpoint"]["strict_model"]
        )
        source_model.eval()
    model.eval()

    dataset = build_dataset(config, config["evaluation"]["split"], augment=False)
    loader = DataLoader(
        dataset, batch_size=config["evaluation"]["batch_size"], shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=config["dataset"]["pin_memory"],
    )
    metrics = SegmentationMetrics(
        config["dataset"]["num_classes"], config["dataset"]["void_class_index"]
    )
    checkpoint_stem = Path(checkpoint_path).stem
    output_dir = (
        Path(config["experiment"]["output_dir"])
        / f"evaluation_{checkpoint_stem}_{config['evaluation']['num_steps']}steps"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    visualized = 0
    for batch_index, (image, _, target) in enumerate(loader):
        maximum_batches = config["evaluation"]["max_batches"]
        if maximum_batches is not None and batch_index >= maximum_batches:
            break
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast_context(config, device):
            prediction = sample_segmentation(model, source_model, image, config)
        metrics.update(prediction, target)
        if config["evaluation"]["save_predictions"]:
            remaining = config["evaluation"]["max_visualizations"] - visualized
            for index in range(min(image.shape[0], max(remaining, 0))):
                save_prediction(
                    image[index], target[index], prediction[index],
                    output_dir / "predictions" / f"{batch_index:04d}_{index:02d}.png",
                    config["augmentation"]["imagenet_normalize"],
                )
                visualized += 1
    result = metrics.compute()
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a DFM checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    arguments = parser.parse_args()
    config = load_config(arguments.config, arguments.set)
    checkpoint = arguments.checkpoint or config["evaluation"]["checkpoint"]
    if not checkpoint:
        raise ValueError("--checkpoint or evaluation.checkpoint must be specified")
    evaluate(config, checkpoint)


if __name__ == "__main__":
    main()

