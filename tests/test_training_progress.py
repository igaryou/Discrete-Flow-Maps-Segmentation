import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import trainer
from config import load_config
from trainer import (
    _DisplayRunningMeans,
    _create_epoch_progress,
    _epoch_total_iterations,
    _format_epoch_summary,
    _gpu_memory_gb,
    _progress_postfix,
)


CONFIG = Path(__file__).parents[1] / "configs" / "debug_diagonal_cityscapes.yaml"


class _FakeProgress:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.postfixes = []
        self.updates = 0
        self.closed = False

    def set_postfix(self, postfix, *, refresh):
        self.postfixes.append((postfix, refresh))

    def update(self, amount):
        self.updates += amount

    def close(self):
        self.closed = True


class _FakeTqdm:
    def __init__(self):
        self.calls = []
        self.writes = []

    def __call__(self, **kwargs):
        progress = _FakeProgress(**kwargs)
        self.calls.append(progress)
        return progress

    def write(self, value):
        self.writes.append(value)


def test_progress_is_created_only_for_rank0_and_uses_resume_epoch(monkeypatch):
    fake_tqdm = _FakeTqdm()
    monkeypatch.setattr(trainer, "tqdm", fake_tqdm)

    progress = _create_epoch_progress(
        epoch_index=7,
        total=200,
        is_main_process=True,
    )
    assert progress is fake_tqdm.calls[0]
    assert progress.kwargs == {
        "total": 200,
        "desc": "epoch 8",
        "dynamic_ncols": True,
        "leave": True,
        "unit": "batch",
        "mininterval": 0.5,
    }

    assert _create_epoch_progress(
        epoch_index=7,
        total=200,
        is_main_process=False,
    ) is None
    assert len(fake_tqdm.calls) == 1


def test_epoch_total_respects_limit_and_full_loader():
    assert _epoch_total_iterations(743, 200, 0) == 200
    assert _epoch_total_iterations(743, 200, 150) == 50
    assert _epoch_total_iterations(743, 200, 200) == 0
    assert _epoch_total_iterations(743, None, 150) == 743


def test_display_running_means_detach_scalars_and_ignore_nonfinite_values():
    meter = _DisplayRunningMeans()
    meter.update({
        "loss_total": torch.tensor(2.0, requires_grad=True),
        "loss_diagonal": 1.0,
        "loss_consistency": 4.0,
        "runtime_iteration_time": 0.5,
    })
    meter.update({
        "loss_total": torch.tensor(4.0, requires_grad=True),
        "loss_diagonal": 3.0,
        "loss_consistency": float("nan"),
        "runtime_iteration_time": 1.5,
    })

    assert meter.means() == {
        "loss_total": 3.0,
        "loss_diagonal": 2.0,
        "loss_consistency": 4.0,
        "runtime_iteration_time": 1.0,
    }
    assert all(isinstance(value, float) for value in meter.sums.values())


@pytest.mark.parametrize("consistency_type", ["ecld", "psd", "csd", "esd"])
def test_progress_postfix_uses_dynamic_consistency_name(consistency_type):
    postfix = _progress_postfix(
        {
            "loss_total": 6.82,
            "loss_diagonal": 2.31,
            "loss_consistency": 11.30,
        },
        consistency_type=consistency_type,
        lr=8.72e-5,
        sec_per_batch=0.78,
        memory_gb=21.4,
    )
    assert list(postfix) == [
        "loss",
        "diag",
        consistency_type,
        "lr",
        "sec/batch",
        "mem",
    ]
    assert postfix[consistency_type] == "11.3000"
    assert postfix["mem"] == "21.4G"


def test_cpu_memory_and_postfix_are_safe():
    assert _gpu_memory_gb(torch.device("cpu")) is None
    postfix = _progress_postfix(
        {
            "loss_total": 1.0,
            "loss_diagonal": 2.0,
            "loss_consistency": 3.0,
        },
        consistency_type="ecld",
        lr=1.0e-4,
        sec_per_batch=0.25,
        memory_gb=None,
    )
    assert "mem" not in postfix


def test_epoch_summary_uses_epoch_means_and_only_existing_optional_metrics():
    summary = _format_epoch_summary(
        epoch=3,
        reduced_epoch={
            "loss_total": torch.tensor(6.734921),
            "loss_diagonal": torch.tensor(2.281443),
            "loss_consistency": torch.tensor(11.151237),
            "consistency_effective_weight": torch.tensor(0.5),
            "loss_ecld_ec": torch.tensor(2.780100),
            "loss_source_align": torch.tensor(0.034221),
            "weighted_align": torch.tensor(0.005133),
            "lr": torch.tensor(9.761234e-5),
            "not_a_summary_metric": torch.tensor(123.0),
        },
        consistency_type="ecld",
        sec_per_batch=0.781234,
    )

    assert summary.startswith(
        "epoch:3 loss_avg:6.734921 loss_total:6.734921 "
        "loss_primary:2.281443 loss_consistency:11.151237 "
        "consistency_type:ecld consistency_weight:0.500000"
    )
    assert "loss_ecld_ec:2.780100" in summary
    assert "loss_source_align:0.034221" in summary
    assert "weighted_align:0.005133" in summary
    assert "lr:9.761234e-05 sec_per_batch:0.781234" in summary
    assert "loss_psd" not in summary
    assert "not_a_summary_metric" not in summary


def test_training_keeps_iteration_jsonl_wandb_and_file_log_with_microbatch_bar(
    tmp_path, monkeypatch, capsys
):
    output_dir = tmp_path / "run"
    config = load_config(
        CONFIG,
        [
            f"experiment.output_dir={output_dir}",
            "runtime.device=cpu",
            "training.epochs=1",
            "training.max_iterations=3",
            "training.grad_accum_steps=2",
            "training.log_interval=2",
            "training.validation_epochs=[]",
            "wandb.enabled=false",
        ],
    )
    batches = [
        (
            torch.full((2, 1), float(index)),
            torch.zeros(2, 1),
            torch.zeros(2, dtype=torch.long),
        )
        for index in range(1, 4)
    ]
    endpoint = nn.Linear(1, 1, bias=False)
    fake_tqdm = _FakeTqdm()

    class _FakeWandb:
        def __init__(self):
            self.logs = []
            self.finished = False

        def log(self, payload, *, step):
            self.logs.append((payload, step))

        def finish(self):
            self.finished = True

    fake_wandb = _FakeWandb()

    def fake_objectives(model, *, operation, **kwargs):
        del operation, kwargs
        parameter = next(model.parameters())
        loss = parameter.square().mean()
        zero = loss.detach() * 0.0
        return {
            "loss": loss,
            "stats": {
                "loss_total": loss.detach(),
                "loss_diagonal": loss.detach(),
                "loss_consistency": zero,
                "loss_source_var": zero,
                "loss_source_align": zero,
                "weighted_var": zero,
                "weighted_align": zero,
                "consistency_effective_weight": zero,
            },
            "consistency_type": "none",
        }

    monkeypatch.setattr(
        trainer,
        "_build_loaders",
        lambda config, context, local_batch_size: (batches, [], None),
    )
    monkeypatch.setattr(
        trainer,
        "build_models",
        lambda config, device: (endpoint.to(device), None),
    )
    monkeypatch.setattr(
        trainer,
        "initialize_or_resume",
        lambda *args, **kwargs: SimpleNamespace(
            start_epoch=0,
            global_step=0,
            best_miou=float("-inf"),
        ),
    )
    monkeypatch.setattr(
        trainer,
        "run_model_training_objectives",
        fake_objectives,
    )
    monkeypatch.setattr(
        trainer,
        "_save_training_checkpoint",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(trainer, "init_wandb", lambda config: fake_wandb)
    monkeypatch.setattr(trainer, "tqdm", fake_tqdm)

    trainer.run_training(config)

    assert len(fake_tqdm.calls) == 1
    progress = fake_tqdm.calls[0]
    assert progress.kwargs["total"] == 3
    assert progress.updates == 3
    assert progress.closed
    assert len(progress.postfixes) == 3
    assert all(refresh is False for _, refresh in progress.postfixes)
    assert "none" in progress.postfixes[-1][0]

    records = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert [record["scope"] for record in records] == [
        "iteration",
        "iteration",
        "epoch",
        "runtime",
    ]
    assert [record["iteration"] for record in records[:2]] == [1, 2]

    assert len(fake_wandb.logs) == 4
    assert [step for _, step in fake_wandb.logs[:2]] == [0, 1]
    assert fake_wandb.finished

    console = capsys.readouterr().err
    assert "epoch=1 iter=" not in console
    file_log = (output_dir / "train_log.txt").read_text()
    assert "epoch=1 iter=1 step=0" in file_log
    assert "epoch=1 iter=2 step=1" in file_log
