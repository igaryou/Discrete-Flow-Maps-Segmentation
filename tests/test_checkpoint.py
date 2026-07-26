from pathlib import Path

import pytest
import torch
import yaml

from checkpoint import (
    SCHEDULER_STEP_UNIT,
    SCHEDULER_VERSION,
    checkpoint_payload,
    initialize_or_resume,
    save_checkpoint,
)
from config import load_config


ROOT = Path(__file__).parents[1]


def configs(tmp_path):
    stage1 = load_config(ROOT / "configs" / "debug_diagonal_cityscapes.yaml")
    raw = yaml.safe_load((ROOT / "configs" / "debug_esd_cityscapes.yaml").read_text())
    raw["checkpoint"]["init_from"] = str(tmp_path / "stage1.pt")
    stage2_path = tmp_path / "stage2.yaml"
    stage2_path.write_text(yaml.safe_dump(raw))
    stage2 = load_config(stage2_path)
    return stage1, stage2


def objects():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, optimizer, scheduler, scaler


def test_stage1_payload_contains_required_stage_information(tmp_path):
    stage1, _ = configs(tmp_path)
    model, optimizer, scheduler, scaler = objects()
    payload = checkpoint_payload(
        config=stage1, epoch=2, global_step=7, model=model, source_model=None,
        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        metrics={"mIoU": 0.25, "best_mIoU": 0.25},
    )
    assert payload["stage"] == "diagonal_pretrain"
    assert payload["source_model"] is None
    assert payload["epoch"] == 2 and payload["global_step"] == 7
    assert payload["scheduler_step_unit"] == SCHEDULER_STEP_UNIT
    assert payload["scheduler_version"] == SCHEDULER_VERSION


def test_init_from_loads_weights_but_not_optimizer_or_epoch(tmp_path):
    stage1, stage2 = configs(tmp_path)
    source_model, source_optimizer, source_scheduler, source_scaler = objects()
    with torch.no_grad():
        source_model.weight.fill_(1.25)
    payload = checkpoint_payload(
        config=stage1, epoch=4, global_step=19, model=source_model, source_model=None,
        optimizer=source_optimizer, scheduler=source_scheduler, scaler=source_scaler,
        metrics={"best_mIoU": 0.4},
    )
    torch.save(payload, stage2["checkpoint"]["init_from"])

    model, optimizer, scheduler, scaler = objects()
    state = initialize_or_resume(
        stage2, model, None, optimizer, scheduler, scaler
    )
    torch.testing.assert_close(model.weight, source_model.weight)
    assert optimizer.state == {}
    assert state.start_epoch == 0 and state.global_step == 0


def test_resume_restores_optimizer_scheduler_epoch_and_step(tmp_path):
    _, stage2 = configs(tmp_path)
    model, optimizer, scheduler, scaler = objects()
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    payload = checkpoint_payload(
        config=stage2, epoch=3, global_step=11, model=model, source_model=None,
        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        metrics={"best_mIoU": 0.5},
    )
    assert payload["config"]["loss"]["consistency"]["esd"] == {
        "formulation": "stabilized_logit_space",
        "source": "discrete_flow_maps",
        "additional_numerical_safeguards": True,
    }
    resume_path = save_checkpoint(payload, tmp_path, "resume.pt")
    stage2["checkpoint"]["init_from"] = None
    stage2["checkpoint"]["resume"] = str(resume_path)
    restored_model, restored_optimizer, restored_scheduler, restored_scaler = objects()
    state = initialize_or_resume(
        stage2, restored_model, None, restored_optimizer, restored_scheduler,
        restored_scaler,
    )
    assert state.start_epoch == 3
    assert state.global_step == 11
    assert restored_optimizer.state
    assert restored_scheduler.last_epoch == scheduler.last_epoch
    torch.testing.assert_close(restored_model.weight, model.weight)


def test_wrong_init_stage_is_rejected(tmp_path):
    stage1, stage2 = configs(tmp_path)
    model, optimizer, scheduler, scaler = objects()
    payload = checkpoint_payload(
        config=stage1, epoch=1, global_step=1, model=model, source_model=None,
        optimizer=optimizer, scheduler=scheduler, scaler=scaler, metrics={},
    )
    payload["stage"] = "esd_distillation"
    torch.save(payload, stage2["checkpoint"]["init_from"])
    with pytest.raises(RuntimeError, match="diagonal_pretrain"):
        initialize_or_resume(stage2, *objects()[:1], None, *objects()[1:])


def test_joint_resume_is_complete_and_rejects_other_stages(tmp_path):
    joint = load_config(ROOT / "configs" / "joint_ecld_cityscapes.yaml")
    joint["training"]["batch_size"] = 2
    model, optimizer, scheduler, scaler = objects()
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    payload = checkpoint_payload(
        config=joint, epoch=4, global_step=13, model=model, source_model=None,
        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        metrics={"best_mIoU": 0.6},
    )
    resume_path = save_checkpoint(payload, tmp_path, "joint.pt")
    joint["checkpoint"]["resume"] = str(resume_path)

    restored_model, restored_optimizer, restored_scheduler, restored_scaler = objects()
    state = initialize_or_resume(
        joint, restored_model, None, restored_optimizer, restored_scheduler,
        restored_scaler,
    )
    torch.testing.assert_close(restored_model.weight, model.weight)
    assert state.start_epoch == 4
    assert state.global_step == 13
    assert state.best_miou == 0.6
    assert restored_optimizer.state

    incompatible = dict(payload)
    incompatible["stage"] = "diagonal_pretrain"
    incompatible_path = tmp_path / "stage1-as-joint.pt"
    torch.save(incompatible, incompatible_path)
    joint["checkpoint"]["resume"] = str(incompatible_path)
    with pytest.raises(RuntimeError, match="Resume stage mismatch"):
        initialize_or_resume(
            joint, restored_model, None, restored_optimizer, restored_scheduler,
            restored_scaler,
        )


def test_legacy_step_scheduler_checkpoint_is_rejected_on_resume(tmp_path):
    joint = load_config(ROOT / "configs" / "joint_ecld_cityscapes.yaml")
    model, optimizer, scheduler, scaler = objects()
    payload = checkpoint_payload(
        config=joint,
        epoch=2,
        global_step=100,
        model=model,
        source_model=None,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        metrics={},
    )
    payload.pop("scheduler_step_unit")
    payload.pop("scheduler_version")
    path = save_checkpoint(payload, tmp_path, "legacy.pt")
    joint["checkpoint"]["resume"] = str(path)

    with pytest.raises(
        RuntimeError,
        match="Legacy optimizer-step scheduler checkpoints cannot be resumed",
    ):
        initialize_or_resume(
            joint,
            model,
            None,
            optimizer,
            scheduler,
            scaler,
        )
