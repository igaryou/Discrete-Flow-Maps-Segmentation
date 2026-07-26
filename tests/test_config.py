from pathlib import Path

import pytest
import yaml

from config import load_config, save_resolved_config
from trainer import log_esd_experiment_metadata


CONFIG = Path(__file__).parents[1] / "configs" / "debug_diagonal_cityscapes.yaml"
ESD_CONFIG = Path(__file__).parents[1] / "configs" / "debug_esd_cityscapes.yaml"
EXPECTED_ESD_METADATA = {
    "formulation": "stabilized_logit_space",
    "source": "discrete_flow_maps",
    "additional_numerical_safeguards": True,
}


def test_yaml_load_and_cli_override():
    config = load_config(CONFIG, ["training.batch_size=2", "runtime.device=cpu"])
    assert config["training"]["batch_size"] == 2
    assert config["runtime"]["device"] == "cpu"
    assert config["flow"]["time_eps"] == 1.0e-5
    assert config["training"]["scheduler"]["step_unit"] == "epoch"
    assert config["training"]["scheduler"]["warmup_start_factor"] == 0.1
    assert config["training"]["max_batches_per_epoch"] is None


def test_scheduler_unit_and_debug_epoch_limit_are_validated():
    with pytest.raises(ValueError, match="step_unit must be epoch"):
        load_config(CONFIG, ["training.scheduler.step_unit=iteration"])
    with pytest.raises(ValueError, match="max_batches_per_epoch"):
        load_config(CONFIG, ["training.max_batches_per_epoch=0"])


def test_missing_required_section_has_clear_error(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    del raw["loss"]
    path = tmp_path / "missing.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Missing required config section: loss"):
        load_config(path)


def test_unknown_key_is_rejected(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    raw["training"]["mystery_option"] = 123
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Unknown config key: training.mystery_option"):
        load_config(path)


def test_unknown_override_is_rejected():
    with pytest.raises(ValueError, match="Unknown override key"):
        load_config(CONFIG, ["training.not_real=1"])


def test_resolved_config_is_saved(tmp_path):
    config = load_config(CONFIG, ["training.batch_size=3"])
    destination = tmp_path / "config_resolved.yaml"
    save_resolved_config(config, destination)
    loaded = yaml.safe_load(destination.read_text())
    assert loaded["training"]["batch_size"] == 3
    assert loaded["runtime"]["config_path"] == str(CONFIG.resolve())


def test_init_from_and_resume_are_mutually_exclusive(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    raw["checkpoint"]["init_from"] = "a.pt"
    raw["checkpoint"]["resume"] = "b.pt"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_config(path)


def test_precision_validation_rejects_fake_psd_jvp_and_bf16_numerics():
    with pytest.raises(ValueError, match="PSD does not use JVP"):
        load_config(
            Path(__file__).parents[1] / "configs" / "debug_ddp_stage2_psd.yaml",
            ["loss.consistency.precision.jvp_dtype=bf16"],
        )
    with pytest.raises(ValueError, match="numerical_dtype must be fp32"):
        load_config(
            Path(__file__).parents[1] / "configs" / "debug_ddp_stage2_ecld.yaml",
            ["loss.consistency.precision.numerical_dtype=bf16"],
        )


def test_bf16_jvp_requires_runtime_bf16_amp():
    with pytest.raises(ValueError, match="bf16 JVP requires"):
        load_config(
            Path(__file__).parents[1] / "configs" / "debug_ddp_stage2_esd.yaml",
            ["runtime.amp=false"],
        )


def test_legacy_single_gpu_yaml_can_omit_distributed(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    del raw["distributed"]
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    assert config["distributed"]["enabled"] == "auto"
    assert config["distributed"]["backend"] == "nccl"


def test_legacy_esd_yaml_without_metadata_uses_defaults_and_resolves(tmp_path):
    raw = yaml.safe_load(ESD_CONFIG.read_text())
    raw["loss"]["consistency"].pop("esd")
    legacy_path = tmp_path / "legacy_esd.yaml"
    legacy_path.write_text(yaml.safe_dump(raw))
    config = load_config(legacy_path)
    assert config["loss"]["consistency"]["esd"] == EXPECTED_ESD_METADATA

    resolved_path = tmp_path / "resolved.yaml"
    save_resolved_config(config, resolved_path)
    resolved = yaml.safe_load(resolved_path.read_text())
    assert resolved["loss"]["consistency"]["esd"] == EXPECTED_ESD_METADATA


def test_esd_metadata_rejects_unimplemented_formulation_and_wrong_source():
    with pytest.raises(ValueError, match="formulation must be"):
        load_config(
            ESD_CONFIG,
            ["loss.consistency.esd.formulation=raw"],
        )
    with pytest.raises(ValueError, match="source must be"):
        load_config(
            ESD_CONFIG,
            ["loss.consistency.esd.source=other"],
        )


def test_esd_startup_log_uses_resolved_metadata_and_safeguard_settings():
    config = load_config(ESD_CONFIG)

    class CapturingLogger:
        def __init__(self):
            self.messages = []

        def info(self, message, *args):
            self.messages.append(message % args if args else message)

    logger = CapturingLogger()
    log_esd_experiment_metadata(config, logger)
    assert logger.messages == [
        "ESD formulation: stabilized_logit_space",
        "ESD source: discrete_flow_maps",
        "ESD additional numerical safeguards: true",
        "ESD invalid teacher strategy: mask_pixel",
        "ESD JVP dtype: bf16",
        "ESD numerical dtype: fp32",
    ]
