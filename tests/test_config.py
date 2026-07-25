from pathlib import Path

import pytest
import yaml

from config import load_config, save_resolved_config


CONFIG = Path(__file__).parents[1] / "configs" / "debug_diagonal_cityscapes.yaml"


def test_yaml_load_and_cli_override():
    config = load_config(CONFIG, ["training.batch_size=2", "runtime.device=cpu"])
    assert config["training"]["batch_size"] == 2
    assert config["runtime"]["device"] == "cpu"
    assert config["flow"]["time_eps"] == 1.0e-5


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

