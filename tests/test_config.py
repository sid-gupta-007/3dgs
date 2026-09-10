"""Tests for configuration system."""

from pathlib import Path
import pytest
from panogs.core.config import PanoGSConfig, load_config


def test_default_config():
    config = PanoGSConfig()
    assert config.hardware.device == "cpu"
    assert config.hardware.num_threads > 0
    assert config.image.normalize_to_float is True
    assert ".jpg" in config.image.supported_formats
    assert config.reconstruction.max_gaussians == 50000


def test_save_and_load_config(tmp_path: Path):
    cfg_file = tmp_path / "custom.yaml"
    cfg = PanoGSConfig()
    cfg.hardware.device = "cpu"
    cfg.reconstruction.resolution = 256
    cfg.save_yaml(cfg_file)

    loaded = load_config(cfg_file)
    assert loaded.hardware.device == "cpu"
    assert loaded.reconstruction.resolution == 256


def test_load_nonexistent_config():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config_path_12345.yaml")
