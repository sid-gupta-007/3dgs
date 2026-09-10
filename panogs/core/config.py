"""
Configuration models and utilities for PanoGS.
Ensures CPU-first, memory-conscious defaults matching hardware constraints.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


@dataclass
class HardwareConfig:
    """Hardware and compute backend configuration."""
    device: str = "cpu"
    num_threads: int = 4
    max_memory_gb: float = 12.0


@dataclass
class ImageConfig:
    """Image loading and preprocessing configuration."""
    max_resolution: int = 1024
    normalize_to_float: bool = True
    supported_formats: list[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png"]
    )


@dataclass
class ReconstructionConfig:
    """Reconstruction and 3DGS defaults (for development phases)."""
    max_gaussians: int = 50000
    iterations: int = 1000
    resolution: int = 512


@dataclass
class PanoGSConfig:
    """Root configuration object for PanoGS."""
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save_yaml(self, path: Path | str) -> None:
        """Save configuration to a YAML file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _deep_update(base_dict: dict, update_dict: dict) -> dict:
    """Recursively update a dictionary."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def load_config(config_path: Optional[Path | str] = None) -> PanoGSConfig:
    """
    Load configuration from YAML file or return defaults if path is None or file does not exist.
    """
    config = PanoGSConfig()
    if config_path is None:
        return config

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Build dataclasses from dictionary
    hw_data = data.get("hardware", {})
    img_data = data.get("image", {})
    recon_data = data.get("reconstruction", {})

    hardware = HardwareConfig(**{k: v for k, v in hw_data.items() if k in HardwareConfig.__dataclass_fields__})
    image = ImageConfig(**{k: v for k, v in img_data.items() if k in ImageConfig.__dataclass_fields__})
    reconstruction = ReconstructionConfig(**{k: v for k, v in recon_data.items() if k in ReconstructionConfig.__dataclass_fields__})

    return PanoGSConfig(
        hardware=hardware,
        image=image,
        reconstruction=reconstruction,
    )
