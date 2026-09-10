"""Shared fixtures and utilities for pytest."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest


@pytest.fixture
def temp_image_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with synthetic test images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1. RGB PNG (100x80)
    rgb_arr = np.zeros((80, 100, 3), dtype=np.uint8)
    rgb_arr[:40, :, 0] = 255  # Red top half
    rgb_arr[40:, :, 1] = 255  # Green bottom half
    Image.fromarray(rgb_arr).save(img_dir / "sample_rgb.png")

    # 2. RGB JPEG (64x64)
    jpg_arr = np.full((64, 64, 3), 128, dtype=np.uint8)
    Image.fromarray(jpg_arr).save(img_dir / "sample_rgb.jpg", quality=95)

    # 3. Grayscale PNG (50x50)
    gray_arr = np.linspace(0, 255, 2500, dtype=np.uint8).reshape((50, 50))
    Image.fromarray(gray_arr, mode="L").save(img_dir / "sample_gray.png")

    # 4. RGBA PNG (32x32)
    rgba_arr = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba_arr[:, :, 2] = 255  # Blue
    rgba_arr[:, :, 3] = 200  # Alpha
    Image.fromarray(rgba_arr, mode="RGBA").save(img_dir / "sample_rgba.png")

    return img_dir
