"""Tests for image loading, metadata, and validation."""

from pathlib import Path
import numpy as np
import pytest
from panogs.io.images import (
    ImageMetadata,
    inspect_image,
    load_image,
    load_image_as_numpy,
    save_image,
    validate_image_path,
)


def test_inspect_image(temp_image_dir: Path):
    png_path = temp_image_dir / "sample_rgb.png"
    meta = inspect_image(png_path)

    assert isinstance(meta, ImageMetadata)
    assert meta.width == 100
    assert meta.height == 80
    assert meta.channels == 3
    assert meta.aspect_ratio == 1.25
    assert meta.raw_memory_bytes == 100 * 80 * 3
    assert "100 x 80" in meta.formatted_summary()


def test_load_image_rgb(temp_image_dir: Path):
    png_path = temp_image_dir / "sample_rgb.png"
    pil_img = load_image(png_path)
    assert pil_img.size == (100, 80)
    assert pil_img.mode == "RGB"


def test_load_image_as_numpy_normalized(temp_image_dir: Path):
    png_path = temp_image_dir / "sample_rgb.png"
    arr = load_image_as_numpy(png_path, normalize_float=True)

    assert arr.shape == (80, 100, 3)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


def test_load_image_as_numpy_uint8(temp_image_dir: Path):
    png_path = temp_image_dir / "sample_rgb.png"
    arr = load_image_as_numpy(png_path, normalize_float=False)

    assert arr.shape == (80, 100, 3)
    assert arr.dtype == np.uint8


def test_load_image_resolution_limit(temp_image_dir: Path):
    png_path = temp_image_dir / "sample_rgb.png"  # 100x80
    arr = load_image_as_numpy(png_path, max_resolution=50)

    # Longest dimension should be scaled down to 50
    assert max(arr.shape[0], arr.shape[1]) == 50


def test_save_and_reload_image(tmp_path: Path):
    out_path = tmp_path / "saved.png"
    sample = np.ones((40, 60, 3), dtype=np.float32)
    save_image(sample, out_path)

    assert out_path.exists()
    reloaded = load_image_as_numpy(out_path, normalize_float=True)
    assert reloaded.shape == (40, 60, 3)
    assert np.allclose(reloaded, 1.0, atol=0.01)


def test_unsupported_extension(tmp_path: Path):
    bad_file = tmp_path / "test.txt"
    bad_file.write_text("not an image")

    with pytest.raises(ValueError, match="Unsupported image format"):
        validate_image_path(bad_file)


def test_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        validate_image_path("does_not_exist_image_9999.png")
