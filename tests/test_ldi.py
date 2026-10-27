"""Tests for Layered Depth Image (LDI) Engine (PanoDreamer Architecture)."""

from pathlib import Path
import numpy as np
import pytest

from panogs.reconstruction.ldi import (
    LDILayer,
    compute_layer_depth_thresholds,
    construct_layered_depth_image,
    reconstruct_from_ldi,
)


def test_compute_layer_depth_thresholds():
    H, W = 100, 100
    # Depth values spanning 1.0m to 10.0m
    d = np.linspace(1.0, 10.0, H * W, dtype=np.float32).reshape(H, W)

    intervals = compute_layer_depth_thresholds(d, num_layers=3)
    assert len(intervals) == 3
    for d_min, d_max in intervals:
        assert d_min < d_max
        assert d_min >= 1.0
        assert d_max <= 10.0


def test_construct_and_reconstruct_ldi(tmp_path: Path):
    H, W = 64, 128
    img = np.random.randint(50, 220, (H, W, 3), dtype=np.uint8)
    depth = np.full((H, W), 4.5, dtype=np.float32)

    # Add foreground object (1.2m depth) in center
    depth[20:45, 40:80] = 1.2

    layers = construct_layered_depth_image(
        img_rgb=img,
        depth_map=depth,
        num_layers=3,
        inpaint_radius=5,
        h_floor=1.5,
        h_ceiling=1.8,
    )

    assert len(layers) == 3
    for layer in layers:
        assert isinstance(layer, LDILayer)
        assert layer.rgb.shape == (H, W, 3)
        assert layer.depth.shape == (H, W)
        assert layer.normals.shape == (H, W, 3)
        assert layer.valid_mask.shape == (H, W)

    out_ply = tmp_path / "test_ldi.ply"
    pc = reconstruct_from_ldi(layers, output_ply=out_ply)

    assert pc.num_points > 0
    assert pc.normals is not None
    assert pc.metadata.get("ldi_enabled") is True
    assert pc.metadata.get("num_ldi_layers") == 3
    assert out_ply.exists()
