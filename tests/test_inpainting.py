"""Tests for AI inpainting and foreground object segmentation."""

import numpy as np
import pytest
from panogs.reconstruction.inpainting import (
    inpaint_background_texture,
    segment_foreground_objects,
)


def test_segment_foreground_objects():
    H, W = 100, 100
    # Background plane at 4.0m
    depth_map = np.full((H, W), 4.0, dtype=np.float32)

    # Foreground object at (30:70, 30:70) at 1.5m (40x40 = 1600 pixels)
    depth_map[30:70, 30:70] = 1.5

    mask = segment_foreground_objects(depth_map, rel_depth_threshold=0.15, min_size_pixels=100)
    assert mask.shape == (H, W)
    assert mask.dtype == np.uint8

    # Center should be marked foreground (255)
    assert mask[50, 50] == 255
    # Corners should be background (0)
    assert mask[5, 5] == 0


def test_inpaint_background_texture():
    H, W = 60, 60
    # Red background
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :] = [200, 50, 50]

    # Green foreground patch in center
    img[20:40, 20:40] = [50, 200, 50]

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[20:40, 20:40] = 255

    inpainted = inpaint_background_texture(img, mask, inpaint_radius=5)
    assert inpainted.shape == (H, W, 3)
    assert inpainted.dtype == np.uint8

    # Center should be restored towards surrounding red background
    center_color = inpainted[30, 30]
    assert center_color[0] > 100  # Red channel restored
