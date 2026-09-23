"""Tests for AI inpainting and foreground object segmentation."""

from pathlib import Path
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


def test_reconstruct_layout_with_inpainting_and_backing(tmp_path):
    from PIL import Image
    from panogs.reconstruction.depth.base import DepthEstimator, DepthResult
    from panogs.reconstruction.panorama.layout import reconstruct_layout_panorama

    class MockEstimator(DepthEstimator):
        def estimate(self, image):
            if isinstance(image, (str, Path)):
                img_np = np.array(Image.open(image))
            elif isinstance(image, Image.Image):
                img_np = np.array(image)
            else:
                img_np = np.asarray(image)
            H, W = img_np.shape[:2]
            d = np.full((H, W), 5.0, dtype=np.float32)
            # Create a foreground object in lower half (chair/table) at depth 1.0m
            d[int(H * 0.55):int(H * 0.85), int(W * 0.4):int(W * 0.6)] = 1.0
            return DepthResult(depth_map=d, is_metric=True, model_name="mock_depth")

    img_arr = np.random.randint(50, 200, (64, 128, 3), dtype=np.uint8)
    img_path = tmp_path / "test_pano.png"
    Image.fromarray(img_arr).save(img_path)

    pc = reconstruct_layout_panorama(
        image_path=img_path,
        depth_estimator=MockEstimator(),
        h_floor=2.0,
        max_resolution=128,
    )

    # Point cloud should have primary points minus pruned boundary pixels plus infilled floor points
    assert pc.num_points > 0.9 * (64 * 128)
    assert pc.normals is not None
    assert "infilled_points" in pc.metadata
    assert pc.metadata["infilled_points"] > 0
