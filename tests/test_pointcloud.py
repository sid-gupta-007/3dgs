"""Tests for 3D point cloud backprojection, disparity conversion, and reconstruction."""

from pathlib import Path
import numpy as np
import pytest
from panogs.core.camera.spherical import equirectangular_rays
from panogs.reconstruction.depth.synthetic import SyntheticDepthEstimator
from panogs.reconstruction.pointcloud import (
    PointCloud,
    backproject_depth_to_points,
    convert_disparity_to_depth,
    reconstruct_from_image,
)


def test_pointcloud_dataclass():
    points = np.array([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    pc = PointCloud(points=points, colors=colors)

    assert pc.num_points == 2
    min_b, max_b = pc.bounds
    assert np.allclose(min_b, [-1.0, 0.0, 1.0])
    assert np.allclose(max_b, [2.0, 3.0, 4.0])
    assert np.allclose(pc.center, [0.5, 1.5, 2.5])


def test_convert_disparity_to_depth_monotonicity():
    # Disparity where pixel 0 is closest (high value) and pixel 1 is far (low value)
    disp = np.array([[100.0, 10.0]], dtype=np.float32)
    depth = convert_disparity_to_depth(disp, min_depth=1.0, max_depth=10.0)

    # Higher disparity -> closer distance (smaller depth)
    assert depth[0, 0] < depth[0, 1]
    assert depth[0, 0] >= 0.99
    assert depth[0, 1] <= 10.01


def test_backproject_depth_to_points_exact():
    # 2x2 grid
    rays = np.zeros((2, 2, 3), dtype=np.float32)
    rays[0, 0] = [0, 0, 1]  # Forward (+Z)
    rays[0, 1] = [1, 0, 0]  # Right (+X)
    rays[1, 0] = [0, 1, 0]  # Up (+Y)
    rays[1, 1] = [0, 0, -1]  # Backward (-Z)

    depth = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    colors = np.full((2, 2, 3), 200, dtype=np.uint8)

    pc = backproject_depth_to_points(rays, depth, colors, camera_center=(1.0, 0.0, 0.0))
    assert pc.num_points == 4

    # Point 0: C + d*R = (1, 0, 0) + 2*(0, 0, 1) = (1, 0, 2)
    assert np.allclose(pc.points[0], [1.0, 0.0, 2.0])
    # Point 1: (1, 0, 0) + 3*(1, 0, 0) = (4, 0, 0)
    assert np.allclose(pc.points[1], [4.0, 0.0, 0.0])


def test_backproject_filtering():
    rays = np.zeros((2, 2, 3), dtype=np.float32)
    rays[:, :, 2] = 1.0  # All forward

    # One NaN, one out-of-range, two valid
    depth = np.array([[np.nan, 0.01], [2.0, 3.0]], dtype=np.float32)
    colors = np.zeros((2, 2, 3), dtype=np.uint8)

    pc = backproject_depth_to_points(rays, depth, colors, min_depth=0.5, max_depth=10.0)
    assert pc.num_points == 2  # Only the two valid depths are retained


def test_reconstruct_from_image_synthetic(temp_image_dir: Path, tmp_path: Path):
    img_path = temp_image_dir / "sample_rgb.png"  # 100x80
    out_ply = tmp_path / "recon.ply"
    estimator = SyntheticDepthEstimator(mode="room", min_depth=1.0, max_depth=5.0)

    pc = reconstruct_from_image(
        image_path=img_path,
        depth_estimator=estimator,
        camera_type="spherical",
        min_depth=1.0,
        max_depth=5.0,
        output_ply=out_ply,
    )

    assert pc.num_points > 0
    assert out_ply.exists()
