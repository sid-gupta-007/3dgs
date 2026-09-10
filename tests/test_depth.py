"""Tests for monocular depth estimation, colormapping, and persistence."""

from pathlib import Path
import numpy as np
import pytest
from panogs.io.depth import load_depth_result, save_depth_result
from panogs.reconstruction.depth import (
    DepthEstimator,
    DepthResult,
    SyntheticDepthEstimator,
    get_depth_estimator,
)


def test_depth_result_dataclass():
    d_map = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    res = DepthResult(depth_map=d_map, is_metric=True, model_name="test_model")

    assert res.height == 2
    assert res.width == 2
    assert res.min_depth == 1.0
    assert res.max_depth == 4.0
    assert res.mean_depth == 2.5
    assert res.is_metric is True

    norm = res.normalized()
    assert norm[0, 0] == 0.0
    assert norm[1, 1] == 1.0


def test_synthetic_depth_estimator(temp_image_dir: Path):
    estimator = SyntheticDepthEstimator(mode="room", min_depth=1.0, max_depth=6.0)
    img_path = temp_image_dir / "sample_rgb.png"  # 100x80

    result = estimator.estimate(img_path)
    assert result.depth_map.shape == (80, 100)
    assert result.depth_map.dtype == np.float32
    assert result.min_depth >= 0.99
    assert result.max_depth <= 6.01
    assert result.is_metric is True


def test_depth_colormapping():
    d_map = np.linspace(0.0, 1.0, 100).reshape((10, 10)).astype(np.float32)
    res = DepthResult(depth_map=d_map, is_metric=False)

    vis = DepthEstimator.colorize(res)
    assert vis.shape == (10, 10, 3)
    assert vis.dtype == np.uint8
    # Check that colors are not all identical (i.e. colormap varies)
    assert not np.array_equal(vis[0, 0], vis[-1, -1])


def test_save_and_load_depth_result(tmp_path: Path):
    d_map = np.random.uniform(1.0, 10.0, (50, 60)).astype(np.float32)
    res = DepthResult(depth_map=d_map, is_metric=True, model_name="unit_test_model")

    npy_path, vis_path, meta_path = save_depth_result(res, tmp_path, prefix="test_depth")
    assert npy_path.exists()
    assert vis_path.exists()
    assert meta_path.exists()

    loaded = load_depth_result(npy_path, meta_path)
    assert loaded.depth_map.shape == (50, 60)
    assert loaded.is_metric is True
    assert loaded.model_name == "unit_test_model"
    assert np.allclose(loaded.depth_map, d_map, atol=1e-6)


def test_get_depth_estimator_factory():
    est = get_depth_estimator("synthetic_room")
    assert isinstance(est, SyntheticDepthEstimator)
    assert est.mode == "room"

    est_grad = get_depth_estimator("synthetic_gradient")
    assert isinstance(est_grad, SyntheticDepthEstimator)
    assert est_grad.mode == "gradient"


def test_cubemap_depth_estimator(temp_image_dir: Path):
    from panogs.reconstruction.depth.cubemap import CubemapDepthEstimator
    est = get_depth_estimator("cubemap_midas")
    assert isinstance(est, CubemapDepthEstimator)
    img_path = temp_image_dir / "sample_rgb.png"
    # Small face size for instant test execution
    est.face_size = 32
    res = est.estimate(img_path)
    assert res.depth_map.shape == (80, 100)
    assert res.depth_map.dtype == np.float32
    assert res.min_depth >= 0.0
    assert res.max_depth <= 1.0

