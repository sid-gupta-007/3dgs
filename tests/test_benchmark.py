"""
Tests for Benchmark and quality measurement suite.
"""

import numpy as np
import pytest
from panogs.benchmarks.benchmark import (
    BenchmarkResult,
    benchmark_scene,
    compute_psnr,
    compute_ssim,
)
from panogs.core.gaussian.model import GaussianModel
from panogs.io.gaussian_ply import save_gaussian_ply


def test_metrics_computation():
    """Test PSNR and SSIM calculations."""
    img1 = np.ones((50, 50, 3), dtype=np.float32) * 0.5
    img2 = np.ones((50, 50, 3), dtype=np.float32) * 0.5

    psnr = compute_psnr(img1, img2)
    assert psnr >= 50.0

    ssim = compute_ssim(img1, img2)
    assert pytest.approx(ssim, rel=1e-3) == 1.0

    # Noisy image
    noisy = img1 + np.random.normal(0, 0.05, img1.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.0, 1.0)
    assert compute_psnr(img1, noisy) < 40.0
    assert compute_ssim(img1, noisy) < 1.0


def test_benchmark_scene(tmp_path):
    """Test benchmark_scene run."""
    N = 50
    xyz = np.random.randn(N, 3).astype(np.float32)
    scales = np.ones((N, 3), dtype=np.float32) * 0.05
    rotations = np.zeros((N, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    colors_rgb = np.ones((N, 3), dtype=np.float32) * 0.8
    opacities = np.ones(N, dtype=np.float32) * 0.9

    model = GaussianModel.from_raw(
        xyz=xyz,
        scales=scales,
        rotation_quats=rotations,
        colors_rgb=colors_rgb,
        opacities=opacities,
    )
    ply_path = tmp_path / "test_scene.ply"
    save_gaussian_ply(ply_path, model)

    result = benchmark_scene(ply_path, num_frames=3, width=64, height=48)
    assert isinstance(result, BenchmarkResult)
    assert result.splat_count == 50
    assert result.avg_render_ms > 0
    assert result.fps_estimate > 0
