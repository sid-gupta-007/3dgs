"""
Tests for LightGaussian significance-based pruning and compression engine.
"""

import numpy as np
import pytest
from panogs.core.compression import (
    CompressionConfig,
    compute_gaussian_importance,
    prune_gaussians_by_importance,
    compress_scene,
)
from panogs.core.gaussian.model import GaussianModel
from panogs.io.gaussian_ply import save_gaussian_ply


@pytest.fixture
def sample_gaussian_model():
    """Create a sample GaussianModel with varying opacities and scales."""
    N = 100
    xyz = np.random.randn(N, 3).astype(np.float32)
    scales = np.random.uniform(0.01, 0.2, (N, 3)).astype(np.float32)
    rotations = np.zeros((N, 4), dtype=np.float32)
    rotations[:, 0] = 1.0  # Identity quaternion
    colors_rgb = np.ones((N, 3), dtype=np.float32) * 0.5
    opacities = np.linspace(0.01, 0.99, N).astype(np.float32)

    return GaussianModel.from_raw(
        xyz=xyz,
        scales=scales,
        rotation_quats=rotations,
        colors_rgb=colors_rgb,
        opacities=opacities,
    )


def test_compute_importance(sample_gaussian_model):
    """Test Gaussian importance calculation."""
    scores = compute_gaussian_importance(sample_gaussian_model)
    assert len(scores) == sample_gaussian_model.num_gaussians
    assert np.all(scores >= 0.0)
    # Higher opacity / volume should have higher score
    assert scores[-1] > scores[0]


def test_prune_gaussians(sample_gaussian_model):
    """Test pruning fraction."""
    pruned, mask = prune_gaussians_by_importance(sample_gaussian_model, prune_ratio=0.30)
    assert pruned.num_gaussians < sample_gaussian_model.num_gaussians
    assert pruned.num_gaussians == np.sum(mask)
    assert pytest.approx(pruned.num_gaussians, abs=5) == 70


def test_compress_scene(tmp_path, sample_gaussian_model):
    """Test full scene compression with file saving."""
    input_ply = tmp_path / "input.ply"
    output_ply = tmp_path / "compressed.ply"

    save_gaussian_ply(input_ply, sample_gaussian_model)

    compressed = compress_scene(input_ply, output_ply, preset="balanced", save_splat=True)
    assert output_ply.exists()
    assert output_ply.with_suffix(".splat").exists()
    assert compressed.num_gaussians < sample_gaussian_model.num_gaussians
