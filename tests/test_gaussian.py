"""Tests for 3D Gaussian Splatting scene model, covariance, adaptive initialization, and PLY/SPLAT I/O."""

from pathlib import Path
import numpy as np
import pytest
from panogs.core.gaussian import GaussianModel, initialize_from_pointcloud
from panogs.core.gaussian.model import (
    logit,
    quaternion_to_rotation_matrix,
    rgb_to_sh0,
    sh0_to_rgb,
    sigmoid,
)
from panogs.io.gaussian_ply import load_gaussian_ply, save_gaussian_ply, save_gaussian_splat
from panogs.reconstruction.pointcloud import PointCloud


def test_sh0_rgb_roundtrip():
    rgb = np.array([[1.0, 0.0, 0.5], [0.2, 0.8, 0.4]], dtype=np.float32)
    sh0 = rgb_to_sh0(rgb)
    reconstructed_rgb = sh0_to_rgb(sh0)
    assert np.allclose(rgb, reconstructed_rgb, atol=1e-5)


def test_sigmoid_logit_roundtrip():
    probs = np.array([0.1, 0.5, 0.8, 0.95], dtype=np.float32)
    logits = logit(probs)
    recovered_probs = sigmoid(logits)
    assert np.allclose(probs, recovered_probs, atol=1e-5)


def test_quaternion_rotation_matrix_properties():
    # Identity quaternion (1, 0, 0, 0)
    q_ident = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    R_ident = quaternion_to_rotation_matrix(q_ident)
    assert np.allclose(R_ident[0], np.eye(3), atol=1e-6)

    # 90-degree rotation around Y axis: qw = cos(45) = sqrt(0.5), qy = sin(45) = sqrt(0.5)
    s = np.sqrt(0.5).astype(np.float32)
    q_rot_y = np.array([[s, 0.0, s, 0.0]], dtype=np.float32)
    R = quaternion_to_rotation_matrix(q_rot_y)[0]

    # Check orthogonality: R @ R^T = I
    assert np.allclose(np.matmul(R, R.T), np.eye(3), atol=1e-6)
    # Check determinant = +1
    assert abs(np.linalg.det(R) - 1.0) < 1e-6


def test_3d_covariance_properties():
    # 2 Gaussians with different positions, rotations, and scales
    xyz = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32)
    scales_log = np.array([[0.0, -1.0, 1.0], [-0.5, 0.5, 0.0]], dtype=np.float32)
    quats = np.array([[1.0, 0, 0, 0], [0.7071, 0.7071, 0, 0]], dtype=np.float32)
    opacities = np.array([[0.0], [1.0]], dtype=np.float32)
    sh0 = np.zeros((2, 3), dtype=np.float32)

    model = GaussianModel(
        xyz=xyz,
        scaling_log=scales_log,
        rotation_quats=quats,
        opacity_logits=opacities,
        features_dc=sh0,
    )

    cov3d = model.get_covariance_3d()
    assert cov3d.shape == (2, 3, 3)

    for i in range(2):
        C = cov3d[i]
        # 1. Check symmetry: C == C^T
        assert np.allclose(C, C.T, atol=1e-6)
        # 2. Check positive semi-definiteness: all eigenvalues >= 0
        eigenvals = np.linalg.eigvalsh(C)
        assert np.all(eigenvals > 0.0)


def test_initialize_from_pointcloud():
    # Grid of 8 points
    x = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.float32)
    z = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
    points = np.stack([x, y, z], axis=-1)
    colors = np.full((8, 3), 200, dtype=np.uint8)
    pc = PointCloud(points=points, colors=colors)

    model = initialize_from_pointcloud(pc, default_opacity=0.8, k_scale_neighbors=3)
    assert model.num_gaussians == 8

    scales = model.get_scaling()
    assert scales.shape == (8, 3)
    assert np.all(scales > 0.0)
    # Unit grid spacing is 1.0
    assert np.allclose(scales, 1.0, atol=0.2)

    opacities = model.get_opacity()
    assert np.allclose(opacities, 0.8, atol=1e-4)


def test_save_and_load_gaussian_ply(tmp_path: Path):
    out_ply = tmp_path / "test_gaussians.ply"
    xyz = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32)
    scales_log = np.array([[-1.0, -2.0, -3.0], [0.1, 0.2, 0.3]], dtype=np.float32)
    quats = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    opacities = np.array([[1.5], [-0.5]], dtype=np.float32)
    sh0 = np.array([[0.5, -0.2, 0.8], [-0.1, 0.9, 0.0]], dtype=np.float32)

    model = GaussianModel(xyz, scales_log, quats, opacities, sh0)
    save_gaussian_ply(out_ply, model)
    assert out_ply.exists()

    loaded = load_gaussian_ply(out_ply)
    assert loaded.num_gaussians == 2
    assert np.allclose(loaded.get_xyz(), xyz, atol=1e-5)
    assert np.allclose(loaded._scaling_log, scales_log, atol=1e-5)
    assert np.allclose(loaded._opacity_logits, opacities, atol=1e-5)
    assert np.allclose(loaded._features_dc, sh0, atol=1e-5)


def test_save_gaussian_splat(tmp_path: Path):
    out_splat = tmp_path / "test.splat"
    xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    scales_log = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    quats = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    opacities = np.array([[1.0]], dtype=np.float32)
    sh0 = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    model = GaussianModel(xyz, scales_log, quats, opacities, sh0)
    save_gaussian_splat(out_splat, model)
    assert out_splat.exists()
    assert out_splat.stat().st_size == 32  # 32 bytes per splat
