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

    # Test sharp V2 initialization
    model = initialize_from_pointcloud(pc, default_opacity=0.8, k_scale_neighbors=3)
    assert model.num_gaussians == 8
    scales = model.get_scaling()
    assert scales.shape == (8, 3)
    assert np.all(scales > 0.0)
    assert np.all(scales < 2.0)

    # Test legacy scale_multiplier=1.0 with sharpness=0.0
    model_legacy = initialize_from_pointcloud(pc, default_opacity=0.8, k_scale_neighbors=3, scale_multiplier=1.0, sharpness=0.0)
    scales_legacy = model_legacy.get_scaling()
    assert np.allclose(scales_legacy, 1.0, atol=0.2)

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


def test_vectorized_rotation_matrix_to_quaternion():
    from panogs.core.gaussian.initialization import rotation_matrix_to_quaternion

    angles = np.linspace(0.0, 2.0 * np.pi, 16, dtype=np.float32)
    mats = []
    for a in angles:
        c, s = np.cos(a), np.sin(a)
        rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
        ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        mats.extend([rx, ry, rz, rx @ ry @ rz])
    R_in = np.stack(mats, axis=0)
    quats = rotation_matrix_to_quaternion(R_in)
    R_out = quaternion_to_rotation_matrix(quats)
    assert np.allclose(R_in, R_out, atol=1e-5)


def test_supersplat_ply_export_and_ensure(tmp_path: Path):
    from panogs.io.gaussian_ply import ensure_supersplat_ply, is_supersplat_compatible_ply
    from panogs.io.ply import write_point_cloud_ply

    xyz = np.array([[1.5, 2.5, -3.0], [-1.0, -0.5, 4.0]], dtype=np.float32)
    scales = np.array([[0.1, 0.2, 0.05], [0.15, 0.1, 0.2]], dtype=np.float32)
    quats = np.array([[1.0, 0.0, 0.0, 0.0], [0.70710678, 0.70710678, 0.0, 0.0]], dtype=np.float32)
    hdr_colors = np.array([[12.0, 4.0, 1.0], [0.2, 0.5, 0.8]], dtype=np.float32)
    opacities = np.array([0.9, 0.8], dtype=np.float32)

    model = GaussianModel.from_raw(xyz, scales, quats, hdr_colors, opacities)
    compat_ply = tmp_path / "compat.ply"
    save_gaussian_ply(compat_ply, model, supersplat_compat=True)
    assert is_supersplat_compatible_ply(compat_ply)

    loaded = load_gaussian_ply(compat_ply)
    assert np.allclose(loaded.get_xyz(), xyz, atol=1e-5)
    assert np.allclose(loaded.get_covariance_3d(), model.get_covariance_3d(), atol=1e-5)

    # Also test upgrading a raw point cloud PLY to SuperSplat 3DGS PLY
    pc_ply = tmp_path / "raw_pc.ply"
    pts = np.random.uniform(-2.0, 2.0, (16, 3)).astype(np.float32)
    cols = np.random.randint(20, 240, (16, 3), dtype=np.uint8)
    write_point_cloud_ply(pc_ply, pts, cols, binary=True)
    assert not is_supersplat_compatible_ply(pc_ply)

    upgraded_ply = tmp_path / "upgraded.ply"
    ensure_supersplat_ply(pc_ply, upgraded_ply)
    assert is_supersplat_compatible_ply(upgraded_ply)
    upgraded_model = load_gaussian_ply(upgraded_ply)
    assert upgraded_model.num_gaussians == 16


def test_tonemap_hdr_to_srgb():
    from panogs.io.images import tonemap_hdr_to_srgb

    hdr = np.zeros((16, 16, 3), dtype=np.float32) + 0.18
    hdr[0, 0] = [5.0, 5.0, 5.0]
    hdr[0, 1] = [50.0, 50.0, 50.0]
    srgb = tonemap_hdr_to_srgb(hdr)
    assert srgb.shape == (16, 16, 3)
    assert np.all(srgb >= 0.0) and np.all(srgb <= 1.0)
    # Brighter HDR peak should remain strictly brighter than lower peak instead of both clipping identically
    assert srgb[0, 1, 0] > srgb[0, 0, 0] > srgb[8, 8, 0]


def test_repair_panoramic_gaussians_heals_voids_and_nadir():
    from panogs.core.camera.spherical import equirectangular_rays
    from panogs.io.gaussian_ply import repair_panoramic_gaussians

    H, W = 40, 80  # 3,200 points (>= 2048 threshold)
    rays = equirectangular_rays(H, W)
    d = np.full((H, W), 4.0, dtype=np.float32)
    # Wide foreground partition at 1.2m with no background points behind it
    d[12:28, 25:45] = 1.2
    # Floating mid-air ribbon at boundary (2.5m between 1.2m and 4.0m)
    d[12:28, 24] = 2.5

    pts = (rays * d[:, :, None]).reshape(-1, 3).astype(np.float32)
    N = len(pts)
    scales = np.full((N, 3), 0.05, dtype=np.float32)
    quats = np.zeros((N, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    colors = np.full((N, 3), 0.6, dtype=np.float32)
    opacities = np.full(N, 0.85, dtype=np.float32)

    model = GaussianModel.from_raw(pts, scales, quats, colors, opacities)
    repaired = repair_panoramic_gaussians(model)

    # Should have synthesized solid background splats + nadir floor cap
    assert repaired.num_gaussians > N
    assert np.all(repaired.get_opacity() >= 0.919)
