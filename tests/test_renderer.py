"""
Unit and integration tests for the PanoGS Camera model and CPU Gaussian Splatting Rasterizer.
"""

import numpy as np
import pytest
from panogs.core.gaussian.model import GaussianModel
from panogs.rendering.camera import Camera, create_lookat_camera, create_orbit_camera
from panogs.rendering.cpu.rasterizer import project_gaussians_to_2d, render_gaussians_cpu


def test_camera_properties():
    """Test camera focal length and principal point computation."""
    cam = Camera(
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        R=np.eye(3, dtype=np.float32),
        t=np.zeros(3, dtype=np.float32),
        width=800,
        height=600,
        fov_x=60.0,
        fov_y=45.0,
    )
    assert cam.width == 800
    assert cam.height == 600
    assert cam.cx == 400.0
    assert cam.cy == 300.0
    assert cam.fx > 0.0
    assert cam.fy > 0.0


def test_lookat_and_orbit_camera():
    """Test camera creation helpers."""
    cam = create_lookat_camera(
        eye=(0.0, 0.0, 5.0),
        target=(0.0, 0.0, 0.0),
        width=256,
        height=256,
        fov=60.0,
    )
    assert cam.position.shape == (3,)
    assert cam.R.shape == (3, 3)
    assert cam.t.shape == (3,)

    # Orbit camera
    orbit_cam = create_orbit_camera(
        target=(0.0, 0.0, 0.0),
        distance=3.0,
        azimuth_deg=45.0,
        elevation_deg=30.0,
        width=128,
        height=128,
    )
    assert orbit_cam.width == 128
    assert orbit_cam.height == 128
    assert np.allclose(np.linalg.norm(orbit_cam.position), 3.0, atol=1e-4)


def test_project_gaussians_to_2d():
    """Test 2D projection and Jacobian covariance transformation."""
    # Create single Gaussian placed directly in front of camera
    xyz = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]], dtype=np.float32)  # One in front, one behind
    scales = np.array([[0.05, 0.05, 0.05], [0.05, 0.05, 0.05]], dtype=np.float32)
    quats = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    opacities = np.array([0.9, 0.9], dtype=np.float32)

    model = GaussianModel.from_raw(xyz, scales, quats, colors, opacities)
    cam = create_lookat_camera(eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 2.0), width=128, height=128)

    vis_idx, depths, screen_pts, cov2d, radii = project_gaussians_to_2d(model, cam, min_depth=0.1)

    # Only 1 Gaussian (the one at +2.0) should be visible
    assert len(vis_idx) == 1
    assert vis_idx[0] == 0
    assert screen_pts.shape == (1, 2)
    # The projected center should be at image center (64, 64)
    assert np.allclose(screen_pts[0], [64.0, 64.0], atol=1.0)
    assert cov2d.shape == (1, 2, 2)
    assert cov2d[0, 0, 0] > 0.0
    assert cov2d[0, 1, 1] > 0.0
    assert radii[0] > 0.0


def test_render_single_gaussian():
    """Test CPU rendering of a single red Gaussian centered in frame."""
    xyz = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)
    scales = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
    quats = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # Pure Red
    opacities = np.array([1.0], dtype=np.float32)

    model = GaussianModel.from_raw(xyz, scales, quats, colors, opacities)
    cam = create_lookat_camera(eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 2.0), width=64, height=64)

    img = render_gaussians_cpu(model, cam, bg_color=(0.0, 0.0, 0.0))  # Black background

    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8

    # Center pixel (32, 32) should be strongly red
    center_pixel = img[32, 32]
    assert center_pixel[0] > 200  # Red
    assert center_pixel[1] < 50   # Green
    assert center_pixel[2] < 50   # Blue

    # Corner pixel (0, 0) should be background (black)
    assert np.all(img[0, 0] == 0)


def test_render_multi_gaussian_depth_sorting():
    """Test front-to-back alpha blending with two overlapping Gaussians."""
    # Blue Gaussian in front (Z=1.5), Red Gaussian in back (Z=3.0)
    xyz = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 3.0]], dtype=np.float32)
    scales = np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]], dtype=np.float32)
    quats = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)  # Blue front, Red back
    opacities = np.array([0.99, 0.99], dtype=np.float32)

    model = GaussianModel.from_raw(xyz, scales, quats, colors, opacities)
    cam = create_lookat_camera(eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 2.0), width=64, height=64)

    img = render_gaussians_cpu(model, cam, bg_color=(0.0, 0.0, 0.0))

    # Center pixel should be dominated by the front (Blue) Gaussian
    center = img[32, 32]
    assert center[2] > center[0]  # Blue > Red
