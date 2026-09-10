"""Tests for equirectangular spherical ray generation."""

import numpy as np
import pytest
from panogs.core.camera.spherical import (
    equirectangular_pixel_to_ray,
    equirectangular_rays,
    pixel_to_spherical,
    spherical_to_ray,
)


def test_pixel_to_spherical_angles():
    # In a 100x50 image:
    # u=50, v=25 should be center: theta ~ 0, phi ~ 0
    theta, phi = pixel_to_spherical(50, 25, 100, 50)
    assert abs(theta) < 0.05
    assert abs(phi) < 0.05

    # v=0 (top edge) -> phi ~ +pi/2
    _, phi_top = pixel_to_spherical(50, 0, 100, 50)
    assert phi_top > 1.5

    # v=49 (bottom edge) -> phi ~ -pi/2
    _, phi_bot = pixel_to_spherical(50, 49, 100, 50)
    assert phi_bot < -1.5


def test_ray_unit_normalization():
    # Generate 64x128 rays
    rays = equirectangular_rays(height=64, width=128)
    assert rays.shape == (64, 128, 3)

    norms = np.linalg.norm(rays, axis=-1)
    # Check that ALL rays have unit length ||d|| = 1 within 1e-6
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_cardinal_directions():
    width, height = 200, 100
    # Center pixel (u = 100, v = 50) -> forward (+Z)
    center_ray = equirectangular_pixel_to_ray(100, 50, width, height)
    assert center_ray[2] > 0.99  # +Z forward
    assert abs(center_ray[0]) < 0.05  # X ~ 0
    assert abs(center_ray[1]) < 0.05  # Y ~ 0

    # Top pixel (u = 100, v = 0) -> +Y (Up)
    top_ray = equirectangular_pixel_to_ray(100, 0, width, height)
    assert top_ray[1] > 0.95  # +Y up

    # Bottom pixel (u = 100, v = 99) -> -Y (Down)
    bottom_ray = equirectangular_pixel_to_ray(100, 99, width, height)
    assert bottom_ray[1] < -0.95  # -Y down
