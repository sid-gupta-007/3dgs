"""
Spherical Camera Geometry and Equirectangular Ray Mapping.

Provides functions to map 2D equirectangular panorama pixels into
spherical coordinates (longitude, latitude) and normalized 3D ray directions.
"""

from typing import Optional, Tuple, Union
import numpy as np


def pixel_to_spherical(
    u: Union[float, np.ndarray],
    v: Union[float, np.ndarray],
    width: int,
    height: int,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert 2D pixel coordinates (u, v) in an equirectangular image to
    spherical angles (theta, phi) in radians using pixel-center alignment.

    Args:
        u: Horizontal pixel index (0 <= u < width) or array.
        v: Vertical pixel index (0 <= v < height) or array.
        width: Image width in pixels (W).
        height: Image height in pixels (H).

    Returns:
        Tuple (theta, phi):
            - theta (longitude): in [-pi, pi], 0 is center-forward (+Z).
            - phi (latitude): in [-pi/2, pi/2], +pi/2 is top (+Y), -pi/2 is bottom (-Y).
    """
    # Pixel center normalized coordinates in (0, 1)
    x_norm = (u + 0.5) / float(width)
    y_norm = (v + 0.5) / float(height)

    # Longitude: x_norm in [0, 1] -> theta in [-pi, pi]
    theta = (x_norm - 0.5) * (2.0 * np.pi)

    # Latitude: y_norm in [0, 1] -> phi in [pi/2, -pi/2]
    phi = (0.5 - y_norm) * np.pi

    return theta, phi


def spherical_to_ray(
    theta: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
) -> np.ndarray:
    """
    Convert spherical angles (theta, phi) in radians to a 3D unit ray direction vector.

    Coordinate system:
        +X: Right
        +Y: Up
        +Z: Forward (center of panorama)

    Args:
        theta: Longitude angle in radians.
        phi: Latitude angle in radians.

    Returns:
        np.ndarray: 3D vector (..., 3) with unit norm ||d|| = 1.
    """
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    dx = cos_phi * sin_theta
    dy = sin_phi
    dz = cos_phi * cos_theta

    if isinstance(theta, np.ndarray) or isinstance(phi, np.ndarray):
        return np.stack([dx, dy, dz], axis=-1).astype(np.float32)
    return np.array([dx, dy, dz], dtype=np.float32)


def equirectangular_pixel_to_ray(
    u: float,
    v: float,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Convert a single pixel (u, v) in an equirectangular image to a 3D unit ray vector.
    """
    theta, phi = pixel_to_spherical(u, v, width, height)
    return spherical_to_ray(theta, phi)


def equirectangular_rays(
    height: int,
    width: int,
    jitter: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a dense grid of unit ray directions for an equirectangular panorama of size (H, W).

    Args:
        height: Image height H.
        width: Image width W.
        jitter: If True, apply sub-pixel jitter to ray sampling coordinates to prevent moire/scanlines.
        rng: Optional NumPy random Generator.

    Returns:
        np.ndarray: Array of shape (H, W, 3) where each vector has unit length.
    """
    # Generate pixel coordinate grids
    u_coords = np.arange(width, dtype=np.float32)
    v_coords = np.arange(height, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords, indexing="xy")

    if jitter:
        gen = rng if rng is not None else np.random.default_rng(42)
        u_jitter = gen.uniform(-0.35, 0.35, size=(height, width)).astype(np.float32)
        v_jitter = gen.uniform(-0.35, 0.35, size=(height, width)).astype(np.float32)
        u_grid = u_grid + u_jitter
        v_grid = np.clip(v_grid + v_jitter, 0.0, height - 1.0)

    theta, phi = pixel_to_spherical(u_grid, v_grid, width, height)
    rays = spherical_to_ray(theta, phi)
    return rays
