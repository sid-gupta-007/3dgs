"""
Dataset and view generator for panoramic 3D Gaussian Splatting training.
Extracts perspective camera views from equirectangular panorama images
using exact spherical ray sampling and bilinear interpolation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
import torch
from PIL import Image

from panogs.io.images import load_image
from panogs.rendering.camera import Camera, create_lookat_camera


@dataclass
class TrainingView:
    """A ground truth training view pairing a Camera with its target RGB image."""
    camera: Camera
    image: torch.Tensor  # (H, W, 3) float32 in [0, 1]
    name: str


def sample_perspective_from_equirectangular(
    pano_rgb: np.ndarray,
    camera: Camera,
) -> np.ndarray:
    """
    Render/sample a perspective view from an equirectangular panorama image.

    Args:
        pano_rgb: (H_pano, W_pano, 3) float32 in [0, 1] or uint8 in [0, 255].
        camera: Perspective Camera defining the viewport.

    Returns:
        np.ndarray: (H, W, 3) float32 sampled perspective image in [0, 1].
    """
    if pano_rgb.dtype == np.uint8:
        pano = pano_rgb.astype(np.float32) / 255.0
    else:
        pano = pano_rgb.astype(np.float32)

    H_pano, W_pano = pano.shape[:2]
    H, W = camera.height, camera.width
    fx, fy = camera.fx, camera.fy
    cx, cy = camera.cx, camera.cy

    # 1. Pixel coordinate grid
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u, v, indexing="xy")

    # 2. Unproject to camera space ray directions: (x, y, 1)
    x_cam = (u_grid - cx) / fx
    y_cam = (v_grid - cy) / fy
    z_cam = np.ones_like(x_cam)

    # (H, W, 3)
    rays_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

    # 3. Transform rays to world space: d_world = R_cam^T @ d_cam
    # camera.R is (3, 3) World-to-Camera, so Camera-to-World is R_cam^T
    rays_world = np.matmul(rays_cam, camera.R)  # (H, W, 3)

    # Normalize rays
    norms = np.linalg.norm(rays_world, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    rays_norm = rays_world / norms

    dx = rays_norm[:, :, 0]
    dy = rays_norm[:, :, 1]
    dz = rays_norm[:, :, 2]

    # 4. Convert unit rays to equirectangular coordinates
    # theta in [-pi, pi], phi in [-pi/2, pi/2]
    theta = np.arctan2(dx, dz)
    phi = np.arcsin(np.clip(dy, -1.0, 1.0))

    # Normalized UVs in [0, 1]
    u_pano_norm = (theta / (2.0 * np.pi)) + 0.5
    v_pano_norm = 0.5 - (phi / np.pi)

    # Pixel coords in panorama
    u_pano_pix = u_pano_norm * (W_pano - 1.0)
    v_pano_pix = v_pano_norm * (H_pano - 1.0)

    # Bilinear interpolation
    u0 = np.floor(u_pano_pix).astype(np.int32)
    u1 = np.clip(u0 + 1, 0, W_pano - 1)
    v0 = np.floor(v_pano_pix).astype(np.int32)
    v1 = np.clip(v0 + 1, 0, H_pano - 1)

    # Wrap u around horizontally
    u0 = np.mod(u0, W_pano)
    u1 = np.mod(u1, W_pano)

    du = (u_pano_pix - u0)[:, :, np.newaxis]
    dv = (v_pano_pix - v0)[:, :, np.newaxis]

    Ia = pano[v0, u0]
    Ib = pano[v0, u1]
    Ic = pano[v1, u0]
    Id = pano[v1, u1]

    # Weighted blend
    sampled = (
        Ia * (1.0 - du) * (1.0 - dv)
        + Ib * du * (1.0 - dv)
        + Ic * (1.0 - du) * dv
        + Id * du * dv
    )
    return np.clip(sampled, 0.0, 1.0).astype(np.float32)


def generate_training_views(
    image_path: Union[str, Path],
    num_views: int = 8,
    view_width: int = 256,
    view_height: int = 256,
    fov: float = 75.0,
    device: str = "cpu",
) -> List[TrainingView]:
    """
    Generate a set of multi-view perspective training pairs from an input panorama.

    Args:
        image_path: Path to input panorama or image.
        num_views: Number of surrounding azimuth angles (e.g., 8 views every 45 deg).
        view_width: Perspective view width in pixels.
        view_height: Perspective view height in pixels.
        fov: Field of view in degrees.
        device: Device to place ground-truth target tensors on.

    Returns:
        List[TrainingView]: List of training views with Cameras and ground-truth image tensors.
    """
    pil_img = load_image(image_path)
    pano_rgb = np.array(pil_img, dtype=np.float32) / 255.0

    views: List[TrainingView] = []
    azimuths = np.linspace(0.0, 360.0, num_views, endpoint=False)

    for i, az in enumerate(azimuths):
        # Create camera looking outwards from origin (0, 0, 0)
        az_rad = np.radians(az)
        # Target direction
        tx = float(np.sin(az_rad))
        ty = 0.0
        tz = float(np.cos(az_rad))

        cam = create_lookat_camera(
            eye=(0.0, 0.0, 0.0),
            target=(tx, ty, tz),
            up=(0.0, 1.0, 0.0),
            width=view_width,
            height=view_height,
            fov=fov,
        )

        gt_rgb_np = sample_perspective_from_equirectangular(pano_rgb, cam)
        gt_tensor = torch.tensor(gt_rgb_np, dtype=torch.float32, device=device)

        views.append(TrainingView(camera=cam, image=gt_tensor, name=f"view_{i:02d}_az_{int(az)}"))

    return views
