"""
Camera model and viewpoint geometry for 3D Gaussian Splatting rendering.
"""

from dataclasses import dataclass
from typing import Tuple, Union
import numpy as np


@dataclass
class Camera:
    """
    Pinhole perspective camera for 3D scene rendering.

    Attributes:
        position: (3,) camera position in world space.
        R: (3, 3) World-to-Camera rotation matrix.
        t: (3,) World-to-Camera translation vector: x_cam = R @ x_world + t.
        width: Image width in pixels.
        height: Image height in pixels.
        fov_x: Horizontal field of view in degrees.
        fov_y: Vertical field of view in degrees.
    """
    position: np.ndarray
    R: np.ndarray
    t: np.ndarray
    width: int
    height: int
    fov_x: float
    fov_y: float

    @property
    def fx(self) -> float:
        """Focal length along X in pixels."""
        return float(self.width / (2.0 * np.tan(np.radians(self.fov_x) / 2.0)))

    @property
    def fy(self) -> float:
        """Focal length along Y in pixels."""
        return float(self.height / (2.0 * np.tan(np.radians(self.fov_y) / 2.0)))

    @property
    def cx(self) -> float:
        """Principal point X in pixels."""
        return float(self.width / 2.0)

    @property
    def cy(self) -> float:
        """Principal point Y in pixels."""
        return float(self.height / 2.0)

    def world_to_camera(self, points: np.ndarray) -> np.ndarray:
        """
        Transform 3D points from world space to camera space:
            p_cam = R @ p_world + t
        """
        pts = np.asarray(points, dtype=np.float32)
        # (N, 3) @ (3, 3)^T + (3,) = (N, 3)
        return (np.matmul(pts, self.R.T) + self.t).astype(np.float32)


def create_lookat_camera(
    eye: Tuple[float, float, float],
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    width: int = 512,
    height: int = 512,
    fov: float = 60.0,
) -> Camera:
    """
    Create a Camera looking from `eye` towards `target` with given `up` vector.

    Camera coordinate convention:
        +X: Right
        +Y: Down (image space) or Up (standard 3DGS view convention: camera looks along +Z, right +X, down +Y)
        +Z: Forward (towards scene)
    """
    eye_pt = np.array(eye, dtype=np.float32)
    target_pt = np.array(target, dtype=np.float32)
    up_vec = np.array(up, dtype=np.float32)

    # Forward direction (+Z in camera space)
    forward = target_pt - eye_pt
    f_norm = np.linalg.norm(forward)
    if f_norm == 0:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        forward = forward / f_norm

    # Right direction (+X in camera space)
    right = np.cross(forward, up_vec)
    r_norm = np.linalg.norm(right)
    if r_norm == 0:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right = right / r_norm

    # True Up direction (-Y in screen space / +Y in camera view)
    cam_up = np.cross(right, forward)

    # Camera rotation matrix: rows are (right, -cam_up, forward) for standard screen Y down
    R = np.stack([right, -cam_up, forward], axis=0).astype(np.float32)
    t = (-np.matmul(R, eye_pt)).astype(np.float32)

    aspect = width / float(height)
    fov_x = fov
    fov_y = float(2.0 * np.degrees(np.arctan(np.tan(np.radians(fov) / 2.0) / aspect)))

    return Camera(
        position=eye_pt,
        R=R,
        t=t,
        width=width,
        height=height,
        fov_x=fov_x,
        fov_y=fov_y,
    )


def create_orbit_camera(
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    distance: float = 2.5,
    azimuth_deg: float = 0.0,
    elevation_deg: float = 0.0,
    width: int = 512,
    height: int = 512,
    fov: float = 60.0,
) -> Camera:
    """
    Create an orbital Camera placed on a sphere around `target`.

    Args:
        target: 3D point the camera is looking at.
        distance: Distance from target in meters.
        azimuth_deg: Horizontal rotation around Y axis in degrees.
        elevation_deg: Vertical elevation angle in degrees (-89 to +89).
        width: Render width.
        height: Render height.
        fov: Field of view in degrees.
    """
    az_rad = np.radians(azimuth_deg)
    el_rad = np.radians(np.clip(elevation_deg, -89.0, 89.0))

    x = target[0] + distance * np.cos(el_rad) * np.sin(az_rad)
    y = target[1] + distance * np.sin(el_rad)
    z = target[2] + distance * np.cos(el_rad) * np.cos(az_rad)

    return create_lookat_camera(
        eye=(x, y, z),
        target=target,
        width=width,
        height=height,
        fov=fov,
    )
