"""
3D Point Cloud Reconstruction Engine.
Combines 2D image colors, spherical/perspective camera rays, and depth maps into 3D point clouds:
    P(u, v) = C + d(u, v) * R(u, v)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image

from panogs.core.camera.spherical import equirectangular_rays
from panogs.core.logging import get_logger
from panogs.io.images import load_image_as_numpy
from panogs.reconstruction.depth.base import DepthEstimator, DepthResult


@dataclass
class PointCloud:
    """
    In-memory representation of a colored 3D point cloud.

    Attributes:
        points: (N, 3) float32 array representing (X, Y, Z) coordinates.
        colors: (N, 3) uint8 array representing RGB colors [0, 255].
        confidence: Optional (N,) float32 array in [0, 1].
        normals: Optional (N, 3) float32 surface normal vectors.
        metadata: Diagnostic and origin metadata.
    """
    points: np.ndarray
    colors: np.ndarray
    confidence: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float32)
        self.colors = np.asarray(self.colors, dtype=np.uint8)

        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"Points must have shape (N, 3), got {self.points.shape}")
        if self.colors.shape[0] != self.points.shape[0] or self.colors.shape[1] != 3:
            raise ValueError(f"Colors shape {self.colors.shape} does not match points {self.points.shape}")

    @property
    def num_points(self) -> int:
        return self.points.shape[0]

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min_xyz, max_xyz) bounding box."""
        if self.num_points == 0:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        return np.min(self.points, axis=0), np.max(self.points, axis=0)

    @property
    def center(self) -> np.ndarray:
        """Return geometric centroid."""
        if self.num_points == 0:
            return np.zeros(3, dtype=np.float32)
        return np.mean(self.points, axis=0)

    def summary(self) -> str:
        min_b, max_b = self.bounds
        return (
            f"PointCloud Summary:\n"
            f"  Points:       {self.num_points:,}\n"
            f"  X Range:      [{min_b[0]:.2f}, {max_b[0]:.2f}]\n"
            f"  Y Range:      [{min_b[1]:.2f}, {max_b[1]:.2f}]\n"
            f"  Z Range:      [{min_b[2]:.2f}, {max_b[2]:.2f}]\n"
            f"  Center:       ({self.center[0]:.2f}, {self.center[1]:.2f}, {self.center[2]:.2f})"
        )


def convert_disparity_to_depth(
    disparity_map: np.ndarray,
    min_depth: float = 0.5,
    max_depth: float = 8.0,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Convert relative disparity (inverse depth) to metric Euclidean distance.

    In disparity maps, larger values indicate closer objects.
    We apply a reciprocal transformation so that:
        max_disparity -> min_depth (close)
        min_disparity -> max_depth (far)
    """
    disp = np.asarray(disparity_map, dtype=np.float32)
    d_min = float(np.nanmin(disp))
    d_max = float(np.nanmax(disp))

    if d_max - d_min < 1e-6:
        return np.full_like(disp, (min_depth + max_depth) / 2.0)

    # Normalize disparity to [0, 1]
    norm_disp = (disp - d_min) / (d_max - d_min)

    # Inverse mapping: distance = min_depth / (1 - (1 - min/max)*norm_disp)
    # Alternatively: 1 / (norm_disp + eps) scaled between [min_depth, max_depth]
    inv = 1.0 / (norm_disp + 0.1)
    inv_min = 1.0 / (1.0 + 0.1)
    inv_max = 1.0 / (0.0 + 0.1)

    norm_inv = (inv - inv_min) / (inv_max - inv_min)
    depth = min_depth + (max_depth - min_depth) * norm_inv
    return depth.astype(np.float32)


def backproject_depth_to_points(
    rays: np.ndarray,
    depth_map: np.ndarray,
    colors: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 50.0,
    depth_edge_threshold: float = 0.0,
    camera_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PointCloud:
    """
    Vectorized backprojection of depth and unit rays to 3D point cloud:
        P = C + d * R

    Args:
        rays: (H, W, 3) float32 unit ray direction vectors.
        depth_map: (H, W) float32 depth values in meters.
        colors: (H, W, 3) uint8 RGB colors.
        min_depth: Minimum valid depth distance for filtering.
        max_depth: Maximum valid depth distance for filtering.
        depth_edge_threshold: Relative depth discontinuity cutoff to eliminate flying edge smear (0 to disable).
        camera_center: (Cx, Cy, Cz) camera position in world space.

    Returns:
        PointCloud: Cleaned 3D point cloud with invalid depths and smear edges filtered out.
    """
    H, W, _ = rays.shape
    C = np.array(camera_center, dtype=np.float32).reshape(1, 1, 3)

    # Compute 3D position for all pixels: P = C + d * R
    d = depth_map[:, :, np.newaxis]
    P = C + d * rays

    # Calculate depth discontinuity edge mask to eliminate flying pixels and rubber-sheet smears
    if depth_edge_threshold > 0:
        diff_x = np.abs(depth_map[:, 1:] - depth_map[:, :-1])
        denom_x = np.minimum(depth_map[:, 1:], depth_map[:, :-1])
        grad_x = np.zeros_like(depth_map)
        grad_x[:, :-1] = diff_x / np.maximum(denom_x, 1e-4)

        diff_y = np.abs(depth_map[1:, :] - depth_map[:-1, :])
        denom_y = np.minimum(depth_map[1:, :], depth_map[:-1, :])
        grad_y = np.zeros_like(depth_map)
        grad_y[:-1, :] = diff_y / np.maximum(denom_y, 1e-4)

        # 360 panorama seam wrap gradient
        diff_wrap = np.abs(depth_map[:, 0] - depth_map[:, -1])
        denom_wrap = np.minimum(depth_map[:, 0], depth_map[:, -1])
        grad_wrap = diff_wrap / np.maximum(denom_wrap, 1e-4)
        grad_x[:, -1] = grad_wrap

        edge_mask = (grad_x > depth_edge_threshold) | (grad_y > depth_edge_threshold)
    else:
        edge_mask = np.zeros_like(depth_map, dtype=bool)

    # Flatten arrays
    flat_points = P.reshape(-1, 3)
    flat_colors = colors.reshape(-1, 3)
    flat_depth = depth_map.reshape(-1)
    flat_edge = edge_mask.reshape(-1)

    # Filter invalid points (NaN, Inf, out-of-range depths, or flying edge discontinuities)
    valid_mask = (
        np.isfinite(flat_points).all(axis=-1)
        & (flat_depth >= min_depth)
        & (flat_depth <= max_depth)
        & (~flat_edge)
    )

    valid_points = flat_points[valid_mask].astype(np.float32)
    valid_colors = flat_colors[valid_mask].astype(np.uint8)

    return PointCloud(
        points=valid_points,
        colors=valid_colors,
        metadata={
            "total_pixels": H * W,
            "valid_points": int(np.sum(valid_mask)),
            "filtered_edge_points": int(np.sum(flat_edge)),
            "min_depth": min_depth,
            "max_depth": max_depth,
        },
    )


def reconstruct_from_image(
    image_path: Union[str, Path],
    depth_estimator: DepthEstimator,
    camera_type: str = "spherical",
    min_depth: float = 0.5,
    max_depth: float = 8.0,
    depth_edge_threshold: float = 0.0,
    max_resolution: Optional[int] = None,
    output_ply: Optional[Union[str, Path]] = None,
) -> PointCloud:
    """
    Complete end-to-end 3D reconstruction pipeline from a single image/panorama:
    Image -> Depth Estimation -> Ray Generation -> 3D Backprojection -> PointCloud -> PLY
    """
    from panogs.io.ply import write_point_cloud_ply

    logger = get_logger("reconstruction.pointcloud")
    image_path = Path(image_path)

    # 1. Load image
    logger.info(f"Loading image for 3D reconstruction: {image_path.name}")
    img_rgb = load_image_as_numpy(image_path, normalize_float=False, max_resolution=max_resolution)
    H, W = img_rgb.shape[:2]

    # 2. Run depth estimation
    logger.info(f"Estimating depth (image size {W}x{H})...")
    depth_res = depth_estimator.estimate(img_rgb)

    # Convert relative disparity to metric depth if needed
    if depth_res.is_metric:
        metric_depth = np.clip(depth_res.depth_map, min_depth, max_depth)
    else:
        logger.info(f"Converting relative disparity map to metric depth [{min_depth}m, {max_depth}m]...")
        metric_depth = convert_disparity_to_depth(
            depth_res.depth_map, min_depth=min_depth, max_depth=max_depth
        )

    # 3. Generate rays based on camera type
    logger.info(f"Generating camera rays ({camera_type})...")
    if camera_type == "spherical":
        rays = equirectangular_rays(H, W)
    else:
        raise ValueError(f"Unsupported camera type '{camera_type}'. Supported: 'spherical'")

    # 4. Backproject to 3D point cloud with depth discontinuity filtering
    logger.info(f"Backprojecting depth and rays into 3D scene point cloud (edge threshold={depth_edge_threshold})...")
    point_cloud = backproject_depth_to_points(
        rays=rays,
        depth_map=metric_depth,
        colors=img_rgb,
        min_depth=min_depth * 0.9,
        max_depth=max_depth * 1.1,
        depth_edge_threshold=depth_edge_threshold,
    )

    logger.info(f"Reconstructed {point_cloud.num_points:,} 3D points.")

    # 5. Export PLY if output path provided
    if output_ply is not None:
        write_point_cloud_ply(
            output_ply,
            point_cloud.points,
            point_cloud.colors,
            binary=True,
        )
        logger.info(f"Exported point cloud PLY to {output_ply}")

    return point_cloud
