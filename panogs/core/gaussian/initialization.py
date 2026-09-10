"""
3D Gaussian Initialization from Point Clouds.
Computes adaptive spatial scale per Gaussian from local k-nearest neighbor spacing.
"""

from typing import Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree

from panogs.core.gaussian.model import GaussianModel, logit, rgb_to_sh0
from panogs.core.logging import get_logger
from panogs.reconstruction.pointcloud import PointCloud


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert (N, 3, 3) rotation matrices to (N, 4) unit quaternions (qw, qx, qy, qz).
    """
    N = R.shape[0]
    quats = np.zeros((N, 4), dtype=np.float32)

    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Standard Shepperd's algorithm or vectorized branchless quaternion conversion
    for i in range(N):
        r = R[i]
        t = tr[i]
        if t > 0.0:
            S = np.sqrt(t + 1.0) * 2.0
            qw = 0.25 * S
            qx = (r[2, 1] - r[1, 2]) / S
            qy = (r[0, 2] - r[2, 0]) / S
            qz = (r[1, 0] - r[0, 1]) / S
        elif (r[0, 0] > r[1, 1]) and (r[0, 0] > r[2, 2]):
            S = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            qw = (r[2, 1] - r[1, 2]) / S
            qx = 0.25 * S
            qy = (r[0, 1] + r[1, 0]) / S
            qz = (r[0, 2] + r[2, 0]) / S
        elif r[1, 1] > r[2, 2]:
            S = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            qw = (r[0, 2] - r[2, 0]) / S
            qx = (r[0, 1] + r[1, 0]) / S
            qy = 0.25 * S
            qz = (r[1, 2] + r[2, 1]) / S
        else:
            S = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            qw = (r[1, 0] - r[0, 1]) / S
            qx = (r[0, 2] + r[2, 0]) / S
            qy = (r[1, 2] + r[2, 1]) / S
            qz = 0.25 * S
        quats[i] = [qw, qx, qy, qz]

    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return quats / norms


def initialize_from_pointcloud(
    point_cloud: PointCloud,
    default_opacity: float = 0.85,
    k_scale_neighbors: int = 3,
    scale_multiplier: float = 1.0,
    min_scale: float = 1e-4,
    max_scale: float = 10.0,
) -> GaussianModel:
    """
    Initialize a 3D Gaussian Splatting scene model from a PointCloud.

    - Positions: Directly set from point cloud XYZ.
    - Scales: Anisotropic scales (sx, sy, sz) where sx, sy cover surface tangent spacing
              and sz is thin normal thickness for flat surfels.
    - Rotations: Quaternions aligned to surface normals (or identity if normals absent).
    - Opacity: Initialized to default_opacity in logit space.
    - Colors: RGB [0, 255] converted to float [0, 1] and mapped to zeroth-order spherical harmonics (SH0).

    Args:
        point_cloud: Input PointCloud.
        default_opacity: Initial opacity value in (0.0, 1.0) (default: 0.85).
        k_scale_neighbors: Number of nearest neighbors to query for adaptive scale (default: 3).
        scale_multiplier: Global scale multiplier (default: 1.0).
        min_scale: Minimum scale clamp in meters (default: 1e-4).
        max_scale: Maximum scale clamp in meters (default: 10.0).

    Returns:
        GaussianModel: Initialized 3D Gaussian model.
    """
    logger = get_logger("gaussian.init")
    N = point_cloud.num_points

    if N == 0:
        raise ValueError("Cannot initialize GaussianModel from an empty PointCloud.")

    logger.info(f"Initializing {N:,} 3D Gaussians from point cloud...")

    # 1. Positions: (N, 3)
    xyz = point_cloud.points.copy()

    # 2. Adaptive Scales from k-NN spacing
    logger.info(f"Computing adaptive scales using k={k_scale_neighbors} nearest neighbors...")
    tree = cKDTree(xyz)
    distances, _ = tree.query(xyz, k=min(k_scale_neighbors + 1, N))

    if distances.shape[1] > 1:
        mean_dists = np.mean(distances[:, 1:], axis=1)
    else:
        mean_dists = np.full(N, 0.05, dtype=np.float32)

    # Avoid zero distances for co-located points
    mean_dists[mean_dists <= 0.0] = 0.01

    adaptive_scales = np.clip(mean_dists * scale_multiplier, min_scale, max_scale).astype(np.float32)

    # 3. Orientations and Anisotropic Scales from Surface Normals
    if point_cloud.normals is not None and len(point_cloud.normals) == N:
        logger.info("Aligning 3D Gaussian orientations and tangential scales with surface normals...")
        normals = point_cloud.normals.astype(np.float32)
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        normals = normals / norms

        # Build orthonormal tangent frame [t1, t2, n]
        # Choose helper vector 'a' not parallel to normal
        helper = np.zeros_like(normals)
        is_y_dominant = np.abs(normals[:, 1]) > 0.9
        helper[is_y_dominant, 0] = 1.0   # (1, 0, 0)
        helper[~is_y_dominant, 1] = 1.0  # (0, 1, 0)

        t1 = np.cross(helper, normals)
        t1_norm = np.linalg.norm(t1, axis=-1, keepdims=True)
        t1_norm[t1_norm == 0] = 1.0
        t1 = t1 / t1_norm

        t2 = np.cross(normals, t1)
        t2_norm = np.linalg.norm(t2, axis=-1, keepdims=True)
        t2_norm[t2_norm == 0] = 1.0
        t2 = t2 / t2_norm

        # R matrix with columns [t1, t2, n]
        R = np.stack([t1, t2, normals], axis=-1)
        rotation_quats = rotation_matrix_to_quaternion(R)

        # Anisotropic scales: sx, sy are tangential to surface (1.2x spacing to cover gaps),
        # sz is thin normal thickness (0.15x) so splats hug walls/floors flatly
        sx = adaptive_scales * 1.25
        sy = adaptive_scales * 1.25
        sz = np.clip(adaptive_scales * 0.15, min_scale, max_scale)
        scales = np.stack([sx, sy, sz], axis=-1).astype(np.float32)
    else:
        # Isotropic initial scales (s, s, s)
        scales = np.stack([adaptive_scales] * 3, axis=-1).astype(np.float32)
        rotation_quats = np.zeros((N, 4), dtype=np.float32)
        rotation_quats[:, 0] = 1.0  # Identity quaternion (qw=1, qx=0, qy=0, qz=0)

    scaling_log = np.log(np.clip(scales, 1e-6, 100.0))

    # 4. Opacity: (N, 1) in logit space
    init_logit = logit(np.array([default_opacity]))
    opacity_logits = np.full((N, 1), init_logit, dtype=np.float32)

    # 5. Colors: convert uint8 RGB [0, 255] -> float [0, 1] -> SH0
    rgb_float = point_cloud.colors.astype(np.float32) / 255.0
    features_dc = rgb_to_sh0(rgb_float)

    logger.info(
        f"Initialization complete: {N:,} Gaussians, scale mean={np.mean(adaptive_scales):.4f}m, opacity={default_opacity:.2f}."
    )

    return GaussianModel(
        xyz=xyz,
        scaling_log=scaling_log,
        rotation_quats=rotation_quats,
        opacity_logits=opacity_logits,
        features_dc=features_dc,
    )
