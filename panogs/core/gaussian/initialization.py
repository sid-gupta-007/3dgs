from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree

from panogs.core.gaussian.model import GaussianModel, logit, rgb_to_sh0
from panogs.core.logging import get_logger

if TYPE_CHECKING:
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
    default_opacity: float = 0.92,
    k_scale_neighbors: int = 8,
    scale_multiplier: float = 0.65,
    min_scale: float = 1e-4,
    max_scale: float = 10.0,
    splat_shape: str = "hybrid",
    sharpness: float = 0.7,
) -> GaussianModel:
    """
    Initialize a 3D Gaussian Splatting scene model from a PointCloud.

    - Positions: Directly set from point cloud XYZ.
    - Scales: Geometric anisotropic shapes:
        * 'hybrid': Planar surfel discs for flat walls/floor + pointy cylindrical needles for pillars/edges.
        * 'cylindrical' / 'needle': Pointy cylindrical elongated splats along surface tangents.
        * 'surfel': Flat planar discs tangent to surface normals.
        * 'isotropic': Classic spherical Gaussians.
    - Rotations: Quaternions aligned to surface normals (or identity if normals absent).
    - Opacity: Initialized to default_opacity in logit space.
    - Colors: RGB [0, 255] converted to float [0, 1] and mapped to zeroth-order spherical harmonics (SH0).
    - sharpness: 0.0 = legacy soft splats, 1.0 = ultra-tight splats (default 0.7).
    """
    logger = get_logger("gaussian.init")
    N = point_cloud.num_points

    if N == 0:
        raise ValueError("Cannot initialize GaussianModel from an empty PointCloud.")

    sharpness = float(np.clip(sharpness, 0.0, 1.0))
    logger.info(f"Initializing {N:,} 3D Gaussians (shape='{splat_shape}', sharpness={sharpness:.2f}) from point cloud...")

    # 1. Positions: (N, 3)
    xyz = point_cloud.points.copy()

    # 2. Adaptive Scales with k-NN, outlier capping, and Angular Ray Footprint Capping
    logger.info(f"Computing adaptive scales for {N:,} points...")
    tree = cKDTree(xyz)
    k_query = min(k_scale_neighbors + 1, N)
    distances, _ = tree.query(xyz, k=k_query)

    if distances.shape[1] > 1:
        mean_dists = np.mean(distances[:, 1:], axis=1)
        base_scales = mean_dists * scale_multiplier
    else:
        base_scales = np.full(N, 0.05, dtype=np.float32)

    # Avoid zero distances for co-located points
    base_scales[base_scales <= 0.0] = 0.01

    # Cap outlier k-NN distances at 3× median to prevent blown-out splats
    median_scale = float(np.median(base_scales))
    outlier_cap = median_scale * 3.0
    base_scales = np.minimum(base_scales, outlier_cap)

    adaptive_scales = np.clip(base_scales, min_scale, max_scale).astype(np.float32)

    # For structured panorama point clouds: cap scale by ray angular footprint to prevent blur across depth steps
    if point_cloud.metadata and "total_pixels" in point_cloud.metadata:
        point_depths = np.linalg.norm(xyz, axis=-1)
        total_pixels = point_cloud.metadata["total_pixels"]
        angular_pixel = (2.0 * np.pi) / max(1, int(np.sqrt(total_pixels) * 2))
        # Tighter angular cap: reduce from 1.05 to lerp(1.05, 0.75, sharpness)
        angular_mult = 1.05 * (1.0 - sharpness) + 0.75 * sharpness
        max_angular_scale = point_depths * angular_pixel * angular_mult * scale_multiplier
        adaptive_scales = np.minimum(adaptive_scales, np.maximum(max_angular_scale, min_scale).astype(np.float32))

    logger.info(
        f"Scale stats: median={median_scale:.5f}m, outlier_cap={outlier_cap:.5f}m, "
        f"after_clip: mean={np.mean(adaptive_scales):.5f}m, max={np.max(adaptive_scales):.5f}m"
    )

    # 3. Orientations and Anisotropic Scales from Surface Normals
    shape_mode = splat_shape.lower().strip()
    if point_cloud.normals is not None and len(point_cloud.normals) == N and shape_mode != "isotropic":
        logger.info(f"Aligning 3D Gaussian orientations and '{shape_mode}' scales with surface normals...")
        normals = point_cloud.normals.astype(np.float32)
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        normals = normals / norms

        # Build orthonormal tangent frame [t1, t2, n]
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

        # R matrix with columns [t1, t2, normals]
        R = np.stack([t1, t2, normals], axis=-1)
        rotation_quats = rotation_matrix_to_quaternion(R)

        edge_mask = point_cloud.metadata.get("edge_mask", None) if point_cloud.metadata else None
        is_edge = np.asarray(edge_mask, dtype=bool) if (edge_mask is not None and len(edge_mask) == N) else np.zeros(N, dtype=bool)

        # Sharpness-aware scale factors: lerp between old (soft) and new (tight) values
        # Old factors: sx=1.02, sy=1.02   New factors: sx=0.72, sy=0.72
        # Old surfel: sx=1.05, sy=1.05    New surfel: sx=0.78, sy=0.78
        def _lerp(soft, tight):
            return soft * (1.0 - sharpness) + tight * sharpness

        if shape_mode in ("cylindrical", "needle", "pointy"):
            # Needle / Cylindrical splats: elongated along tangent t1, tight along t2 and normal
            sx = adaptive_scales * _lerp(1.80, 1.30)
            sy = np.clip(adaptive_scales * _lerp(0.35, 0.20), min_scale, max_scale)
            sz = np.clip(adaptive_scales * _lerp(0.12, 0.06), min_scale, max_scale)
        elif shape_mode == "surfel":
            # Flat planar surfel discs
            sx = adaptive_scales * _lerp(1.05, 0.78)
            sy = adaptive_scales * _lerp(1.05, 0.78)
            sz = np.clip(adaptive_scales * _lerp(0.12, 0.05), min_scale, max_scale)
        else:
            # Hybrid: surfel discs for flat walls/floors + pointy cylindrical needles for pillars/edges
            sx = adaptive_scales * _lerp(1.02, 0.72)
            sy = adaptive_scales * _lerp(1.02, 0.72)
            sz = np.clip(adaptive_scales * _lerp(0.15, 0.06), min_scale, max_scale)

            # Pointy cylindrical needles at edge contours and pillars
            if np.any(is_edge):
                sx[is_edge] = adaptive_scales[is_edge] * _lerp(1.60, 1.10)
                sy[is_edge] = np.clip(adaptive_scales[is_edge] * _lerp(0.30, 0.15), min_scale, max_scale)
                sz[is_edge] = np.clip(adaptive_scales[is_edge] * _lerp(0.05, 0.02), min_scale, max_scale)

        scales = np.stack([sx, sy, sz], axis=-1).astype(np.float32)
    else:
        # Isotropic initial scales (s, s, s)
        iso_factor = 1.0 * (1.0 - sharpness) + 0.70 * sharpness
        scales = np.stack([adaptive_scales * iso_factor] * 3, axis=-1).astype(np.float32)
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
