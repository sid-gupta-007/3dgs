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


def estimate_dominant_light_direction(point_cloud: PointCloud) -> np.ndarray:
    """
    Estimate the primary directional light vector in world coordinates
    from the brightest highlights in the scene point cloud colors and positions.
    Defaults to overhead ceiling illumination [0.0, 1.0, 0.0] if unconstrained.
    """
    if point_cloud.colors is not None and len(point_cloud.colors) > 0:
        rgb = point_cloud.colors.astype(np.float32)
        luminance = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        bright_cutoff = np.percentile(luminance, 97)
        bright_indices = np.where(luminance >= bright_cutoff)[0]

        if len(bright_indices) >= 10:
            bright_pts = point_cloud.points[bright_indices]
            mean_pos = np.mean(bright_pts, axis=0)
            norm = np.linalg.norm(mean_pos)
            if norm > 1e-3:
                return (mean_pos / norm).astype(np.float32)

    return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def initialize_from_pointcloud(
    point_cloud: PointCloud,
    default_opacity: float = 0.95,
    k_scale_neighbors: int = 8,
    scale_multiplier: float = 1.10,
    min_scale: float = 1e-4,
    max_scale: float = 10.0,
    splat_shape: str = "hybrid",
    sharpness: float = 0.5,
) -> GaussianModel:
    """
    Initialize a 3D Gaussian Splatting scene model from a PointCloud.

    - Positions: Directly set from point cloud XYZ.
    - Scales: Continuous surfel discs and light-aligned pointy needles.
    - Rotations: Quaternions aligned to surface normals and dominant light directions.
    - Opacity: Initialized to default_opacity in logit space.
    - Colors: RGB [0, 255] converted to float [0, 1] and mapped to zeroth-order spherical harmonics (SH0).
    """
    logger = get_logger("gaussian.init")
    N = point_cloud.num_points

    if N == 0:
        raise ValueError("Cannot initialize GaussianModel from an empty PointCloud.")

    sharpness = float(np.clip(sharpness, 0.0, 1.0))
    logger.info(f"Initializing {N:,} 3D Gaussians (shape='{splat_shape}', scale={scale_multiplier:.2f}) from point cloud...")

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

    # For structured panorama point clouds: cap scale by ray angular footprint
    if point_cloud.metadata and "total_pixels" in point_cloud.metadata:
        point_depths = np.linalg.norm(xyz, axis=-1)
        total_pixels = point_cloud.metadata["total_pixels"]
        angular_pixel = (2.0 * np.pi) / max(1, int(np.sqrt(total_pixels) * 2))
        max_angular_scale = point_depths * angular_pixel * 1.30 * scale_multiplier
        adaptive_scales = np.minimum(adaptive_scales, np.maximum(max_angular_scale, min_scale).astype(np.float32))

    logger.info(
        f"Scale stats: median={median_scale:.5f}m, outlier_cap={outlier_cap:.5f}m, "
        f"after_clip: mean={np.mean(adaptive_scales):.5f}m, max={np.max(adaptive_scales):.5f}m"
    )

    # 3. Orientations and Anisotropic Scales from Surface Normals & Light Flow
    shape_mode = splat_shape.lower().strip()
    if point_cloud.normals is not None and len(point_cloud.normals) == N and shape_mode != "isotropic":
        logger.info(f"Aligning 3D Gaussian orientations and '{shape_mode}' scales with surface normals...")
        normals = point_cloud.normals.astype(np.float32)
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        normals = normals / norms

        # Build continuous orthonormal tangent frame [t1, t2, normals] using Duff et al. (JCGT 2017)
        nx = normals[:, 0]
        ny = normals[:, 1]
        nz = normals[:, 2]

        sign = np.where(nz >= 0.0, 1.0, -1.0).astype(np.float32)
        a = -1.0 / (sign + nz + 1e-7)
        b = nx * ny * a

        t1_x = 1.0 + sign * nx * nx * a
        t1_y = sign * b
        t1_z = -sign * nx
        t1 = np.stack([t1_x, t1_y, t1_z], axis=-1)

        t2_x = b
        t2_y = sign + ny * ny * a
        t2_z = -ny
        t2 = np.stack([t2_x, t2_y, t2_z], axis=-1)

        # Light-source-aware tangent elongation: align t1 with projected light vector
        L = estimate_dominant_light_direction(point_cloud)
        dot_LN = np.sum(normals * L, axis=-1, keepdims=True)
        t_light = L - dot_LN * normals
        t_light_len = np.linalg.norm(t_light, axis=-1, keepdims=True)
        use_light_t = (t_light_len > 0.05)[:, 0]

        t1_final = t1.copy()
        t1_final[use_light_t] = (t_light[use_light_t] / t_light_len[use_light_t]).astype(np.float32)

        t2_final = np.cross(normals, t1_final)
        t2_final_norm = np.linalg.norm(t2_final, axis=-1, keepdims=True)
        t2_final_norm[t2_final_norm == 0] = 1.0
        t2_final = (t2_final / t2_final_norm).astype(np.float32)

        # R matrix with columns [t1_final, t2_final, normals]
        R = np.stack([t1_final, t2_final, normals], axis=-1)
        rotation_quats = rotation_matrix_to_quaternion(R)

        edge_mask = point_cloud.metadata.get("edge_mask", None) if point_cloud.metadata else None
        is_edge = np.asarray(edge_mask, dtype=bool) if (edge_mask is not None and len(edge_mask) == N) else np.zeros(N, dtype=bool)

        if shape_mode in ("cylindrical", "needle", "pointy", "light_pointy"):
            # Pointy needles aligned with light source reflection vectors
            sx = adaptive_scales * 1.55
            sy = np.clip(adaptive_scales * 0.45, min_scale, max_scale)
            sz = np.clip(adaptive_scales * 0.08, min_scale, max_scale)
        elif shape_mode == "surfel":
            # Flat planar surfel discs with continuous overlap
            sx = adaptive_scales * 1.15
            sy = adaptive_scales * 1.15
            sz = np.clip(adaptive_scales * 0.08, min_scale, max_scale)
        else:
            # Hybrid: continuous surfel discs for flat walls/floors + pointy needles for edges & highlights
            sx = adaptive_scales * 1.15
            sy = adaptive_scales * 1.15
            sz = np.clip(adaptive_scales * 0.08, min_scale, max_scale)

            # Pointy cylindrical needles at edge contours and specular highlights
            if np.any(is_edge):
                sx[is_edge] = adaptive_scales[is_edge] * 1.45
                sy[is_edge] = np.clip(adaptive_scales[is_edge] * 0.35, min_scale, max_scale)
                sz[is_edge] = np.clip(adaptive_scales[is_edge] * 0.05, min_scale, max_scale)

        scales = np.stack([sx, sy, sz], axis=-1).astype(np.float32)
    else:
        # Isotropic initial scales (s, s, s)
        scales = np.stack([adaptive_scales * 1.15] * 3, axis=-1).astype(np.float32)
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
