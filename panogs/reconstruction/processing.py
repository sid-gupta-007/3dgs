"""
Point Cloud Processing and Filtering Pipeline.
Provides voxel grid downsampling, statistical outlier removal, and surface normal estimation.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

from panogs.core.logging import get_logger
from panogs.reconstruction.pointcloud import PointCloud


def voxel_downsample(
    point_cloud: PointCloud,
    voxel_size: float = 0.04,
) -> PointCloud:
    """
    Uniformly downsample a point cloud using an integer 3D voxel grid.
    Computes exact centroid positions and averaged RGB colors within each voxel cell.

    Args:
        point_cloud: Input PointCloud.
        voxel_size: Edge length of each cubic voxel in meters (default: 0.04m / 4cm).

    Returns:
        PointCloud: Downsampled point cloud with uniform spatial density.
    """
    if voxel_size <= 0.0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    if point_cloud.num_points == 0:
        return point_cloud

    points = point_cloud.points
    colors = point_cloud.colors

    # Discretize coordinates into integer voxel cell indices
    voxel_coords = np.floor(points / voxel_size).astype(np.int64)

    # Find unique voxels and inverse index mapping
    _, unique_indices, inverse_indices, counts = np.unique(
        voxel_coords, axis=0, return_index=True, return_inverse=True, return_counts=True
    )

    num_voxels = len(unique_indices)

    # Vectorized centroid position accumulation
    pos_accum = np.zeros((num_voxels, 3), dtype=np.float64)
    np.add.at(pos_accum, inverse_indices, points.astype(np.float64))
    centroid_points = (pos_accum / counts[:, None]).astype(np.float32)

    # Vectorized mean color accumulation
    col_accum = np.zeros((num_voxels, 3), dtype=np.float64)
    np.add.at(col_accum, inverse_indices, colors.astype(np.float64))
    mean_colors = np.clip(np.round(col_accum / counts[:, None]), 0, 255).astype(np.uint8)

    return PointCloud(
        points=centroid_points,
        colors=mean_colors,
        metadata={
            **point_cloud.metadata,
            "voxel_size": voxel_size,
            "original_points": point_cloud.num_points,
            "downsampled_points": num_voxels,
        },
    )


def remove_statistical_outliers(
    point_cloud: PointCloud,
    k_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> PointCloud:
    """
    Remove sparse outlier points using Statistical Outlier Removal (SOR).

    For each point, computes the average Euclidean distance to its k nearest neighbors.
    Points with mean distance > (global_mean + std_ratio * global_std) are discarded.

    Args:
        point_cloud: Input PointCloud.
        k_neighbors: Number of nearest neighbors to query (default: 20).
        std_ratio: Standard deviation multiplier threshold (default: 2.0).

    Returns:
        PointCloud: Cleaned point cloud without isolated noise artifacts.
    """
    if point_cloud.num_points <= k_neighbors:
        return point_cloud

    from scipy.spatial import cKDTree

    points = point_cloud.points
    tree = cKDTree(points)

    # Query (k+1) neighbors since point 0 is the query point itself
    distances, _ = tree.query(points, k=k_neighbors + 1)
    mean_dists = np.mean(distances[:, 1:], axis=1)

    mu = float(np.mean(mean_dists))
    sigma = float(np.std(mean_dists))
    threshold = mu + std_ratio * sigma

    inlier_mask = mean_dists <= threshold

    filtered_points = points[inlier_mask]
    filtered_colors = point_cloud.colors[inlier_mask]
    filtered_normals = point_cloud.normals[inlier_mask] if point_cloud.normals is not None else None

    return PointCloud(
        points=filtered_points,
        colors=filtered_colors,
        normals=filtered_normals,
        metadata={
            **point_cloud.metadata,
            "sor_k_neighbors": k_neighbors,
            "sor_std_ratio": std_ratio,
            "sor_dropped_points": int(np.sum(~inlier_mask)),
        },
    )


def estimate_surface_normals(
    point_cloud: PointCloud,
    k_neighbors: int = 15,
    camera_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PointCloud:
    """
    Estimate unit surface normals for each point using Principal Component Analysis (PCA)
    over local k-nearest neighbor covariance matrices.
    Normals are consistently oriented towards the camera center:
        n . (C - p) > 0

    Args:
        point_cloud: Input PointCloud.
        k_neighbors: Neighborhood size for local plane fitting (default: 15).
        camera_center: (Cx, Cy, Cz) position of the camera in world coordinates.

    Returns:
        PointCloud: PointCloud with (N, 3) float32 surface normals.
    """
    if point_cloud.num_points < 3:
        return point_cloud

    from scipy.spatial import cKDTree

    points = point_cloud.points
    N = points.shape[0]
    k = min(k_neighbors, N)

    tree = cKDTree(points)
    _, neighbor_idx = tree.query(points, k=k)

    # Gather neighbor clusters: shape (N, k, 3)
    clusters = points[neighbor_idx]

    # Center clusters: X = P_neighbors - mean(P_neighbors)
    cluster_means = np.mean(clusters, axis=1, keepdims=True)
    centered = clusters - cluster_means

    # Covariance matrices: C = X^T X / k, shape (N, 3, 3)
    # Using batched matrix multiplication: (N, 3, k) @ (N, k, 3)
    cov = np.matmul(centered.transpose(0, 2, 1), centered) / float(k)

    # Compute eigenvalues & eigenvectors: eigh returns ascending eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Smallest eigenvalue corresponds to index 0 (the surface normal direction)
    normals = eigenvectors[:, :, 0].astype(np.float32)

    # Normalize vectors
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = normals / norms

    # Orient normals towards camera center (C - p)
    C = np.array(camera_center, dtype=np.float32).reshape(1, 3)
    view_dirs = C - points
    dot_products = np.sum(normals * view_dirs, axis=-1, keepdims=True)
    flip_mask = (dot_products < 0.0).flatten()
    normals[flip_mask] = -normals[flip_mask]

    return PointCloud(
        points=points,
        colors=point_cloud.colors,
        normals=normals,
        confidence=point_cloud.confidence,
        metadata={
            **point_cloud.metadata,
            "normals_estimated": True,
            "normals_k_neighbors": k,
        },
    )


def process_point_cloud(
    point_cloud: PointCloud,
    voxel_size: Optional[float] = 0.04,
    remove_outliers: bool = True,
    k_outliers: int = 20,
    std_outliers: float = 2.0,
    compute_normals: bool = True,
    k_normals: int = 15,
    camera_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PointCloud:
    """
    Unified sequential point cloud processing pipeline:
        1. Voxel downsampling (uniform spatial reduction)
        2. Statistical outlier removal (noise suppression)
        3. Local PCA surface normal estimation
    """
    logger = get_logger("reconstruction.processing")
    pc = point_cloud

    logger.info(f"Starting point cloud processing on {pc.num_points:,} points...")

    # Step 1: Voxel Downsampling
    if voxel_size is not None and voxel_size > 0.0:
        logger.info(f"Applying voxel downsampling (voxel_size = {voxel_size}m)...")
        pc = voxel_downsample(pc, voxel_size=voxel_size)
        logger.info(f"Voxel downsampling complete: {pc.num_points:,} points remaining.")

    # Step 2: Statistical Outlier Removal
    if remove_outliers and pc.num_points > k_outliers:
        logger.info(f"Applying statistical outlier removal (k={k_outliers}, std_ratio={std_outliers})...")
        pc = remove_statistical_outliers(pc, k_neighbors=k_outliers, std_ratio=std_outliers)
        logger.info(f"Outlier removal complete: {pc.num_points:,} points remaining.")

    # Step 3: Normal Estimation
    if compute_normals and pc.num_points >= 3:
        logger.info(f"Estimating surface normals (k={k_normals})...")
        pc = estimate_surface_normals(pc, k_neighbors=k_normals, camera_center=camera_center)
        logger.info("Surface normal estimation complete.")

    return pc
