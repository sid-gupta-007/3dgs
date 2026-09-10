"""Tests for point cloud processing: voxel downsampling, statistical outlier removal, and normal estimation."""

from pathlib import Path
import numpy as np
import pytest
from panogs.reconstruction.pointcloud import PointCloud
from panogs.reconstruction.processing import (
    estimate_surface_normals,
    process_point_cloud,
    remove_statistical_outliers,
    voxel_downsample,
)


def test_voxel_downsample_clustering():
    # 8 points in a 0.01m cube, voxel_size = 0.05m -> should downsample to 1 point
    points = np.random.uniform(0.0, 0.01, (8, 3)).astype(np.float32)
    colors = np.full((8, 3), 100, dtype=np.uint8)
    pc = PointCloud(points=points, colors=colors)

    downsampled = voxel_downsample(pc, voxel_size=0.05)
    assert downsampled.num_points == 1
    assert np.allclose(downsampled.points[0], np.mean(points, axis=0), atol=1e-5)
    assert np.array_equal(downsampled.colors[0], [100, 100, 100])


def test_voxel_downsample_grid_preservation():
    # 2 distinct clusters separated by 1.0m
    cluster1 = np.random.uniform(0.0, 0.05, (20, 3)).astype(np.float32)
    cluster2 = np.random.uniform(1.0, 1.05, (20, 3)).astype(np.float32)
    points = np.vstack([cluster1, cluster2])
    colors = np.full((40, 3), 255, dtype=np.uint8)
    pc = PointCloud(points=points, colors=colors)

    downsampled = voxel_downsample(pc, voxel_size=0.1)
    # Should yield approximately 2 voxel centroids (one for each cluster)
    assert 2 <= downsampled.num_points <= 4


def test_remove_statistical_outliers():
    # Dense cluster of 100 points near (0, 0, 0) + 1 distant isolated noise point at (10, 10, 10)
    cluster = np.random.normal(0.0, 0.1, (100, 3)).astype(np.float32)
    outlier = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
    points = np.vstack([cluster, outlier])
    colors = np.full((101, 3), 200, dtype=np.uint8)
    pc = PointCloud(points=points, colors=colors)

    cleaned = remove_statistical_outliers(pc, k_neighbors=10, std_ratio=1.5)
    # Outlier should be removed
    assert cleaned.num_points == 100
    assert np.max(cleaned.points) < 2.0


def test_estimate_surface_normals_plane():
    # Generate points on horizontal plane Y = 2.0
    x = np.linspace(-1.0, 1.0, 10)
    z = np.linspace(-1.0, 1.0, 10)
    xx, zz = np.meshgrid(x, z)
    yy = np.full_like(xx, 2.0)

    points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1).astype(np.float32)
    colors = np.zeros_like(points, dtype=np.uint8)
    pc = PointCloud(points=points, colors=colors)

    # Camera at (0, 0, 0)
    with_normals = estimate_surface_normals(pc, k_neighbors=8, camera_center=(0.0, 0.0, 0.0))
    assert with_normals.normals is not None
    assert with_normals.normals.shape == points.shape

    # For plane at Y = 2 with camera at Y = 0: normal facing camera should point down along -Y ([0, -1, 0])
    # Interior points should have near-perfect vertical normals
    interior_normals = with_normals.normals.reshape(10, 10, 3)[2:8, 2:8, :].reshape(-1, 3)
    assert np.all(interior_normals[:, 1] < -0.9)  # strongly negative Y component (towards camera)
    assert np.all(np.abs(interior_normals[:, 0]) < 0.15)
    assert np.all(np.abs(interior_normals[:, 2]) < 0.15)


def test_process_point_cloud_pipeline():
    points = np.random.uniform(-1.0, 1.0, (500, 3)).astype(np.float32)
    colors = np.random.randint(0, 255, (500, 3), dtype=np.uint8)
    pc = PointCloud(points=points, colors=colors)

    processed = process_point_cloud(
        pc, voxel_size=0.1, remove_outliers=True, compute_normals=True
    )
    assert processed.num_points < 500
    assert processed.normals is not None
    assert processed.normals.shape == processed.points.shape
