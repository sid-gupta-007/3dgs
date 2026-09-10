"""Tests for PLY point cloud writer and reader."""

from pathlib import Path
import numpy as np
import pytest
from panogs.io.ply import read_point_cloud_ply, write_point_cloud_ply


def test_write_and_read_binary_ply(tmp_path: Path):
    ply_path = tmp_path / "test_binary.ply"
    points = np.array([
        [0.0, 1.0, 2.0],
        [-1.5, 0.5, 3.2],
        [4.0, -2.0, 0.0],
    ], dtype=np.float32)
    colors = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
    ], dtype=np.uint8)

    write_point_cloud_ply(ply_path, points, colors, binary=True)
    assert ply_path.exists()

    read_points, read_colors = read_point_cloud_ply(ply_path)
    assert read_points.shape == (3, 3)
    assert read_colors.shape == (3, 3)
    assert np.allclose(read_points, points, atol=1e-5)
    assert np.array_equal(read_colors, colors)


def test_write_and_read_ascii_ply(tmp_path: Path):
    ply_path = tmp_path / "test_ascii.ply"
    points = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ], dtype=np.float32)
    colors = np.array([
        [100, 150, 200],
        [50, 60, 70],
    ], dtype=np.uint8)

    write_point_cloud_ply(ply_path, points, colors, binary=False)
    assert ply_path.exists()

    read_points, read_colors = read_point_cloud_ply(ply_path)
    assert read_points.shape == (2, 3)
    assert read_colors.shape == (2, 3)
    assert np.allclose(read_points, points, atol=1e-5)
    assert np.array_equal(read_colors, colors)


def test_float_colors_scaling(tmp_path: Path):
    ply_path = tmp_path / "test_float_colors.ply"
    points = np.zeros((1, 3), dtype=np.float32)
    colors = np.array([[1.0, 0.5, 0.0]], dtype=np.float32)  # Float [0, 1]

    write_point_cloud_ply(ply_path, points, colors, binary=True)
    _, read_colors = read_point_cloud_ply(ply_path)
    assert read_colors[0, 0] == 255
    assert abs(int(read_colors[0, 1]) - 127) <= 1
    assert read_colors[0, 2] == 0
