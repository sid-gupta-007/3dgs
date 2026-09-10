"""
PLY (Polygon File Format / Stanford Triangle Format) I/O module.
Supports high-speed binary and ASCII point cloud export/import with positions and colors.
"""

from pathlib import Path
from typing import Any, Optional, Tuple, Union
import numpy as np


def write_point_cloud_ply(
    file_path: Union[str, Path],
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    binary: bool = True,
) -> Path:
    """
    Write a 3D point cloud with optional RGB colors and normals to a .ply file.
    """
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points array must have shape (N, 3), got {points.shape}")

    num_vertices = points.shape[0]

    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape[0] != num_vertices or colors.shape[1] != 3:
            raise ValueError(
                f"Colors array shape {colors.shape} does not match points count {num_vertices}"
            )
        if np.issubdtype(colors.dtype, np.floating):
            colors = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            colors = np.clip(colors, 0, 255).astype(np.uint8)
    else:
        colors = np.full((num_vertices, 3), 255, dtype=np.uint8)

    has_normals = normals is not None
    if has_normals:
        normals = np.asarray(normals, dtype=np.float32)
        if normals.shape != (num_vertices, 3):
            raise ValueError(f"Normals array shape {normals.shape} does not match points {num_vertices}")

    # Construct PLY Header
    fmt_str = "binary_little_endian 1.0" if binary else "ascii 1.0"
    header_lines = [
        "ply",
        f"format {fmt_str}",
        f"comment PanoGS Point Cloud Generator",
        f"element vertex {num_vertices}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_normals:
        header_lines.extend([
            "property float nx",
            "property float ny",
            "property float nz",
        ])
    header_lines.extend([
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header\n",
    ])
    header = "\n".join(header_lines).encode("ascii")

    if binary:
        # Create structured numpy array for zero-copy binary serialization
        fields = [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
        ]
        if has_normals:
            fields.extend([
                ("nx", "<f4"),
                ("ny", "<f4"),
                ("nz", "<f4"),
            ])
        fields.extend([
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ])
        vertex_dtype = np.dtype(fields)
        vertex_data = np.empty(num_vertices, dtype=vertex_dtype)
        vertex_data["x"] = points[:, 0]
        vertex_data["y"] = points[:, 1]
        vertex_data["z"] = points[:, 2]
        if has_normals:
            vertex_data["nx"] = normals[:, 0]
            vertex_data["ny"] = normals[:, 1]
            vertex_data["nz"] = normals[:, 2]
        vertex_data["red"] = colors[:, 0]
        vertex_data["green"] = colors[:, 1]
        vertex_data["blue"] = colors[:, 2]

        with open(target, "wb") as f:
            f.write(header)
            vertex_data.tofile(f)
    else:
        with open(target, "w", encoding="ascii") as f:
            f.write("\n".join(header_lines))
            for i in range(num_vertices):
                x, y, z = points[i]
                r, g, b = colors[i]
                if has_normals:
                    nx, ny, nz = normals[i]
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {nx:.6f} {ny:.6f} {nz:.6f} {int(r)} {int(g)} {int(b)}\n")
                else:
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

    return target


def write_pointcloud(file_path: Union[str, Path], point_cloud: Any, binary: bool = True) -> Path:
    """Convenience helper to write a PointCloud instance to PLY."""
    return write_point_cloud_ply(
        file_path=file_path,
        points=point_cloud.points,
        colors=point_cloud.colors,
        normals=point_cloud.normals,
        binary=binary,
    )


def read_point_cloud_ply(
    file_path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a point cloud .ply file and return (points, colors).

    Args:
        file_path: Path to .ply file.

    Returns:
        Tuple: (points: np.ndarray (N, 3) float32, colors: np.ndarray (N, 3) uint8)
    """
    target = Path(file_path)
    if not target.exists():
        raise FileNotFoundError(f"PLY file not found: {target}")

    with open(target, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # Parse header properties
        num_vertices = 0
        is_binary = False
        properties = []
        for line in header_lines:
            if line.startswith("format binary_little_endian"):
                is_binary = True
            elif line.startswith("element vertex"):
                num_vertices = int(line.split()[2])
            elif line.startswith("property"):
                parts = line.split()
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_type, prop_name))

        if num_vertices == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

        if is_binary:
            field_map = {
                "float": "<f4",
                "float32": "<f4",
                "double": "<f8",
                "uchar": "u1",
                "uint8": "u1",
                "int": "<i4",
                "uint": "<u4",
            }
            dtype_fields = []
            for p_type, p_name in properties:
                np_fmt = field_map.get(p_type.lower(), "<f4")
                dtype_fields.append((p_name, np_fmt))

            vertex_dtype = np.dtype(dtype_fields)
            data = np.fromfile(f, dtype=vertex_dtype, count=num_vertices)
            points = np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float32)
            if "red" in data.dtype.names and "green" in data.dtype.names and "blue" in data.dtype.names:
                colors = np.stack([data["red"], data["green"], data["blue"]], axis=-1).astype(np.uint8)
            else:
                colors = np.full((num_vertices, 3), 255, dtype=np.uint8)
            return points, colors
        else:
            # ASCII parsing
            content = f.read().decode("ascii").strip().splitlines()
            points = []
            colors = []
            for row in content[:num_vertices]:
                tokens = row.split()
                points.append([float(tokens[0]), float(tokens[1]), float(tokens[2])])
                if len(tokens) >= 6:
                    colors.append([int(tokens[-3]), int(tokens[-2]), int(tokens[-1])])
                else:
                    colors.append([255, 255, 255])
            return np.array(points, dtype=np.float32), np.array(colors, dtype=np.uint8)
