"""
Standard 3D Gaussian Splatting PLY and SPLAT serialization module.
Outputs format compatible with SuperSplat, SIBR, PlayCanvas, and Nerfstudio.
"""

from pathlib import Path
from typing import Union
import numpy as np

from panogs.core.gaussian.model import GaussianModel, logit, rgb_to_sh0, sh0_to_rgb, sigmoid
from panogs.core.logging import get_logger


def save_gaussian_ply(
    file_path: Union[str, Path],
    model: GaussianModel,
) -> Path:
    """
    Save a GaussianModel to standard 3DGS binary PLY format.

    Properties exported:
        - x, y, z: Position
        - nx, ny, nz: Normals (default 0)
        - f_dc_0, f_dc_1, f_dc_2: Zeroth-order spherical harmonics (SH0)
        - opacity: Logit opacity
        - scale_0, scale_1, scale_2: Log scale
        - rot_0, rot_1, rot_2, rot_3: Quaternion (qw, qx, qy, qz)

    Args:
        file_path: Output file path.
        model: GaussianModel instance.

    Returns:
        Path: Output file path.
    """
    logger = get_logger("io.gaussian_ply")
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    N = model.num_gaussians
    xyz = model.get_xyz()
    normals = np.zeros((N, 3), dtype=np.float32)
    f_dc = model.get_features_dc()
    opacity = model._opacity_logits
    scales = model._scaling_log
    rotations = model.get_rotation_quats()

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment PanoGS 3D Gaussian Splatting Exporter",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header\n",
    ]
    header = "\n".join(header_lines).encode("ascii")

    # Structured array definition (all float32)
    dtype_fields = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ]
    data = np.empty(N, dtype=dtype_fields)

    data["x"] = xyz[:, 0]
    data["y"] = xyz[:, 1]
    data["z"] = xyz[:, 2]
    data["nx"] = normals[:, 0]
    data["ny"] = normals[:, 1]
    data["nz"] = normals[:, 2]
    data["f_dc_0"] = f_dc[:, 0]
    data["f_dc_1"] = f_dc[:, 1]
    data["f_dc_2"] = f_dc[:, 2]
    data["opacity"] = opacity[:, 0]
    data["scale_0"] = scales[:, 0]
    data["scale_1"] = scales[:, 1]
    data["scale_2"] = scales[:, 2]
    data["rot_0"] = rotations[:, 0]
    data["rot_1"] = rotations[:, 1]
    data["rot_2"] = rotations[:, 2]
    data["rot_3"] = rotations[:, 3]

    with open(target, "wb") as f:
        f.write(header)
        data.tofile(f)

    logger.info(f"Saved {N:,} 3D Gaussians to {target} ({target.stat().st_size / (1024*1024):.2f} MB)")
    return target


def load_gaussian_ply(file_path: Union[str, Path]) -> GaussianModel:
    """
    Load a 3D Gaussian Splatting scene model from a standard 3DGS binary PLY file.
    """
    target = Path(file_path)
    if not target.exists():
        raise FileNotFoundError(f"Gaussian PLY not found: {target}")

    with open(target, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        num_vertices = 0
        for line in header_lines:
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[2])

        if num_vertices == 0:
            raise ValueError(f"Empty PLY file: {target}")

        dtype_fields = [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
            ("opacity", "<f4"),
            ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
            ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
        ]

        data = np.fromfile(f, dtype=dtype_fields, count=num_vertices)

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=-1)
    f_dc = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=-1)
    opacity = data["opacity"][:, np.newaxis]
    scales = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=-1)
    rotations = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=-1)

    return GaussianModel(
        xyz=xyz,
        scaling_log=scales,
        rotation_quats=rotations,
        opacity_logits=opacity,
        features_dc=f_dc,
    )


def save_gaussian_splat(
    file_path: Union[str, Path],
    model: GaussianModel,
) -> Path:
    """
    Save Gaussians in the 32-byte .splat binary format:
        - Position: 3 x float32 (12 bytes)
        - Scale: 3 x float32 (12 bytes)
        - Color RGBA: 4 x uint8 (4 bytes)
        - Rotation: 4 x uint8 (4 bytes, mapped from [-1, 1] to [0, 255])
    """
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    N = model.num_gaussians
    xyz = model.get_xyz().astype(np.float32)
    scales = model.get_scaling().astype(np.float32)
    rgb = model.get_rgb_uint8()
    alpha = (model.get_opacity().flatten() * 255.0).astype(np.uint8)[:, np.newaxis]
    rgba = np.hstack([rgb, alpha])

    quats = model.get_rotation_quats()
    # Pack [-1.0, 1.0] quaternion into [0, 255] uint8
    quats_u8 = np.clip(np.round((quats * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)

    splat_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("sx", "<f4"), ("sy", "<f4"), ("sz", "<f4"),
        ("rgba", "u1", (4,)),
        ("rot", "u1", (4,)),
    ])
    splat_data = np.empty(N, dtype=splat_dtype)
    splat_data["x"] = xyz[:, 0]
    splat_data["y"] = xyz[:, 1]
    splat_data["z"] = xyz[:, 2]
    splat_data["sx"] = scales[:, 0]
    splat_data["sy"] = scales[:, 1]
    splat_data["sz"] = scales[:, 2]
    splat_data["rgba"] = rgba
    splat_data["rot"] = quats_u8

    with open(target, "wb") as f:
        splat_data.tofile(f)

    return target
