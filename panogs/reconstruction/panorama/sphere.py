"""
Synthetic Spherical Point Cloud Generation for Panorama Geometry Validation.

Maps equirectangular image pixels to a spherical shell in 3D space:
    P = C + radius * d
where C = (0, 0, 0) and d is the normalized ray vector.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

from panogs.core.camera.spherical import equirectangular_rays
from panogs.core.logging import get_logger
from panogs.io.images import load_image_as_numpy
from panogs.io.ply import write_point_cloud_ply


def generate_synthetic_sphere(
    image_path: Union[str, Path],
    radius: float = 1.0,
    max_resolution: Optional[int] = None,
    output_ply: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[Path]]:
    """
    Generate a colored synthetic 3D spherical shell point cloud from an equirectangular panorama.

    Args:
        image_path: Path to equirectangular panorama image.
        radius: Sphere radius in world units (default: 1.0).
        max_resolution: Optional maximum dimension constraint for memory efficiency.
        output_ply: Optional file path to write PLY point cloud.

    Returns:
        Tuple: (points (N, 3), colors (N, 3), output_ply_path)
    """
    logger = get_logger("reconstruction.sphere")
    image_path = Path(image_path)
    logger.info(f"Loading equirectangular panorama: {image_path.name}")

    # Load RGB image as uint8 (0..255)
    img_rgb = load_image_as_numpy(image_path, normalize_float=False, max_resolution=max_resolution)
    height, width = img_rgb.shape[:2]
    logger.info(f"Generating rays for resolution {width}x{height} (radius={radius})")

    # Generate unit rays (H, W, 3)
    rays = equirectangular_rays(height, width)

    # 3D points: P = radius * d
    points_3d = (rays * radius).reshape(-1, 3).astype(np.float32)
    colors_rgb = img_rgb.reshape(-1, 3).astype(np.uint8)

    logger.info(f"Constructed spherical point cloud with {points_3d.shape[0]:,} vertices")

    saved_path = None
    if output_ply is not None:
        saved_path = write_point_cloud_ply(output_ply, points_3d, colors_rgb, binary=True)
        logger.info(f"Saved spherical point cloud to {saved_path}")

    return points_3d, colors_rgb, saved_path
