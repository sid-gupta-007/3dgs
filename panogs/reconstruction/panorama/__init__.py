"""Panorama reconstruction subpackage."""
from panogs.reconstruction.panorama.sphere import generate_synthetic_sphere
from panogs.reconstruction.panorama.layout import (
    compute_cuboid_room_geometry,
    reconstruct_layout_panorama,
)

# Backward compatibility alias
compute_cuboid_room_depth = compute_cuboid_room_geometry

__all__ = [
    "generate_synthetic_sphere",
    "compute_cuboid_room_geometry",
    "compute_cuboid_room_depth",
    "reconstruct_layout_panorama",
]

