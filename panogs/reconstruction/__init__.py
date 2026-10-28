"""Reconstruction modules for PanoGS."""

__all__ = [
    "PointCloud",
    "backproject_depth_to_points",
    "convert_disparity_to_depth",
    "reconstruct_from_image",
    "voxel_downsample",
    "remove_statistical_outliers",
    "estimate_surface_normals",
    "process_point_cloud",
    "create_virtual_camera_rig",
    "synthesize_multi_view_scene",
    "LDILayer",
    "construct_layered_depth_image",
    "reconstruct_from_ldi",
]


def __getattr__(name: str):
    if name in {"PointCloud", "backproject_depth_to_points", "convert_disparity_to_depth", "reconstruct_from_image"}:
        from panogs.reconstruction import pointcloud
        return getattr(pointcloud, name)
    if name in {"estimate_surface_normals", "process_point_cloud", "remove_statistical_outliers", "voxel_downsample"}:
        from panogs.reconstruction import processing
        return getattr(processing, name)
    if name in {"create_virtual_camera_rig", "synthesize_multi_view_scene"}:
        from panogs.reconstruction import synthesis
        return getattr(synthesis, name)
    if name in {"LDILayer", "construct_layered_depth_image", "reconstruct_from_ldi"}:
        from panogs.reconstruction import ldi
        return getattr(ldi, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
