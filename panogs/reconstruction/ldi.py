"""
Layered Depth Image (LDI) Reconstruction Engine for Panoramic 3D Scenes.
Inspired by PanoDreamer (ArXiv 2412.04827).

Decomposes a single 360 panorama into decoupled, continuous depth layers
(Foreground Objects + Complete Inpainted Background Room Shell), inpainting
occluded geometry behind foreground objects to enable parallax-rich, hole-free
and lightweight 60 FPS 3D Gaussian navigation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from panogs.core.camera.spherical import equirectangular_rays
from panogs.core.logging import get_logger
from panogs.reconstruction.inpainting import (
    inpaint_background_depth,
    inpaint_background_texture,
    segment_foreground_objects,
)
from panogs.reconstruction.pointcloud import PointCloud


@dataclass
class LDILayer:
    """A single continuous layer in a Layered Depth Image representation."""
    layer_index: int
    rgb: np.ndarray             # (H, W, 3) uint8 RGB image
    depth: np.ndarray           # (H, W) float32 metric depth map
    normals: np.ndarray         # (H, W, 3) float32 surface normals
    valid_mask: np.ndarray      # (H, W) bool mask of active pixels in this layer
    depth_range: Tuple[float, float]  # (min_depth, max_depth) for this tier


def compute_layer_depth_thresholds(
    depth_map: np.ndarray,
    num_layers: int = 2,
    min_depth: float = 0.3,
    max_depth: float = 25.0,
) -> List[Tuple[float, float]]:
    """
    Compute optimal depth split intervals for LDI slicing using scene depth quantiles.
    """
    valid_depths = depth_map[(depth_map >= min_depth) & (depth_map <= max_depth)]
    if len(valid_depths) == 0:
        return [(min_depth, max_depth)]

    if num_layers <= 1:
        return [(float(np.min(valid_depths)), float(np.max(valid_depths)))]

    percentiles = np.linspace(0, 100, num_layers + 1)
    quantiles = np.percentile(valid_depths, percentiles)

    intervals = []
    for i in range(num_layers):
        d_min = float(quantiles[i])
        d_max = float(quantiles[i + 1])
        if i == num_layers - 1:
            d_max = max(d_max, float(np.max(valid_depths)))
        intervals.append((d_min, d_max))

    return intervals


def construct_layered_depth_image(
    img_rgb: np.ndarray,
    depth_map: np.ndarray,
    num_layers: int = 2,
    inpaint_radius: int = 9,
    h_floor: float = 1.4,
    h_ceiling: float = 1.7,
) -> List[LDILayer]:
    """
    Deconstruct an equirectangular panorama into an LDI stack of decoupled layers.

    Guarantees no redundant splat duplication (lightweight 60 FPS WebGL rendering)
    and zero black voids or horizon step discontinuities.
    """
    logger = get_logger("reconstruction.ldi")
    H, W = depth_map.shape[:2]
    rays = equirectangular_rays(H, W, jitter=False)

    intervals = compute_layer_depth_thresholds(depth_map, num_layers=num_layers)
    logger.info(f"Decomposing scene into {len(intervals)} LDI layers: {intervals}")

    layers: List[LDILayer] = []
    current_rgb = img_rgb.copy()
    current_depth = depth_map.copy()

    for idx, (d_min, d_max) in enumerate(intervals):
        is_foreground = (idx < len(intervals) - 1)

        if is_foreground:
            if idx == 0:
                layer_active = (current_depth <= d_max)
            else:
                layer_active = (current_depth >= d_min) & (current_depth <= d_max)

            # Erode boundary slightly to prevent rubber-sheet stretching
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            clean_mask = cv2.erode(layer_active.astype(np.uint8), kernel).astype(bool)

            # Compute layer surface normals
            P_layer = current_depth[:, :, np.newaxis] * rays
            Tu = (np.roll(P_layer, -1, axis=1) - np.roll(P_layer, 1, axis=1)) * 0.5
            Tv = np.zeros_like(P_layer)
            Tv[1:-1, :, :] = (P_layer[2:, :, :] - P_layer[:-2, :, :]) * 0.5
            Tv[0, :, :] = P_layer[1, :, :] - P_layer[0, :, :]
            Tv[-1, :, :] = P_layer[-1, :, :] - P_layer[-2, :, :]

            normals_grad = np.cross(Tu, Tv)
            n_len = np.linalg.norm(normals_grad, axis=-1, keepdims=True)
            n_len[n_len == 0] = 1.0
            normals_grad = (normals_grad / n_len).astype(np.float32)

            facing = np.sum(normals_grad * rays, axis=-1, keepdims=True)
            normals_grad[facing[:, :, 0] > 0] *= -1.0

            layers.append(
                LDILayer(
                    layer_index=idx,
                    rgb=current_rgb.copy(),
                    depth=current_depth.copy(),
                    normals=normals_grad,
                    valid_mask=clean_mask,
                    depth_range=(d_min, d_max),
                )
            )
            logger.info(f"Layer {idx} (Foreground [{d_min:.2f}m - {d_max:.2f}m]): {np.count_nonzero(clean_mask):,} active splats.")

            # Inpaint background behind this layer for deeper layers
            occlusion_mask = layer_active.astype(np.uint8) * 255
            current_rgb = inpaint_background_texture(current_rgb, occlusion_mask, inpaint_radius=inpaint_radius, method="telea")
            current_depth = inpaint_background_depth(current_depth, occlusion_mask, smooth_sigma=3.0)

        else:
            # Deepest Background Room Shell (Walls, Ceiling, Floor)
            P_bg = current_depth[:, :, np.newaxis] * rays
            Tu = (np.roll(P_bg, -1, axis=1) - np.roll(P_bg, 1, axis=1)) * 0.5
            Tv = np.zeros_like(P_bg)
            Tv[1:-1, :, :] = (P_bg[2:, :, :] - P_bg[:-2, :, :]) * 0.5
            Tv[0, :, :] = P_bg[1, :, :] - P_bg[0, :, :]
            Tv[-1, :, :] = P_bg[-1, :, :] - P_bg[-2, :, :]

            normals_bg = np.cross(Tu, Tv)
            nb_len = np.linalg.norm(normals_bg, axis=-1, keepdims=True)
            nb_len[nb_len == 0] = 1.0
            normals_bg = (normals_bg / nb_len).astype(np.float32)

            facing = np.sum(normals_bg * rays, axis=-1, keepdims=True)
            normals_bg[facing[:, :, 0] > 0] *= -1.0

            # Planar normal snapping for ceiling and floor
            ry = rays[:, :, 1]
            is_ceil = (ry > 0.55) & (P_bg[:, :, 1] > h_ceiling * 0.70)
            is_floor = (ry < -0.55) & (P_bg[:, :, 1] < -h_floor * 0.70)
            normals_bg[is_ceil] = [0.0, -1.0, 0.0]
            normals_bg[is_floor] = [0.0, 1.0, 0.0]

            bg_valid = np.ones((H, W), dtype=bool)

            layers.append(
                LDILayer(
                    layer_index=idx,
                    rgb=current_rgb,
                    depth=current_depth,
                    normals=normals_bg,
                    valid_mask=bg_valid,
                    depth_range=(d_min, d_max),
                )
            )
            logger.info(f"Layer {idx} (Background Room Shell [{d_min:.2f}m - {d_max:.2f}m]): {np.count_nonzero(bg_valid):,} active splats.")

    return layers


def reconstruct_from_ldi(
    layers: List[LDILayer],
    output_ply: Optional[Union[str, Path]] = None,
) -> PointCloud:
    """
    Backproject all LDI layers into 3D world space and assemble a unified multi-layer PointCloud.

    Args:
        layers: List of LDILayer objects.
        output_ply: Optional path to write point cloud PLY.

    Returns:
        Unified PointCloud containing continuous foreground, midground, and background splats.
    """
    logger = get_logger("reconstruction.ldi")
    all_points = []
    all_colors = []
    all_normals = []
    all_layer_ids = []

    H, W = layers[0].depth.shape[:2]
    rays = equirectangular_rays(H, W, jitter=False)

    for layer in layers:
        mask = layer.valid_mask
        if not np.any(mask):
            continue

        d = layer.depth[mask, np.newaxis]
        r = rays[mask]
        pts = (r * d).astype(np.float32)
        cols = layer.rgb[mask].astype(np.uint8)
        norms = layer.normals[mask].astype(np.float32)
        layer_ids = np.full(len(pts), layer.layer_index, dtype=np.int32)

        all_points.append(pts)
        all_colors.append(cols)
        all_normals.append(norms)
        all_layer_ids.append(layer_ids)

    merged_points = np.vstack(all_points).astype(np.float32)
    merged_colors = np.vstack(all_colors).astype(np.uint8)
    merged_normals = np.vstack(all_normals).astype(np.float32)
    merged_layers = np.concatenate(all_layer_ids)

    pc = PointCloud(
        points=merged_points,
        colors=merged_colors,
        normals=merged_normals,
        metadata={
            "total_pixels": H * W,
            "num_ldi_layers": len(layers),
            "layer_indices": merged_layers,
            "ldi_enabled": True,
        },
    )

    logger.info(f"LDI Reconstruction assembled {pc.num_points:,} total 3D Gaussians across {len(layers)} layers.")

    if output_ply is not None:
        from panogs.io.ply import write_point_cloud_ply
        write_point_cloud_ply(
            output_ply,
            pc.points,
            pc.colors,
            normals=pc.normals,
            binary=True,
        )
        logger.info(f"Exported LDI point cloud PLY to {output_ply}")

    return pc
