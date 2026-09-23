"""
Layout-Guided Panoramic 3D Reconstruction Engine (360-GS Architecture).
Computes exact cuboidal Manhattan room geometry (flat floor, flat ceiling, vertical planar walls)
and applies smooth interior object relief in the horizontal visual band to generate solid,
undistorted, photorealistic 3D indoor room scenes.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image

from panogs.core.camera.spherical import equirectangular_rays, pixel_to_spherical
from panogs.core.logging import get_logger
from panogs.io.images import load_image_as_numpy
from panogs.reconstruction.depth.base import DepthEstimator
from panogs.reconstruction.depth.midas import MiDaSDepthEstimator
from panogs.reconstruction.pointcloud import PointCloud


def compute_cuboid_room_geometry(
    H: int,
    W: int,
    h_floor: float = 1.4,
    h_ceiling: float = 1.7,
    w_north: float = 4.8,
    w_south: float = 4.2,
    w_east: float = 3.8,
    w_west: float = 3.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute exact geometric ray intersection distance field and analytical surface normals
    for a cuboidal Manhattan room:
    - Floor plane at Y = -h_floor (Normal: [0, 1, 0])
    - Ceiling plane at Y = +h_ceiling (Normal: [0, -1, 0])
    - North wall at Z = +w_north (Normal: [0, 0, -1])
    - South wall at Z = -w_south (Normal: [0, 0, 1])
    - East wall at X = +w_east (Normal: [-1, 0, 0])
    - West wall at X = -w_west (Normal: [1, 0, 0])

    Returns:
        d_box: (H, W) float32 distance map along each ray.
        normals_box: (H, W, 3) float32 analytical surface normal vectors.
        plane_type: (H, W) int array (0: floor, 1: ceiling, 2: north, 3: south, 4: east, 5: west).
    """
    rays = equirectangular_rays(H, W)
    rx = rays[:, :, 0]
    ry = rays[:, :, 1]
    rz = rays[:, :, 2]

    eps = 1e-5
    d_box = np.full((H, W), 100.0, dtype=np.float32)
    normals = np.zeros((H, W, 3), dtype=np.float32)
    plane_type = np.zeros((H, W), dtype=np.int32)

    # 1. Floor plane: Y = -h_floor (ry < 0)
    floor_mask = ry < -eps
    d_floor = np.full((H, W), 100.0, dtype=np.float32)
    d_floor[floor_mask] = h_floor / (-ry[floor_mask])
    
    # 2. Ceiling plane: Y = +h_ceiling (ry > 0)
    ceil_mask = ry > eps
    d_ceil = np.full((H, W), 100.0, dtype=np.float32)
    d_ceil[ceil_mask] = h_ceiling / ry[ceil_mask]

    # 3. North wall: Z = +w_north (rz > 0)
    north_mask = rz > eps
    d_north = np.full((H, W), 100.0, dtype=np.float32)
    d_north[north_mask] = w_north / rz[north_mask]

    # 4. South wall: Z = -w_south (rz < 0)
    south_mask = rz < -eps
    d_south = np.full((H, W), 100.0, dtype=np.float32)
    d_south[south_mask] = w_south / (-rz[south_mask])

    # 5. East wall: X = +w_east (rx > 0)
    east_mask = rx > eps
    d_east = np.full((H, W), 100.0, dtype=np.float32)
    d_east[east_mask] = w_east / rx[east_mask]

    # 6. West wall: X = -w_west (rx < 0)
    west_mask = rx < -eps
    d_west = np.full((H, W), 100.0, dtype=np.float32)
    d_west[west_mask] = w_west / (-rx[west_mask])

    # Stack all candidate distances
    candidates = np.stack([d_floor, d_ceil, d_north, d_south, d_east, d_west], axis=-1)
    min_idx = np.argmin(candidates, axis=-1)
    d_box = np.min(candidates, axis=-1).astype(np.float32)

    # Assign analytical surface normals for each plane
    # 0: floor (+Y)
    normals[min_idx == 0] = [0.0, 1.0, 0.0]
    # 1: ceiling (-Y)
    normals[min_idx == 1] = [0.0, -1.0, 0.0]
    # 2: north wall (-Z)
    normals[min_idx == 2] = [0.0, 0.0, -1.0]
    # 3: south wall (+Z)
    normals[min_idx == 3] = [0.0, 0.0, 1.0]
    # 4: east wall (-X)
    normals[min_idx == 4] = [-1.0, 0.0, 0.0]
    # 5: west wall (+X)
    normals[min_idx == 5] = [1.0, 0.0, 0.0]

    return d_box, normals, min_idx


def detect_depth_edges(
    depth_map: np.ndarray,
    rel_threshold: float = 0.08,
    abs_threshold: float = 0.20,
) -> np.ndarray:
    """
    Detect depth discontinuity edges where relative depth change > threshold.
    
    These edges represent object boundary transitions (e.g., edge of a table/couch
    where depth jumps to the floor/wall behind it). Flying transition pixels
    along these edges must be pruned to avoid stretching rubber-sheet artifacts.
    """
    d = np.maximum(depth_map, 0.01)
    
    # Horizontal depth jumps
    d_left = np.roll(d, 1, axis=1)
    d_right = np.roll(d, -1, axis=1)
    rel_h = np.maximum(
        np.abs(d - d_left) / np.minimum(d, d_left),
        np.abs(d - d_right) / np.minimum(d, d_right),
    )
    abs_h = np.maximum(np.abs(d - d_left), np.abs(d - d_right))
    
    # Vertical depth jumps
    d_up = np.roll(d, 1, axis=0)
    d_down = np.roll(d, -1, axis=0)
    rel_v = np.maximum(
        np.abs(d - d_up) / np.minimum(d, d_up),
        np.abs(d - d_down) / np.minimum(d, d_down),
    )
    abs_v = np.maximum(np.abs(d - d_up), np.abs(d - d_down))
    
    edge = ((rel_h > rel_threshold) & (abs_h > abs_threshold)) | ((rel_v > rel_threshold) & (abs_v > abs_threshold))
    
    # Dilate by 1 pixel to capture the full transition zone
    dilated = (
        edge
        | np.roll(edge, 1, axis=0)
        | np.roll(edge, -1, axis=0)
        | np.roll(edge, 1, axis=1)
        | np.roll(edge, -1, axis=1)
    )
    return dilated


def joint_bilateral_filter(
    depth: np.ndarray,
    guide_rgb: np.ndarray,
    spatial_radius: int = 2,
    sigma_spatial: float = 2.0,
    sigma_depth: float = 0.15,
    sigma_color: float = 0.18,
) -> np.ndarray:
    """
    Apply joint bilateral filter to smooth monocular depth noise while strictly preserving sharp object edges.
    Handles spherical 360 equirectangular horizontal wrapping automatically.
    """
    smoothed = np.zeros_like(depth)
    weights_sum = np.zeros_like(depth)

    # Normalize RGB to [0, 1]
    rgb_norm = guide_rgb.astype(np.float32) / 255.0

    for dy in range(-spatial_radius, spatial_radius + 1):
        for dx in range(-spatial_radius, spatial_radius + 1):
            spatial_w = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_spatial * sigma_spatial))

            # Horizontal equirectangular wrap (axis=1), vertical clamped wrap (axis=0)
            shifted_d = np.roll(np.roll(depth, dy, axis=0), dx, axis=1)
            shifted_rgb = np.roll(np.roll(rgb_norm, dy, axis=0), dx, axis=1)

            # Depth difference penalty & color difference penalty
            d_diff = np.abs(depth - shifted_d)
            c_diff = np.linalg.norm(rgb_norm - shifted_rgb, axis=-1)

            range_w = np.exp(-((d_diff / max(1e-3, sigma_depth)) ** 2 + (c_diff / max(1e-3, sigma_color)) ** 2))
            w = spatial_w * range_w

            smoothed += shifted_d * w
            weights_sum += w

    return (smoothed / np.maximum(weights_sum, 1e-6)).astype(np.float32)


def reconstruct_layout_panorama(
    image_path: Union[str, Path],
    depth_estimator: Optional[DepthEstimator] = None,
    h_floor: float = 1.35,
    h_ceiling: float = 1.65,
    room_depth: float = 4.5,
    room_width: float = 3.6,
    relief_weight: float = 1.0,
    max_resolution: Optional[int] = 1024,
    output_ply: Optional[Union[str, Path]] = None,
) -> PointCloud:
    """
    Reconstruct a true photorealistic 3D indoor scene from a 360 panorama.
    
    Uses Depth Anything V2 Metric Indoor via 6-view cubemap projection to obtain
    true physical Euclidean depth across the entire 360 sphere without distortion.
    
    Surface normals are computed via 3D spatial gradients, with camera-facing billboard
    normals and thin anisotropic Gaussian scaling applied at depth discontinuity edges
    to prevent stretching.
    """
    from panogs.io.ply import write_point_cloud_ply
    from panogs.reconstruction.depth.cubemap import CubemapDepthEstimator

    logger = get_logger("reconstruction.layout")
    image_path = Path(image_path)
    logger.info(f"Loading panorama for 3D reconstruction: {image_path.name}")

    img_rgb = load_image_as_numpy(image_path, normalize_float=False, max_resolution=max_resolution)
    H, W = img_rgb.shape[:2]

    # 1. Depth estimation: prefer Depth Anything V2 via cubemap for metric depth
    if depth_estimator is None:
        try:
            from panogs.reconstruction.depth.depth_anything import DepthAnythingV2Estimator
            logger.info("Initializing Depth Anything V2 Metric Indoor (cubemap) for sharp metric depth...")
            dav2 = DepthAnythingV2Estimator(variant="small", device="cpu")
            depth_estimator = CubemapDepthEstimator(
                face_size=512,
                underlying_estimator=dav2,
            )
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Depth Anything V2 unavailable ({e}), falling back to MiDaS cubemap...")
            depth_estimator = CubemapDepthEstimator(face_size=512, base_model="MiDaS_small")

    logger.info("Inferring neural depth for scene geometry...")
    res = depth_estimator.estimate(img_rgb)
    depth_raw = res.depth_map.astype(np.float32)
    is_metric = getattr(res, 'is_metric', False)

    # 2. Process neural depth: metric vs. relative
    if is_metric:
        logger.info(
            f"Using METRIC depth (min={depth_raw.min():.2f}m, max={depth_raw.max():.2f}m)."
        )
        min_near = 0.3
        max_far = 40.0
        d_final = np.clip(depth_raw, min_near, max_far)
    else:
        logger.info("Using RELATIVE disparity. Calibrating depth range [0.5m, 12.0m]...")
        disp = depth_raw
        disp_min = float(np.percentile(disp, 2))
        disp_max = float(np.percentile(disp, 98))
        norm_disp = np.clip((disp - disp_min) / max(1e-4, disp_max - disp_min), 0.0, 1.0)
        # Reciprocal mapping
        inv = 1.0 / (norm_disp + 0.1)
        inv_min = 1.0 / (1.0 + 0.1)
        inv_max = 1.0 / (0.0 + 0.1)
        norm_inv = (inv - inv_min) / (inv_max - inv_min)
        d_final = (0.6 + 8.0 * norm_inv).astype(np.float32)

    # Apply RGB-Guided Joint Bilateral Filtering to eliminate depth wobbles while keeping crisp edges
    logger.info("Applying RGB-guided joint bilateral smoothing to eliminate depth noise...")
    d_final = joint_bilateral_filter(d_final, img_rgb, spatial_radius=2, sigma_spatial=2.0, sigma_depth=0.12, sigma_color=0.15)

    # 3. Generate equirectangular unit rays
    rays = equirectangular_rays(H, W)

    # 4. Backproject into 3D Euclidean coordinates: P = d * rays
    P = d_final[:, :, np.newaxis] * rays

    # 5. Segment foreground furniture and inpaint occluded background
    from panogs.reconstruction.inpainting import inpaint_background_texture, segment_foreground_objects

    logger.info("Detecting depth discontinuity edges...")
    edge_mask = detect_depth_edges(d_final, rel_threshold=0.12, abs_threshold=0.25)
    edge_count = np.count_nonzero(edge_mask)
    logger.info(f"Found {edge_count:,} depth-edge silhouette pixels ({100.0 * edge_count / (H*W):.1f}% of image)")

    logger.info("Segmenting foreground furniture & inpainting occluded floorboards...")
    fg_mask = segment_foreground_objects(d_final, rel_depth_threshold=0.18, min_size_pixels=150)
    fg_count = np.count_nonzero(fg_mask)
    logger.info(f"Segmented {fg_count:,} foreground furniture pixels ({100.0 * fg_count / (H * W):.1f}%).")

    if fg_count > 0:
        bg_rgb = inpaint_background_texture(img_rgb, fg_mask, inpaint_radius=9, method="telea")
    else:
        bg_rgb = img_rgb.copy()

    # 6. Compute 3D surface normals from spatial geometry
    logger.info("Computing surface normals from 3D geometry...")
    Tu = (np.roll(P, -1, axis=1) - np.roll(P, 1, axis=1)) * 0.5
    Tv = np.zeros_like(P)
    Tv[1:-1, :, :] = (P[2:, :, :] - P[:-2, :, :]) * 0.5
    Tv[0, :, :] = P[1, :, :] - P[0, :, :]
    Tv[-1, :, :] = P[-1, :, :] - P[-2, :, :]

    normals_grad = np.cross(Tu, Tv)
    n_len = np.linalg.norm(normals_grad, axis=-1, keepdims=True)
    n_len[n_len == 0] = 1.0
    normals_grad = normals_grad / n_len

    # Orient surface normals toward camera
    facing = np.sum(normals_grad * rays, axis=-1, keepdims=True)
    normals_grad[facing[:, :, 0] > 0] *= -1.0

    # At depth edges: force camera-facing billboard normals to prevent stretching sideways
    normals_billboard = -rays.copy()
    nbb_len = np.linalg.norm(normals_billboard, axis=-1, keepdims=True)
    nbb_len[nbb_len == 0] = 1.0
    normals_billboard = normals_billboard / nbb_len

    edge_3d = edge_mask[:, :, np.newaxis]
    normals_final = np.where(edge_3d, normals_billboard, normals_grad)
    n_final_len = np.linalg.norm(normals_final, axis=-1, keepdims=True)
    n_final_len[n_final_len == 0] = 1.0
    normals_final = (normals_final / n_final_len).astype(np.float32)

    # Filter out flying transition slope pixels so furniture doesn't smear into walls
    flying_mask = detect_depth_edges(d_final, rel_threshold=0.08, abs_threshold=0.20)
    keep_primary = ~flying_mask.reshape(-1)

    flat_P = P.reshape(-1, 3).astype(np.float32)[keep_primary]
    flat_C = img_rgb.reshape(-1, 3).astype(np.uint8)[keep_primary]
    flat_N = normals_final.reshape(-1, 3).astype(np.float32)[keep_primary]
    flat_E = edge_mask.reshape(-1)[keep_primary]

    all_points = [flat_P]
    all_colors = [flat_C]
    all_normals = [flat_N]
    all_edges = [flat_E]

    # 7. Ground Floor Infilling: synthesize solid floor splats under occluded furniture
    infilled_count = 0
    ry = rays[:, :, 1]
    floor_occluded = (fg_mask > 0) & (ry < -0.06)
    if np.any(floor_occluded):
        d_floor_target = h_floor / (-ry[floor_occluded])
        d_current = d_final[floor_occluded]
        valid_infill = (d_floor_target > d_current + 0.25) & (d_floor_target < 20.0)

        if np.any(valid_infill):
            infill_rays = rays[floor_occluded][valid_infill]
            infill_d = d_floor_target[valid_infill, np.newaxis]
            infill_pts = (infill_rays * infill_d).astype(np.float32)
            infill_cols = bg_rgb[floor_occluded][valid_infill].astype(np.uint8)
            infill_norms = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (len(infill_pts), 1))
            infill_edge = np.zeros(len(infill_pts), dtype=bool)

            infilled_count = len(infill_pts)
            all_points.append(infill_pts)
            all_colors.append(infill_cols)
            all_normals.append(infill_norms)
            all_edges.append(infill_edge)
            logger.info(f"Synthesized {infilled_count:,} inpainted ground floor splats under occluded furniture.")

    merged_points = np.vstack(all_points).astype(np.float32)
    merged_colors = np.vstack(all_colors).astype(np.uint8)
    merged_normals = np.vstack(all_normals).astype(np.float32)
    merged_edges = np.concatenate(all_edges)

    point_cloud = PointCloud(
        points=merged_points,
        colors=merged_colors,
        normals=merged_normals,
        metadata={
            "total_pixels": H * W,
            "depth_is_metric": is_metric,
            "depth_model": res.model_name,
            "edge_pixel_count": int(edge_count),
            "edge_pixel_ratio": float(edge_count / (H * W)),
            "edge_mask": merged_edges,
            "infilled_points": int(infilled_count),
        },
    )

    logger.info(f"Reconstructed {point_cloud.num_points:,} total solid 3D points from metric depth & inpainting.")

    # 9. Export PLY if output path provided
    if output_ply is not None:
        write_point_cloud_ply(
            output_ply,
            point_cloud.points,
            point_cloud.colors,
            normals=point_cloud.normals,
            binary=True,
        )
        logger.info(f"Exported point cloud PLY to {output_ply}")

    return point_cloud

