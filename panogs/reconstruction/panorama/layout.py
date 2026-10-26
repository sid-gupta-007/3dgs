"""
Layout-Guided Panoramic 3D Reconstruction Engine (360-GS Architecture).
Computes exact cuboidal Manhattan room geometry (flat floor, flat ceiling, vertical planar walls)
and applies smooth interior object relief in the horizontal visual band to generate solid,
undistorted, photorealistic 3D indoor room scenes.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from panogs.core.camera.spherical import equirectangular_rays, pixel_to_spherical
from panogs.core.logging import get_logger
from panogs.io.images import load_hdr_radiance, load_image_as_numpy, srgb_to_linear
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


def regularize_panoramic_depth(
    depth_map: np.ndarray,
    rays: np.ndarray,
) -> np.ndarray:
    """
    Eliminate monocular depth tearing on lattice partitions, window muntins, hanging cords,
    and glowing ceiling fixtures while preserving all real 3D furniture volumes:
    1. Thin foreground blade/cord suppression (MORPH_CLOSE): prevents narrow (<3.5% width)
       see-through lattice bars, hanging cords, and ceiling LED strips from jutting meters
       into mid-air and shredding when viewed off-axis.
    2. Narrow pane/aperture sealing (MORPH_OPEN): prevents narrow see-through panes
       from spiking backward between frame bars.
    3. Shock edge snapping: snaps transition ramps cleanly to foreground or background depth
       so zero pixels float in mid-air as shredded ribbons.
    """
    H, W = depth_map.shape
    d = np.maximum(depth_map.astype(np.float32), 0.1)
    ry = rays[:, :, 1]

    # 1. Suppress narrow 1-2px floating cord/muntin spikes without flattening real walls or pillars
    k_thin = 5
    pad = k_thin
    d_pad = np.pad(d, ((0, 0), (pad, pad)), mode="wrap")
    kernel_thin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_thin, k_thin))
    d_closed = cv2.morphologyEx(d_pad, cv2.MORPH_CLOSE, kernel_thin)[:, pad : pad + W]
    is_thin_blade = ((d_closed - d) / np.maximum(d, 0.1)) > 0.22
    d = np.where(is_thin_blade, d_closed, d)

    # 2. Suppress narrow 1-2px depth pinholes
    d_opened = cv2.morphologyEx(d, cv2.MORPH_OPEN, kernel_thin)
    pane_spike = (d - d_opened) > np.maximum(0.45, d_opened * 0.22)
    d = np.where(pane_spike, d_opened, d)

    # 3. Snap 1-2px transition ramps at sharp silhouette boundaries
    d = sharpen_depth_edges(d, rel_jump=0.16, abs_jump=0.28, radius=2)
    return d.astype(np.float32)


def sharpen_depth_edges(
    depth_map: np.ndarray,
    rel_jump: float = 0.14,
    abs_jump: float = 0.22,
    radius: int = 4,
) -> np.ndarray:
    """
    Shock-filter depth transition ramps at object boundaries so bilinear/neural
    transition slopes snap cleanly to either the foreground object or background wall
    instead of stretching across mid-air as floating ribbons.
    """
    d = np.maximum(depth_map.astype(np.float32), 0.05)
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    d_min = cv2.erode(d, kernel)
    d_max = cv2.dilate(d, kernel)

    span = d_max - d_min
    is_ramp = ((span / np.maximum(d_min, 0.05)) > rel_jump) & (span > abs_jump)
    if not np.any(is_ramp):
        return d

    pos = (d - d_min) / np.maximum(span, 1e-4)
    snapped = np.where(pos < 0.48, d_min, d_max)
    return np.where(is_ramp, snapped, d).astype(np.float32)


def detect_depth_edges(
    depth_map: np.ndarray,
    rel_threshold: float = 0.08,
    abs_threshold: float = 0.20,
) -> np.ndarray:
    """
    Detect depth discontinuity edges where relative depth change > threshold.
    
    Checks both 1-pixel and 2-pixel neighborhood jumps so multi-pixel transition
    slopes at object silhouettes are caught completely and never form floating ribbons.
    """
    d = np.maximum(depth_map, 0.01)
    edge = np.zeros(d.shape, dtype=bool)

    for step in (1, 2):
        thresh_r = rel_threshold * (1.0 if step == 1 else 1.25)
        thresh_a = abs_threshold * (1.0 if step == 1 else 1.25)

        d_left = np.roll(d, step, axis=1)
        d_right = np.roll(d, -step, axis=1)
        rel_h = np.maximum(
            np.abs(d - d_left) / np.minimum(d, d_left),
            np.abs(d - d_right) / np.minimum(d, d_right),
        )
        abs_h = np.maximum(np.abs(d - d_left), np.abs(d - d_right))

        d_up = np.roll(d, step, axis=0)
        d_down = np.roll(d, -step, axis=0)
        # Do not wrap vertical top-to-bottom poles across axis=0
        d_up[:step, :] = d[:step, :]
        d_down[-step:, :] = d[-step:, :]
        rel_v = np.maximum(
            np.abs(d - d_up) / np.minimum(d, d_up),
            np.abs(d - d_down) / np.minimum(d, d_down),
        )
        abs_v = np.maximum(np.abs(d - d_up), np.abs(d - d_down))

        edge |= ((rel_h > thresh_r) & (abs_h > thresh_a)) | ((rel_v > thresh_r) & (abs_v > thresh_a))

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
    solid_shell: bool = True,
    jitter: bool = False,
    max_resolution: Optional[int] = None,
    output_ply: Optional[Union[str, Path]] = None,
) -> PointCloud:
    """
    Reconstruct a true photorealistic 3D indoor scene from a 360 panorama (V3 Architecture).
    
    Features:
    - Depth Anything V2 Metric Indoor via 6-view cubemap projection.
    - 360-equirectangular wrapped RGB-guided joint bilateral depth filtering.
    - Sub-pixel anti-moire ray jittering.
    - Planar normal snapping for ceilings, floors, and walls.
    - Edge floater purging along steep depth transitions.
    - Full 6-surface solid Manhattan room shell backing behind all occluded objects.
    """
    from panogs.io.ply import write_point_cloud_ply
    from panogs.reconstruction.depth.cubemap import CubemapDepthEstimator

    logger = get_logger("reconstruction.layout")
    image_path = Path(image_path)
    logger.info(f"Loading panorama for 3D reconstruction: {image_path.name}")

    img_rgb = load_image_as_numpy(image_path, normalize_float=False, max_resolution=max_resolution)
    if image_path.suffix.lower() in {".hdr", ".exr"}:
        img_radiance = load_hdr_radiance(image_path, max_resolution=max_resolution)
    else:
        img_radiance = srgb_to_linear(img_rgb.astype(np.float32) / 255.0)
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

    # 3. Generate equirectangular unit rays (with anti-moire jittering if enabled)
    rays = equirectangular_rays(H, W, jitter=jitter)

    # Regularize ceiling fixtures, lattice partitions, and transition ramps before bilateral filtering
    logger.info("Regularizing panoramic depth structure (ceiling fixtures, lattice panes, and silhouette ramps)...")
    d_final = regularize_panoramic_depth(d_final, rays)

    # Apply RGB-Guided Joint Bilateral Filtering to eliminate depth wobbles while keeping crisp edges
    logger.info("Applying RGB-guided joint bilateral smoothing to eliminate depth noise...")
    d_final = joint_bilateral_filter(d_final, img_rgb, spatial_radius=2, sigma_spatial=1.8, sigma_depth=0.10, sigma_color=0.12)
    d_final = sharpen_depth_edges(d_final, rel_jump=0.16, abs_jump=0.25, radius=2)

    # 4. Compute Manhattan cuboid room geometry for architectural planar bounds & solid shell
    d_box, normals_box, plane_type = compute_cuboid_room_geometry(
        H, W,
        h_floor=h_floor,
        h_ceiling=h_ceiling,
        w_north=room_depth,
        w_south=room_depth,
        w_east=room_width,
        w_west=room_width,
    )

    # 5. Backproject into 3D Euclidean coordinates: P = d * rays (100% preserving real 3D object geometry)
    P = d_final[:, :, np.newaxis] * rays

    # 6. Segment foreground furniture and inpaint occluded background (RGB + Depth)
    from panogs.reconstruction.inpainting import inpaint_background_texture, segment_foreground_objects

    logger.info("Detecting depth discontinuity edges...")
    edge_mask = detect_depth_edges(d_final, rel_threshold=0.10, abs_threshold=0.20)
    edge_count = np.count_nonzero(edge_mask)
    logger.info(f"Found {edge_count:,} depth-edge silhouette pixels ({100.0 * edge_count / (H*W):.1f}% of image)")

    logger.info("Segmenting foreground furniture & inpainting occluded background...")
    fg_mask = segment_foreground_objects(d_final, rel_depth_threshold=0.18, min_size_pixels=120, background_depth=None)
    fg_count = np.count_nonzero(fg_mask)
    logger.info(f"Segmented {fg_count:,} foreground furniture pixels ({100.0 * fg_count / (H * W):.1f}%).")

    inpaint_mask = (fg_mask > 0).astype(np.uint8) * 255
    if solid_shell and np.any(inpaint_mask > 0):
        bg_rgb = inpaint_background_texture(img_rgb, inpaint_mask, inpaint_radius=7, method="telea")
        # Inpaint continuous metric depth behind foreground objects using scene's own depth envelope
        from panogs.reconstruction.inpainting import inpaint_background_depth
        d_background = inpaint_background_depth(d_final, inpaint_mask, cuboid_envelope=None, smooth_sigma=2.5)
    else:
        bg_rgb = img_rgb.copy()
        d_background = d_final.copy()

    # 7. Compute 3D surface normals directly from accurate 3D geometry
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

    # Gentle normal & planar height refinement ONLY for un-occluded pure ceiling and floor (never touching 3D objects)
    ry = rays[:, :, 1]
    is_pure_ceil = (ry > 0.70) & (normals_grad[:, :, 1] < -0.80) & (fg_mask == 0)
    is_pure_floor = (ry < -0.70) & (normals_grad[:, :, 1] > 0.80) & (fg_mask == 0)

    if relief_weight < 1.0:
        w_rel = float(np.clip(relief_weight, 0.0, 1.0))
        bg_blend = fg_mask == 0
        d_blended = np.where(bg_blend, w_rel * d_final + (1.0 - w_rel) * d_box, d_final).astype(np.float32)
        P = d_blended[:, :, np.newaxis] * rays
    else:
        if np.any(is_pure_floor):
            median_floor_y = float(np.median(P[is_pure_floor, 1]))
            floor_target_d = np.clip(-median_floor_y / np.maximum(-ry, 1e-4), 0.3, 40.0)
            floor_alpha = np.clip((-ry - 0.70) / 0.25, 0.0, 1.0) * 0.35
            d_floor_reg = (1.0 - floor_alpha) * d_final + floor_alpha * floor_target_d
            P[is_pure_floor] = (d_floor_reg[is_pure_floor, np.newaxis] * rays[is_pure_floor]).astype(np.float32)
        if np.any(is_pure_ceil):
            median_ceil_y = float(np.median(P[is_pure_ceil, 1]))
            ceil_target_d = np.clip(median_ceil_y / np.maximum(ry, 1e-4), 0.3, 40.0)
            ceil_alpha = np.clip((ry - 0.70) / 0.25, 0.0, 1.0) * 0.35
            d_ceil_reg = (1.0 - ceil_alpha) * d_final + ceil_alpha * ceil_target_d
            P[is_pure_ceil] = (d_ceil_reg[is_pure_ceil, np.newaxis] * rays[is_pure_ceil]).astype(np.float32)

    normals_snapped = normals_grad.copy()
    normals_snapped[is_pure_ceil] = [0.0, -1.0, 0.0]
    normals_snapped[is_pure_floor] = [0.0, 1.0, 0.0]

    # At depth edges: force camera-facing billboard normals to prevent stretching sideways
    normals_billboard = -rays.copy()
    nbb_len = np.linalg.norm(normals_billboard, axis=-1, keepdims=True)
    nbb_len[nbb_len == 0] = 1.0
    normals_billboard = normals_billboard / nbb_len

    edge_3d = edge_mask[:, :, np.newaxis]
    normals_final = np.where(edge_3d, normals_billboard, normals_snapped)
    n_final_len = np.linalg.norm(normals_final, axis=-1, keepdims=True)
    n_final_len[n_final_len == 0] = 1.0
    normals_final = (normals_final / n_final_len).astype(np.float32)

    # 8. Filter out extreme flying transition slope pixels so furniture doesn't smear into walls
    flying_mask = detect_depth_edges(d_final, rel_threshold=0.12, abs_threshold=0.25)
    # Never prune nadir/zenith poles (|ry| > 0.88) to prevent black floor/ceiling holes
    flying_mask &= (np.abs(ry) <= 0.88)
    keep_primary = ~flying_mask.reshape(-1)

    flat_P = P.reshape(-1, 3).astype(np.float32)[keep_primary]
    flat_C = img_rgb.reshape(-1, 3).astype(np.uint8)[keep_primary]
    flat_R = img_radiance.reshape(-1, 3).astype(np.float32)[keep_primary]
    flat_N = normals_final.reshape(-1, 3).astype(np.float32)[keep_primary]
    flat_E = edge_mask.reshape(-1)[keep_primary]

    all_points = [flat_P]
    all_colors = [flat_C]
    all_radiance = [flat_R]
    all_normals = [flat_N]
    all_edges = [flat_E]

    # 9. Solid Room Shell Backing: Synthesize solid background points behind occluded furniture
    infilled_count = 0
    if solid_shell and fg_count > 0:
        logger.info("Synthesizing solid backing surfaces behind occluded foreground furniture...")
        occluded_mask = (fg_mask > 0) & (d_background > d_final + 0.20) & (d_background < 25.0)

        if np.any(occluded_mask):
            shell_rays = rays[occluded_mask]
            shell_d = d_background[occluded_mask, np.newaxis]
            shell_pts = (shell_rays * shell_d).astype(np.float32)
            shell_cols = bg_rgb[occluded_mask].astype(np.uint8)

            # Use analytical room box normals for background walls/floors/ceilings so surfels lie flat on the wall
            shell_norms = normals_box[occluded_mask].astype(np.float32)
            sn_len = np.linalg.norm(shell_norms, axis=-1, keepdims=True)
            sn_len[sn_len == 0] = 1.0
            shell_norms = (shell_norms / sn_len).astype(np.float32)

            shell_edge = np.zeros(len(shell_pts), dtype=bool)

            infilled_count = len(shell_pts)
            all_points.append(shell_pts)
            all_colors.append(shell_cols)
            all_radiance.append(srgb_to_linear(shell_cols.astype(np.float32) / 255.0))
            all_normals.append(shell_norms)
            all_edges.append(shell_edge)
            logger.info(f"Synthesized {infilled_count:,} solid backing splats behind occluded furniture.")

    merged_points = np.vstack(all_points).astype(np.float32)
    merged_colors = np.vstack(all_colors).astype(np.uint8)
    merged_radiance = np.vstack(all_radiance).astype(np.float32)
    merged_normals = np.vstack(all_normals).astype(np.float32)
    merged_edges = np.concatenate(all_edges)

    point_cloud = PointCloud(
        points=merged_points,
        colors=merged_colors,
        radiance=merged_radiance,
        normals=merged_normals,
        metadata={
            "total_pixels": H * W,
            "depth_is_metric": is_metric,
            "depth_model": res.model_name,
            "edge_pixel_count": int(edge_count),
            "edge_pixel_ratio": float(edge_count / (H * W)),
            "edge_mask": merged_edges,
            "infilled_points": int(infilled_count),
            "solid_shell": solid_shell,
            "jitter": jitter,
        },
    )

    logger.info(f"Reconstructed {point_cloud.num_points:,} total solid 3D points (V3 engine).")

    # 10. Export PLY if output path provided
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
