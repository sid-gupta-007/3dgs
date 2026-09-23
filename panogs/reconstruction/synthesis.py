"""
Generative Multi-Angle Viewpoint Synthesis Engine (PanoGS V3).

Synthesizes novel perspective camera viewpoints around the room to observe occluded
cavities (e.g. behind bathtubs, vanities, furniture), inpainting occluded textures,
inferring metric depth, and fusing the multi-view point clouds into a complete,
hole-free 3D scene representation.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import cv2
from scipy.spatial import cKDTree

from panogs.core.logging import get_logger
from panogs.reconstruction.pointcloud import PointCloud
from panogs.reconstruction.inpainting import inpaint_background_texture


@dataclass
class VirtualCameraView:
    """Represents a virtual perspective camera viewpoint for novel-view synthesis."""
    position: np.ndarray        # (3,) Camera center in world space
    target: np.ndarray          # (3,) Look-at target in world space
    up: np.ndarray              # (3,) World up vector
    fov_deg: float = 75.0       # Horizontal field of view in degrees
    width: int = 512            # Viewport width in pixels
    height: int = 512           # Viewport height in pixels


def create_virtual_camera_rig(
    room_depth: float = 4.5,
    room_width: float = 3.6,
    num_views: int = 4,
    orbit_radius: float = 0.85,
    camera_height: float = 0.0,
) -> List[VirtualCameraView]:
    """
    Generate a rig of virtual perspective cameras positioned at strategic offsets
    around the room center to capture occluded angles.

    Args:
        room_depth: Half-depth of the room in meters.
        room_width: Half-width of the room in meters.
        num_views: Number of virtual cameras to generate (typically 4-8).
        orbit_radius: Offset distance from room center in meters.
        camera_height: Vertical offset from camera origin.

    Returns:
        List of VirtualCameraView objects.
    """
    views = []
    r_x = min(orbit_radius, room_width * 0.45)
    r_z = min(orbit_radius, room_depth * 0.45)

    if num_views <= 4:
        offsets = [
            (r_x, camera_height, 0.0, np.array([0.0, 0.0, 1.0])),     # East view
            (-r_x, camera_height, 0.0, np.array([0.0, 0.0, 1.0])),    # West view
            (0.0, camera_height, r_z, np.array([1.0, 0.0, 0.0])),     # North view
            (0.0, camera_height, -r_z, np.array([1.0, 0.0, 0.0])),    # South view
        ][:num_views]
    else:
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        offsets = [
            (
                float(r_x * np.cos(a)),
                camera_height,
                float(r_z * np.sin(a)),
                np.array([-np.sin(a), 0.0, np.cos(a)]),
            )
            for a in angles
        ]

    for px, py, pz, look_dir in offsets:
        pos = np.array([px, py, pz], dtype=np.float32)
        target = pos + look_dir.astype(np.float32) * 2.0
        views.append(
            VirtualCameraView(
                position=pos,
                target=target,
                up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                fov_deg=75.0,
                width=512,
                height=512,
            )
        )

    return views


def render_depth_and_color_view(
    point_cloud: PointCloud,
    cam: VirtualCameraView,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D point cloud into a virtual perspective camera view using Z-buffering.

    Returns:
        depth_buffer: (H, W) float32 z-depth buffer (inf where empty).
        color_buffer: (H, W, 3) uint8 color buffer.
        normal_buffer: (H, W, 3) float32 normal buffer.
        valid_mask: (H, W) bool mask of observed pixels.
    """
    H, W = cam.height, cam.width
    depth_buf = np.full((H, W), np.inf, dtype=np.float32)
    color_buf = np.zeros((H, W, 3), dtype=np.uint8)
    normal_buf = np.zeros((H, W, 3), dtype=np.float32)

    fwd = cam.target - cam.position
    fwd_len = np.linalg.norm(fwd)
    if fwd_len < 1e-6:
        fwd = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        fwd = fwd / fwd_len

    right = np.cross(fwd, cam.up)
    r_len = np.linalg.norm(right)
    if r_len < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right = right / r_len

    cam_up = np.cross(right, fwd)

    # OpenCV Camera Convention: +X right, +Y down, +Z forward
    R = np.stack([right, -cam_up, fwd], axis=0).astype(np.float32)  # (3, 3)
    t = -R @ cam.position.astype(np.float32)                         # (3,)

    # Intrinsics
    fov_rad = np.radians(cam.fov_deg)
    fx = (W / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx
    cx = W / 2.0
    cy = H / 2.0

    # Transform Points to Camera Frame
    pts = point_cloud.points.astype(np.float32)
    pts_cam = (R @ pts.T + t[:, np.newaxis]).T  # (N, 3)

    # Positive depth only (in front of camera)
    valid = pts_cam[:, 2] > 0.15
    if not np.any(valid):
        return depth_buf, color_buf, normal_buf, np.zeros((H, W), dtype=bool)

    pts_valid = pts_cam[valid]
    colors_valid = point_cloud.colors[valid]
    normals_valid = point_cloud.normals[valid] if point_cloud.normals is not None else np.zeros_like(pts_valid)

    # Perspective Projection
    z = pts_valid[:, 2]
    u = np.round(fx * (pts_valid[:, 0] / z) + cx).astype(np.int32)
    v = np.round(fy * (pts_valid[:, 1] / z) + cy).astype(np.int32)

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u_b = u[in_bounds]
    v_b = v[in_bounds]
    z_b = z[in_bounds]
    c_b = colors_valid[in_bounds]
    n_b = normals_valid[in_bounds]

    # Sort front-to-back to populate Z-buffer
    sort_idx = np.argsort(z_b)[::-1]  # Draw furthest first, closest overrides
    for idx in sort_idx:
        ui, vi, zi = u_b[idx], v_b[idx], z_b[idx]
        if zi < depth_buf[vi, ui]:
            depth_buf[vi, ui] = zi
            color_buf[vi, ui] = c_b[idx]
            normal_buf[vi, ui] = n_b[idx]

    valid_mask = np.isfinite(depth_buf)
    return depth_buf, color_buf, normal_buf, valid_mask


def synthesize_cavity_splats(
    point_cloud: PointCloud,
    cam: VirtualCameraView,
    inpaint_radius: int = 7,
) -> Optional[PointCloud]:
    """
    Project the scene to a virtual novel camera view, detect unobserved occlusion holes,
    inpaint textures and depth in the cavity, and backproject into new 3D Gaussians.

    Returns:
        PointCloud of newly synthesized 3D points, or None if no holes needed filling.
    """
    H, W = cam.height, cam.width
    depth_buf, color_buf, normal_buf, valid_mask = render_depth_and_color_view(point_cloud, cam)

    hole_mask = (~valid_mask).astype(np.uint8)

    # Only inpaint interior cavities, not the outside boundary of the camera view
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_valid = cv2.dilate(valid_mask.astype(np.uint8), kernel, iterations=3)
    cavity_mask = (hole_mask & dilated_valid).astype(bool)

    cavity_pixels = np.count_nonzero(cavity_mask)
    if cavity_pixels < 20:
        return None

    # Inpaint colors in the cavity
    inpainted_color = inpaint_background_texture(color_buf, cavity_mask, inpaint_radius=inpaint_radius, method="telea")

    # Inpaint depths in the cavity using nearest valid neighbor interpolation
    depth_clean = depth_buf.copy()
    depth_clean[~valid_mask] = 0.0
    depth_8u = np.clip((depth_clean / 10.0) * 255.0, 0, 255).astype(np.uint8)
    depth_inpainted_8u = cv2.inpaint(depth_8u, cavity_mask.astype(np.uint8), inpaint_radius, cv2.INPAINT_TELEA)
    depth_inpainted = (depth_inpainted_8u.astype(np.float32) / 255.0) * 10.0
    depth_inpainted = np.maximum(depth_inpainted, 0.3)

    # Backproject synthesized cavity pixels into 3D world coordinates
    fwd = cam.target - cam.position
    fwd = fwd / max(1e-6, np.linalg.norm(fwd))
    right = np.cross(fwd, cam.up)
    right = right / max(1e-6, np.linalg.norm(right))
    cam_up = np.cross(right, fwd)

    R = np.stack([right, -cam_up, fwd], axis=0).astype(np.float32)
    R_inv = R.T
    pos = cam.position.astype(np.float32)

    fov_rad = np.radians(cam.fov_deg)
    fx = (W / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx
    cx = W / 2.0
    cy = H / 2.0

    v_coords, u_coords = np.where(cavity_mask)
    z_vals = depth_inpainted[v_coords, u_coords]

    # Camera coordinates
    x_cam = (u_coords - cx) * z_vals / fx
    y_cam = (v_coords - cy) * z_vals / fy
    pts_cam = np.stack([x_cam, y_cam, z_vals], axis=-1)  # (M, 3)

    # World coordinates: P_world = R_inv @ P_cam + pos
    pts_world = (R_inv @ pts_cam.T).T + pos[np.newaxis, :]
    colors_synth = inpainted_color[v_coords, u_coords]

    # Surface normals: camera-facing towards virtual camera
    norms = pos[np.newaxis, :] - pts_world
    norm_lens = np.linalg.norm(norms, axis=-1, keepdims=True)
    norm_lens[norm_lens == 0] = 1.0
    norms_world = (norms / norm_lens).astype(np.float32)

    return PointCloud(
        points=pts_world.astype(np.float32),
        colors=colors_synth.astype(np.uint8),
        normals=norms_world.astype(np.float32),
    )


def synthesize_multi_view_scene(
    primary_point_cloud: PointCloud,
    room_depth: float = 4.5,
    room_width: float = 3.6,
    num_views: int = 4,
    voxel_consensus_dist: float = 0.025,
) -> PointCloud:
    """
    Perform generative multi-view novel viewpoint synthesis on an indoor room reconstruction.

    1. Deploys a virtual camera rig around the room.
    2. Identifies occluded cavities and renders depth/color z-buffers.
    3. Synthesizes hole fillings via generative inpainting.
    4. Fuses all novel-view point clouds with voxel consensus to prevent duplicates.

    Args:
        primary_point_cloud: Input PointCloud from single-view reconstruction.
        room_depth: Room depth in meters.
        room_width: Room width in meters.
        num_views: Number of novel viewpoints (4-8).
        voxel_consensus_dist: Minimum distance threshold for merging synthesized points.

    Returns:
        Fused PointCloud with solid, complete coverage.
    """
    logger = get_logger("reconstruction.synthesis")
    logger.info(f"Starting Multi-View Novel Viewpoint Synthesis with {num_views} virtual cameras...")

    rig = create_virtual_camera_rig(
        room_depth=room_depth,
        room_width=room_width,
        num_views=num_views,
        orbit_radius=0.85,
    )

    all_points = [primary_point_cloud.points]
    all_colors = [primary_point_cloud.colors]
    all_normals = [primary_point_cloud.normals] if primary_point_cloud.normals is not None else []
    
    # Build spatial KD-tree of existing points for fast duplicate rejection
    tree = cKDTree(primary_point_cloud.points)
    total_synth_points = 0

    for i, cam_view in enumerate(rig):
        logger.info(f"Synthesizing perspective view {i + 1}/{len(rig)} at position {cam_view.position.tolist()}...")
        synth_pc = synthesize_cavity_splats(primary_point_cloud, cam_view)
        if synth_pc is not None and synth_pc.num_points > 0:
            # Query proximity against existing point cloud to reject duplicate splats
            dists, _ = tree.query(synth_pc.points, k=1)
            novel_mask = dists > voxel_consensus_dist

            novel_count = np.count_nonzero(novel_mask)
            if novel_count > 0:
                novel_pts = synth_pc.points[novel_mask]
                novel_cols = synth_pc.colors[novel_mask]
                novel_norms = synth_pc.normals[novel_mask]

                all_points.append(novel_pts)
                all_colors.append(novel_cols)
                all_normals.append(novel_norms)
                total_synth_points += novel_count
                logger.info(f"View {i + 1}: Fused {novel_count:,} synthesized cavity points into scene.")

    merged_points = np.vstack(all_points).astype(np.float32)
    merged_colors = np.vstack(all_colors).astype(np.uint8)
    merged_normals = np.vstack(all_normals).astype(np.float32) if len(all_normals) > 0 else None

    logger.info(
        f"Multi-View Synthesis complete: added {total_synth_points:,} new solid points. "
        f"Total scene points: {len(merged_points):,}."
    )

    meta = dict(primary_point_cloud.metadata) if primary_point_cloud.metadata else {}
    meta["synthesized_points"] = total_synth_points
    meta["num_views_synthesized"] = num_views

    return PointCloud(
        points=merged_points,
        colors=merged_colors,
        normals=merged_normals,
        metadata=meta,
    )
