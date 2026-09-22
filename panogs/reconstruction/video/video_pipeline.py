"""
Multi-View Video-to-3DGS Reconstruction Pipeline.
Processes walking video clips (.mp4/.mov), estimates camera trajectory,
fuses multi-view metric depth, and reconstructs solid 3D Gaussian Splatting scenes.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from panogs.core.logging import get_logger
from panogs.io.ply import write_point_cloud_ply
from panogs.io.video_loader import extract_video_keyframes, VideoFrame
from panogs.reconstruction.depth.base import DepthEstimator
from panogs.reconstruction.pointcloud import PointCloud
from panogs.reconstruction.processing import process_point_cloud


def estimate_camera_motion(
    prev_rgb: np.ndarray,
    curr_rgb: np.ndarray,
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate relative camera rotation R and unit translation t between adjacent frames
    using ORB feature matching and the Essential Matrix 5-point RANSAC algorithm.

    Args:
        prev_rgb: (H, W, 3) previous RGB frame.
        curr_rgb: (H, W, 3) current RGB frame.
        K: (3, 3) camera intrinsic matrix.

    Returns:
        R: (3, 3) relative rotation matrix.
        t: (3, 1) relative translation direction vector.
    """
    gray1 = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return np.eye(3, dtype=np.float32), np.array([[0.0], [0.0], [0.1]], dtype=np.float32)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    if len(matches) < 8:
        return np.eye(3, dtype=np.float32), np.array([[0.0], [0.0], [0.1]], dtype=np.float32)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:200]])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:200]])

    E, inliers = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        return np.eye(3, dtype=np.float32), np.array([[0.0], [0.0], [0.1]], dtype=np.float32)

    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R.astype(np.float32), t.astype(np.float32)


def reconstruct_from_video(
    video_path: Union[str, Path],
    depth_estimator: Optional[DepthEstimator] = None,
    target_fps: float = 2.0,
    max_frames: int = 60,
    voxel_size: float = 0.03,
    output_ply: Optional[Union[str, Path]] = None,
) -> PointCloud:
    """
    Complete end-to-end Video-to-3DGS Reconstruction Pipeline:
    Video -> Keyframes -> Depth Estimation -> Camera Tracking -> Multi-View Point Fusion -> PLY

    Args:
        video_path: Path to video file (.mp4, .mov).
        depth_estimator: Depth estimator instance (defaults to Depth Anything V2).
        target_fps: Sampling rate in FPS.
        max_frames: Maximum keyframes to process.
        voxel_size: Voxel downsampling grid size in meters.
        output_ply: Target PLY path.

    Returns:
        PointCloud: Unified multi-view reconstructed 3D point cloud.
    """
    logger = get_logger("reconstruction.video")
    video_path = Path(video_path)

    # 1. Extract sharp video keyframes
    keyframes = extract_video_keyframes(
        video_path=video_path,
        target_fps=target_fps,
        max_frames=max_frames,
    )

    if len(keyframes) == 0:
        raise ValueError(f"No usable keyframes found in video: {video_path}")

    # 2. Depth estimator instantiation
    if depth_estimator is None:
        from panogs.reconstruction.depth.depth_anything import DepthAnythingV2Estimator
        logger.info("Initializing Depth Anything V2 Metric Indoor estimator...")
        depth_estimator = DepthAnythingV2Estimator(variant="small", device="cpu")

    H, W = keyframes[0].image_rgb.shape[:2]
    # Standard pinhole intrinsic matrix approximation (FOV ~ 65 deg)
    fov_deg = 65.0
    focal = (W / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
    K = np.array([
        [focal, 0.0, W / 2.0],
        [0.0, focal, H / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    all_world_points: List[np.ndarray] = []
    all_world_colors: List[np.ndarray] = []

    # Camera trajectory state (World-to-Camera pose: R_cw, C_w)
    R_cw = np.eye(3, dtype=np.float32)
    C_w = np.zeros((3, 1), dtype=np.float32)

    logger.info(f"Fusing {len(keyframes)} multi-view video frames into 3D scene...")

    # Ray grid in camera space
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u, v)
    x_cam = (u_grid - K[0, 2]) / K[0, 0]
    y_cam = (v_grid - K[1, 2]) / K[1, 1]
    rays_cam = np.stack([x_cam, y_cam, np.ones_like(x_cam)], axis=-1)  # (H, W, 3)

    for i, kf in enumerate(keyframes):
        rgb = kf.image_rgb

        # Estimate camera motion relative to previous frame
        if i > 0:
            R_rel, t_rel = estimate_camera_motion(keyframes[i - 1].image_rgb, rgb, K)
            # Accumulate pose: R_curr = R_rel @ R_prev, C_curr = C_prev + R_prev.T @ (t_rel * scale)
            step_scale = 0.15  # Average walking step ~15cm per keyframe
            C_w = C_w + np.matmul(R_cw.T, t_rel * step_scale)
            R_cw = np.matmul(R_rel, R_cw)

        # Infer metric depth
        res = depth_estimator.estimate(rgb)
        depth = res.depth_map.astype(np.float32)
        depth = np.clip(depth, 0.3, 15.0)

        # 3D points in camera coordinates: P_cam = depth * rays_cam
        P_cam = (depth[:, :, np.newaxis] * rays_cam).reshape(-1, 3)
        colors_flat = rgb.reshape(-1, 3)

        # Transform to world coordinates: P_world = R_cw^T @ P_cam + C_w
        P_world = np.matmul(P_cam, R_cw) + C_w.T

        # Subsample each frame for memory efficiency
        step = 2
        all_world_points.append(P_world[::step].astype(np.float32))
        all_world_colors.append(colors_flat[::step].astype(np.uint8))

    merged_points = np.vstack(all_world_points)
    merged_colors = np.vstack(all_world_colors)

    logger.info(f"Merged {len(merged_points):,} raw points from {len(keyframes)} video views.")

    # 3. Clean and downsample unified point cloud
    raw_pc = PointCloud(points=merged_points, colors=merged_colors)
    cleaned_pc = process_point_cloud(
        point_cloud=raw_pc,
        voxel_size=voxel_size,
        remove_outliers=True,
        compute_normals=True,
    )

    logger.info(
        f"Multi-view video reconstruction complete: {cleaned_pc.num_points:,} filtered points "
        f"with surface normals."
    )

    # 4. Export PLY if path provided
    if output_ply is not None:
        write_point_cloud_ply(
            output_ply,
            cleaned_pc.points,
            cleaned_pc.colors,
            normals=cleaned_pc.normals,
            binary=True,
        )
        logger.info(f"Saved video reconstruction PLY to {output_ply}")

    return cleaned_pc
