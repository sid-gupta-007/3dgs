"""
CPU Gaussian Splatting Rasterizer.
Implements perspective Jacobian projection, 2D covariance calculation,
depth sorting, and front-to-back alpha compositing in vectorized NumPy.
"""

from typing import Optional, Tuple
import numpy as np

from panogs.core.gaussian.model import GaussianModel
from panogs.core.logging import get_logger
from panogs.rendering.camera import Camera


def project_gaussians_to_2d(
    model: GaussianModel,
    camera: Camera,
    min_depth: float = 0.05,
    low_pass_filter: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D Gaussians into 2D screen space:
        1. World-to-Camera coordinate transformation
        2. Frustum near-plane culling (tz > min_depth)
        3. Pinhole screen projection
        4. Jacobian affine approximation: Sigma_2D = J * W * Sigma_3D * W^T * J^T + low_pass * I

    Args:
        model: GaussianModel instance.
        camera: Camera instance.
        min_depth: Near plane cutoff in meters.
        low_pass_filter: Filter added to 2D covariance diagonal for subpixel antialiasing.

    Returns:
        Tuple:
            - visible_indices: (M,) indices of visible Gaussians.
            - depths: (M,) camera-space Z depths (tz).
            - screen_pts: (M, 2) 2D screen coordinates (xs, ys).
            - cov2d: (M, 2, 2) 2D screen covariance matrices.
            - radii: (M,) bounding box radii in pixels.
    """
    xyz = model.get_xyz()
    cov3d = model.get_covariance_3d()

    # 1. Transform positions to camera space: t = R @ p + t_cam
    p_cam = camera.world_to_camera(xyz)  # (N, 3)
    tx = p_cam[:, 0]
    ty = p_cam[:, 1]
    tz = p_cam[:, 2]

    # 2. Near-plane culling
    visible_mask = tz > min_depth
    if not np.any(visible_mask):
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2, 2), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    vis_idx = np.nonzero(visible_mask)[0]
    tx = tx[vis_idx]
    ty = ty[vis_idx]
    tz = tz[vis_idx]
    cov3d_vis = cov3d[vis_idx]

    fx = camera.fx
    fy = camera.fy
    cx = camera.cx
    cy = camera.cy

    # 3. Perspective projection
    xs = (fx * tx / tz) + cx
    ys = (fy * ty / tz) + cy
    screen_pts = np.stack([xs, ys], axis=-1).astype(np.float32)

    # 4. Transform 3D covariance to camera view frame: Sigma_cam = R_view @ Sigma_3D @ R_view^T
    # R_view: (3, 3)
    R_view = camera.R  # (3, 3)
    # cov3d_cam = R_view @ cov3d_vis @ R_view^T
    cov3d_cam = np.matmul(R_view[np.newaxis, :, :], np.matmul(cov3d_vis, R_view.T[np.newaxis, :, :]))

    # 5. Jacobian of projection J: (M, 2, 3)
    M = len(vis_idx)
    tz2 = tz * tz

    J = np.zeros((M, 2, 3), dtype=np.float32)
    J[:, 0, 0] = fx / tz
    J[:, 0, 2] = - (fx * tx) / tz2
    J[:, 1, 1] = fy / tz
    J[:, 1, 2] = - (fy * ty) / tz2

    # 6. Compute 2D covariance: Sigma_2D = J @ cov3d_cam @ J^T
    cov2d = np.matmul(J, np.matmul(cov3d_cam, J.transpose(0, 2, 1)))

    # Add low-pass filter to diagonal
    cov2d[:, 0, 0] += low_pass_filter
    cov2d[:, 1, 1] += low_pass_filter

    # 7. Compute bounding box radii (3 * sqrt(max(cov_xx, cov_yy)))
    r_x = 3.0 * np.sqrt(np.maximum(cov2d[:, 0, 0], 1e-4))
    r_y = 3.0 * np.sqrt(np.maximum(cov2d[:, 1, 1], 1e-4))
    radii = np.maximum(r_x, r_y).astype(np.float32)

    return vis_idx, tz, screen_pts, cov2d, radii


def render_gaussians_cpu(
    model: GaussianModel,
    camera: Camera,
    bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_depth: float = 0.05,
) -> np.ndarray:
    """
    Render 3D Gaussians from the viewpoint of `camera` on CPU using front-to-back alpha compositing.

    Args:
        model: GaussianModel containing 3D Gaussians.
        camera: Viewpoint camera.
        bg_color: RGB background color in [0, 1] (default: white (1, 1, 1)).
        min_depth: Near plane clipping distance.

    Returns:
        np.ndarray: (H, W, 3) uint8 RGB rendered image.
    """
    logger = get_logger("rendering.cpu")
    H, W = camera.height, camera.width

    # 1. Project Gaussians to 2D screen space
    vis_idx, depths, screen_pts, cov2d, radii = project_gaussians_to_2d(
        model, camera, min_depth=min_depth
    )

    if len(vis_idx) == 0:
        logger.warning("No Gaussians visible in current camera frustum.")
        bg_rgb = (np.array(bg_color, dtype=np.float32) * 255.0).astype(np.uint8)
        return np.full((H, W, 3), bg_rgb, dtype=np.uint8)

    # 2. Front-to-back depth sorting (ascending order of camera depth tz)
    sort_order = np.argsort(depths)
    vis_idx = vis_idx[sort_order]
    screen_pts = screen_pts[sort_order]
    cov2d = cov2d[sort_order]
    radii = radii[sort_order]

    # Pre-fetch colors and opacities for sorted visible Gaussians
    all_colors = model.get_rgb()       # (N, 3) float [0, 1]
    all_opacities = model.get_opacity().flatten()  # (N,) float [0, 1]

    sorted_colors = all_colors[vis_idx]
    sorted_opacities = all_opacities[vis_idx]

    # Precompute 2D covariance inverse determinants: det = a*c - b^2
    a = cov2d[:, 0, 0]
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1]
    dets = a * c - b * b
    valid_dets = dets > 1e-6

    # Buffers for alpha compositing
    # Accumulated color C (H, W, 3) float32 in [0, 1]
    acc_color = np.zeros((H, W, 3), dtype=np.float32)
    # Remaining transmittance T (H, W) float32 in [0, 1]
    transmittance = np.ones((H, W), dtype=np.float32)

    M = len(vis_idx)
    rendered_count = 0

    # 3. Rasterize and alpha composite each Gaussian
    for i in range(M):
        if not valid_dets[i]:
            continue

        cx, cy = screen_pts[i]
        rad = radii[i]
        det = dets[i]

        # Bounding box in integer pixel coordinates
        u_min = max(0, int(np.floor(cx - rad)))
        u_max = min(W - 1, int(np.ceil(cx + rad)))
        v_min = max(0, int(np.floor(cy - rad)))
        v_max = min(H - 1, int(np.ceil(cy + rad)))

        if u_min > u_max or v_min > v_max:
            continue

        # Check if bounding box region is already fully saturated (transmittance near 0)
        T_box = transmittance[v_min:v_max + 1, u_min:u_max + 1]
        if np.max(T_box) < 1e-4:
            continue

        # Generate 2D pixel coordinate grid for the bounding box
        u_range = np.arange(u_min, u_max + 1, dtype=np.float32)
        v_range = np.arange(v_min, v_max + 1, dtype=np.float32)
        u_grid, v_grid = np.meshgrid(u_range, v_range, indexing="xy")

        dx = u_grid - cx
        dy = v_grid - cy

        # Inverse 2D covariance quadratic form:
        # Delta^T * Sigma^-1 * Delta = (c * dx^2 - 2 * b * dx * dy + a * dy^2) / det
        inv_a = a[i]
        inv_b = b[i]
        inv_c = c[i]
        power = -0.5 * (inv_c * dx * dx - 2.0 * inv_b * dx * dy + inv_a * dy * dy) / det

        # Keep points with power <= 0
        inside = power <= 0.0
        if not np.any(inside):
            continue

        # Gaussian density G(p) in [0, 1]
        G = np.zeros_like(power)
        G[inside] = np.exp(power[inside])

        # Alpha = opacity * G
        alpha = sorted_opacities[i] * G

        # Skip subpixel contributions (< 1/255)
        alpha_mask = alpha >= (1.0 / 255.0)
        if not np.any(alpha_mask):
            continue

        # Over-operator compositing weight: w = alpha * T
        w = alpha * T_box
        color_i = sorted_colors[i]  # (3,)

        # Accumulate color: C += w * color_i
        acc_color[v_min:v_max + 1, u_min:u_max + 1] += w[:, :, np.newaxis] * color_i[np.newaxis, np.newaxis, :]
        # Update transmittance: T *= (1 - alpha)
        transmittance[v_min:v_max + 1, u_min:u_max + 1] *= (1.0 - alpha)

        rendered_count += 1

    # 4. Composite remaining background
    bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
    final_rgb = acc_color + transmittance[:, :, np.newaxis] * bg
    final_uint8 = (np.clip(final_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    logger.debug(f"Rendered {rendered_count:,} / {M:,} visible Gaussians to {W}x{H} frame.")
    return final_uint8
