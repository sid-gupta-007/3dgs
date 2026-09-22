"""
Cubemap-based Multi-Perspective Depth Estimator for 360 Panoramas.
Eliminates equirectangular barrel distortion by projecting the 360 panorama
into 6 rectilinear pinhole cubemap faces, running perspective neural depth inference,
and seamlessly fusing the depth maps back into equirectangular space.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from panogs.core.camera.spherical import equirectangular_rays
from panogs.core.logging import get_logger
from panogs.io.images import load_image
from panogs.reconstruction.depth.base import DepthEstimator, DepthResult
from panogs.reconstruction.depth.midas import MiDaSDepthEstimator
from panogs.rendering.camera import Camera, create_lookat_camera
from panogs.training.dataset import sample_perspective_from_equirectangular


class CubemapDepthEstimator(DepthEstimator):
    """
    Estimates undistorted 3D scene depth by decomposing the 360 panorama into
    6 rectilinear perspective cubemap views (Front, Right, Back, Left, Top, Bottom),
    inferring perspective depth, and re-projecting into equirectangular coordinates.
    
    Supports any underlying DepthEstimator (MiDaS, Depth Anything V2, etc).
    When the underlying estimator produces metric depth, this estimator preserves
    the metric scale and does NOT normalize to [0,1].
    """

    def __init__(
        self,
        base_model: str = "midas_small",
        device: str = "cpu",
        face_size: int = 512,
        underlying_estimator: Optional[DepthEstimator] = None,
    ):
        self.device = device
        self.face_size = face_size
        self.logger = get_logger("depth.cubemap")
        
        if underlying_estimator is not None:
            self.underlying_estimator = underlying_estimator
            self.model_name = f"cubemap_{type(underlying_estimator).__name__}"
            self._is_metric = getattr(underlying_estimator, 'is_metric', False)
        else:
            self.underlying_estimator = MiDaSDepthEstimator(model_type=base_model, device=device)
            self.model_name = f"cubemap_{base_model}"
            self._is_metric = False

    @property
    def is_metric(self) -> bool:
        return self._is_metric

    def estimate(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
    ) -> DepthResult:
        """
        Estimate undistorted depth map for a 360 panorama using 6-face cubemap projection.
        """
        if isinstance(image_input, (str, Path)):
            pil_img = load_image(image_input)
            pano_rgb = np.array(pil_img, dtype=np.float32) / 255.0
        elif isinstance(image_input, Image.Image):
            pano_rgb = np.array(image_input.convert("RGB"), dtype=np.float32) / 255.0
        else:
            if image_input.dtype == np.uint8:
                pano_rgb = image_input.astype(np.float32) / 255.0
            else:
                pano_rgb = image_input.astype(np.float32)

        H_pano, W_pano = pano_rgb.shape[:2]
        F = self.face_size

        self.logger.info(f"Decomposing {W_pano}x{H_pano} panorama into 6 rectilinear cubemap faces ({F}x{F})...")

        # Define 6 cubemap cameras: Front, Right, Back, Left, Top, Bottom
        face_configs = [
            ("front",  (0.0, 0.0, 1.0),  (0.0, 1.0, 0.0)),
            ("right",  (1.0, 0.0, 0.0),  (0.0, 1.0, 0.0)),
            ("back",   (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
            ("left",   (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ("top",    (0.0, 1.0, 0.0),  (0.0, 0.0, -1.0)),
            ("bottom", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
        ]

        face_cameras: List[Camera] = []
        face_disparities: List[np.ndarray] = []

        for name, target, up in face_configs:
            cam = create_lookat_camera(
                eye=(0.0, 0.0, 0.0),
                target=target,
                up=up,
                width=F,
                height=F,
                fov=90.0,
            )
            # Sample perspective RGB face
            face_rgb = sample_perspective_from_equirectangular(pano_rgb, cam)
            face_pil = Image.fromarray((face_rgb * 255.0).astype(np.uint8))

            # Run depth estimation on perspective face (straight walls & flat planes)
            res = self.underlying_estimator.estimate(face_pil)
            disp = res.depth_map

            if self._is_metric:
                # Preserve metric depth values (meters) — no normalization
                face_cameras.append(cam)
                face_disparities.append(disp)
            else:
                # Normalize non-metric disparity to [0, 1]
                disp_norm = (disp - disp.min()) / max(1e-6, disp.max() - disp.min())
                face_cameras.append(cam)
                face_disparities.append(disp_norm)

        self.logger.info("Fusing 6 cubemap depth maps into continuous equirectangular disparity...")

        # Generate equirectangular rays: (H, W, 3)
        rays = equirectangular_rays(H_pano, W_pano)
        dx = rays[:, :, 0]
        dy = rays[:, :, 1]
        dz = rays[:, :, 2]

        abs_x = np.abs(dx)
        abs_y = np.abs(dy)
        abs_z = np.abs(dz)

        # Fused disparity map
        fused_disparity = np.zeros((H_pano, W_pano), dtype=np.float32)

        # Vectorized reprojection for each ray
        # Face selection based on major axis
        is_front = (dz >= abs_x) & (dz >= abs_y) & (dz > 0)
        is_back = (-dz >= abs_x) & (-dz >= abs_y) & (dz < 0)
        is_right = (dx >= abs_y) & (dx >= abs_z) & (dx > 0)
        is_left = (-dx >= abs_y) & (-dx >= abs_z) & (dx < 0)
        is_top = (dy >= abs_x) & (dy >= abs_z) & (dy > 0)
        is_bottom = (-dy >= abs_x) & (-dy >= abs_z) & (dy < 0)

        masks = [is_front, is_right, is_back, is_left, is_top, is_bottom]

        for i, mask in enumerate(masks):
            if not np.any(mask):
                continue
            cam = face_cameras[i]
            disp_face = face_disparities[i]

            # Transform rays to camera space
            rays_sub = rays[mask]
            # p_cam = R @ ray
            p_cam = np.matmul(rays_sub, cam.R.T)
            tx = p_cam[:, 0]
            ty = p_cam[:, 1]
            tz = np.maximum(p_cam[:, 2], 1e-4)

            # Perspective projection onto face
            u_proj = (cam.fx * tx / tz) + cam.cx
            v_proj = (cam.fy * ty / tz) + cam.cy

            # Sub-pixel bilinear interpolation
            u_clamped = np.clip(u_proj, 0.0, float(F - 1))
            v_clamped = np.clip(v_proj, 0.0, float(F - 1))

            x0 = np.floor(u_clamped).astype(np.int32)
            x1 = np.clip(x0 + 1, 0, F - 1)
            y0 = np.floor(v_clamped).astype(np.int32)
            y1 = np.clip(y0 + 1, 0, F - 1)

            wx = (u_clamped - x0).astype(np.float32)
            wy = (v_clamped - y0).astype(np.float32)

            Ia = disp_face[y0, x0]
            Ib = disp_face[y0, x1]
            Ic = disp_face[y1, x0]
            Id = disp_face[y1, x1]

            sampled = (
                Ia * (1.0 - wx) * (1.0 - wy)
                + Ib * wx * (1.0 - wy)
                + Ic * (1.0 - wx) * wy
                + Id * wx * wy
            )

            if self._is_metric:
                # Convert pinhole planar Z-depth to Euclidean radial ray distance: r = Z / tz
                sampled_radial = sampled / tz
                fused_disparity[mask] = sampled_radial
            else:
                fused_disparity[mask] = sampled

        if self._is_metric:
            # Metric depth: preserve absolute meter values
            return DepthResult(
                depth_map=fused_disparity.astype(np.float32),
                is_metric=True,
                model_name=self.model_name,
            )
        else:
            # Non-metric: normalize fused disparity to [0, 1]
            disp_min = fused_disparity.min()
            disp_max = fused_disparity.max()
            fused_norm = (fused_disparity - disp_min) / max(1e-6, disp_max - disp_min)
            return DepthResult(
                depth_map=fused_norm.astype(np.float32),
                is_metric=False,
                model_name=self.model_name,
            )
