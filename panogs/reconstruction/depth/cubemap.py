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


def _bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample a 2D map at floating point pixel coordinates."""
    height, width = image.shape
    u = np.clip(u, 0.0, width - 1.0)
    v = np.clip(v, 0.0, height - 1.0)
    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = u - x0
    wy = v - y0
    return (
        image[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + image[y0, x1] * wx * (1.0 - wy)
        + image[y1, x0] * (1.0 - wx) * wy
        + image[y1, x1] * wx * wy
    )


def _align_relative_faces(
    depth_maps: List[np.ndarray], cameras: List[Camera], face_size: int
) -> List[np.ndarray]:
    """Align relative depth predictions between overlapping perspective faces.

    Monocular relative-depth models can choose a different affine depth range for
    each input image. Estimate pairwise affine mappings from rays visible in both
    faces, then propagate mappings from the front face through the overlap graph.
    """
    count = len(depth_maps)
    focal = cameras[0].fx
    cx, cy = cameras[0].cx, cameras[0].cy
    axis = np.arange(face_size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    rays_cam = np.stack(((xx - cx) / focal, (yy - cy) / focal,
                         np.ones_like(xx)), axis=-1)
    rays_world = [rays_cam.reshape(-1, 3) @ camera.R for camera in cameras]

    # Each edge stores a robust fit mapping raw child-face depth to raw parent depth.
    adjacency: List[List[Tuple[int, float, float]]] = [[] for _ in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            rays_i = rays_world[i]
            rays_j_cam = rays_i @ cameras[j].R.T
            z = rays_j_cam[:, 2]
            valid = z > 1e-5
            u = cameras[j].fx * rays_j_cam[:, 0] / np.maximum(z, 1e-5) + cameras[j].cx
            v = cameras[j].fy * rays_j_cam[:, 1] / np.maximum(z, 1e-5) + cameras[j].cy
            valid &= (u >= 1) & (u < face_size - 1) & (v >= 1) & (v < face_size - 1)
            ids = np.flatnonzero(valid)
            if len(ids) < 256:
                continue
            # Bound the calibration cost while retaining broad spatial coverage.
            if len(ids) > 12000:
                ids = ids[np.linspace(0, len(ids) - 1, 12000).astype(np.int32)]
            di = depth_maps[i].reshape(-1)[ids]
            dj = _bilinear_sample(depth_maps[j], u[ids], v[ids])
            finite = np.isfinite(di) & np.isfinite(dj)
            di, dj = di[finite], dj[finite]
            if len(di) < 256:
                continue

            # Robustly fit di ~= a*dj+b by trimming large residuals twice.
            keep = np.ones(len(di), dtype=bool)
            a, b = 1.0, 0.0
            for _ in range(3):
                if np.count_nonzero(keep) < 128:
                    break
                design = np.column_stack((dj[keep], np.ones(np.count_nonzero(keep))))
                a, b = np.linalg.lstsq(design, di[keep], rcond=None)[0]
                residual = np.abs(di - (a * dj + b))
                cutoff = np.percentile(residual, 80)
                keep = residual <= max(float(cutoff), 1e-6)
            if np.isfinite(a) and np.isfinite(b) and a > 1e-5:
                adjacency[i].append((j, float(a), float(b)))
                adjacency[j].append((i, float(1.0 / a), float(-b / a)))

    transforms: List[Optional[Tuple[float, float]]] = [None] * count
    transforms[0] = (1.0, 0.0)
    queue = [0]
    while queue:
        parent = queue.pop(0)
        parent_scale, parent_bias = transforms[parent]
        for child, a, b in adjacency[parent]:
            if transforms[child] is not None:
                continue
            transforms[child] = (parent_scale * a, parent_scale * b + parent_bias)
            queue.append(child)

    # Disconnected faces retain their own robust range; this is a fallback for
    # unusual camera layouts or very small face sizes, not the normal path.
    aligned = []
    for depth_map, transform in zip(depth_maps, transforms):
        if transform is None:
            lo, hi = np.percentile(depth_map, (2, 98))
            scale = 1.0 / max(float(hi - lo), 1e-6)
            aligned.append(((depth_map - lo) * scale).astype(np.float32))
        else:
            scale, bias = transform
            aligned.append((depth_map * scale + bias).astype(np.float32))
    return aligned


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
                # A small overlap lets neighboring perspective estimates be
                # calibrated against common content and blended at cube seams.
                fov=100.0,
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
                face_cameras.append(cam)
                face_disparities.append(disp.astype(np.float32))

        if not self._is_metric:
            self.logger.info("Aligning relative depth scales across overlapping panorama views...")
            face_disparities = _align_relative_faces(face_disparities, face_cameras, F)

        self.logger.info("Fusing 6 cubemap depth maps into continuous equirectangular disparity...")

        # Generate equirectangular rays: (H, W, 3)
        rays = equirectangular_rays(H_pano, W_pano)
        # Blend all overlapping views. Process chunks to keep memory bounded for
        # 4K panoramas; only two full-resolution accumulators are retained.
        flat_rays = rays.reshape(-1, 3)
        fused = np.zeros(len(flat_rays), dtype=np.float32)
        chunk_size = 262144
        feather = max(4.0, F * 0.12)
        for start in range(0, len(flat_rays), chunk_size):
            end = min(start + chunk_size, len(flat_rays))
            ray_chunk = flat_rays[start:end]
            chunk_sum = np.zeros(end - start, dtype=np.float32)
            chunk_weight = np.zeros(end - start, dtype=np.float32)
            for cam, depth_face in zip(face_cameras, face_disparities):
                p_cam = ray_chunk @ cam.R.T
                tz = p_cam[:, 2]
                safe_z = np.maximum(tz, 1e-5)
                u = cam.fx * p_cam[:, 0] / safe_z + cam.cx
                v = cam.fy * p_cam[:, 1] / safe_z + cam.cy
                valid = (tz > 1e-5) & (u >= 0.0) & (u <= F - 1.0) & (v >= 0.0) & (v <= F - 1.0)
                if not np.any(valid):
                    continue
                uv, vv = u[valid], v[valid]
                edge = np.minimum.reduce((uv, vv, F - 1.0 - uv, F - 1.0 - vv))
                t = np.clip(edge / feather, 0.0, 1.0)
                weight = (t * t * (3.0 - 2.0 * t) + 0.02).astype(np.float32)
                sampled = _bilinear_sample(depth_face, uv, vv)
                if self._is_metric:
                    sampled = sampled / safe_z[valid]
                chunk_sum[valid] += sampled * weight
                chunk_weight[valid] += weight
            fused[start:end] = chunk_sum / np.maximum(chunk_weight, 1e-6)
        fused_disparity = fused.reshape(H_pano, W_pano)

        if self._is_metric:
            # Metric depth: preserve absolute meter values
            return DepthResult(
                depth_map=fused_disparity.astype(np.float32),
                is_metric=True,
                model_name=self.model_name,
            )
        else:
            # Non-metric: normalize fused disparity to [0, 1]
            disp_min, disp_max = np.percentile(fused_disparity, (2.0, 98.0))
            fused_norm = np.clip(
                (fused_disparity - disp_min) / max(1e-6, disp_max - disp_min), 0.0, 1.0
            )
            return DepthResult(
                depth_map=fused_norm.astype(np.float32),
                is_metric=False,
                model_name=self.model_name,
            )
