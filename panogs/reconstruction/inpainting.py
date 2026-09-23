"""
AI Inpainting & Semantic Layering Engine for 360 Panoramas.
Extracts foreground furniture objects, inpaints occluded background floorboards and walls,
and produces decoupled 3D layers to eliminate 2D wallpaper sticker artifacts.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from panogs.core.logging import get_logger


def segment_foreground_objects(
    depth_map: np.ndarray,
    rel_depth_threshold: float = 0.15,
    min_size_pixels: int = 100,
    background_depth: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Segment freestanding foreground objects (furniture, chairs, tables, plants)
    based on local depth discontinuities against background surfaces.

    Args:
        depth_map: (H, W) float32 depth map in meters.
        rel_depth_threshold: Relative depth difference to declare foreground.
        min_size_pixels: Minimum pixel area to keep as an object.
        background_depth: Optional reference background depth map (e.g. room envelope).

    Returns:
        np.ndarray: (H, W) uint8 binary mask (255 for foreground objects, 0 for background).
    """
    H, W = depth_map.shape
    d = np.maximum(depth_map, 0.01)

    if background_depth is not None:
        bg = np.maximum(background_depth, d)
    else:
        # Morphological dilation to span across foreground objects without exceeding local room envelope
        k_size = max(45, min(75, int(min(H, W) * 0.10)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        bg = cv2.dilate(d, kernel)

    # Foreground is significantly closer than local background
    depth_diff = (bg - d) / np.maximum(d, 0.1)
    raw_fg = (depth_diff > rel_depth_threshold).astype(np.uint8) * 255

    # Morphological cleanup (closing small gaps, removing speckle noise)
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_fg = cv2.morphologyEx(raw_fg, cv2.MORPH_CLOSE, clean_kernel)
    clean_fg = cv2.morphologyEx(clean_fg, cv2.MORPH_OPEN, clean_kernel)

    # Connected component filtering to remove tiny noise islands
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_fg, connectivity=8)
    final_mask = np.zeros((H, W), dtype=np.uint8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size_pixels:
            final_mask[labels == i] = 255

    return final_mask


def inpaint_background_texture(
    image_rgb: np.ndarray,
    object_mask: np.ndarray,
    inpaint_radius: int = 7,
    method: str = "telea",
) -> np.ndarray:
    """
    Inpaint occluded background regions (floor under tables/sofas, walls behind chairs)
    to generate seamless background texture without furniture stickers.

    Args:
        image_rgb: (H, W, 3) uint8 or float32 RGB image.
        object_mask: (H, W) uint8 binary mask (255 where objects are located).
        inpaint_radius: Radius around masked pixels to sample inpaint source colors.
        method: "telea" (Fast Marching) or "ns" (Navier-Stokes fluid inpainting).

    Returns:
        np.ndarray: (H, W, 3) inpainted RGB image with objects removed.
    """
    from scipy.ndimage import distance_transform_edt

    if image_rgb.dtype != np.uint8:
        img_u8 = (np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        was_float = True
    else:
        img_u8 = image_rgb.copy()
        was_float = False

    # Ensure object_mask is uint8 binary (255 / 0)
    if object_mask.dtype != np.uint8:
        mask_u8 = (object_mask > 0).astype(np.uint8) * 255
    else:
        mask_u8 = np.where(object_mask > 0, np.uint8(255), np.uint8(0))

    # Dilate mask slightly to cover edge antialiasing fringes
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(mask_u8, dilate_kernel)

    # Fast multi-scale processing for large panoramas
    orig_h, orig_w = img_u8.shape[:2]
    max_dim = max(orig_h, orig_w)
    if max_dim > 1024:
        scale = 1024.0 / max_dim
        proc_w = int(orig_w * scale)
        proc_h = int(orig_h * scale)
        small_img = cv2.resize(img_u8, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(dilated_mask, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)

        img_bgr = cv2.cvtColor(small_img, cv2.COLOR_RGB2BGR)
        flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
        inpainted_small_bgr = cv2.inpaint(img_bgr, small_mask, inpaint_radius, flag)
        inpainted_small_rgb = cv2.cvtColor(inpainted_small_bgr, cv2.COLOR_BGR2RGB)

        inpainted_rgb = cv2.resize(inpainted_small_rgb, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        # Keep original pixels where mask was not applied
        mask_3d = (dilated_mask > 0)[:, :, np.newaxis]
        inpainted_rgb = np.where(mask_3d, inpainted_rgb, img_u8)
    else:
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
        flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
        inpainted_bgr = cv2.inpaint(img_bgr, dilated_mask, inpaint_radius, flag)
        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

    # Fallback nearest-neighbor boundary fill if inpainting left empty/black holes
    is_empty = (np.mean(inpainted_rgb, axis=-1) < 1.0) & (dilated_mask > 0)
    if np.any(is_empty):
        valid_bg = (dilated_mask == 0)
        if np.any(valid_bg):
            _, indices = distance_transform_edt(~valid_bg, return_indices=True)
            fallback_rgb = img_u8[indices[0], indices[1]]
            inpainted_rgb[is_empty] = fallback_rgb[is_empty]

    if was_float:
        return inpainted_rgb.astype(np.float32) / 255.0
    return inpainted_rgb


def inpaint_background_depth(
    depth_map: np.ndarray,
    object_mask: np.ndarray,
    cuboid_envelope: Optional[np.ndarray] = None,
    smooth_sigma: float = 3.0,
) -> np.ndarray:
    """
    Inpaint occluded metric depth behind foreground objects.
    Uses distance transform boundary propagation and Gaussian smoothing to guarantee
    100% valid, continuous, non-zero depth behind any size occluding structure.

    Args:
        depth_map: (H, W) float32 metric depth map in meters.
        object_mask: (H, W) uint8 binary mask (255 where objects are located).
        cuboid_envelope: Optional (H, W) float32 room box distance prior.
        smooth_sigma: Gaussian smoothing sigma for inpainted regions.

    Returns:
        (H, W) float32 continuous inpainted background depth map.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    d = depth_map.copy().astype(np.float32)
    mask = (object_mask > 0)
    if not np.any(mask):
        return d

    # Dilate mask slightly so boundary transitions blend cleanly
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)

    # Valid background: pixels outside dilated mask with valid positive depth
    valid_bg = (~dilated_mask) & (d > 0.1)
    if not np.any(valid_bg):
        return d if cuboid_envelope is None else cuboid_envelope.copy()

    # Propagate nearest valid background depth into masked region
    _, indices = distance_transform_edt(~valid_bg, return_indices=True)
    d_propagated = d[indices[0], indices[1]]

    # Smooth the propagated depth to eliminate Voronoi boundary ridges
    d_smoothed = gaussian_filter(d_propagated, sigma=smooth_sigma)

    # Inpaint result combines original background with smoothed propagated depth
    d_inpaint = np.where(dilated_mask, d_smoothed, d)

    # If a cuboid room envelope prior is provided, ensure background depth doesn't exceed room envelope
    if cuboid_envelope is not None:
        d_inpaint = np.where(dilated_mask, np.maximum(d_inpaint, cuboid_envelope * 0.95), d_inpaint)

    return d_inpaint.astype(np.float32)

