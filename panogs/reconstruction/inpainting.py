"""
AI Inpainting & Semantic Layering Engine for 360 Panoramas.
Extracts foreground furniture objects, inpaints occluded background floorboards and walls,
and produces decoupled 3D layers to eliminate 2D wallpaper sticker artifacts and occlusion voids.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from panogs.core.logging import get_logger


def _estimate_room_envelope_depth(
    depth_map: np.ndarray,
    background_depth: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate the outer background wall/floor/ceiling depth envelope across wide occluders.

    Standard small-kernel morphological dilation (e.g. 75px) fails on wide foreground
    partitions, curtains, beds, or sofas (>100px wide), leaving their interiors unsegmented.
    We combine a coarse multi-scale morphological closing/dilation (spanning ~25% of panorama
    width) with fine local dilation so foreground structures of any width are captured.
    """
    H, W = depth_map.shape
    d = np.maximum(depth_map.astype(np.float32), 0.05)

    # Fine-scale local dilation
    k_fine = max(45, min(85, int(min(H, W) * 0.11)) | 1)
    kernel_fine = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_fine, k_fine))
    bg_fine = cv2.dilate(d, kernel_fine)

    # Coarse-scale dilation (handles wider partitions, curtains, tables without bleeding across rooms)
    scale = max(1, min(6, min(H, W) // 48))
    if scale > 1:
        small_h = max(8, H // scale)
        small_w = max(16, W // scale)
        d_small = cv2.resize(d, (small_w, small_h), interpolation=cv2.INTER_AREA)
        pad_w = max(4, small_w // 6)
        d_padded = np.pad(d_small, ((0, 0), (pad_w, pad_w)), mode="wrap")
        k_coarse_w = max(9, min(25, int(small_w * 0.14)) | 1)
        k_coarse_h = max(9, min(21, int(small_h * 0.16)) | 1)
        kernel_coarse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_coarse_w, k_coarse_h))
        bg_small_pad = cv2.dilate(d_padded, kernel_coarse)
        bg_small = bg_small_pad[:, pad_w : pad_w + small_w]
        bg_small_smooth = cv2.GaussianBlur(bg_small, (0, 0), sigmaX=1.5, sigmaY=1.5)
        bg_coarse = cv2.resize(bg_small_smooth, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        bg_coarse = bg_fine

    bg = np.maximum(bg_fine, bg_coarse)
    if background_depth is not None:
        # Clamp wild over-dilations while allowing real room envelope depth
        bg = np.minimum(bg, np.maximum(background_depth * 1.25, d))
    return bg.astype(np.float32)


def segment_foreground_objects(
    depth_map: np.ndarray,
    rel_depth_threshold: float = 0.15,
    min_size_pixels: int = 100,
    background_depth: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Segment freestanding foreground objects (furniture, chairs, tables, plants, partitions)
    based on multi-scale depth discontinuities against background room surfaces.

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

    bg = _estimate_room_envelope_depth(d, background_depth=background_depth)

    # Foreground is significantly closer than local background envelope
    depth_diff = (bg - d) / np.maximum(d, 0.1)
    raw_fg = (depth_diff > rel_depth_threshold).astype(np.uint8) * 255

    # Morphological cleanup (closing interior holes, removing speckle noise)
    close_k = max(5, min(15, int(min(H, W) * 0.02)) | 1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_fg = cv2.morphologyEx(raw_fg, cv2.MORPH_CLOSE, close_kernel)
    clean_fg = cv2.morphologyEx(clean_fg, cv2.MORPH_OPEN, open_kernel)

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
    Inpaint occluded background regions (floor under tables/sofas, walls behind partitions/chairs)
    to generate seamless background texture without furniture stickers or black holes.

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

    if object_mask.dtype != np.uint8:
        mask_u8 = (object_mask > 0).astype(np.uint8) * 255
    else:
        mask_u8 = np.where(object_mask > 0, np.uint8(255), np.uint8(0))

    # Dilate mask slightly to cover foreground silhouette antialiasing fringes
    # so foreground white/dark fringes never bleed into the inpainted background wall.
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated_mask = cv2.dilate(mask_u8, dilate_kernel)

    orig_h, orig_w = img_u8.shape[:2]
    max_dim = max(orig_h, orig_w)
    if max_dim > 768:
        scale = 768.0 / max_dim
        proc_w = max(16, int(orig_w * scale))
        proc_h = max(8, int(orig_h * scale))
        small_img = cv2.resize(img_u8, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(dilated_mask, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)

        img_bgr = cv2.cvtColor(small_img, cv2.COLOR_RGB2BGR)
        flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
        inpainted_small_bgr = cv2.inpaint(img_bgr, small_mask, max(inpaint_radius, 9), flag)
        inpainted_small_rgb = cv2.cvtColor(inpainted_small_bgr, cv2.COLOR_BGR2RGB)
        # Lightly smooth deep interior of wide inpainted regions to prevent patchy seams
        blurred_small = cv2.GaussianBlur(inpainted_small_rgb, (0, 0), sigmaX=3.0)
        inner_mask = cv2.erode(small_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
        inpainted_small_rgb = np.where(inner_mask[:, :, None], blurred_small, inpainted_small_rgb)

        inpainted_rgb = cv2.resize(inpainted_small_rgb, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
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
        valid_bg = dilated_mask == 0
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
    Uses inverse-depth (disparity) harmonic interpolation across masked regions so
    planar walls and floors remain flat behind wide occluding structures.

    Args:
        depth_map: (H, W) float32 metric depth map in meters.
        object_mask: (H, W) uint8 binary mask (255 where objects are located).
        cuboid_envelope: Optional (H, W) float32 room box distance prior.
        smooth_sigma: Gaussian smoothing sigma for inpainted regions.

    Returns:
        (H, W) float32 continuous inpainted background depth map.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    d = np.maximum(depth_map.copy().astype(np.float32), 0.1)
    mask = object_mask > 0
    if not np.any(mask):
        return d

    # Dilate mask generously so foreground depth ramps never contaminate the background seed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)

    valid_bg = (~dilated_mask) & (d > 0.1)
    if not np.any(valid_bg):
        return d if cuboid_envelope is None else cuboid_envelope.copy()

    # Work in inverse depth (1/d) because planar surfaces have linear 1/d in perspective/spherical rays
    inv_d = 1.0 / d
    _, indices = distance_transform_edt(~valid_bg, return_indices=True)
    inv_prop = inv_d[indices[0], indices[1]]

    # Multi-pass harmonic relaxation in inverse-depth space inside dilated_mask
    sigma_large = max(float(smooth_sigma) * 3.0, 8.0)
    inv_smooth = gaussian_filter(inv_prop, sigma=sigma_large, mode=("nearest", "wrap"))
    inv_prop = np.where(dilated_mask, inv_smooth, inv_d)
    inv_smooth2 = gaussian_filter(inv_prop, sigma=max(float(smooth_sigma), 3.0), mode=("nearest", "wrap"))
    inv_final = np.where(dilated_mask, inv_smooth2, inv_d)

    d_inpaint = 1.0 / np.maximum(inv_final, 1e-3)

    # Ensure inpainted background is strictly behind the foreground object
    envelope = _estimate_room_envelope_depth(d, background_depth=cuboid_envelope)
    d_inpaint = np.where(dilated_mask, np.maximum(d_inpaint, envelope * 0.92), d)

    if cuboid_envelope is not None:
        d_inpaint = np.where(dilated_mask, np.maximum(d_inpaint, cuboid_envelope * 0.90), d_inpaint)

    return d_inpaint.astype(np.float32)
