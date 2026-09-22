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
        # Multi-scale morphological background estimation
        k_size = max(15, int(min(H, W) * 0.45) | 1)
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
    if image_rgb.dtype != np.uint8:
        img_u8 = (np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        was_float = True
    else:
        img_u8 = image_rgb.copy()
        was_float = False

    # Dilate mask slightly to cover edge antialiasing fringes
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(object_mask, dilate_kernel)

    # OpenCV inpaint works on BGR uint8
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

    flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    inpainted_bgr = cv2.inpaint(img_bgr, dilated_mask, inpaint_radius, flag)

    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

    if was_float:
        return inpainted_rgb.astype(np.float32) / 255.0
    return inpainted_rgb
