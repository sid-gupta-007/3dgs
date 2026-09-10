"""
Synthetic depth estimator for deterministic unit testing and baseline verification.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
from PIL import Image

from panogs.io.images import load_image_as_numpy
from panogs.reconstruction.depth.base import DepthEstimator, DepthResult


class SyntheticDepthEstimator(DepthEstimator):
    """
    Generates deterministic synthetic depth geometries (e.g. box room or radial depth).
    """

    def __init__(self, mode: str = "room", min_depth: float = 0.5, max_depth: float = 5.0):
        self.mode = mode
        self.min_val = min_depth
        self.max_val = max_depth

    def estimate(self, image: Union[np.ndarray, Image.Image, Path, str]) -> DepthResult:
        """Estimate synthetic depth matching the input image dimensions."""
        if isinstance(image, (str, Path)):
            arr = load_image_as_numpy(image)
            height, width = arr.shape[:2]
        elif isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, np.ndarray):
            height, width = image.shape[:2]
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        # Create coordinate grids
        y, x = np.mgrid[0:height, 0:width]
        y_norm = (y - height / 2.0) / (height / 2.0)
        x_norm = (x - width / 2.0) / (width / 2.0)

        if self.mode == "room":
            # Synthetic perspective room depth (corners further, center closer)
            radial = np.sqrt(x_norm**2 + y_norm**2)
            d_map = self.min_val + (self.max_val - self.min_val) * (1.0 - 1.0 / (1.0 + radial))
        elif self.mode == "gradient":
            # Vertical gradient (sky far, floor close)
            d_map = self.min_val + (self.max_val - self.min_val) * (1.0 - (y / float(height)))
        else:
            d_map = np.full((height, width), (self.min_val + self.max_val) / 2.0, dtype=np.float32)

        d_map = d_map.astype(np.float32)
        confidence = np.ones((height, width), dtype=np.float32)

        return DepthResult(
            depth_map=d_map,
            is_metric=True,
            confidence=confidence,
            model_name=f"synthetic_{self.mode}",
            metadata={"min_val": self.min_val, "max_val": self.max_val, "mode": self.mode},
        )
