"""
Base interfaces and data structures for monocular depth estimation.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
from PIL import Image


@dataclass
class DepthResult:
    """
    Standard container for depth estimation results.

    Attributes:
        depth_map: 2D numpy array of shape (H, W) with float32 values.
        is_metric: True if values represent metric depth (in meters),
                   False if values represent relative inverse depth (disparity).
        confidence: Optional 2D array of shape (H, W) in [0.0, 1.0].
        model_name: Identifier for the model/method used.
        metadata: Additional model or inference diagnostics.
    """
    depth_map: np.ndarray
    is_metric: bool = False
    confidence: Optional[np.ndarray] = None
    model_name: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.depth_map = np.asarray(self.depth_map, dtype=np.float32)
        if self.depth_map.ndim != 2:
            raise ValueError(f"Depth map must be a 2D array (H, W), got shape {self.depth_map.shape}")
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=np.float32)
            if self.confidence.shape != self.depth_map.shape:
                raise ValueError(
                    f"Confidence shape {self.confidence.shape} does not match depth shape {self.depth_map.shape}"
                )

    @property
    def height(self) -> int:
        return self.depth_map.shape[0]

    @property
    def width(self) -> int:
        return self.depth_map.shape[1]

    @property
    def min_depth(self) -> float:
        return float(np.nanmin(self.depth_map))

    @property
    def max_depth(self) -> float:
        return float(np.nanmax(self.depth_map))

    @property
    def mean_depth(self) -> float:
        return float(np.nanmean(self.depth_map))

    def normalized(self) -> np.ndarray:
        """Return depth normalized to [0.0, 1.0] for visualization."""
        d = self.depth_map.copy()
        d_min = np.nanmin(d)
        d_max = np.nanmax(d)
        if d_max - d_min < 1e-8:
            return np.zeros_like(d)
        return (d - d_min) / (d_max - d_min)


def apply_turbo_colormap(normalized_depth: np.ndarray) -> np.ndarray:
    """
    Apply Google Turbo colormap to normalized values in [0, 1].
    Implemented in pure vectorized NumPy for maximum speed and zero extra dependencies.

    Returns:
        np.ndarray: (H, W, 3) uint8 RGB array.
    """
    x = np.clip(normalized_depth, 0.0, 1.0)

    # 4th-order polynomial approximation coefficients for Google Turbo colormap
    r = (
        0.1357 + x * (4.61539 - x * (42.6603 - x * (132.131 - x * (152.552 - x * 60.106))))
    )
    g = (
        0.0914 + x * (2.19418 + x * (16.4223 - x * (57.4583 - x * (59.213 - x * 19.387))))
    )
    b = (
        0.1067 + x * (12.5833 - x * (86.8774 - x * (272.766 - x * (346.505 - x * 153.228))))
    )

    r = np.clip(r, 0.0, 1.0)
    g = np.clip(g, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


class DepthEstimator(ABC):
    """Abstract base class for all monocular depth estimators."""

    @abstractmethod
    def estimate(self, image: Union[np.ndarray, Image.Image, Path, str]) -> DepthResult:
        """
        Estimate depth from an input image.

        Args:
            image: Image as numpy array (H, W, 3), PIL Image, or file path.

        Returns:
            DepthResult: Object containing depth map and metadata.
        """
        pass

    @staticmethod
    def colorize(depth_result: DepthResult) -> np.ndarray:
        """
        Create a high-contrast RGB visual representation of a depth map.

        Returns:
            np.ndarray: (H, W, 3) uint8 RGB image.
        """
        norm_d = depth_result.normalized()
        return apply_turbo_colormap(norm_d)
