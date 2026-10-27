"""
Depth Anything V2 Metric Indoor Depth Estimator.
Uses the HuggingFace transformers pipeline to produce true metric depth (meters)
with dramatically sharper object boundaries than MiDaS.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
from PIL import Image

from panogs.core.logging import get_logger
from panogs.reconstruction.depth.base import DepthEstimator, DepthResult


class DepthAnythingV2Estimator(DepthEstimator):
    """
    Metric indoor monocular depth estimator using Depth Anything V2.
    Outputs true metric depth in meters (trained on Hypersim indoor dataset).
    
    Model variants (HuggingFace):
        - Small (25M params): "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
        - Base  (98M params): "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
        - Large (335M params): "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    """

    MODEL_IDS = {
        "small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
        "large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    }

    def __init__(
        self,
        variant: str = "small",
        device: str = "cpu",
    ):
        self.variant = variant.lower()
        self.device_name = device
        self._pipe = None
        self.logger = get_logger("depth.depth_anything_v2")

        if self.variant not in self.MODEL_IDS:
            raise ValueError(
                f"Unknown Depth Anything V2 variant '{variant}'. "
                f"Choose from: {list(self.MODEL_IDS.keys())}"
            )
        self.model_id = self.MODEL_IDS[self.variant]

    def _load_pipeline(self):
        """Lazy-load the HuggingFace depth estimation pipeline."""
        if self._pipe is not None:
            return

        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise RuntimeError(
                "The 'transformers' package is required for Depth Anything V2. "
                "Install it with: pip install transformers>=4.45.0"
            )

        self.logger.info(
            f"Loading Depth Anything V2 Metric Indoor ({self.variant}) on '{self.device_name}'..."
        )
        self._pipe = hf_pipeline(
            task="depth-estimation",
            model=self.model_id,
            device=self.device_name if self.device_name != "cpu" else -1,
        )
        self.logger.info("Depth Anything V2 model loaded successfully.")

    @property
    def is_metric(self) -> bool:
        return True

    def estimate(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
    ) -> DepthResult:
        """
        Estimate metric depth (meters) from a single perspective image.
        
        Args:
            image_input: PIL Image, numpy array (H, W, 3), or file path.
            
        Returns:
            DepthResult with is_metric=True, depth_map in meters.
        """
        self._load_pipeline()

        # Convert input to PIL Image
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype == np.float32 or image_input.dtype == np.float64:
                arr = (np.clip(image_input, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                arr = image_input.astype(np.uint8)
            pil_img = Image.fromarray(arr, "RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Run inference
        result = self._pipe(pil_img)
        
        # The pipeline returns {"depth": PIL.Image, "predicted_depth": torch.Tensor}
        if hasattr(result, "get") and "predicted_depth" in result:
            import torch
            depth_tensor = result["predicted_depth"]
            if isinstance(depth_tensor, torch.Tensor):
                depth_map = depth_tensor.squeeze().cpu().numpy()
            else:
                depth_map = np.array(depth_tensor, dtype=np.float32)
        elif hasattr(result, "get") and "depth" in result:
            depth_pil = result["depth"]
            depth_map = np.array(depth_pil, dtype=np.float32)
            if depth_map.max() > 100:
                depth_map = depth_map / 1000.0
        else:
            raise RuntimeError(f"Unexpected depth estimation result format: {type(result)}")

        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze()

        # Resize to match input if needed
        input_h, input_w = pil_img.height, pil_img.width
        if depth_map.shape != (input_h, input_w):
            from scipy.ndimage import zoom
            scale_h = input_h / depth_map.shape[0]
            scale_w = input_w / depth_map.shape[1]
            depth_map = zoom(depth_map, (scale_h, scale_w), order=1)

        return DepthResult(
            depth_map=depth_map.astype(np.float32),
            is_metric=True,
            model_name=f"depth_anything_v2_metric_indoor_{self.variant}",
            metadata={
                "variant": self.variant,
                "model_id": self.model_id,
                "max_depth_m": float(depth_map.max()),
                "min_depth_m": float(depth_map.min()),
            },
        )
