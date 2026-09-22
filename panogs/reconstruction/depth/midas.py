"""
MiDaS Monocular Depth Estimator implementation.
Uses lightweight MiDaS_small (~45MB) with CPU execution.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
from PIL import Image

from panogs.core.logging import get_logger
from panogs.io.images import load_image_as_numpy
from panogs.reconstruction.depth.base import DepthEstimator, DepthResult


class MiDaSDepthEstimator(DepthEstimator):
    """
    Monocular depth estimator using MiDaS (Small) via PyTorch Hub.
    Produces relative inverse depth (disparity).
    """

    def __init__(
        self,
        model_type: str = "MiDaS_small",
        device: str = "cpu",
    ):
        model_mapping = {
            "midas_small": "MiDaS_small",
            "dpt_large": "DPT_Large",
            "dpt_hybrid": "DPT_Hybrid",
        }
        self.model_type = model_mapping.get(model_type.lower(), model_type)
        self.device_name = device
        self._model = None
        self._transforms = None
        self.logger = get_logger("depth.midas")

    def _load_model(self):
        """Lazy load MiDaS model and transforms."""
        if self._model is not None:
            return

        import os
        import torch

        # Ensure trusted repos are recorded to avoid interactive prompt freezes
        hub_dir = torch.hub.get_dir()
        os.makedirs(hub_dir, exist_ok=True)
        trusted_file = os.path.join(hub_dir, "trusted_list")
        existing_trusted = set()
        if os.path.exists(trusted_file):
            with open(trusted_file, "r", encoding="utf-8") as f:
                existing_trusted = {line.strip() for line in f}
        
        needed = {"intel-isl_MiDaS", "rwightman_gen-efficientnet-pytorch"}
        to_add = needed - existing_trusted
        if to_add:
            with open(trusted_file, "a", encoding="utf-8") as f:
                for item in to_add:
                    f.write(item + "\n")

        self.logger.info(f"Loading MiDaS model '{self.model_type}' on device '{self.device_name}'...")
        try:
            self._model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
            self._model.to(torch.device(self.device_name))
            self._model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            if self.model_type in ("DPT_Large", "DPT_Hybrid"):
                self._transforms = midas_transforms.dpt_transform
            else:
                self._transforms = midas_transforms.small_transform
            self.logger.info("MiDaS model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load MiDaS model from PyTorch Hub: {e}")
            raise RuntimeError(
                f"Could not load MiDaS model '{self.model_type}'. Ensure internet connectivity for initial download or PyTorch installation: {e}"
            )

    def estimate(self, image: Union[np.ndarray, Image.Image, Path, str]) -> DepthResult:
        """
        Estimate relative inverse depth from an input image.

        Args:
            image: Image array (H, W, 3) in [0, 1] or [0, 255], PIL Image, or path.

        Returns:
            DepthResult: Object containing float32 relative depth map (H, W).
        """
        import torch

        self._load_model()

        # Load RGB image uint8 in [0, 255]
        if isinstance(image, (str, Path)):
            img_np = load_image_as_numpy(image, normalize_float=False)
        elif isinstance(image, Image.Image):
            img_np = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            if image.dtype in (np.float32, np.float64):
                img_np = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                img_np = image.astype(np.uint8)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        orig_h, orig_w = img_np.shape[:2]

        # Apply MiDaS input preprocessing
        input_batch = self._transforms(img_np).to(torch.device(self.device_name))

        with torch.no_grad():
            prediction = self._model(input_batch)

            # Resize disparity map back to original image resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth_np = prediction.cpu().numpy().astype(np.float32)

        return DepthResult(
            depth_map=depth_np,
            is_metric=False,  # MiDaS outputs relative disparity / inverse depth
            confidence=None,
            model_name=f"midas_{self.model_type.lower()}",
            metadata={
                "device": self.device_name,
                "input_resolution": f"{orig_w}x{orig_h}",
                "metric": False,
            },
        )
