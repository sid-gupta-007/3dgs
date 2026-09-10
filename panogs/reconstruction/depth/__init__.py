"""Monocular depth estimation interfaces and models."""

from typing import Optional
from panogs.reconstruction.depth.base import DepthEstimator, DepthResult
from panogs.reconstruction.depth.synthetic import SyntheticDepthEstimator


def get_depth_estimator(model_name: str = "midas_small", device: str = "cpu") -> DepthEstimator:
    """
    Factory function to instantiate depth estimators by name.

    Supported model names:
        - "midas_small" (default, lightweight MiDaS CPU)
        - "synthetic_room" (deterministic test box room)
        - "synthetic_gradient" (vertical gradient)
    """
    model_name_lower = model_name.lower()
    if model_name_lower.startswith("synthetic"):
        mode = "gradient" if "gradient" in model_name_lower else "room"
        return SyntheticDepthEstimator(mode=mode)
    elif "cubemap" in model_name_lower:
        from panogs.reconstruction.depth.cubemap import CubemapDepthEstimator
        base = "MiDaS_small"
        if "hybrid" in model_name_lower:
            base = "DPT_Hybrid"
        return CubemapDepthEstimator(base_model=base, device=device)
    elif "midas" in model_name_lower:
        from panogs.reconstruction.depth.midas import MiDaSDepthEstimator
        variant = "MiDaS_small"
        if "hybrid" in model_name_lower:
            variant = "DPT_Hybrid"
        elif "large" in model_name_lower:
            variant = "DPT_Large"
        return MiDaSDepthEstimator(model_type=variant, device=device)
    else:
        raise ValueError(f"Unknown depth estimator model '{model_name}'. Choose 'cubemap_midas', 'midas_small', or 'synthetic_room'.")


__all__ = [
    "DepthEstimator",
    "DepthResult",
    "SyntheticDepthEstimator",
    "get_depth_estimator",
]
