"""
Depth map I/O and persistence module.
Saves raw float32 arrays, metadata JSON, and colormapped visualizations.
"""

import json
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image

from panogs.reconstruction.depth.base import DepthEstimator, DepthResult


def save_depth_result(
    result: DepthResult,
    output_dir: Union[str, Path],
    prefix: str = "depth",
) -> Tuple[Path, Path, Path]:
    """
    Save depth results to a folder:
        - {prefix}.npy: Exact uncompressed float32 numpy array
        - {prefix}_vis.png: High-contrast RGB visualization image
        - {prefix}_metadata.json: Metadata, metric flag, and range diagnostics

    Returns:
        Tuple[Path, Path, Path]: (npy_path, vis_png_path, meta_json_path)
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    npy_path = target_dir / f"{prefix}.npy"
    vis_path = target_dir / f"{prefix}_vis.png"
    meta_path = target_dir / f"{prefix}_metadata.json"

    # 1. Save raw float32 numpy array
    np.save(npy_path, result.depth_map.astype(np.float32))

    # 2. Save colormapped visualization PNG
    vis_rgb = DepthEstimator.colorize(result)
    Image.fromarray(vis_rgb).save(vis_path)

    # 3. Save metadata JSON
    meta_data = {
        "model_name": result.model_name,
        "is_metric": result.is_metric,
        "height": result.height,
        "width": result.width,
        "min_depth": result.min_depth,
        "max_depth": result.max_depth,
        "mean_depth": result.mean_depth,
        "has_confidence": result.confidence is not None,
        "custom_metadata": result.metadata,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    return npy_path, vis_path, meta_path


def load_depth_result(
    npy_path: Union[str, Path],
    meta_path: Optional[Union[str, Path]] = None,
) -> DepthResult:
    """
    Load a raw depth array and optional metadata file into a DepthResult.
    """
    npy_file = Path(npy_path)
    if not npy_file.exists():
        raise FileNotFoundError(f"Depth .npy file not found: {npy_file}")

    depth_arr = np.load(npy_file).astype(np.float32)

    is_metric = False
    model_name = "loaded"
    metadata = {}

    if meta_path is not None and Path(meta_path).exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            is_metric = meta.get("is_metric", False)
            model_name = meta.get("model_name", "loaded")
            metadata = meta.get("custom_metadata", {})

    return DepthResult(
        depth_map=depth_arr,
        is_metric=is_metric,
        model_name=model_name,
        metadata=metadata,
    )
