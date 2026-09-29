"""
LightGaussian: Unbounded 3D Gaussian Compression and Significance-Based Pruning.
Implements importance-based pruning and distillation techniques (Fan et al., arXiv:2311.17245)
to reduce model size, improve framerates, and remove visual floaters with minimal PSNR/SSIM loss.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

from panogs.core.gaussian.model import GaussianModel
from panogs.core.logging import get_logger
from panogs.io.gaussian_ply import load_gaussian_ply, save_gaussian_ply, save_gaussian_splat


@dataclass
class CompressionConfig:
    """Configuration for LightGaussian importance-based pruning and compression."""
    prune_ratio: float = 0.30  # Fraction of lowest-importance Gaussians to prune (0.0 to 0.8)
    min_opacity_threshold: float = 0.02  # Absolute cutoff for low-opacity floaters
    volume_weight: float = 1.0  # Weight for 3D spatial volume in significance score
    opacity_weight: float = 1.0  # Weight for opacity in significance score
    max_scale_clamp: float = 0.5  # Max scale clamp to prune unconstrained giant sky floaters


def compute_gaussian_importance(
    model: GaussianModel,
    config: Optional[CompressionConfig] = None,
) -> np.ndarray:
    """
    Calculate importance/significance score for each Gaussian in the model.
    Significance score S_i = (opacity_i ^ opacity_weight) * (Volume_i ^ volume_weight)
    Gaussians with near-zero opacity or negligible/degenerate footprint receive low scores.

    Args:
        model: GaussianModel instance.
        config: Compression configuration.

    Returns:
        np.ndarray: (N,) importance score array in range [0, 1].
    """
    if config is None:
        config = CompressionConfig()

    opacity = np.clip(model.get_opacity().ravel(), 0.0, 1.0)
    scales = model.get_scaling()  # (N, 3)

    # 3D Gaussian Volume: proportional to s_x * s_y * s_z
    volume = np.prod(np.clip(scales, 1e-6, 10.0), axis=1)

    # Normalize volume to [0, 1] robustly using percentiles to avoid outlier domination
    v_min, v_max = np.percentile(volume, 1.0), np.percentile(volume, 99.0)
    v_norm = np.clip((volume - v_min) / max(1e-6, v_max - v_min), 1e-4, 1.0)

    # Importance formula from LightGaussian:
    # S_i = opacity^w_op * volume^w_vol
    importance = (opacity ** config.opacity_weight) * (v_norm ** config.volume_weight)

    # Penalize unconstrained giant floaters (splats with giant scale but isolated in space)
    giant_mask = np.any(scales > config.max_scale_clamp, axis=1)
    importance[giant_mask] *= 0.1

    # Penalize sub-threshold opacities
    low_alpha_mask = opacity < config.min_opacity_threshold
    importance[low_alpha_mask] = 0.0

    return importance.astype(np.float32)


def prune_gaussians_by_importance(
    model: GaussianModel,
    prune_ratio: float = 0.30,
    min_opacity: float = 0.02,
) -> Tuple[GaussianModel, np.ndarray]:
    """
    Prune a fraction of least significant 3D Gaussians.

    Args:
        model: Input GaussianModel.
        prune_ratio: Percentage of Gaussians to prune (0.0 = keep all, 0.5 = keep top 50%).
        min_opacity: Hard minimum opacity threshold.

    Returns:
        Tuple: (pruned GaussianModel, keep_mask boolean array)
    """
    logger = get_logger("core.compression")
    num_before = model.num_gaussians

    if num_before == 0 or prune_ratio <= 0.0:
        return model, np.ones(num_before, dtype=bool)

    cfg = CompressionConfig(prune_ratio=prune_ratio, min_opacity_threshold=min_opacity)
    scores = compute_gaussian_importance(model, cfg)

    target_keep_count = max(1, int(num_before * (1.0 - prune_ratio)))
    sorted_indices = np.argsort(scores)
    keep_indices = sorted_indices[-target_keep_count:]

    keep_mask = np.zeros(num_before, dtype=bool)
    keep_mask[keep_indices] = True

    pruned_model = GaussianModel(
        xyz=model._xyz[keep_mask],
        scaling_log=model._scaling_log[keep_mask],
        rotation_quats=model._rotation_quats[keep_mask],
        opacity_logits=model._opacity_logits[keep_mask],
        features_dc=model._features_dc[keep_mask],
    )

    num_after = pruned_model.num_gaussians
    pct_saved = ((num_before - num_after) / num_before) * 100
    logger.info(
        f"LightGaussian Pruning: {num_before:,} -> {num_after:,} splats "
        f"({pct_saved:.1f}% reduction, retained {100 - pct_saved:.1f}%)."
    )

    return pruned_model, keep_mask


def compress_scene(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    preset: str = "balanced",
    save_splat: bool = True,
) -> GaussianModel:
    """
    Compress a 3D Gaussian Splatting scene (.ply or .splat) using quality presets.

    Presets:
        - "crisp": 15% prune ratio, keeps maximum detail and subpixel edges.
        - "balanced": 30% prune ratio, optimal quality/speed tradeoff.
        - "fast": 50% prune ratio, maximizes WebGL framerates.

    Args:
        input_path: Path to input .ply or .splat.
        output_path: Destination path for compressed model.
        preset: "crisp", "balanced", or "fast".
        save_splat: Also save binary WebGL .splat file.

    Returns:
        GaussianModel: Compressed model.
    """
    logger = get_logger("core.compression")
    input_path = Path(input_path)
    output_path = Path(output_path)

    preset_ratios = {
        "crisp": 0.15,
        "balanced": 0.30,
        "fast": 0.50,
    }
    prune_ratio = preset_ratios.get(preset.lower(), 0.30)

    logger.info(f"Loading scene {input_path} for compression (preset: {preset})...")
    model = load_gaussian_ply(input_path)

    compressed_model, _ = prune_gaussians_by_importance(model, prune_ratio=prune_ratio)

    save_gaussian_ply(output_path, compressed_model)
    logger.info(f"Saved compressed PLY to {output_path}")

    if save_splat:
        splat_out = output_path.with_suffix(".splat")
        save_gaussian_splat(splat_out, compressed_model)
        logger.info(f"Saved compressed SPLAT to {splat_out}")

    return compressed_model
