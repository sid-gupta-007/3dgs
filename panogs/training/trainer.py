"""
End-to-end 3D Gaussian Splatting Trainer.
Orchestrates training views, gradient descent, densification, metrics logging,
and standard 3DGS PLY / SPLAT export.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import List, Optional, Tuple, Union
import numpy as np
import torch

from panogs.core.gaussian.model import GaussianModel
from panogs.core.logging import get_logger
from panogs.io.gaussian_ply import load_gaussian_ply, save_gaussian_ply, save_gaussian_splat
from panogs.rendering.torch.rasterizer import TorchGaussianModel, render_gaussians_torch
from panogs.training.dataset import TrainingView, generate_training_views
from panogs.training.losses import combined_loss
from panogs.training.optimizer import GaussianOptimizer


@dataclass
class TrainingConfig:
    """Configuration parameters for 3DGS training."""
    iterations: int = 150
    lr_xyz: float = 1.6e-4
    lr_scale: float = 5.0e-3
    lr_rotation: float = 1.0e-3
    lr_opacity: float = 5.0e-2
    lr_feature: float = 2.5e-3
    lambda_ssim: float = 0.2
    num_views: int = 8
    view_resolution: int = 128
    fov: float = 75.0
    densify_from: int = 10
    densify_until: int = 120
    densify_interval: int = 25
    opacity_reset_interval: int = 50
    grad_threshold: float = 0.0002
    min_opacity: float = 0.05
    device: str = "cpu"


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def train_gaussians(
    image_path: Union[str, Path],
    init_ply_path: Union[str, Path],
    output_ply_path: Union[str, Path],
    config: Optional[TrainingConfig] = None,
    save_splat: bool = True,
) -> Tuple[GaussianModel, List[dict]]:
    """
    Execute 3D Gaussian Splatting scene optimization against perspective views extracted from input image.

    Args:
        image_path: Path to input panorama or image.
        init_ply_path: Initialized 3DGS PLY path.
        output_ply_path: Target output path for optimized PLY.
        config: Training configuration.
        save_splat: Whether to also save .splat binary file.

    Returns:
        Tuple: (optimized GaussianModel, history log list)
    """
    if config is None:
        config = TrainingConfig()

    logger = get_logger("training.trainer")
    device = config.device

    image_path = Path(image_path)
    init_ply_path = Path(init_ply_path)
    output_ply_path = Path(output_ply_path)

    logger.info(f"Loading initial 3D Gaussians from {init_ply_path}...")
    np_model = load_gaussian_ply(init_ply_path)
    torch_model = TorchGaussianModel.from_numpy(np_model, device=device)

    logger.info(
        f"Generating {config.num_views} perspective training views ({config.view_resolution}x{config.view_resolution}) "
        f"from {image_path}..."
    )
    training_views = generate_training_views(
        image_path=image_path,
        num_views=config.num_views,
        view_width=config.view_resolution,
        view_height=config.view_resolution,
        fov=config.fov,
        device=device,
    )

    optimizer = GaussianOptimizer(
        model=torch_model,
        lr_xyz=config.lr_xyz,
        lr_scale=config.lr_scale,
        lr_rotation=config.lr_rotation,
        lr_opacity=config.lr_opacity,
        lr_feature=config.lr_feature,
    )

    logger.info(
        f"Starting 3DGS optimization: {config.iterations} iterations on {device.upper()} "
        f"({torch_model.num_gaussians:,} initial Gaussians)..."
    )

    history: List[dict] = []
    start_time = time.time()

    for step in range(1, config.iterations + 1):
        # Pick viewpoint round-robin or randomly
        view_idx = (step - 1) % len(training_views)
        view = training_views[view_idx]

        optimizer.zero_grad()

        # Differentiable forward render
        pred_rgb, render_info = render_gaussians_torch(
            model=torch_model,
            camera=view.camera,
            bg_color=(1.0, 1.0, 1.0),
        )

        # Calculate loss
        loss, l1_val, ssim_val = combined_loss(
            pred=pred_rgb,
            target=view.image,
            lambda_ssim=config.lambda_ssim,
        )

        if loss.requires_grad:
            loss.backward()
            # Positional gradient tracking for densification
            optimizer.accumulate_gradients(render_info)
            optimizer.step()

        # Densification step
        if (
            config.densify_from <= step <= config.densify_until
            and step % config.densify_interval == 0
        ):
            optimizer.densify_and_prune(
                grad_threshold=config.grad_threshold,
                min_opacity=config.min_opacity,
            )

        # Periodic opacity reset
        if step % config.opacity_reset_interval == 0 and step < config.densify_until:
            optimizer.reset_opacity(value=0.1)

        psnr_val = calculate_psnr(pred_rgb, view.image)

        # Logging
        if step == 1 or step % 10 == 0 or step == config.iterations:
            elapsed = time.time() - start_time
            logger.info(
                f"Step {step:03d}/{config.iterations} | Loss: {loss.item():.4f} (L1: {l1_val.item():.4f}, SSIM: {ssim_val.item():.3f}) "
                f"| PSNR: {psnr_val:.2f} dB | Gaussians: {torch_model.num_gaussians:,} | Time: {elapsed:.1f}s"
            )

        history.append({
            "step": step,
            "loss": loss.item(),
            "l1": l1_val.item(),
            "ssim": ssim_val.item(),
            "psnr": psnr_val,
            "num_gaussians": torch_model.num_gaussians,
        })

    total_time = time.time() - start_time
    logger.info(f"Optimization finished in {total_time:.2f}s.")

    # Export final model
    final_np_model = torch_model.to_numpy_model()
    save_gaussian_ply(output_ply_path, final_np_model)
    logger.info(f"Saved optimized 3DGS PLY to {output_ply_path}")

    if save_splat:
        splat_path = output_ply_path.with_suffix(".splat")
        save_gaussian_splat(splat_path, final_np_model)
        logger.info(f"Saved optimized WebGL SPLAT to {splat_path}")

    return final_np_model, history
