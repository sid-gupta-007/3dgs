"""
Training and optimization subpackage for 3D Gaussian Splatting.
"""

from panogs.training.losses import ssim, l1_loss, combined_loss
from panogs.training.dataset import TrainingView, generate_training_views
from panogs.training.optimizer import GaussianOptimizer
from panogs.training.trainer import TrainingConfig, train_gaussians, calculate_psnr

__all__ = [
    "ssim",
    "l1_loss",
    "combined_loss",
    "TrainingView",
    "generate_training_views",
    "GaussianOptimizer",
    "TrainingConfig",
    "train_gaussians",
    "calculate_psnr",
]
