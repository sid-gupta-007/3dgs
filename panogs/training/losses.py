"""
Loss functions for 3D Gaussian Splatting optimization.
Implements L1 photometric loss, differentiable Structural Similarity Index (SSIM),
and combined (1 - lambda) * L1 + lambda * (1 - SSIM) loss.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_1d_kernel(window_size: int, sigma: float) -> torch.Tensor:
    """Generate 1D normalized Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    gauss = torch.exp(-coords**2 / (2.0 * sigma**2))
    return gauss / gauss.sum()


def _create_gaussian_2d_window(window_size: int = 11, channel: int = 3, sigma: float = 1.5) -> torch.Tensor:
    """Create 2D separable Gaussian smoothing window tensor for depthwise conv2d."""
    _1d = _gaussian_1d_kernel(window_size, sigma).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    channel: int = 3,
    size_average: bool = True,
) -> torch.Tensor:
    """
    Calculate Structural Similarity Index (SSIM) between two image batches.

    Args:
        img1: (B, C, H, W) or (C, H, W) tensor in [0, 1].
        img2: (B, C, H, W) or (C, H, W) tensor in [0, 1].
        window_size: Gaussian kernel window size (default: 11).
        sigma: Gaussian kernel std (default: 1.5).
        channel: Number of color channels (default: 3).
        size_average: Return scalar mean if True.

    Returns:
        torch.Tensor: SSIM value in [0, 1].
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    device = img1.device
    dtype = img1.dtype
    window = _create_gaussian_2d_window(window_size, channel, sigma).to(device=device, dtype=dtype)

    # Constants for numerical stability (C1 = (K1*L)^2, C2 = (K2*L)^2 with L=1.0)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(dim=(1, 2, 3))


def l1_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Mean absolute error (L1 loss) between predicted and target images."""
    return torch.abs(img1 - img2).mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_ssim: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standard 3D Gaussian Splatting loss:
        L = (1 - lambda_ssim) * L1 + lambda_ssim * (1 - SSIM)

    Args:
        pred: (H, W, 3) or (B, 3, H, W) float tensor in [0, 1].
        target: (H, W, 3) or (B, 3, H, W) float tensor in [0, 1].
        lambda_ssim: Weight for SSIM loss component (default: 0.2).

    Returns:
        Tuple: (total_loss, l1_val, ssim_val)
    """
    # Ensure (B, C, H, W)
    if pred.dim() == 3 and pred.shape[-1] == 3:  # (H, W, 3)
        p = pred.permute(2, 0, 1).unsqueeze(0)
        t = target.permute(2, 0, 1).unsqueeze(0)
    elif pred.dim() == 3:  # (C, H, W)
        p = pred.unsqueeze(0)
        t = target.unsqueeze(0)
    else:
        p = pred
        t = target

    loss_l1 = l1_loss(p, t)
    ssim_val = ssim(p, t)
    loss_ssim = 1.0 - ssim_val

    total_loss = (1.0 - lambda_ssim) * loss_l1 + lambda_ssim * loss_ssim
    return total_loss, loss_l1, ssim_val
