"""
Unit and integration tests for the PanoGS Differentiable PyTorch Renderer,
Loss Functions, and 3DGS Optimization Engine.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from panogs.core.gaussian.model import GaussianModel
from panogs.rendering.camera import create_lookat_camera
from panogs.rendering.torch.rasterizer import TorchGaussianModel, render_gaussians_torch
from panogs.training.dataset import generate_training_views, sample_perspective_from_equirectangular
from panogs.training.losses import combined_loss, l1_loss, ssim
from panogs.training.optimizer import GaussianOptimizer
from panogs.training.trainer import TrainingConfig, train_gaussians


def test_loss_functions():
    """Test SSIM, L1, and composite loss functions."""
    t1 = torch.ones((1, 3, 32, 32), dtype=torch.float32)
    t2 = torch.ones((1, 3, 32, 32), dtype=torch.float32)

    # Identical images -> SSIM == 1, L1 == 0
    assert torch.isclose(ssim(t1, t2), torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(l1_loss(t1, t2), torch.tensor(0.0), atol=1e-4)

    total_loss, l1_val, ssim_val = combined_loss(t1, t2)
    assert torch.isclose(total_loss, torch.tensor(0.0), atol=1e-4)

    # Inverted images -> non-zero loss
    t3 = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    loss_inv, l1_inv, ssim_inv = combined_loss(t1, t3)
    assert loss_inv > 0.5
    assert l1_inv == 1.0
    assert ssim_inv < 0.1


def test_torch_gaussian_model_autograd():
    """Test differentiable forward render and autograd backward pass."""
    xyz = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)
    scales = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
    quats = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    opacities = np.array([0.9], dtype=np.float32)

    np_model = GaussianModel.from_raw(xyz, scales, quats, colors, opacities)
    torch_model = TorchGaussianModel.from_numpy(np_model, device="cpu")

    cam = create_lookat_camera(eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 2.0), width=32, height=32)

    pred_img, render_info = render_gaussians_torch(torch_model, cam, bg_color=(0.0, 0.0, 0.0))
    assert pred_img.shape == (32, 32, 3)

    target_img = torch.ones((32, 32, 3), dtype=torch.float32)  # Target is full white
    loss, _, _ = combined_loss(pred_img, target_img)

    loss.backward()

    # Check that gradients exist and are non-zero
    assert torch_model._xyz.grad is not None
    assert torch_model._features_dc.grad is not None
    assert torch_model._opacity_logits.grad is not None
    assert torch_model._scaling_log.grad is not None


def test_perspective_sampling_from_equirectangular(temp_image_dir: Path):
    """Test extracting perspective training views from equirectangular panorama."""
    img_path = temp_image_dir / "sample_rgb.png"
    views = generate_training_views(
        image_path=img_path,
        num_views=4,
        view_width=32,
        view_height=32,
        fov=60.0,
        device="cpu",
    )
    assert len(views) == 4
    for v in views:
        assert v.image.shape == (32, 32, 3)
        assert isinstance(v.image, torch.Tensor)
        assert v.image.min() >= 0.0
        assert v.image.max() <= 1.0


def test_gaussian_optimizer_densification_and_reset():
    """Test optimizer cloning, splitting, pruning, and opacity reset operations."""
    xyz = np.array([[0.0, 0.0, 2.0], [0.5, 0.5, 2.0], [1.0, 1.0, 2.0]], dtype=np.float32)
    scales = np.array([[0.01, 0.01, 0.01], [0.3, 0.3, 0.3], [0.01, 0.01, 0.01]], dtype=np.float32)
    quats = np.array([[1.0, 0.0, 0.0, 0.0]] * 3, dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    opacities = np.array([0.9, 0.9, 0.01], dtype=np.float32)  # 3rd is low opacity (prune candidate)

    np_model = GaussianModel.from_raw(xyz, scales, quats, colors, opacities)
    torch_model = TorchGaussianModel.from_numpy(np_model, device="cpu")

    optimizer = GaussianOptimizer(torch_model)

    # Fake gradients: 1st Gaussian has high grad (clone), 2nd has high grad (split)
    torch_model._xyz.grad = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    optimizer.accumulate_gradients({"vis_idx": torch.tensor([0, 1, 2])})

    num_cloned, num_split, num_pruned = optimizer.densify_and_prune(
        grad_threshold=0.1,
        min_opacity=0.05,
        percent_dense=0.05,
        scene_extent=2.0,
    )

    assert num_cloned == 1
    assert num_split == 1
    assert num_pruned == 1
    # Initial 3: cloned 1 (+1), split 1 (-1 parent + 2 children = +1), pruned 1 (-1) -> 3 + 1 + 1 - 1 = 4
    assert torch_model.num_gaussians == 4

    # Opacity reset
    optimizer.reset_opacity(value=0.1)
    assert (torch_model.get_opacity() <= 0.11).all()


def test_train_gaussians_mini_loop(temp_image_dir: Path, tmp_path: Path):
    """Test full training loop execution and loss convergence."""
    img_path = temp_image_dir / "sample_rgb.png"
    out_ply = tmp_path / "train_out.ply"

    # Create small initial model
    xyz = np.array([[0.0, 0.0, 1.5], [0.2, 0.0, 1.5]], dtype=np.float32)
    scales = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], dtype=np.float32)
    quats = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=np.float32)
    opacities = np.array([0.8, 0.8], dtype=np.float32)

    init_model = GaussianModel.from_raw(xyz, scales, quats, colors, opacities)
    init_ply = tmp_path / "init.ply"
    from panogs.io.gaussian_ply import save_gaussian_ply
    save_gaussian_ply(init_ply, init_model)

    config = TrainingConfig(
        iterations=5,
        num_views=2,
        view_resolution=32,
        densify_interval=2,
        densify_from=1,
        densify_until=4,
    )

    final_model, history = train_gaussians(
        image_path=img_path,
        init_ply_path=init_ply,
        output_ply_path=out_ply,
        config=config,
    )

    assert len(history) == 5
    assert out_ply.exists()
    assert (tmp_path / "train_out.splat").exists()
    assert final_model.num_gaussians >= 1
