"""
3D Gaussian Splatting Optimizer and Adaptive Densification Engine.
Implements multi-parameter Adam optimization, 2D positional gradient accumulation,
cloning, splitting, opacity resets, and pruning.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.optim as optim

from panogs.core.gaussian.model import GaussianModel, logit
from panogs.core.logging import get_logger
from panogs.rendering.torch.rasterizer import TorchGaussianModel


class GaussianOptimizer:
    """
    Manages parameter updates, learning rate schedules, and adaptive densification for 3DGS.
    """

    def __init__(
        self,
        model: TorchGaussianModel,
        lr_xyz: float = 1.6e-4,
        lr_scale: float = 5.0e-3,
        lr_rotation: float = 1.0e-3,
        lr_opacity: float = 5.0e-2,
        lr_feature: float = 2.5e-3,
    ):
        self.model = model
        self.logger = get_logger("training.optimizer")

        self.lr_xyz = lr_xyz
        self.lr_scale = lr_scale
        self.lr_rotation = lr_rotation
        self.lr_opacity = lr_opacity
        self.lr_feature = lr_feature

        # Initialize parameter groups and optimizer
        self._setup_optimizer()

        # Gradient accumulation state for densification
        self.xyz_grad_accum = torch.zeros((model.num_gaussians, 1), device=model._xyz.device)
        self.denom = torch.zeros((model.num_gaussians, 1), device=model._xyz.device)

    def _setup_optimizer(self):
        """Create PyTorch Adam optimizer with separate parameter learning rates."""
        param_groups = [
            {"params": [self.model._xyz], "lr": self.lr_xyz, "name": "xyz"},
            {"params": [self.model._scaling_log], "lr": self.lr_scale, "name": "scaling"},
            {"params": [self.model._rotation_quats], "lr": self.lr_rotation, "name": "rotation"},
            {"params": [self.model._opacity_logits], "lr": self.lr_opacity, "name": "opacity"},
            {"params": [self.model._features_dc], "lr": self.lr_feature, "name": "feature"},
        ]
        self.optimizer = optim.Adam(param_groups, eps=1e-15)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        self.optimizer.step()

    def accumulate_gradients(self, render_info: dict):
        """
        Accumulate screen-space or 3D positional gradients for densification metrics.
        """
        vis_idx = render_info.get("vis_idx")
        if self.model._xyz.grad is not None:
            # Norm of 3D gradient for visible points
            grads = torch.norm(self.model._xyz.grad, dim=-1, keepdim=True)
            if vis_idx is not None and len(vis_idx) > 0:
                self.xyz_grad_accum[vis_idx] += grads[vis_idx]
                self.denom[vis_idx] += 1
            else:
                self.xyz_grad_accum += grads
                self.denom += 1

    def reset_opacity(self, value: float = 0.1):
        """Reset opacities to a lower threshold to purge floaters."""
        with torch.no_grad():
            new_logits = logit(np.array([value], dtype=np.float32))[0]
            curr_opacities = self.model.get_opacity()
            # Set to min(current_opacity, value)
            reset_mask = curr_opacities > value
            if torch.any(reset_mask):
                self.model._opacity_logits.data[reset_mask] = float(new_logits)
        self.logger.debug(f"Reset opacities to {value:.2f}")

    def densify_and_prune(
        self,
        grad_threshold: float = 0.0002,
        min_opacity: float = 0.05,
        max_scale: float = 0.5,
        percent_dense: float = 0.01,
        scene_extent: float = 3.0,
    ) -> Tuple[int, int, int]:
        """
        Perform cloning, splitting, and pruning on the Gaussian model.

        Returns:
            Tuple[int, int, int]: (num_cloned, num_split, num_pruned)
        """
        device = self.model._xyz.device

        # Compute average gradients
        valid_denom = torch.clamp(self.denom, min=1.0)
        avg_grads = (self.xyz_grad_accum / valid_denom).squeeze(-1)  # (N,)

        scales = self.model.get_scaling()  # (N, 3)
        max_scales = torch.max(scales, dim=-1).values  # (N,)
        scale_threshold = percent_dense * scene_extent

        # High gradient masks
        high_grad_mask = avg_grads >= grad_threshold

        # 1. Clone candidates: high gradient & small scale
        clone_mask = high_grad_mask & (max_scales < scale_threshold)
        num_cloned = int(clone_mask.sum().item())

        # 2. Split candidates: high gradient & large scale
        split_mask = high_grad_mask & (max_scales >= scale_threshold)
        num_split = int(split_mask.sum().item())

        with torch.no_grad():
            # Clone new Gaussians
            new_xyz_clone = self.model._xyz[clone_mask]
            new_scale_clone = self.model._scaling_log[clone_mask]
            new_rot_clone = self.model._rotation_quats[clone_mask]
            new_op_clone = self.model._opacity_logits[clone_mask]
            new_feat_clone = self.model._features_dc[clone_mask]

            # Split: create 2 child Gaussians per parent with scale / 1.6 and slight noise
            parent_xyz = self.model._xyz[split_mask]
            parent_scale = self.model._scaling_log[split_mask] - np.log(1.6)
            parent_rot = self.model._rotation_quats[split_mask]
            parent_op = self.model._opacity_logits[split_mask]
            parent_feat = self.model._features_dc[split_mask]

            # Child 1 & Child 2
            noise1 = torch.randn_like(parent_xyz) * torch.exp(parent_scale) * 0.1
            noise2 = torch.randn_like(parent_xyz) * torch.exp(parent_scale) * 0.1
            new_xyz_split = torch.cat([parent_xyz + noise1, parent_xyz + noise2], dim=0)
            new_scale_split = torch.cat([parent_scale, parent_scale], dim=0)
            new_rot_split = torch.cat([parent_rot, parent_rot], dim=0)
            new_op_split = torch.cat([parent_op, parent_op], dim=0)
            new_feat_split = torch.cat([parent_feat, parent_feat], dim=0)

            # Combine old (excluding split parents) + cloned + split children
            keep_mask = ~split_mask

            # 3. Prune low opacity or oversized Gaussians
            opacities = self.model.get_opacity().squeeze(-1)
            prune_mask = (opacities < min_opacity) | (max_scales > max_scale)
            keep_mask = keep_mask & (~prune_mask)
            num_pruned = int(prune_mask.sum().item())

            combined_xyz = torch.cat([self.model._xyz[keep_mask], new_xyz_clone, new_xyz_split], dim=0)
            combined_scale = torch.cat([self.model._scaling_log[keep_mask], new_scale_clone, new_scale_split], dim=0)
            combined_rot = torch.cat([self.model._rotation_quats[keep_mask], new_rot_clone, new_rot_split], dim=0)
            combined_op = torch.cat([self.model._opacity_logits[keep_mask], new_op_clone, new_op_split], dim=0)
            combined_feat = torch.cat([self.model._features_dc[keep_mask], new_feat_clone, new_feat_split], dim=0)

        # Update model parameters
        self.model._xyz = torch.nn.Parameter(combined_xyz)
        self.model._scaling_log = torch.nn.Parameter(combined_scale)
        self.model._rotation_quats = torch.nn.Parameter(combined_rot)
        self.model._opacity_logits = torch.nn.Parameter(combined_op)
        self.model._features_dc = torch.nn.Parameter(combined_feat)

        # Re-initialize optimizer and gradient accumulators
        self._setup_optimizer()
        self.xyz_grad_accum = torch.zeros((self.model.num_gaussians, 1), device=device)
        self.denom = torch.zeros((self.model.num_gaussians, 1), device=device)

        self.logger.info(
            f"Densification step: Cloned {num_cloned:,}, Split {num_split:,}, Pruned {num_pruned:,} "
            f"-> Total: {self.model.num_gaussians:,} Gaussians"
        )
        return num_cloned, num_split, num_pruned
