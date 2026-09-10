"""
PyTorch Differentiable 3D Gaussian Splatting Rasterizer.
Provides TorchGaussianModel with autograd parameter tensors and
differentiable perspective projection, covariance computation, and alpha compositing.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from panogs.core.gaussian.model import GaussianModel, SH_C0, rgb_to_sh0, sh0_to_rgb
from panogs.rendering.camera import Camera


def quaternion_to_rotation_matrix_torch(quats: torch.Tensor) -> torch.Tensor:
    """
    Convert (N, 4) unit quaternions (qw, qx, qy, qz) into (N, 3, 3) rotation matrices in PyTorch.
    """
    q = F.normalize(quats, p=2, dim=-1, eps=1e-8)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    N = quats.shape[0]
    R = torch.zeros((N, 3, 3), dtype=quats.dtype, device=quats.device)

    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - r * z)
    R[:, 0, 2] = 2.0 * (x * z + r * y)

    R[:, 1, 0] = 2.0 * (x * y + r * z)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - r * x)

    R[:, 2, 0] = 2.0 * (x * z - r * y)
    R[:, 2, 1] = 2.0 * (y * z + r * x)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)

    return R


class TorchGaussianModel(nn.Module):
    """
    Differentiable PyTorch container for 3D Gaussian Splats.
    """

    def __init__(
        self,
        xyz: torch.Tensor,
        scaling_log: torch.Tensor,
        rotation_quats: torch.Tensor,
        opacity_logits: torch.Tensor,
        features_dc: torch.Tensor,
    ):
        super().__init__()
        self._xyz = nn.Parameter(xyz.float())
        self._scaling_log = nn.Parameter(scaling_log.float())
        self._rotation_quats = nn.Parameter(rotation_quats.float())
        self._opacity_logits = nn.Parameter(opacity_logits.float())
        self._features_dc = nn.Parameter(features_dc.float())

    @property
    def num_gaussians(self) -> int:
        return self._xyz.shape[0]

    @classmethod
    def from_numpy(cls, model: GaussianModel, device: str = "cpu") -> "TorchGaussianModel":
        """Construct a TorchGaussianModel from a NumPy GaussianModel."""
        return cls(
            xyz=torch.tensor(model._xyz, device=device),
            scaling_log=torch.tensor(model._scaling_log, device=device),
            rotation_quats=torch.tensor(model._rotation_quats, device=device),
            opacity_logits=torch.tensor(model._opacity_logits, device=device),
            features_dc=torch.tensor(model._features_dc, device=device),
        )

    def to_numpy_model(self) -> GaussianModel:
        """Export current PyTorch parameters to a NumPy GaussianModel."""
        with torch.no_grad():
            return GaussianModel(
                xyz=self._xyz.detach().cpu().numpy(),
                scaling_log=self._scaling_log.detach().cpu().numpy(),
                rotation_quats=self._rotation_quats.detach().cpu().numpy(),
                opacity_logits=self._opacity_logits.detach().cpu().numpy(),
                features_dc=self._features_dc.detach().cpu().numpy(),
            )

    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    def get_scaling(self) -> torch.Tensor:
        """Standard deviation scales sigma = exp(s_log)."""
        return torch.exp(self._scaling_log)

    def get_rotation_quats(self) -> torch.Tensor:
        return F.normalize(self._rotation_quats, p=2, dim=-1, eps=1e-8)

    def get_rotation_matrices(self) -> torch.Tensor:
        return quaternion_to_rotation_matrix_torch(self._rotation_quats)

    def get_opacity(self) -> torch.Tensor:
        """Alpha opacity in (0, 1)."""
        return torch.sigmoid(self._opacity_logits)

    def get_rgb(self) -> torch.Tensor:
        """RGB color in [0, 1] from SH0."""
        return torch.clamp(self._features_dc * SH_C0 + 0.5, 0.0, 1.0)

    def get_covariance_3d(self) -> torch.Tensor:
        """
        Compute (N, 3, 3) 3D covariance Sigma = R S S^T R^T.
        """
        R = self.get_rotation_matrices()  # (N, 3, 3)
        scales = self.get_scaling()       # (N, 3)
        # M = R * S
        M = R * scales.unsqueeze(1)       # (N, 3, 3)
        cov3d = torch.bmm(M, M.transpose(1, 2))
        return cov3d


def render_gaussians_torch(
    model: TorchGaussianModel,
    camera: Camera,
    bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_depth: float = 0.05,
    low_pass_filter: float = 0.3,
) -> Tuple[torch.Tensor, dict]:
    """
    Differentiable rendering of 3D Gaussians for a camera viewpoint using PyTorch.

    Args:
        model: TorchGaussianModel instance.
        camera: Camera viewpoint.
        bg_color: RGB background color.
        min_depth: Near plane clipping.
        low_pass_filter: Antialiasing constant.

    Returns:
        Tuple:
            - rendered_image: (H, W, 3) float32 torch.Tensor in [0, 1].
            - render_info: Dictionary containing 'screen_pts', 'vis_idx' for gradient tracking.
    """
    device = model._xyz.device
    dtype = model._xyz.dtype
    H, W = camera.height, camera.width

    xyz = model.get_xyz()
    cov3d = model.get_covariance_3d()
    colors = model.get_rgb()
    opacities = model.get_opacity().squeeze(-1)

    # 1. Transform positions to camera coordinate space
    R_cam = torch.tensor(camera.R, device=device, dtype=dtype)  # (3, 3)
    t_cam = torch.tensor(camera.t, device=device, dtype=dtype)  # (3,)

    # p_cam = xyz @ R_cam^T + t_cam
    p_cam = torch.matmul(xyz, R_cam.t()) + t_cam  # (N, 3)
    tx = p_cam[:, 0]
    ty = p_cam[:, 1]
    tz = p_cam[:, 2]

    # 2. Frustum near plane filter
    vis_mask = tz > min_depth
    if not torch.any(vis_mask):
        bg = torch.tensor(bg_color, device=device, dtype=dtype).view(1, 1, 3).expand(H, W, 3)
        return bg, {"screen_pts": None, "vis_idx": None}

    vis_idx = torch.nonzero(vis_mask, as_tuple=False).squeeze(-1)
    tx_vis = tx[vis_idx]
    ty_vis = ty[vis_idx]
    tz_vis = tz[vis_idx]
    cov3d_vis = cov3d[vis_idx]
    colors_vis = colors[vis_idx]
    opacities_vis = opacities[vis_idx]

    fx = camera.fx
    fy = camera.fy
    cx = camera.cx
    cy = camera.cy

    # 3. Perspective projection
    xs = (fx * tx_vis / tz_vis) + cx
    ys = (fy * ty_vis / tz_vis) + cy
    screen_pts = torch.stack([xs, ys], dim=-1)  # (M, 2)

    # Enable grad tracking on screen points for densification
    if screen_pts.requires_grad:
        screen_pts.retain_grad()

    # 4. Transform 3D covariance to camera view frame: cov_cam = R_cam @ cov3d @ R_cam^T
    cov3d_cam = torch.matmul(
        R_cam.unsqueeze(0),
        torch.matmul(cov3d_vis, R_cam.t().unsqueeze(0))
    )

    # 5. Jacobian of projection J: (M, 2, 3)
    M = vis_idx.shape[0]
    tz2 = tz_vis * tz_vis

    J = torch.zeros((M, 2, 3), device=device, dtype=dtype)
    J[:, 0, 0] = fx / tz_vis
    J[:, 0, 2] = - (fx * tx_vis) / tz2
    J[:, 1, 1] = fy / tz_vis
    J[:, 1, 2] = - (fy * ty_vis) / tz2

    # 6. 2D Covariance: Sigma_2D = J @ cov3d_cam @ J^T + low_pass * I
    cov2d = torch.bmm(J, torch.bmm(cov3d_cam, J.transpose(1, 2)))
    cov2d[:, 0, 0] = cov2d[:, 0, 0] + low_pass_filter
    cov2d[:, 1, 1] = cov2d[:, 1, 1] + low_pass_filter

    # Covariance components
    a = cov2d[:, 0, 0]
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1]
    det = a * c - b * b
    valid_det = det > 1e-6

    r_x = 3.0 * torch.sqrt(torch.clamp(a, min=1e-4))
    r_y = 3.0 * torch.sqrt(torch.clamp(c, min=1e-4))
    radii = torch.maximum(r_x, r_y)

    # Filter invalid Gaussians
    valid_mask = valid_det & (radii > 0.5)
    if not torch.any(valid_mask):
        bg = torch.tensor(bg_color, device=device, dtype=dtype).view(1, 1, 3).expand(H, W, 3)
        return bg, {"screen_pts": screen_pts, "vis_idx": vis_idx}

    # 7. Depth sorting (front-to-back)
    sort_order = torch.argsort(tz_vis)
    sort_order = sort_order[valid_mask[sort_order]]

    sorted_screen_pts = screen_pts[sort_order]
    sorted_a = a[sort_order]
    sorted_b = b[sort_order]
    sorted_c = c[sort_order]
    sorted_det = det[sort_order]
    sorted_radii = radii[sort_order]
    sorted_colors = colors_vis[sort_order]
    sorted_opacities = opacities_vis[sort_order]

    # Initialize compositing buffers
    acc_color = torch.zeros((H, W, 3), device=device, dtype=dtype)
    transmittance = torch.ones((H, W), device=device, dtype=dtype)

    num_to_render = sorted_screen_pts.shape[0]

    # Compositing loop
    for i in range(num_to_render):
        cx_i = sorted_screen_pts[i, 0]
        cy_i = sorted_screen_pts[i, 1]
        rad_i = sorted_radii[i].item()

        u_min = max(0, int(np.floor(cx_i.item() - rad_i)))
        u_max = min(W - 1, int(np.ceil(cx_i.item() + rad_i)))
        v_min = max(0, int(np.floor(cy_i.item() - rad_i)))
        v_max = min(H - 1, int(np.ceil(cy_i.item() + rad_i)))

        if u_min > u_max or v_min > v_max:
            continue

        if transmittance[v_min:v_max + 1, u_min:u_max + 1].max().item() < 1e-4:
            continue

        # Pixel coordinate grid
        u_range = torch.arange(u_min, u_max + 1, device=device, dtype=dtype)
        v_range = torch.arange(v_min, v_max + 1, device=device, dtype=dtype)
        v_grid, u_grid = torch.meshgrid(v_range, u_range, indexing="ij")

        dx = u_grid - cx_i
        dy = v_grid - cy_i

        # Quadratic form: Delta^T * Sigma_2D^-1 * Delta
        det_i = sorted_det[i]
        inv_a = sorted_a[i]
        inv_b = sorted_b[i]
        inv_c = sorted_c[i]

        power = -0.5 * (inv_c * dx * dx - 2.0 * inv_b * dx * dy + inv_a * dy * dy) / det_i
        power_clamped = torch.clamp(power, max=0.0)
        G = torch.exp(power_clamped)

        alpha_box = sorted_opacities[i] * G

        # Pad alpha_box to full image (H, W) out-of-place for autograd integrity
        pad_left = u_min
        pad_right = W - 1 - u_max
        pad_top = v_min
        pad_bottom = H - 1 - v_max
        alpha_full = F.pad(alpha_box, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)

        # Out-of-place accumulation
        w = alpha_full * transmittance
        color_i = sorted_colors[i]  # (3,)

        acc_color = acc_color + w.unsqueeze(-1) * color_i.view(1, 1, 3)
        transmittance = transmittance * (1.0 - alpha_full)

    # Composite background
    bg = torch.tensor(bg_color, device=device, dtype=dtype).view(1, 1, 3)
    final_image = acc_color + transmittance.unsqueeze(-1) * bg
    final_clamped = torch.clamp(final_image, 0.0, 1.0)

    render_info = {
        "screen_pts": screen_pts,
        "vis_idx": vis_idx,
    }
    return final_clamped, render_info
