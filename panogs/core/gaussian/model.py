"""
3D Gaussian Splatting Scene Model.
Encapsulates positions, scales (log-space), rotations (unit quaternions),
opacities (logit-space), and zeroth-order spherical harmonics (SH0) colors.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np

SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))


def rgb_to_sh0(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB in [0.0, 1.0] to zeroth-order spherical harmonics (SH0).
    Formula: SH0 = (RGB - 0.5) / SH_C0
    """
    rgb_clipped = np.clip(rgb, 0.0, 1.0)
    return (rgb_clipped - 0.5) / SH_C0


def sh0_to_rgb(sh0: np.ndarray) -> np.ndarray:
    """
    Convert zeroth-order spherical harmonics (SH0) back to RGB in [0.0, 1.0].
    Formula: RGB = clamp(SH0 * SH_C0 + 0.5, 0.0, 1.0)
    """
    return np.clip(sh0 * SH_C0 + 0.5, 0.0, 1.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Numerically stable logit (inverse-sigmoid) function."""
    p_clamped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clamped / (1.0 - p_clamped))


def quaternion_to_rotation_matrix(quats: np.ndarray) -> np.ndarray:
    """
    Convert (N, 4) unit quaternions (qw, qx, qy, qz) into (N, 3, 3) 3D rotation matrices.
    """
    # Normalize quaternions
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    q = quats / norms

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R = np.zeros((quats.shape[0], 3, 3), dtype=np.float32)

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


class GaussianModel:
    """
    Container and operations for a set of N 3D Gaussians.
    """

    def __init__(
        self,
        xyz: np.ndarray,
        scaling_log: np.ndarray,
        rotation_quats: np.ndarray,
        opacity_logits: np.ndarray,
        features_dc: np.ndarray,
    ):
        """
        Initialize GaussianModel with unconstrained internal parameter tensors.

        Args:
            xyz: (N, 3) float32 positions in world space.
            scaling_log: (N, 3) float32 log-scale parameters.
            rotation_quats: (N, 4) float32 quaternions (qw, qx, qy, qz).
            opacity_logits: (N, 1) float32 inverse-sigmoid opacity logits.
            features_dc: (N, 3) float32 zeroth-order spherical harmonics (SH0).
        """
        self._xyz = np.asarray(xyz, dtype=np.float32)
        self._scaling_log = np.asarray(scaling_log, dtype=np.float32)
        self._rotation_quats = np.asarray(rotation_quats, dtype=np.float32)
        self._opacity_logits = np.asarray(opacity_logits, dtype=np.float32)
        if self._opacity_logits.ndim == 1:
            self._opacity_logits = self._opacity_logits[:, np.newaxis]
        self._features_dc = np.asarray(features_dc, dtype=np.float32)

        N = self._xyz.shape[0]
        if self._scaling_log.shape != (N, 3):
            raise ValueError(f"Scaling shape {self._scaling_log.shape} != ({N}, 3)")
        if self._rotation_quats.shape != (N, 4):
            raise ValueError(f"Rotation shape {self._rotation_quats.shape} != ({N}, 4)")
        if self._opacity_logits.shape != (N, 1):
            raise ValueError(f"Opacity shape {self._opacity_logits.shape} != ({N}, 1)")
        if self._features_dc.shape != (N, 3):
            raise ValueError(f"Features DC shape {self._features_dc.shape} != ({N}, 3)")

    @classmethod
    def from_raw(
        cls,
        xyz: np.ndarray,
        scales: np.ndarray,
        rotation_quats: np.ndarray,
        colors_rgb: np.ndarray,
        opacities: np.ndarray,
    ) -> "GaussianModel":
        """
        Create a GaussianModel from physical domain values (RGB [0, 1], opacities [0, 1], positive scales).
        """
        scaling_log = np.log(np.maximum(scales, 1e-6))
        features_dc = rgb_to_sh0(colors_rgb)
        opacity_logits = logit(opacities)
        return cls(
            xyz=xyz,
            scaling_log=scaling_log,
            rotation_quats=rotation_quats,
            opacity_logits=opacity_logits,
            features_dc=features_dc,
        )

    @property
    def num_gaussians(self) -> int:
        return self._xyz.shape[0]

    # ─────────────────────────────────────────────────────────────
    # Parameter Accessors (Mapped to Valid Physical Ranges)
    # ─────────────────────────────────────────────────────────────
    def get_xyz(self) -> np.ndarray:
        """Return (N, 3) 3D Gaussian center positions."""
        return self._xyz

    def get_scaling(self) -> np.ndarray:
        """Return (N, 3) strictly positive scaling standard deviations sigma = exp(s_log)."""
        return np.exp(self._scaling_log)

    def get_rotation_quats(self) -> np.ndarray:
        """Return (N, 4) normalized unit quaternions (qw, qx, qy, qz)."""
        norms = np.linalg.norm(self._rotation_quats, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return self._rotation_quats / norms

    def get_rotation_matrices(self) -> np.ndarray:
        """Return (N, 3, 3) 3D orthonormal rotation matrices R."""
        return quaternion_to_rotation_matrix(self._rotation_quats)

    def get_opacity(self) -> np.ndarray:
        """Return (N, 1) opacities alpha in (0.0, 1.0)."""
        return sigmoid(self._opacity_logits)

    def get_features_dc(self) -> np.ndarray:
        """Return (N, 3) zeroth-order spherical harmonics (SH0)."""
        return self._features_dc

    def get_rgb(self) -> np.ndarray:
        """Return (N, 3) float32 RGB colors in [0.0, 1.0]."""
        return sh0_to_rgb(self._features_dc)

    def get_rgb_uint8(self) -> np.ndarray:
        """Return (N, 3) uint8 RGB colors in [0, 255]."""
        return (self.get_rgb() * 255.0).astype(np.uint8)

    # ─────────────────────────────────────────────────────────────
    # 3D Covariance Calculation
    # ─────────────────────────────────────────────────────────────
    def get_covariance_3d(self) -> np.ndarray:
        """
        Compute the (N, 3, 3) 3D covariance matrix Sigma for each Gaussian:
            Sigma = R * S * S^T * R^T = M * M^T
        where M = R * S.

        Returns:
            np.ndarray: (N, 3, 3) symmetric positive semi-definite covariance matrices.
        """
        R = self.get_rotation_matrices()  # (N, 3, 3)
        scales = self.get_scaling()       # (N, 3)

        # Scale columns of R: M[:, :, j] = R[:, :, j] * scale[j]
        # (N, 3, 3) * (N, 1, 3) via broadcasting
        M = R * scales[:, np.newaxis, :]  # (N, 3, 3)

        # Sigma = M @ M^T: (N, 3, 3) @ (N, 3, 3)
        cov3d = np.matmul(M, M.transpose(0, 2, 1))
        return cov3d.astype(np.float32)

    def summary(self) -> str:
        """Return human-readable diagnostic summary."""
        scales = self.get_scaling()
        opacities = self.get_opacity()
        min_pos = np.min(self._xyz, axis=0)
        max_pos = np.max(self._xyz, axis=0)

        return (
            f"GaussianModel Summary:\n"
            f"  Total Gaussians:   {self.num_gaussians:,}\n"
            f"  Position Bounds:   [{min_pos[0]:.2f}, {min_pos[1]:.2f}, {min_pos[2]:.2f}] to [{max_pos[0]:.2f}, {max_pos[1]:.2f}, {max_pos[2]:.2f}]\n"
            f"  Scale Range:       [{np.min(scales):.4f}, {np.max(scales):.4f}] (mean: {np.mean(scales):.4f})\n"
            f"  Opacity Range:     [{np.min(opacities):.3f}, {np.max(opacities):.3f}] (mean: {np.mean(opacities):.3f})"
        )
