# 3D Gaussian Mathematics & Representation

This document details the mathematical formulation, parameterization spaces, 3D covariance computation, and spherical harmonics representation used in PanoGS.

---

## 1. 3D Gaussian Definition

A 3D Gaussian in world space is defined by its center mean $\mathbf{\mu} \in \mathbb{R}^3$ and a $3 \times 3$ symmetric positive semi-definite covariance matrix $\mathbf{\Sigma} \in \mathbb{R}^{3 \times 3}$:

$$G(\mathbf{x}) = \exp\left( -\frac{1}{2} (\mathbf{x} - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu}) \right)$$

---

## 2. Numerically Stable Parameterization

To ensure gradient updates remain in valid physical domains, Gaussians are represented using unconstrained internal parameters:

### A. Scale $\mathbf{s} \in \mathbb{R}^3$ (Log-Space)
- Internal parameter: $\mathbf{s}_{\log} = (\ln s_x, \ln s_y, \ln s_z)$
- Physical scale: $\mathbf{s} = \exp(\mathbf{s}_{\log}) > 0$

### B. Rotation $\mathbf{q} \in \mathbb{H}$ (Unit Quaternions)
- Internal quaternion: $\mathbf{q} = (q_w, q_x, q_y, q_z)$
- Normalized unit quaternion: $\tilde{\mathbf{q}} = \frac{\mathbf{q}}{\|\mathbf{q}\|_2}$
- Rotation matrix $\mathbf{R} \in \text{SO}(3)$:
  $$\mathbf{R} = \begin{bmatrix}
  1 - 2(y^2 + z^2) & 2(xy - rz) & 2(xz + ry) \\
  2(xy + rz) & 1 - 2(x^2 + z^2) & 2(yz - rx) \\
  2(xz - ry) & 2(yz + rx) & 1 - 2(x^2 + y^2)
  \end{bmatrix}$$
  where $(r, x, y, z) = (\tilde{q}_w, \tilde{q}_x, \tilde{q}_y, \tilde{q}_z)$.

### C. Opacity $\alpha \in (0, 1)$ (Logit-Space)
- Internal parameter: $o_{\text{logit}} = \ln \frac{\alpha}{1 - \alpha}$
- Physical opacity: $\alpha = \sigma(o_{\text{logit}}) = \frac{1}{1 + \exp(-o_{\text{logit}})}$

### D. Color Representation (Spherical Harmonics $SH_0$)
- Zeroth-order constant: $C_0 = \frac{1}{2\sqrt{\pi}} \approx 0.28209479177387814$
- RGB to $SH_0$: $\mathbf{f}_{\text{dc}} = \frac{\text{RGB} - 0.5}{C_0}$
- $SH_0$ to RGB: $\text{RGB} = \text{clamp}(\mathbf{f}_{\text{dc}} \cdot C_0 + 0.5, 0.0, 1.0)$

---

## 3. 3D Covariance Matrix Formulation

The 3D covariance matrix $\mathbf{\Sigma}$ is parameterized via scale $\mathbf{S} = \text{diag}(s_x, s_y, s_z)$ and rotation $\mathbf{R}$:

$$\mathbf{M} = \mathbf{R} \mathbf{S}$$
$$\mathbf{\Sigma} = \mathbf{M} \mathbf{M}^T = \mathbf{R} \mathbf{S} \mathbf{S}^T \mathbf{R}^T$$

This matrix decomposition guarantees that $\mathbf{\Sigma}$ is symmetric and positive semi-definite ($\forall \mathbf{v} \neq 0, \mathbf{v}^T \mathbf{\Sigma} \mathbf{v} \ge 0$).

---

## 4. Adaptive Initialization from Point Clouds

Rather than assigning an arbitrary constant scale to all Gaussians, PanoGS initializes each Gaussian's scale $s_i$ dynamically based on the average Euclidean distance to its $k=3$ nearest spatial neighbors via `cKDTree`:

$$s_i = \text{clamp}\left( \frac{1}{k} \sum_{j \in \mathcal{N}_k(i)} \|\mathbf{p}_i - \mathbf{p}_j\|_2, \; s_{\min}, \; s_{\max} \right)$$
