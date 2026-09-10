# 3D Scene Reconstruction & Point Cloud Backprojection

This document details the mathematical formulation and architecture of the 3D scene backprojection engine in PanoGS.

---

## 1. Backprojection Formulation

Given:
- **$\mathbf{C} \in \mathbb{R}^3$**: Camera optical center in world coordinates (default $\mathbf{C} = (0, 0, 0)^T$).
- **$\mathbf{R}(u, v) \in \mathbb{R}^3$**: Normalized unit ray vector for pixel $(u, v)$ with $\|\mathbf{R}(u, v)\|_2 = 1.0$.
- **$d(u, v) \in \mathbb{R}^+$**: Metric Euclidean depth distance along the ray.

The 3D point $\mathbf{P}(u, v)$ is computed via:
$$\mathbf{P}(u, v) = \mathbf{C} + d(u, v) \cdot \mathbf{R}(u, v)$$

---

## 2. Disparity to Metric Depth Inversion

For relative monocular depth models (e.g. MiDaS) where prediction $D(u, v)$ represents uncalibrated inverse depth (disparity):
- High disparity $D_{\max} \implies$ closest distance $d_{\min}$.
- Low disparity $D_{\min} \implies$ furthest distance $d_{\max}$.

We map normalized disparity $\tilde{D}(u, v) = \frac{D(u, v) - D_{\min}}{D_{\max} - D_{\min}} \in [0, 1]$ to metric depth via:
$$d(u, v) = d_{\min} + (d_{\max} - d_{\min}) \cdot \frac{\frac{1}{\tilde{D}(u, v) + \epsilon} - \frac{1}{1.0 + \epsilon}}{\frac{1}{0.0 + \epsilon} - \frac{1}{1.0 + \epsilon}}$$
where $\epsilon = 0.1$ prevents near-zero division instability.

---

## 3. Invalid Point Filtering

During backprojection, points are filtered if:
1. $\mathbf{P}(u, v)$ contains `NaN` or `Inf`.
2. $d(u, v) < d_{\min}$ or $d(u, v) > d_{\max}$.
3. Depth confidence (if provided) is below threshold $\tau_{\text{conf}}$.
