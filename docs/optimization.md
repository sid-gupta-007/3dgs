# 3D Gaussian Splatting Optimization & Adaptive Densification

The PanoGS optimization engine refines 3D Gaussian positions, anisotropic scales, 3D rotations, opacities, and spherical harmonics colors against ground-truth multi-view perspective supervisory images sampled from panoramas.

---

## 1. Differentiable Photometric Loss

At each training step, a perspective camera view $\mathbf{I}_{\text{pred}}$ is rendered via the differentiable PyTorch rasterizer and evaluated against the ground-truth perspective image $\mathbf{I}_{\text{gt}}$:

$$\mathcal{L} = (1 - \lambda) \mathcal{L}_1(\mathbf{I}_{\text{pred}}, \mathbf{I}_{\text{gt}}) + \lambda \mathcal{L}_{\text{SSIM}}(\mathbf{I}_{\text{pred}}, \mathbf{I}_{\text{gt}})$$

Where:
- $\lambda = 0.2$ balances pixel-wise accuracy and structural detail.
- $\mathcal{L}_1 = \frac{1}{HW} \sum_{\mathbf{p}} |\mathbf{I}_{\text{pred}}(\mathbf{p}) - \mathbf{I}_{\text{gt}}(\mathbf{p})|$.
- $\mathcal{L}_{\text{SSIM}} = 1.0 - \text{SSIM}(\mathbf{I}_{\text{pred}}, \mathbf{I}_{\text{gt}})$.

---

## 2. Multi-Parameter Adam Optimization

Each attribute group in `TorchGaussianModel` is updated with independent Adam learning rates:

| Parameter | Tensor Shape | Learning Rate | Representation |
| :--- | :--- | :--- | :--- |
| **Position $\mathbf{\mu}$** | $(N, 3)$ | $1.6 \times 10^{-4}$ | Unconstrained world coordinates |
| **Scale $\mathbf{s}_{\log}$** | $(N, 3)$ | $5.0 \times 10^{-3}$ | Log-space $\mathbf{s} = \exp(\mathbf{s}_{\log})$ |
| **Rotation $\mathbf{q}$** | $(N, 4)$ | $1.0 \times 10^{-3}$ | Unit quaternion $\mathbf{q} / \|\mathbf{q}\|_2$ |
| **Opacity $o_{\text{logit}}$** | $(N, 1)$ | $5.0 \times 10^{-2}$ | Logit $\alpha = \sigma(o_{\text{logit}})$ |
| **Color $SH_0$** | $(N, 3)$ | $2.5 \times 10^{-3}$ | Zeroth-order spherical harmonics |

---

## 3. Adaptive Densification & Pruning

Positional gradients are accumulated across training views:
$$\bar{\nabla}_{\mathbf{\mu}} = \frac{1}{K} \sum_{k=1}^K \|\nabla_{\mathbf{\mu}} \mathcal{L}_k\|$$

When $\bar{\nabla}_{\mathbf{\mu}} \ge \tau_{\text{grad}}$ (default: $2 \times 10^{-4}$):
1. **Cloning (Under-reconstruction)**:
   If $\max(\mathbf{s}) < \text{scale\_threshold}$: Duplicate the small Gaussian to provide additional spatial capacity.
2. **Splitting (Over-reconstruction)**:
   If $\max(\mathbf{s}) \ge \text{scale\_threshold}$: Replace the large Gaussian with two child Gaussians with scales divided by 1.6:
   $$\mathbf{s}_{\text{child}} = \frac{\mathbf{s}_{\text{parent}}}{1.6}, \quad \mathbf{\mu}_{\text{child}} = \mathbf{\mu}_{\text{parent}} + \mathcal{N}(0, \mathbf{\Sigma})$$
3. **Pruning**:
   Remove any Gaussian where $\alpha < \alpha_{\min}$ (default: $0.05$) or $\max(\mathbf{s}) > s_{\max}$ (default: $0.5\text{m}$).
4. **Opacity Reset**:
   Every $N_{\text{reset}}$ steps, reset $\alpha = \min(\alpha, 0.1)$ to eliminate floaters and transparent artifacts.

---

## 4. CLI Training Command

```bash
# Optimize 3D Gaussians from an initial point cloud / PLY against panorama perspective views
panogs train examples/data/abcd.png --init-ply output/abcd_gaussians.ply -o output/abcd_optimized.ply --iterations 100 --views 8 --res 128
```
