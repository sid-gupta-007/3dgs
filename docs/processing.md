# Point Cloud Processing & Filtering

This document details the geometric filtering, spatial decimation, and surface normal estimation pipeline in PanoGS.

---

## 1. Voxel Grid Downsampling

Raw monocular backprojections generate dense point concentrations that overload memory and slow down Gaussian optimization.

### Method
1. Discretize coordinates $\mathbf{p} = (x, y, z)$ into integer voxel keys:
   $$\mathbf{v} = \left\lfloor \frac{\mathbf{p}}{s} \right\rfloor$$
   where $s$ is the voxel cell size in meters (default $s = 0.04\text{m}$).
2. Compute the exact centroid $\bar{\mathbf{p}}_k$ and averaged color $\bar{\mathbf{c}}_k$ for all points falling into voxel cell $k$:
   $$\bar{\mathbf{p}}_k = \frac{1}{|V_k|} \sum_{\mathbf{p} \in V_k} \mathbf{p}, \quad \bar{\mathbf{c}}_k = \frac{1}{|V_k|} \sum_{\mathbf{c} \in V_k} \mathbf{c}$$

---

## 2. Statistical Outlier Removal (SOR)

Floating artifacts and depth edge discontinuities are purged using a statistical distance filter:
1. For each point $\mathbf{p}_i$, find its $k$ nearest neighbors ($k = 20$) and compute mean distance $\bar{d}_i$:
   $$\bar{d}_i = \frac{1}{k} \sum_{j \in \mathcal{N}_k(i)} \|\mathbf{p}_i - \mathbf{p}_j\|_2$$
2. Calculate the global mean $\mu$ and standard deviation $\sigma$ across all $\bar{d}_i$.
3. Retain points satisfying:
   $$\bar{d}_i \le \mu + \alpha \cdot \sigma \quad (\alpha = 2.0)$$

---

## 3. Surface Normal Estimation via Local PCA

Each Gaussian in 3DGS is initialized with an orientation aligned to the underlying scene surface.

### Method
1. For each point $\mathbf{p}_i$, compute the local $3 \times 3$ covariance matrix over its $k$ nearest neighbors:
   $$\mathbf{C}_i = \frac{1}{k} \sum_{j \in \mathcal{N}_k(i)} (\mathbf{p}_j - \bar{\mathbf{p}}_i)(\mathbf{p}_j - \bar{\mathbf{p}}_i)^T$$
2. Decompose $\mathbf{C}_i = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$ with eigenvalues $\lambda_1 \le \lambda_2 \le \lambda_3$.
3. The surface normal $\mathbf{n}_i$ is the eigenvector $\mathbf{v}_1$ corresponding to the smallest eigenvalue $\lambda_1$.
4. **Camera Consistency**: Ensure normals point towards the camera center $\mathbf{C}_{\text{cam}}$:
   $$\text{If } \mathbf{n}_i \cdot (\mathbf{C}_{\text{cam}} - \mathbf{p}_i) < 0 \implies \mathbf{n}_i \leftarrow -\mathbf{n}_i$$
