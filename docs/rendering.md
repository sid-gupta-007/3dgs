# 3D Gaussian Splatting CPU Renderer & Camera Geometry

The PanoGS rendering pipeline implements perspective camera geometry, affine Jacobian projection, 2D covariance computation, depth sorting, and front-to-back Over-operator alpha compositing.

---

## 1. Camera Projection & Viewport Geometry

### Pinhole Perspective Model
Given image dimensions $(W, H)$ and horizontal field of view $\text{FoV}_x$, the focal lengths and principal point are computed as:
$$f_x = \frac{W}{2 \tan(\text{FoV}_x / 2)}, \quad f_y = \frac{H}{2 \tan(\text{FoV}_y / 2)}$$
$$c_x = \frac{W}{2}, \quad c_y = \frac{H}{2}$$

World coordinates $\mathbf{P}_w \in \mathbb{R}^3$ are mapped to camera coordinate space $\mathbf{P}_c = (t_x, t_y, t_z)^T$:
$$\mathbf{P}_c = \mathbf{R}_{view} \mathbf{P}_w + \mathbf{t}_{view}$$

The 2D screen coordinate $(x_s, y_s)$ is obtained via central projection:
$$x_s = f_x \frac{t_x}{t_z} + c_x, \quad y_s = f_y \frac{t_y}{t_z} + c_y$$

---

## 2. Jacobian Matrix & 2D Covariance

Following the formulation in *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (Kerbl et al., SIGGRAPH 2023), the 3D Gaussian covariance $\mathbf{\Sigma} \in \mathbb{R}^{3 \times 3}$ is projected into 2D image plane covariance $\mathbf{\Sigma}' \in \mathbb{R}^{2 \times 2}$ using the local affine approximation:

$$\mathbf{\Sigma}' = \mathbf{J} \, \mathbf{W} \, \mathbf{\Sigma} \, \mathbf{W}^T \mathbf{J}^T + s \mathbf{I}_{2 \times 2}$$

Where:
- $\mathbf{W} = \mathbf{R}_{view}$ is the camera rotation transformation.
- $\mathbf{J} \in \mathbb{R}^{2 \times 3}$ is the Jacobian matrix of the perspective projection evaluated at $\mathbf{P}_c = (t_x, t_y, t_z)$:
$$\mathbf{J} = \begin{bmatrix} \frac{f_x}{t_z} & 0 & -\frac{f_x t_x}{t_z^2} \\ 0 & \frac{f_y}{t_z} & -\frac{f_y t_y}{t_z^2} \end{bmatrix}$$
- $s = 0.3$ is a low-pass Gaussian antialiasing filter applied to the diagonal of $\mathbf{\Sigma}'$ to prevent sub-pixel sampling artifacts.

The 2D Gaussian bounding box radius is bounded by:
$$r = 3.0 \times \sqrt{\max(\mathbf{\Sigma}'_{00}, \mathbf{\Sigma}'_{11})}$$

---

## 3. Front-to-Back Alpha Compositing (The Over Operator)

Gaussians in front of the camera ($t_z > \text{min\_depth}$) are sorted in strictly ascending order of depth $t_z$:
$$\mathcal{G}_{(1)}, \mathcal{G}_{(2)}, \dots, \mathcal{G}_{(M)} \quad \text{where} \quad t_{z,(1)} \le t_{z,(2)} \le \dots \le t_{z,(M)}$$

For each pixel $\mathbf{p} = (u, v)^T$ within the bounding box of Gaussian $i$:
1. **Quadratic Form Evaluation**:
   $$\Delta = \mathbf{p} - \mathbf{x}_{s,i}$$
   $$G_i(\mathbf{p}) = \exp\left( -\frac{1}{2} \Delta^T (\mathbf{\Sigma}'_i)^{-1} \Delta \right)$$
2. **Alpha Density**:
   $$\alpha_i(\mathbf{p}) = \sigma(o_i) \cdot G_i(\mathbf{p})$$
3. **Color Accumulation & Transmittance**:
   $$C(\mathbf{p}) \leftarrow C(\mathbf{p}) + c_i \cdot \alpha_i(\mathbf{p}) \cdot T(\mathbf{p})$$
   $$T(\mathbf{p}) \leftarrow T(\mathbf{p}) \cdot (1 - \alpha_i(\mathbf{p}))$$
   where $T(\mathbf{p})$ initializes to $1.0$.

When $T(\mathbf{p}) < 10^{-4}$, the pixel is fully saturated and early-ray-termination skips subsequent Gaussians.

---

## 4. CLI Usage

Render a perspective view from an orbital camera:
```bash
# Render to default 512x512 with white background
panogs render output/gaussians.ply -o output/rendered.png --distance 2.0 --elevation 10.0 --azimuth 30.0

# Render with black background
panogs render output/gaussians.ply -o output/rendered.png --bg black --distance 1.5
```
