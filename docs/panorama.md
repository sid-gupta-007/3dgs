# Spherical Camera & Panorama Mathematics

This document outlines the geometric and mathematical foundations of the Equirectangular Spherical Camera model used in PanoGS.

---

## 1. Coordinate System

PanoGS adheres to the standard right-handed 3D camera coordinate convention:
- **$+X$**: Right
- **$+Y$**: Up
- **$+Z$**: Forward (pointing into the center of the equirectangular panorama)

---

## 2. Equirectangular Projection Mapping

An equirectangular image of resolution $W \times H$ spans:
- **Horizontal field of view (Longitude $\theta$)**: $360^\circ$ ($2\pi$ radians)
- **Vertical field of view (Latitude $\phi$)**: $180^\circ$ ($\pi$ radians)

### Discrete Pixel Centering
For discrete pixel coordinates $(u, v)$ where $0 \le u < W$ and $0 \le v < H$:
$$x_{\text{norm}} = \frac{u + 0.5}{W} \in (0, 1)$$
$$y_{\text{norm}} = \frac{v + 0.5}{H} \in (0, 1)$$

### Spherical Angle Transformations
$$\theta = (x_{\text{norm}} - 0.5) \cdot 2\pi \in (-\pi, \pi)$$
$$\phi = (0.5 - y_{\text{norm}}) \cdot \pi \in \left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$$

---

## 3. Unit Ray Vector Generation

Given spherical angles $(\theta, \phi)$, the 3D unit ray direction vector $\mathbf{d} = (d_x, d_y, d_z)^T$ is given by:

$$\begin{aligned}
d_x &= \cos(\phi) \sin(\theta) \\
d_y &= \sin(\phi) \\
d_z &= \cos(\phi) \cos(\theta)
\end{aligned}$$

### Verification of Unit Length
$$\|\mathbf{d}\|_2^2 = d_x^2 + d_y^2 + d_z^2 = \cos^2(\phi)\sin^2(\theta) + \sin^2(\phi) + \cos^2(\phi)\cos^2(\theta) = \cos^2(\phi) + \sin^2(\phi) = 1$$

---

## 4. Synthetic Sphere Reconstruction

For geometric verification without depth, each pixel is projected onto a sphere of radius $R$:
$$\mathbf{P}(u, v) = \mathbf{C} + R \cdot \mathbf{d}(u, v)$$
where $\mathbf{C} = (0, 0, 0)$ is the camera optical center.
