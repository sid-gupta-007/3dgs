# Monocular Depth Estimation in PanoGS

This document details the monocular depth estimation design, relative versus metric depth distinction, and visualization standards in PanoGS.

---

## 1. Relative Inverse Depth vs. Metric Depth

A central principle in 3D reconstruction is never conflating **relative inverse depth (disparity)** with **metric Euclidean depth**:

| Property | Relative Inverse Depth (e.g. MiDaS) | Metric Depth (e.g. LiDAR, Calibrated RGB-D) |
| :--- | :--- | :--- |
| **Units** | Dimensionless / uncalibrated disparity $d \propto \frac{1}{Z}$ | Meters ($m$) |
| **Monotonicity** | Higher values mean *closer* to camera | Higher values mean *further* from camera |
| **PanoGS Flag** | `is_metric: False` | `is_metric: True` |
| **Reconstruction** | Requires inversion & scale/shift calibration | Direct Euclidean backprojection $\mathbf{P} = d \cdot \mathbf{R}$ |

---

## 2. Abstraction Interface: `DepthEstimator`

All depth models implement the unified interface:

```python
class DepthEstimator(ABC):
    @abstractmethod
    def estimate(self, image: Union[np.ndarray, Image.Image, Path, str]) -> DepthResult:
        """Estimate depth from an input image."""
        pass
```

### `DepthResult` Structure
- `depth_map`: Float32 array $(H, W)$ without quantization or clipping.
- `is_metric`: Explicit boolean flag.
- `confidence`: Optional $(H, W)$ array in $[0.0, 1.0]$.
- `model_name`: Model identifier string.
- `metadata`: Diagnostic runtime details.

---

## 3. High-Contrast Vectorized Colormapping

Depth visualizations use the **Google Turbo** colormap computed via a 4th-order polynomial approximation directly in NumPy:

```python
vis_rgb = DepthEstimator.colorize(depth_result)
```

Outputs an $(H, W, 3)$ uint8 RGB image where near/far structures are clearly separated for human inspection.

---

## 4. File Formats & Persistence

Running `panogs depth <image>` outputs:
1. `depth.npy`: Exact 32-bit floating point matrix.
2. `depth_vis.png`: High-resolution colormapped visualization.
3. `depth_metadata.json`: Model version, metric status, and min/max/mean depth values.
