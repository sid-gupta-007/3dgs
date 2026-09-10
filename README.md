# PanoGS

> **A local-first, unified image/panorama/video → 3D Gaussian Splatting reconstruction engine with an interactive viewer.**

---

## 📌 Development Status: Milestone 0 & 1 Complete

- **Milestone 0**: Engineering foundation (CLI, logging, YAML configuration system, tests, CPU-first defaults)
- **Milestone 1**: Image loader & metadata inspection (`panogs inspect <image>`)

---

## ⚙️ Hardware Compatibility

PanoGS is designed CPU-first and memory-conscious by default, ensuring full functionality without requiring CUDA/NVIDIA GPUs or paid cloud infrastructure.

- **Primary Target**: CPU (Intel Core i5, 16 GB RAM)
- **Memory Safety**: Default resolution clamping and memory footprint analysis

---

## 🚀 Quickstart & Installation

### 1. Set up Virtual Environment
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```

### 2. Install PanoGS
```bash
pip install -e ".[dev]"
```

### 3. Usage

#### Show CLI Help
```bash
panogs --help
```

#### Inspect an Image
```bash
panogs inspect examples/sample.jpg
```

Example output:
```text
Image Inspection: sample.jpg
────────────────────────────────────────
  Path:             C:\path\to\sample.jpg
  Format:           JPEG
  Dimensions:       512 × 512 (W × H)
  Aspect Ratio:     1.00
  Channels:         3
  Color Space:      RGB
  Data Type:        uint8
  Disk Size:        34.20 KB
  Raw Memory:       0.75 MB (786,432 bytes)
```

---

## 🧪 Running Tests

```bash
pytest
```

---

## 🗺️ Roadmap & Milestones

- [x] **Milestone 0**: Engineering Foundation
- [x] **Milestone 1**: Image Loader & Inspection
- [ ] **Milestone 2**: Panorama Ray Engine (Equirectangular to Spherical Rays & Synthetic Shell PLY)
- [ ] **Milestone 3**: Monocular Depth Estimation
- [ ] **Milestone 4**: Depth + Rays → 3D Point Cloud Reconstruction
- [ ] **Milestone 5**: Point Cloud Processing (Outliers, Normals, Downsampling)
- [ ] **Milestone 6**: 3D Gaussian Scene Representation
- [ ] **Milestone 7**: Gaussian Mathematics & CPU Renderer
- [ ] **Milestone 8**: Differentiable PyTorch Renderer
- [ ] **Milestone 9**: 3DGS Optimization (Densification & Pruning)
- [ ] **Milestone 10**: Multi-View Image Pipeline
- [ ] **Milestone 11**: Video Reconstruction
- [ ] **Milestone 12**: Performance Backends (Optional GPU/CUDA)
- [ ] **Milestone 13**: Interactive Viewer
