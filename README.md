---
title: PanoGS Studio
emoji: 🌐
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.29.0
python_version: '3.12'
pinned: false
app_file: app.py
---

# PanoGS

> **A local-first, unified image/panorama/video → 3D Gaussian Splatting reconstruction engine with an interactive viewer.**

This repository can also run as a Hugging Face Gradio Space. Its `app.py` hosts
PanoGS Studio, runs panorama reconstruction on the Space, and serves the bundled
SuperSplat editor. You do not need Docker installed on your own computer.

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

### Deploy the proof of concept to Hugging Face

This branch is set up for a **Gradio Space**. Push it to the Space repository to
deploy the app. The first reconstruction downloads the Depth Anything V2 Metric
Indoor Small model, so its first run takes longer while the model is cached.
The pipeline currently runs on CPU; selecting GPU hardware does not speed up this
version yet.

By default, uploads, generated scenes, and downloaded model files live on the
Space's temporary disk/cache and may be removed when it restarts. To retain them,
enable persistent storage in the Space settings and set `PANOGS_OUTPUT_DIR` to a
path on the attached volume (for example `/data/panogs-output`) and `HF_HOME` to
a cache path there (for example `/data/.cache/huggingface`).

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

#### Reconstruct a 360-degree scene from one panorama

PanoGS makes overlapping perspective crops from the panorama, estimates depth, blends the views, and exports a surround Gaussian scene. See [the single-panorama pipeline](docs/single_panorama_pipeline.md) for the command and its capture-center movement limits.

```powershell
panogs reconstruct "path\to\room_2k.hdr" --model cubemap_depth_anything --shape surfel -o output\room_360.ply
```

#### Upload and reconstruct from the GUI

Launch PanoGS Studio to open a lightweight start page. Choose **Upload an HDR panorama** to run reconstruction and create a Gaussian PLY, or **Open a Gaussian PLY** to load an existing file in SuperSplat. The Studio page does not initialize the PanoGS splat viewer or load its scene catalog.

```powershell
panogs studio
```

SuperSplat Editor is bundled from the user-provided PlayCanvas open-source checkout. Its MIT license is included at `panogs/apps/viewer/SUPERSPLAT_LICENSE.txt`. SuperSplat requires a browser/device with WebGPU support.

#### Generate an AI-assisted navigable world from a panorama

See [AI world generation](docs/ai_world_generation.md) for the World Labs API key setup, HDR handling, and command options. This optional workflow uploads an LDR preview and uses provider credits.

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
