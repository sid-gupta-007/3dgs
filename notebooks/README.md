# PanoGS: Kaggle GPU Acceleration & Training Guide

This guide explains how to run **PanoGS** on Kaggle with free GPU acceleration (Tesla T4 or P100) to reconstruct high-density 3D Gaussian Splatting scenes from videos and panoramas in seconds.

---

## 🚀 Quick Start on Kaggle

1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.
2. Under **Notebook settings** (right sidebar):
   - Set **Accelerator** to **GPU T4 x2** or **GPU P100**.
   - Set **Internet** to **On** (required for downloading pre-trained depth weights).
3. Import or copy the notebook cells from [`panogs_kaggle_gpu.ipynb`](./panogs_kaggle_gpu.ipynb).
4. Upload your video (`.mp4`, `.mov`) or panorama image (`.hdr`, `.jpg`, `.png`) to `/kaggle/working/` or use Kaggle Datasets.
5. Run the cells:
   ```bash
   # Multi-view video reconstruction with GPU Depth-Anything-V2 & Anisotropic Splats
   !panogs video my_room_tour.mp4 -o output/room.ply --fps 3.0 --max-frames 80 --voxel-size 0.02 --shape hybrid
   ```
6. Download the resulting `room.splat` and view it locally on your computer with 60 FPS Three.js:
   ```bash
   panogs view output/room.splat
   ```

---

## ⚡ What GPU Acceleration Provides

| Feature | CPU (Local) | Kaggle GPU (T4 / P100) |
| :--- | :--- | :--- |
| **Depth-Anything-V2 Inference** | ~1.5s per frame | **~0.04s per frame (35x faster)** |
| **Max Keyframes Sampled** | 20 - 40 frames | **100 - 300+ frames** |
| **Gaussian Splat Density** | 200k - 500k splats | **1M - 5M+ splats** |
| **Reconstruction Time** | ~1-2 minutes | **~8-15 seconds** |

---

## 🎮 Interactive Three.js Controls

Once you download your `.splat` file to your PC:
- **`W`, `A`, `S`, `D`**: Walk through room with realistic wall and furniture collision sliding.
- **`Space`**: Jump.
- **`C` / `Ctrl`**: Crouch / Sneak.
- **`Shift`**: Sprint (2.2x speed boost).
- **`V`**: Toggle Collision on / off (noclip fly mode).
- **Mouse**: Look around (Click canvas for pointer lock).
