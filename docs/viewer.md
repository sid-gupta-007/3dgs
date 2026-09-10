# PanoGS 3D Gaussian Splatting Interactive Web Viewer

The PanoGS viewer provides real-time, hardware-accelerated 60 FPS WebGL scene inspection for 3D Gaussian Splats directly in modern web browsers.

---

## 1. Features

- **High-Performance WebGL 2.0 Engine**:
  - Direct 32-byte `.splat` binary streaming.
  - Instanced quad splatting with smooth exponential Gaussian falloff $G(r) = \exp(-2r^2)$.
- **Intuitive Camera Navigation**:
  - **Orbit Navigation**: Left-click drag to rotate around the target point.
  - **Pan Navigation**: Right-click drag or Shift + Left-click drag to slide camera center.
  - **Zoom**: Mouse wheel / trackpad scroll.
  - **First-Person Fly Navigation**: <kbd>W</kbd> <kbd>A</kbd> <kbd>S</kbd> <kbd>D</kbd> keys for spatial walkthroughs.
- **Glassmorphic Control Center**:
  - **Splat Scale Slider**: Adjust standard deviation scaling multiplier ($0.1\times - 3.0\times$).
  - **Field of View (FOV)**: Dynamic perspective adjustment ($30^\circ - 110^\circ$).
  - **Opacity Cutoff**: Filter low-alpha or transparent floaters on the fly.
  - **Environment Themes**: Pitch Dark, Studio Gray, Clean White, Deep Space.
  - **Snapshot Capture**: One-click high-resolution PNG snapshot download.
  - **Drag-and-Drop Loader**: Drop any external `.splat` or `.ply` file onto the window to switch scenes instantly.

---

## 2. CLI Usage

Start the interactive viewer for any trained scene:
```bash
# Launch viewer in browser (default: http://127.0.0.1:8080)
panogs view output/abcd_optimized.splat

# View PLY model directly (auto-converted to WebGL stream)
panogs view output/abcd_optimized.ply --port 8085

# Headless server (do not open browser automatically)
panogs view output/abcd_optimized.splat --no-browser
```
