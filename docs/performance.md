# Performance and Reconstruction Quality Plan

The viewer and reconstruction pipeline have different bottlenecks. Treat frame time and scene fidelity as separate targets: a faster renderer cannot restore geometry that a single panorama or inaccurate camera poses never observed.

## What the current code tells us

- The WebGL viewer performs visibility collection, a 16-bit depth histogram sort, and several million CPU-side attribute writes on the main thread when the camera moves (`panogs/apps/viewer/index.html`). The scene in the supplied screenshots has about 2.05 million splats, with 0.87–1.30 million visible and 18–45 FPS.
- The current spatial LOD samples every second or fourth item in each octree leaf. Input order is not a screen-space quality metric, so this can produce uneven detail and shimmer.
- Panorama training views are perspective crops from one shared camera center (`panogs/training/dataset.py`). They do not contain parallax observations. The video pipeline estimates translation direction with monocular essential-matrix recovery, then assigns every step a fixed 15 cm scale (`panogs/reconstruction/video/video_pipeline.py`). That scale is not measured; pose drift and inconsistent depth can create doubled surfaces, stretched splats, and floaters.
- The default training loop uses 150 iterations and 128-pixel views (`panogs/training/trainer.py`). That is a development-scale optimizer, not a substitute for multi-view reconstruction and validated convergence.

## Work in order

1. **Measure a fixed baseline.** Keep a small, medium, and large scene. Record camera path, resolution, device-pixel ratio, splat count, load time, median/p95 frame time, and sort time. For quality, render fixed held-out views and compare PSNR/SSIM plus visual inspection. Do not compare FPS across different camera paths or quality settings.
2. **Reduce viewer CPU work.** Keep the current order between camera updates, avoid per-update temporary allocations, and replace index-stride LOD with deterministic, spatially balanced or projected-size selection. Track visible splats and sort duration. The initial viewer change caps depth resorting at 12.5 Hz and caps device-pixel ratio at 1.5; this is a first mitigation, not a measured guarantee.
3. **Add a real accelerated rasterizer.** The original 3DGS renderer uses tile-based visibility-aware rasterization. For an optional NVIDIA path, evaluate `gsplat` rather than extending a Python/JavaScript per-splat CPU loop indefinitely. Keep the CPU path for compatibility and make hardware support explicit. A browser GPU backend needs its own WebGPU/WebGL implementation and device benchmarks.
4. **Fix capture geometry before tuning splat count.** For video, estimate calibrated camera intrinsics and full camera poses with a multi-view SfM pipeline, reject low-confidence pose/depth frames, and optimize against actual captured views. Keep single-panorama depth as a clearly labelled preview/2.5D mode. Do not invent metric translation from a fixed constant.
5. **Address rendering artifacts with the right methods.** Evaluate Mip-Splatting for scale-dependent aliasing and dilation; evaluate StopThePop for view-dependent popping and hierarchical resorting. These methods address different rendering artifacts. Neither fixes incorrect camera poses or hallucinated monocular depth.
6. **Compress only against quality measurements.** Evaluate importance-based pruning/distillation (for example LightGaussian) on held-out views and publish the quality/size/speed tradeoff for each preset.

## Research starting points

- Kerbl et al., [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079): reference representation, density control, and tile-based renderer.
- Yu et al., [Mip-Splatting: Alias-free 3D Gaussian Splatting](https://arxiv.org/abs/2311.16493): 3D frequency constraints and scale-aware filtering.
- Radl et al., [StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering](https://arxiv.org/abs/2402.00525): hierarchical resorting for reduced view inconsistency.
- Fan et al., [LightGaussian: Unbounded 3D Gaussian Compression](https://arxiv.org/abs/2311.17245): pruning and distillation to reduce model cost.
- [gsplat documentation](https://docs.gsplat.studio/main/apis/rasterization.html): GPU rasterization controls and implementation reference.

## Acceptance targets

Set targets after collecting the baseline on the intended machine. At minimum, report FPS and p95 frame time at 1080p for the same fixed path, along with splat count, file size, and held-out image quality. A performance preset is only acceptable if its quality loss is measured and visible to the user.
