# Single panorama to 360-degree Gaussian scene

PanoGS can reconstruct a full surround scene from one equirectangular panorama. The result is intended for looking around from the panorama's capture center, with only small camera translations recommended.

## Pipeline

1. Project the panorama into six rectilinear perspective views. Each view shares the same optical center and uses a 100-degree field of view, so neighboring views overlap.
2. Estimate depth for each perspective view. The default `cubemap_depth_anything` path uses Depth Anything V2 Metric Indoor; `cubemap_midas` is available when relative depth is preferred.
3. For relative-depth models, estimate affine scale and offset mappings between views using the rays visible in both images. For metric models, preserve metric predictions.
4. Project predictions back to the equirectangular ray grid. Feather and blend the overlapping predictions; convert perspective Z-depth to radial ray distance for metric models.
5. Back-project panorama colors and HDR radiance along those rays, remove depth-edge transition points, initialize shape-adaptive Gaussians, and export `.ply`, `.splat`, and `.hdrsplat` files. An inferred background shell is off by default; enable it with `--solid-shell` only when you specifically need guessed surfaces behind foreground objects.

All perspective views are derived from the same panorama. They add no parallax observations. The scene therefore supports a 360-degree look-around from the capture point, but geometry revealed by walking far from that point is inferred from monocular depth and can stretch or be absent. This pipeline does not generate new unseen images or claim a metrically complete room.

## Run

```powershell
panogs reconstruct "path\to\room_2k.hdr" `
  --model cubemap_depth_anything `
  --shape hybrid `
  --sharpness 0.75 `
  --scale-factor 0.8 `
  -o output\room_360.ply
```

For the local GUI workflow, run `panogs studio` and use its lightweight start page to upload an `.hdr` or `.exr`. Studio calls the same `panogs reconstruct` CLI pipeline with the hybrid, no-shell panorama settings. Reconstruction runs on this start page; the PanoGS splat viewer and its scene catalog are not initialized. When reconstruction finishes, open the generated Gaussian PLY in the bundled SuperSplat editor. Existing Gaussian PLY files can also be opened from the same start page.

The Gaussian exports use the same stem: `room_360_gaussians.ply`, `room_360.splat`, and `room_360_gaussians.hdrsplat`. The HDR splat keeps linear radiance values for the viewer's HDR exposure controls.

## Current first-run artifact

`output/brown_photostudio_360_overlap_2k_v1_gaussians.hdrsplat` is the first original-resolution run using the overlap-and-blend fusion path. It has 2,397,901 Gaussians. `output/brown_photostudio_360_overlap_v1_gaussians.hdrsplat` is a 1024-pixel preview run with 611,797 Gaussians.

Depth inference currently runs at 512 pixels per perspective face, even when the source panorama is 2K. The original-resolution run preserves the panorama's color sampling density; it does not make the depth model itself operate at 2K. Depth-face size and quality/performance comparisons remain future tuning work.

The Brown Photostudio comparison model `output/brown_photostudio_360_hybrid_noshell_v2_gaussians.hdrsplat` uses hybrid splat shapes and disables the inferred background shell. This removes the long translucent backing layer visible in the earlier render. It does not remove all wavy stripes: these are depth/geometry artifacts from estimating a navigable scene from one panorama, and are most visible when translating away from the capture center. The shell is generated from inpainting and can add ghost geometry, so it is intentionally opt-in.
