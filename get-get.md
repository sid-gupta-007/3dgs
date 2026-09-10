We're going to build **one complete project**, not a collection of disconnected experiments:

> **A local-first, unified image/panorama/video → 3D Gaussian Splatting reconstruction engine with an interactive viewer.**

**3DGS is the finish line for now.** 4DGS/5DGS are explicitly out of scope until the static 3DGS system is genuinely working.

I also want the agent to work **incrementally**, because an agent given "build a 3DGS engine" will otherwise install 14 libraries, clone three repositories, create 2000 lines of code, and tell you it's done. We don't want that.

---

# 1. What we're actually building

Final system:

```text
                         PanoGS
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
     IMAGE              PANORAMA             VIDEO
 JPG/PNG/HDR/EXR      Equirectangular      MP4/MOV
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                           ▼
                  INPUT ANALYSIS
                           │
                           ▼
                CAMERA / GEOMETRY
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
         DEPTH           POSES        FEATURES
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                    3D RECONSTRUCTION
                           │
                           ▼
                     POINT CLOUD
                           │
                           ▼
                GAUSSIAN INITIALIZATION
                           │
                           ▼
                    3D GAUSSIANS
                           │
                           ▼
              DIFFERENTIABLE RENDERER
                           │
                           ▼
                      OPTIMIZATION
                           │
                    ┌──────┴──────┐
                    ▼             ▼
                DENSIFY         PRUNE
                    │             │
                    └──────┬──────┘
                           ▼
                    FINAL 3DGS SCENE
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
               .PLY/.SPLAT       VIEWER
```

---

# 2. The three input modes

## A — Single image

```text
room.jpg
   ↓
monocular depth
   ↓
camera assumption
   ↓
3D points
   ↓
Gaussians
   ↓
3DGS
```

This will be our simplest mode.

---

## B — Panorama

```text
room.png
room.hdr
room.exr
       ↓
equirectangular detection
       ↓
spherical camera
       ↓
360° depth
       ↓
3D reconstruction
       ↓
Gaussians
       ↓
3DGS
```

This is your original idea.

---

## C — Video

```text
room.mp4
     ↓
frame extraction
     ↓
keyframe selection
     ↓
feature tracking
     ↓
camera pose estimation
     ↓
multi-view geometry
     ↓
depth / point cloud
     ↓
Gaussians
     ↓
3DGS
```

**Important:** our first video mode assumes a **static scene with a moving camera**.

We are NOT doing dynamic objects yet.

---

# 3. What the user ultimately gets

Something like:

```bash
panogs reconstruct room.mp4
```

and:

```text
PanoGS Reconstruction
────────────────────────────

Input: room.mp4
Type: Video
Frames: 482
Resolution: 1920×1080

[1/6] Extracting frames       ✓
[2/6] Estimating camera poses ✓
[3/6] Reconstructing geometry ✓
[4/6] Initializing Gaussians  ✓
[5/6] Optimizing 3DGS         ███████░░░ 72%
[6/6] Exporting scene         ✓

Gaussians: 1,284,392
Output: output/room/
```

Then open:

```bash
panogs view output/room/
```

and get:

```text
┌───────────────────────────────────────┐
│                                       │
│             reconstructed             │
│                scene                  │
│                                       │
│          ● ● ● ● ● ●                 │
│       ● ● ● ● ● ● ● ●                │
│                                       │
│                                       │
└───────────────────────────────────────┘

Orbit | Pan | Zoom | FPS
```

---

# 4. Architecture

The codebase should eventually become:

```text
PanoGS/
│
├── apps/
│   ├── cli/
│   └── viewer/
│
├── core/
│   ├── camera/
│   ├── geometry/
│   ├── math/
│   ├── scene/
│   └── gaussian/
│
├── reconstruction/
│   ├── image/
│   ├── panorama/
│   ├── video/
│   ├── depth/
│   ├── poses/
│   └── fusion/
│
├── rendering/
│   ├── cpu/
│   └── gpu/
│
├── optimization/
│   ├── losses/
│   ├── optimizer/
│   ├── densification/
│   └── pruning/
│
├── io/
│   ├── images/
│   ├── video/
│   ├── ply/
│   └── scene/
│
├── tests/
│
├── examples/
│
├── docs/
│
├── configs/
│
├── scripts/
│
└── README.md
```

But **we do not build all of this immediately**.

The agent must create only what's needed for the current milestone.

---

# 5. The development phases

This is the important part.

## PHASE 0 — Engineering foundation

Build:

```text
project
 ↓
Python environment
 ↓
configuration
 ↓
CLI
 ↓
logging
 ↓
tests
```

Deliverable:

```bash
panogs --help
```

works.

---

# PHASE 1 — Image pipeline

Support:

```text
JPG
PNG
```

Build:

```text
image
 ↓
loader
 ↓
metadata
 ↓
normalization
 ↓
visualization
```

Learn:

* image arrays
* RGB
* resolution
* color spaces
* float vs uint8
* memory layout

Deliverable:

```bash
panogs inspect image.jpg
```

---

# PHASE 2 — Panorama engine

Support:

```text
equirectangular PNG
JPG
HDR
EXR
```

Build:

```text
(u,v)
 ↓
longitude
latitude
 ↓
ray direction
```

Then:

```text
panorama
 ↓
ray map
 ↓
synthetic sphere
 ↓
PLY
```

This is our **first 3D output**.

Deliverable:

```bash
panogs panorama test.jpg
```

produces:

```text
sphere.ply
```

---

# PHASE 3 — Depth

Add monocular depth.

```text
image
 ↓
depth model
 ↓
depth map
```

We need:

```text
depth visualization
confidence
normalization
```

Do NOT immediately trust the model.

We compare:

```text
RGB
Depth
```

visually.

---

# PHASE 4 — Depth → 3D

Now:

```text
ray
+
depth
 ↓
3D point
```

Equation:

```text
P = C + dR
```

Build:

```text
RGB + depth
       ↓
point cloud
       ↓
PLY
```

Deliverable:

```bash
panogs reconstruct-room room.jpg
```

produces an actual approximate scene.

---

# PHASE 5 — Point-cloud processing

Add:

```text
outlier removal
normal estimation
confidence filtering
voxel downsampling
depth consistency
```

Now our geometry becomes usable for Gaussian initialization.

---

# PHASE 6 — Gaussian representation

Turn:

```text
point
```

into:

```text
Gaussian
```

Each Gaussian:

```text
position
scale
rotation
opacity
color
```

Eventually:

```text
SH coefficients
```

Build the data structure.

Export:

```text
gaussians.ply
```

---

# PHASE 7 — Tiny Gaussian renderer

This is a **major milestone**.

We implement a CPU renderer.

Pipeline:

```text
3D Gaussian
      ↓
camera transform
      ↓
projection
      ↓
2D covariance
      ↓
2D ellipse
      ↓
depth sorting
      ↓
alpha compositing
      ↓
pixel
```

At first it will be slow.

That's fine.

The objective is understanding and correctness.

---

# PHASE 8 — Differentiable rendering

Move the renderer into PyTorch where practical.

```text
Gaussians
 ↓
render
 ↓
image
 ↓
loss
 ↓
gradient
 ↓
Gaussian update
```

Now the system actually learns.

---

# PHASE 9 — 3DGS optimization

Implement:

```text
position optimization
scale optimization
rotation optimization
opacity optimization
color / SH optimization
```

Then:

```text
densification
pruning
opacity reset
learning-rate scheduling
```

This is where we can honestly say:

# **We built a 3DGS engine.**

---

# PHASE 10 — Multi-view images

Support:

```text
images/
    001.jpg
    002.jpg
    003.jpg
    ...
```

Need:

```text
camera intrinsics
camera extrinsics
poses
```

Then:

```text
multi-view images
       ↓
common coordinate system
       ↓
3DGS
```

---

# PHASE 11 — Video

Now:

```text
video
 ↓
frame sampling
 ↓
keyframes
 ↓
feature matching
 ↓
camera tracking
 ↓
poses
 ↓
reconstruction
 ↓
3DGS
```

This is where the engine becomes much more useful.

---

# PHASE 12 — Performance

Only after correctness.

Optimize:

```text
CPU
 ↓
vectorization
 ↓
multithreading
 ↓
Numba/C++
 ↓
GPU backend
```

Potential GPU implementation:

```text
CUDA
```

but this isn't something we'll force onto your Iris Xe.

The architecture should support:

```text
--device cpu
--device cuda
```

without changing the scene representation.

---

# PHASE 13 — Viewer

Build a proper viewer.

Initial:

```text
Open3D
```

Later:

```text
WebGL/WebGPU
```

Capabilities:

```text
orbit
pan
zoom
FPS camera
Gaussian size
opacity
background
point/Gaussian mode
```

---

# 6. What we DON'T build yet

Explicitly:

```text
❌ 4DGS
❌ dynamic reconstruction
❌ 5DGS
❌ neural rendering research
❌ custom CUDA optimization
❌ distributed training
❌ mobile deployment
❌ fancy AI UI
```

Not because they aren't interesting.

Because:

```text
3DGS isn't finished yet.
```

---

# 7. How we'll learn while building

This is important because you specifically wanted to **learn the concepts**.

Every milestone follows:

```text
CONCEPT
   ↓
WHY?
   ↓
MATH
   ↓
TINY EXPERIMENT
   ↓
CODE
   ↓
VISUALIZATION
   ↓
TEST
   ↓
INTEGRATION
```

For example, before implementing spherical projection:

I'll teach:

```text
Cartesian coordinates
spherical coordinates
longitude
latitude
unit vectors
equirectangular projection
```

Then we implement it.

Then visualize rays.

Then test it.

Then move forward.

So you're not just accumulating code.

---

# 8. Testing philosophy

Every important mathematical component gets tests.

For example:

```text
tests/
├── test_camera.py
├── test_projection.py
├── test_rays.py
├── test_depth.py
├── test_pointcloud.py
├── test_gaussian.py
├── test_covariance.py
├── test_renderer.py
└── test_io.py
```

Example:

```text
ray normalization:

||R|| ≈ 1
```

Projection:

```text
3D → image → expected pixel
```

Gaussian:

```text
scale > 0
rotation valid
opacity ∈ [0,1]
```

This becomes extremely important once optimization starts.

---

# 9. Data format

Internally, we'll eventually have our own scene format:

```text
.scene
```

containing:

```text
metadata
camera
gaussians
images
poses
depth
```

And export:

```text
PLY
SPLAT
```

This means the user isn't locked into our internal format.

---

# 10. Hardware strategy

Your laptop:

```text
i5-12500H
16GB RAM
Iris Xe
```

is our **development machine**.

We'll make:

```text
CPU-first
memory-conscious
low-resolution testing
small Gaussian counts
```

the default.

Example development configuration:

```yaml
resolution: 512
max_gaussians: 50000
iterations: 1000
device: cpu
```

Later:

```yaml
resolution: 1920
max_gaussians: 2000000
iterations: 30000
device: cuda
```

if we have access to an NVIDIA GPU.

**No paid cloud requirement for development.**

---

# 11. What Antigravity should and shouldn't do

This is where I'd be strict.

The agent should:

**DO**

* create project structure
* write modular code
* write tests
* run tests
* inspect errors
* benchmark
* document decisions
* create visual debugging outputs
* keep CPU compatibility
* use existing libraries where appropriate
* explain major implementation decisions

**DO NOT**

* clone an entire 3DGS repository and call it ours
* blindly install massive dependencies
* implement everything in one file
* silently replace our implementation with an external black box
* claim GPU support without testing it
* skip tests
* move to the next milestone when the current one fails
* introduce 4DGS
* optimize before correctness
* download giant models without asking/recording why
* consume huge amounts of disk space unnecessarily

---

# 12. Give this to Antigravity

You can paste the following as the **master project instruction**.

# PanoGS — Master Development Specification

## 1. Project Objective

Build a complete, modular, local-first 3D reconstruction engine named `PanoGS`.

The final target is:

```text
JPG / PNG / HDR / EXR / Panorama / Video
                ↓
       Scene Reconstruction
                ↓
       3D Gaussian Initialization
                ↓
       Differentiable Gaussian Rendering
                ↓
       3DGS Optimization
                ↓
       Exportable Gaussian Scene
                ↓
       Interactive 3D Viewer
```

The project is intended both as a functional software system and as a learning/research implementation.

The current final scope is STATIC 3DGS.

Do NOT implement 4DGS, dynamic reconstruction, 5DGS, or other temporal Gaussian representations yet.

---

## 2. Developer Hardware Constraint

Primary development machine:

* CPU: Intel Core i5-12500H
* RAM: 16 GB
* GPU: Intel Iris Xe integrated graphics
* No NVIDIA CUDA GPU
* No Thunderbolt/eGPU dependency

Therefore:

1. The project MUST work in CPU mode.
2. Development defaults MUST use low-resolution inputs and small scenes.
3. Do not assume CUDA availability.
4. GPU support may be added later behind a separate backend.
5. Do not make paid cloud GPU services a dependency.
6. Avoid unnecessary large downloads and memory-heavy defaults.

---

## 3. Development Philosophy

Build incrementally.

For every major component use this sequence:

```text
Concept
→ Mathematical specification
→ Minimal experiment
→ Implementation
→ Unit test
→ Visualization/debug output
→ Integration
→ Benchmark
```

Do not implement the entire system in one pass.

Do not claim a milestone is complete unless it runs successfully and has tests.

Prefer simple, transparent implementations over opaque abstractions during early development.

---

## 4. Architecture

Use a modular architecture approximately like:

```text
PanoGS/
├── apps/
│   ├── cli/
│   └── viewer/
│
├── core/
│   ├── camera/
│   ├── geometry/
│   ├── math/
│   ├── scene/
│   └── gaussian/
│
├── reconstruction/
│   ├── image/
│   ├── panorama/
│   ├── video/
│   ├── depth/
│   ├── poses/
│   └── fusion/
│
├── rendering/
│   ├── cpu/
│   └── gpu/
│
├── optimization/
│   ├── losses/
│   ├── optimizer/
│   ├── densification/
│   └── pruning/
│
├── io/
│   ├── images/
│   ├── video/
│   ├── ply/
│   └── scene/
│
├── tests/
├── examples/
├── docs/
├── configs/
├── scripts/
└── README.md
```

Do not create unnecessary empty modules. Create directories as functionality is implemented.

---

# 5. Input Modes

The final engine should support:

### Single image

```text
JPG
JPEG
PNG
```

Pipeline:

```text
Image
→ Depth
→ Camera assumption
→ 3D initialization
→ Gaussians
→ 3DGS
```

### Panorama

Support:

```text
Equirectangular JPG
PNG
HDR
EXR
```

Pipeline:

```text
Panorama
→ Equirectangular camera
→ Spherical ray generation
→ Depth
→ 3D reconstruction
→ Gaussian initialization
→ 3DGS
```

### Video

Support common formats such as:

```text
MP4
MOV
```

Pipeline:

```text
Video
→ Frame extraction
→ Keyframe selection
→ Feature matching/tracking
→ Camera pose estimation
→ Multi-view geometry
→ 3D initialization
→ Gaussian initialization
→ 3DGS
```

Initial video assumption:

STATIC SCENE + MOVING CAMERA.

Do not implement dynamic objects yet.

---

# 6. Phase 0 — Project Foundation

Implement:

* Python package
* virtual environment documentation
* CLI
* logging
* configuration system
* test framework
* basic dependency management

The CLI should eventually support commands such as:

```bash
panogs --help
panogs inspect input.jpg
panogs panorama input.jpg
panogs depth input.jpg
panogs reconstruct input.jpg
panogs train scene/
panogs view scene/
```

Initially only implement commands relevant to the current milestone.

---

# 7. Phase 1 — Image Loader

Implement:

```text
JPG
JPEG
PNG
```

Provide:

* image loading
* RGB conversion
* resolution inspection
* dtype normalization
* memory-aware processing
* basic metadata

Create:

```bash
panogs inspect image.jpg
```

It should report:

```text
format
width
height
channels
dtype
color space
estimated memory
```

Add unit tests.

---

# 8. Phase 2 — Panorama Engine

Implement equirectangular panorama support.

Core mathematical mapping:

```text
pixel (u,v)
→ longitude
→ latitude
→ normalized 3D ray
```

Use pixel-center coordinates.

Implement:

```python
equirectangular_pixel_to_ray(...)
```

and a vectorized:

```python
equirectangular_rays(...)
```

The output should be:

```text
H × W × 3
```

with approximately unit-length vectors.

Test:

```text
||ray|| ≈ 1
```

for representative pixels.

---

# 9. Panorama Validation

Before depth reconstruction, create a synthetic validation pipeline:

```text
Panorama
→ spherical rays
→ constant synthetic depth
→ 3D points
→ colored point cloud
→ PLY
```

This should produce a colored spherical shell.

This is intentionally NOT a real reconstruction.

It validates the spherical geometry.

Add visualization/debugging capability.

---

# 10. Phase 3 — Depth

Introduce a monocular depth model.

The model should be isolated behind an interface such as:

```python
DepthEstimator
```

Do not couple the entire engine to a single depth model.

Support:

```text
input image
→ depth map
→ confidence if available
```

The depth module must support CPU execution.

Provide:

```bash
panogs depth input.jpg
```

Outputs:

```text
depth.npy
depth_visualization.png
```

Important:

* distinguish relative depth from metric depth
* document model limitations
* preserve floating-point depth
* never silently treat relative depth as metric
* allow normalization/scale alignment later

---

# 11. Phase 4 — Depth + Rays → 3D

For each pixel:

```text
P = C + dR
```

where:

```text
C = camera position
d = estimated depth
R = ray direction
```

Implement a vectorized reconstruction function.

Output:

```text
XYZ
RGB
confidence
```

Then create:

```text
colored point cloud
```

and export to:

```text
PLY
```

Add tests for known synthetic rays and depths.

---

# 12. Phase 5 — Point Cloud Processing

Implement modular processing:

* confidence filtering
* invalid-depth filtering
* outlier removal
* voxel downsampling
* normal estimation
* optional depth smoothing
* optional sky/background masking

All operations should be configurable.

Do not destroy original reconstruction data.

Use separate processed outputs.

---

# 13. Phase 6 — Gaussian Representation

Implement a Gaussian scene representation.

At minimum each Gaussian must contain:

```text
position
scale
rotation
opacity
color
```

Later support:

```text
spherical harmonics
covariance
```

Use numerically stable parameterizations.

Scale must remain positive.

Opacity should be safely parameterized.

Rotation must remain valid.

Implement conversion:

```text
PointCloud
→ GaussianSet
```

Initial scale should be estimated from local point spacing when possible rather than using one arbitrary global constant.

Export a debug representation.

---

# 14. Phase 7 — Gaussian Mathematics

Implement and test:

```text
3D Gaussian
covariance
scale + rotation parameterization
camera transformation
projection
2D covariance
2D Gaussian footprint
```

Do not hide the mathematics behind an external library during this educational implementation.

Document the equations in:

```text
docs/gaussian_math.md
```

---

# 15. Phase 8 — CPU Gaussian Renderer

Build a small, understandable CPU renderer.

Pipeline:

```text
3D Gaussian
→ world-to-camera transform
→ projection
→ 2D Gaussian
→ depth ordering
→ alpha compositing
→ output image
```

Correctness comes before speed.

The renderer should initially support a small number of Gaussians.

Create synthetic tests with known Gaussian positions.

Render simple scenes such as:

```text
single Gaussian
two Gaussians
depth-ordered Gaussians
colored Gaussian grid
```

---

# 16. Phase 9 — Differentiable Rendering

Introduce PyTorch where appropriate.

Goal:

```text
Gaussian parameters
→ differentiable render
→ loss
→ gradient
→ parameter update
```

Start with a minimal RGB reconstruction loss.

For example:

```text
L_rgb = L1(rendered_image, target_image)
```

Then expand carefully.

Do not optimize everything simultaneously at first.

Test gradient flow numerically.

---

# 17. Phase 10 — Actual 3DGS Optimization

Implement optimization for:

```text
position
scale
rotation
opacity
color
```

Then implement:

```text
learning-rate schedules
opacity reset
densification
splitting
cloning
pruning
```

Use configurable thresholds.

Track:

```text
loss
Gaussian count
memory
iteration time
render time
```

Save checkpoints.

---

# 18. Phase 11 — Multi-view Images

Support:

```text
images/
    000001.jpg
    000002.jpg
    ...
```

Implement a camera abstraction containing:

```text
intrinsics
extrinsics
resolution
projection type
```

Support:

```text
perspective camera
equirectangular camera
```

Convert all observations into a common world coordinate system.

---

# 19. Phase 12 — Video Reconstruction

Implement:

```text
video
→ frames
→ keyframes
→ feature detection
→ feature matching/tracking
→ camera poses
→ sparse geometry
→ depth/multi-view reconstruction
→ Gaussian initialization
```

Do not process every video frame by default.

Use configurable sampling.

Provide diagnostics:

```text
number of frames
number of keyframes
feature matches
estimated camera trajectory
reconstruction confidence
```

The estimated camera trajectory should be visualizable.

---

# 20. Phase 13 — Viewer

Start with an existing reliable 3D visualization library if necessary.

The first viewer should support:

```text
orbit
pan
zoom
camera movement
```

Later create a browser viewer using WebGL/WebGPU.

Viewer should eventually load:

```text
.scene
PLY
SPLAT
```

---

# 21. Scene Format

Create an internal scene representation.

Eventually support:

```text
.scene
```

containing:

```text
metadata
camera parameters
camera poses
Gaussian parameters
training configuration
source information
```

Also support export:

```text
PLY
SPLAT
```

Keep the internal representation independent of any external viewer.

---

# 22. Hardware Backends

The system must expose:

```bash
--device cpu
```

and eventually:

```bash
--device cuda
```

CPU must remain functional.

GPU code must be optional.

Never assume NVIDIA CUDA exists.

Do not make CUDA a requirement for installation.

---

# 23. Testing

Every mathematical core must have tests.

Required categories:

```text
camera
projection
ray generation
depth
point reconstruction
PLY
Gaussian representation
covariance
projection
renderer
loss
optimization
```

Use synthetic deterministic tests whenever possible.

Example:

```text
known pixel
→ expected ray
```

and:

```text
known ray + depth
→ expected XYZ
```

---

# 24. Benchmarking

Create small reproducible benchmarks.

Track:

```text
input resolution
number of points
number of Gaussians
memory usage
initialization time
render time
training iteration time
loss
```

Do not optimize without measurements.

---

# 25. Documentation

Maintain:

```text
README.md
docs/architecture.md
docs/camera_geometry.md
docs/panorama.md
docs/depth.md
docs/pointcloud.md
docs/gaussian_math.md
docs/rendering.md
docs/optimization.md
docs/video_pipeline.md
```

Documentation should explain both:

1. what the code does
2. why it exists mathematically

---

# 26. Strict Milestone Rule

Do NOT move to the next phase until:

1. Current implementation runs.
2. Tests pass.
3. Output is visually inspected.
4. Failure modes are documented.
5. Code is committed or otherwise checkpointed.

Use milestone tags or commits such as:

```text
v0.1-input
v0.2-panorama-rays
v0.3-depth
v0.4-pointcloud
v0.5-gaussian
v0.6-renderer
v0.7-differentiable
v0.8-3dgs
```

---

# 27. Current Milestone

Start ONLY with:

## Milestone 0 + Milestone 1

Build:

```text
PanoGS/
→ Python package
→ CLI
→ configuration
→ logging
→ tests
→ image loader
→ image inspection command
```

Do NOT implement:

* depth
* Gaussian splatting
* 3DGS
* video
* viewer
* CUDA

yet.

After Milestone 1 works, stop and report:

```text
Files created
Dependencies installed
Tests run
Tests passed
CLI commands available
Known limitations
Next milestone
```

Then wait for approval before continuing.

---

# 28. Agent Behavior

When encountering a technical choice:

1. Prefer simple solutions first.
2. Explain why the choice is being made.
3. Avoid unnecessary dependencies.
4. Avoid huge model downloads.
5. Keep CPU compatibility.
6. Write tests alongside implementation.
7. Never claim success without running the relevant code.
8. Never hide an external black-box implementation behind a function that appears to be our own implementation.
9. Clearly identify external libraries and what functionality they provide.
10. Keep the architecture replaceable so experimental implementations can later be swapped in.

The goal is not merely to produce a working demo.

The goal is to build a real, understandable 3DGS reconstruction engine incrementally.

---

# 13. One more thing: don't let Antigravity run away

After you give it that master instruction, **do not ask it "build the whole thing."**

Give it this next:

Start Milestone 0 + Milestone 1 only.

Before writing code:

1. Inspect the current workspace.
2. Check the installed Python version.
3. Check available RAM and GPU information.
4. Check whether an existing project structure already exists.
5. Do not delete or overwrite unrelated files.

Then create the minimal PanoGS project foundation and image loader.

Requirements:

* Python package structure
* CLI entry point
* configuration system
* logging
* pytest setup
* image loader for JPG/JPEG/PNG
* `panogs --help`
* `panogs inspect <image>`
* unit tests for image loading and metadata
* README with setup instructions
* no depth model yet
* no 3DGS yet
* no video pipeline yet
* no CUDA dependency
* no unnecessary large downloads

Run the tests yourself.

Run the CLI yourself on a small test image.

Do not proceed beyond Milestone 1.

At the end, report:

* files created
* dependencies installed
* commands run
* test results
* CLI output
* known limitations
* exact next milestone

That's where I'd start.

**And importantly, don't let the agent decide the architecture for us after that.** We can review each milestone together. When it reaches the panorama stage, I'll teach you the spherical-camera mathematics before we have it implement the reconstruction. When it reaches Gaussian rendering, we'll stop and learn covariance/projection/rasterization before moving into optimization.

The eventual destination is **a complete 3DGS system**, but we're going to earn it one layer at a time.
