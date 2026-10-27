"""Tests for Generative Multi-Angle Viewpoint Synthesis (PanoGS V3)."""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from panogs.core.camera.spherical import equirectangular_rays
from panogs.reconstruction.pointcloud import PointCloud
from panogs.reconstruction.synthesis import (
    create_virtual_camera_rig,
    render_depth_and_color_view,
    synthesize_cavity_splats,
    synthesize_multi_view_scene,
    VirtualCameraView,
)
from panogs.reconstruction.panorama.layout import reconstruct_layout_panorama
from panogs.reconstruction.depth.base import DepthEstimator, DepthResult


def test_equirectangular_rays_jitter():
    H, W = 32, 64
    rays_standard = equirectangular_rays(H, W, jitter=False)
    rays_jittered = equirectangular_rays(H, W, jitter=True)

    assert rays_standard.shape == (H, W, 3)
    assert rays_jittered.shape == (H, W, 3)

    # Unit norm check
    norms = np.linalg.norm(rays_jittered, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    # Jittered rays should differ slightly from standard grid
    diff = np.abs(rays_jittered - rays_standard)
    assert np.max(diff) > 0.001


def test_create_virtual_camera_rig():
    rig = create_virtual_camera_rig(room_depth=4.0, room_width=3.5, num_views=4)
    assert len(rig) == 4
    for view in rig:
        assert isinstance(view, VirtualCameraView)
        assert view.position.shape == (3,)
        assert view.target.shape == (3,)
        assert view.fov_deg == 75.0


def test_render_and_synthesize_cavity():
    # Construct a simple synthetic point cloud with a foreground block
    pts = []
    cols = []
    norms = []

    # Background wall at Z = 3.0
    for x in np.linspace(-2.0, 2.0, 30):
        for y in np.linspace(-1.5, 1.5, 30):
            pts.append([x, y, 3.0])
            cols.append([200, 200, 200])
            norms.append([0.0, 0.0, -1.0])

    # Foreground occluding block at Z = 1.2
    for x in np.linspace(-0.5, 0.5, 15):
        for y in np.linspace(-0.5, 0.5, 15):
            pts.append([x, y, 1.2])
            cols.append([50, 150, 220])
            norms.append([0.0, 0.0, -1.0])

    pc = PointCloud(
        points=np.array(pts, dtype=np.float32),
        colors=np.array(cols, dtype=np.uint8),
        normals=np.array(norms, dtype=np.float32),
    )

    cam = VirtualCameraView(
        position=np.array([0.8, 0.0, 0.0], dtype=np.float32),
        target=np.array([0.0, 0.0, 2.0], dtype=np.float32),
        up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        width=128,
        height=128,
    )

    depth_buf, color_buf, normal_buf, valid_mask = render_depth_and_color_view(pc, cam)
    assert depth_buf.shape == (128, 128)
    assert color_buf.shape == (128, 128, 3)
    assert np.any(valid_mask)

    synth_pc = synthesize_cavity_splats(pc, cam)
    if synth_pc is not None:
        assert synth_pc.num_points > 0
        assert synth_pc.points.shape[1] == 3
        assert synth_pc.colors.shape[1] == 3


def test_synthesize_multi_view_scene():
    pts = np.random.uniform(-2, 2, (500, 3)).astype(np.float32)
    cols = np.random.randint(0, 255, (500, 3)).astype(np.uint8)
    norms = np.tile(np.array([[0, 1, 0]], dtype=np.float32), (500, 1))

    pc = PointCloud(points=pts, colors=cols, normals=norms)
    fused_pc = synthesize_multi_view_scene(pc, room_depth=4.0, room_width=3.5, num_views=4)

    assert fused_pc.num_points >= pc.num_points
    assert fused_pc.normals is not None
    assert "num_views_synthesized" in fused_pc.metadata


def test_reconstruct_layout_solid_shell_v3(tmp_path):
    class MockEstimator(DepthEstimator):
        def estimate(self, image):
            if isinstance(image, (str, Path)):
                img_np = np.array(Image.open(image))
            elif isinstance(image, Image.Image):
                img_np = np.array(image)
            else:
                img_np = np.asarray(image)
            H, W = img_np.shape[:2]
            d = np.full((H, W), 4.5, dtype=np.float32)
            # Add bathtub/furniture at 1.2m
            d[int(H * 0.4):int(H * 0.8), int(W * 0.3):int(W * 0.7)] = 1.2
            return DepthResult(depth_map=d, is_metric=True, model_name="mock_depth")

    img_arr = np.random.randint(50, 200, (64, 128, 3), dtype=np.uint8)
    img_path = tmp_path / "test_v3_pano.png"
    Image.fromarray(img_arr).save(img_path)

    pc = reconstruct_layout_panorama(
        image_path=img_path,
        depth_estimator=MockEstimator(),
        h_floor=1.5,
        h_ceiling=1.8,
        room_depth=4.5,
        room_width=4.0,
        solid_shell=True,
        jitter=True,
        max_resolution=128,
    )

    assert pc.num_points > 0.9 * (64 * 128)
    assert pc.metadata["solid_shell"] is True
    assert pc.metadata["jitter"] is True
    assert pc.metadata["infilled_points"] > 0
