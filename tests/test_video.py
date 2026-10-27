"""Tests for video ingestion, keyframe extraction, and video-to-3DGS pipeline."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from panogs.io.video_loader import (
    calculate_blur_score,
    extract_video_keyframes,
    VideoFrame,
)
from panogs.reconstruction.video.video_pipeline import (
    estimate_camera_motion,
    reconstruct_from_video,
)
from panogs.reconstruction.depth.synthetic import SyntheticDepthEstimator


@pytest.fixture
def sample_video_path(tmp_path: Path) -> Path:
    """Create a short 10-frame synthetic MP4 video."""
    video_file = tmp_path / "test_walk.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_file), fourcc, 10.0, (120, 80))

    for i in range(10):
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        # Moving box with textured features for ORB matching
        x_offset = int(i * 3)
        cv2.rectangle(frame, (10 + x_offset, 20), (50 + x_offset, 60), (200, 200, 50), -1)
        cv2.circle(frame, (30 + x_offset, 40), 10, (50, 50, 200), -1)
        out.write(frame)

    out.release()
    return video_file


def test_extract_video_keyframes(sample_video_path: Path):
    frames = extract_video_keyframes(sample_video_path, target_fps=5.0, max_frames=5)
    assert len(frames) > 0
    assert isinstance(frames[0], VideoFrame)
    assert frames[0].image_rgb.shape == (80, 120, 3)
    assert frames[0].blur_score >= 0.0


def test_estimate_camera_motion():
    img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img2 = np.roll(img1, 2, axis=1)
    K = np.eye(3, dtype=np.float32)

    R, t = estimate_camera_motion(img1, img2, K)
    assert R.shape == (3, 3)
    assert t.shape == (3, 1)


def test_reconstruct_from_video(sample_video_path: Path, tmp_path: Path):
    estimator = SyntheticDepthEstimator(mode="room")
    out_ply = tmp_path / "video_reconstructed.ply"

    pc = reconstruct_from_video(
        video_path=sample_video_path,
        depth_estimator=estimator,
        target_fps=5.0,
        max_frames=4,
        voxel_size=0.05,
        output_ply=out_ply,
    )

    assert pc.num_points > 0
    assert out_ply.exists()
    assert pc.normals is not None
