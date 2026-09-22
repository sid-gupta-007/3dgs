"""
Video Frame Extraction and Keyframe Selection Module.
Supports MP4, MOV, AVI formats with blur filtering and adaptive frame sampling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from panogs.core.logging import get_logger

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class VideoFrame:
    """A single extracted video frame with timestamp and blur metadata."""
    frame_index: int
    timestamp_sec: float
    image_rgb: np.ndarray  # (H, W, 3) uint8
    blur_score: float

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.image_rgb.shape[1], self.image_rgb.shape[0]  # W, H


def calculate_blur_score(image_rgb: np.ndarray) -> float:
    """Calculate Laplacian variance to measure image sharpness (higher is sharper)."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_video_keyframes(
    video_path: Union[str, Path],
    target_fps: float = 2.0,
    max_frames: int = 150,
    min_blur_score: float = 30.0,
    max_resolution: Optional[int] = 1024,
) -> List[VideoFrame]:
    """
    Extract sharp, temporally-spaced keyframes from a video file.

    Args:
        video_path: Path to video file (.mp4, .mov, .avi).
        target_fps: Target frame extraction frequency in Hz (default: 2.0 frames/sec).
        max_frames: Maximum number of frames to return.
        min_blur_score: Minimum sharpness threshold (rejects motion blur).
        max_resolution: Optional maximum dimension downscaling.

    Returns:
        List[VideoFrame]: Extracted sharp video keyframes.
    """
    logger = get_logger("io.video_loader")
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video format: {video_path.suffix}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_native_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(native_fps / target_fps)))

    logger.info(
        f"Processing video '{video_path.name}': {total_native_frames} frames, "
        f"{native_fps:.1f} FPS, sampling every {frame_interval} frames (~{target_fps:.1f} FPS)..."
    )

    frames: List[VideoFrame] = []
    frame_idx = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, bgr_frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            if max_resolution is not None:
                h, w = rgb_frame.shape[:2]
                max_dim = max(h, w)
                if max_dim > max_resolution:
                    scale = max_resolution / float(max_dim)
                    new_w = max(1, int(round(w * scale)))
                    new_h = max(1, int(round(h * scale)))
                    rgb_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            blur_score = calculate_blur_score(rgb_frame)
            timestamp = frame_idx / native_fps

            if blur_score >= min_blur_score or len(frames) == 0:
                frames.append(
                    VideoFrame(
                        frame_index=frame_idx,
                        timestamp_sec=timestamp,
                        image_rgb=rgb_frame,
                        blur_score=blur_score,
                    )
                )

        frame_idx += 1

    cap.release()
    logger.info(f"Extracted {len(frames)} sharp keyframes from video.")
    return frames
