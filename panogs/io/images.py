"""
Image loading, inspection, and format handling module.
Supports JPG, JPEG, and PNG formats with strict RGB normalization,
metadata extraction, and memory-aware calculations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".hdr", ".exr"}


@dataclass
class ImageMetadata:
    """Metadata describing an image file and its memory footprint."""
    file_path: str
    format: str
    width: int
    height: int
    channels: int
    dtype: str
    color_space: str
    file_size_bytes: int
    raw_memory_bytes: int

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return (width, height)."""
        return self.width, self.height

    @property
    def aspect_ratio(self) -> float:
        """Return width / height."""
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def raw_memory_mb(self) -> float:
        """Return raw uncompressed memory in Megabytes."""
        return self.raw_memory_bytes / (1024 * 1024)

    def formatted_summary(self) -> str:
        """Generate human-readable inspection output."""
        return (
            f"Image Inspection: {Path(self.file_path).name}\n"
            f"----------------------------------------\n"
            f"  Path:             {self.file_path}\n"
            f"  Format:           {self.format}\n"
            f"  Dimensions:       {self.width} x {self.height} (W x H)\n"
            f"  Aspect Ratio:     {self.aspect_ratio:.2f}\n"
            f"  Channels:         {self.channels}\n"
            f"  Color Space:      {self.color_space}\n"
            f"  Data Type:        {self.dtype}\n"
            f"  Disk Size:        {self.file_size_bytes / 1024:.2f} KB\n"
            f"  Raw Memory:       {self.raw_memory_mb:.2f} MB ({self.raw_memory_bytes:,} bytes)"
        )


def validate_image_path(path: Union[str, Path]) -> Path:
    """Validate that path exists and has a supported extension."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a regular file: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{file_path.suffix}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return file_path


def inspect_image(path: Union[str, Path]) -> ImageMetadata:
    """
    Inspect an image file and extract its dimensional, channel, and format metadata.
    """
    file_path = validate_image_path(path)
    file_size = file_path.stat().st_size
    ext = file_path.suffix.lower()

    if ext in {".hdr", ".exr"}:
        import cv2
        hdr_data = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if hdr_data is None:
            raise ValueError(f"Failed to read HDR image: {file_path}")
        height, width = hdr_data.shape[:2]
        channels = 3 if hdr_data.ndim == 3 else 1
        return ImageMetadata(
            file_path=str(file_path.resolve()),
            format="HDR" if ext == ".hdr" else "EXR",
            width=width,
            height=height,
            channels=channels,
            dtype="float32",
            color_space="Linear Radiance HDR",
            file_size_bytes=file_size,
            raw_memory_bytes=width * height * channels * 4,
        )

    with Image.open(file_path) as img:
        img_format = img.format or file_path.suffix[1:].upper()
        width, height = img.size
        mode = img.mode

        # Determine standard channels and color space
        if mode == "RGB":
            channels = 3
            color_space = "RGB"
        elif mode == "RGBA":
            channels = 4
            color_space = "RGBA (Alpha)"
        elif mode == "L":
            channels = 1
            color_space = "Grayscale"
        elif mode == "LA":
            channels = 2
            color_space = "Grayscale + Alpha"
        elif mode == "CMYK":
            channels = 4
            color_space = "CMYK"
        elif mode == "P":
            channels = 3  # Palette mapped
            color_space = "Palette (Indexed)"
        else:
            channels = len(mode)
            color_space = mode

        # Raw memory in uint8 RGB format
        raw_memory = width * height * 3

        return ImageMetadata(
            file_path=str(file_path.resolve()),
            format=img_format,
            width=width,
            height=height,
            channels=channels,
            dtype="uint8",
            color_space=color_space,
            file_size_bytes=file_size,
            raw_memory_bytes=raw_memory,
        )


def load_image(
    path: Union[str, Path],
    max_resolution: Optional[int] = None,
) -> Image.Image:
    """
    Load an image using PIL or OpenCV (for HDR/EXR), convert to RGB mode, and optionally downscale if exceeding max_resolution.
    """
    file_path = validate_image_path(path)
    ext = file_path.suffix.lower()

    if ext in {".hdr", ".exr"}:
        import cv2
        hdr_bgr = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if hdr_bgr is None:
            raise ValueError(f"Failed to read HDR/EXR image: {file_path}")
        if hdr_bgr.ndim == 2:
            hdr_rgb = np.stack([hdr_bgr] * 3, axis=-1)
        else:
            hdr_rgb = cv2.cvtColor(hdr_bgr, cv2.COLOR_BGR2RGB)

        # Tone map linear radiance to display RGB via Reinhard operator
        toned = hdr_rgb / (1.0 + hdr_rgb)
        uint8_rgb = (np.clip(toned, 0.0, 1.0) * 255.0).astype(np.uint8)
        img = Image.fromarray(uint8_rgb)
    else:
        img = Image.open(file_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

    # Downscale if max_resolution is specified and image exceeds it
    if max_resolution is not None:
        w, h = img.size
        max_dim = max(w, h)
        if max_dim > max_resolution:
            scale = max_resolution / float(max_dim)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return img


def load_image_as_numpy(
    path: Union[str, Path],
    normalize_float: bool = True,
    max_resolution: Optional[int] = None,
) -> np.ndarray:
    """
    Load an image as a numpy array in H x W x 3 RGB format.

    Args:
        path: Path to the image file.
        normalize_float: If True, returns float32 array normalized to [0.0, 1.0].
                         If False, returns uint8 array in [0, 255].
        max_resolution: Optional maximum dimension constraint for memory safety.

    Returns:
        np.ndarray: Array with shape (H, W, 3).
    """
    pil_img = load_image(path, max_resolution=max_resolution)
    arr = np.asarray(pil_img)

    # Ensure array is contiguous and 3-channel
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    if normalize_float:
        return (arr.astype(np.float32) / 255.0).copy()
    return arr.copy()


def save_image(
    arr: np.ndarray,
    output_path: Union[str, Path],
) -> Path:
    """
    Save a numpy array (H, W, 3) or (H, W) as an image file.
    Automatically handles float32 in [0, 1] or uint8 in [0, 255].
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if arr.dtype == np.float32 or arr.dtype == np.float64:
        clipped = np.clip(arr, 0.0, 1.0)
        uint8_arr = (clipped * 255.0).astype(np.uint8)
    elif arr.dtype == np.uint8:
        uint8_arr = arr
    else:
        uint8_arr = arr.astype(np.uint8)

    img = Image.fromarray(uint8_arr)
    img.save(target)
    return target
