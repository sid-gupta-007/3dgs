"""Generate test images and synthetic equirectangular panoramas for PanoGS."""
from pathlib import Path
import numpy as np
from PIL import Image


def generate_sample_image(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Generate a gradient image (512x512 RGB)
    x = np.linspace(0, 1, 512)
    y = np.linspace(0, 1, 512)
    xx, yy = np.meshgrid(x, y)

    r = (np.sin(xx * np.pi) * 255).astype(np.uint8)
    g = (np.cos(yy * np.pi) * 255).astype(np.uint8)
    b = ((xx + yy) / 2.0 * 255).astype(np.uint8)

    rgb = np.stack([r, g, b], axis=-1)
    img = Image.fromarray(rgb)
    img.save(output_path, quality=95)
    print(f"Sample test image generated at: {output_path}")


def generate_sample_panorama(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Equirectangular 2:1 aspect ratio (512x256)
    width, height = 512, 256
    u = np.linspace(0, 1, width)
    v = np.linspace(0, 1, height)
    uu, vv = np.meshgrid(u, v)

    # Synthetic room pattern (sky blue on top, floor grid on bottom, colored walls)
    r = (np.sin(uu * 2 * np.pi) * 0.5 + 0.5) * 255
    g = (vv * 255)
    b = (np.cos(uu * 2 * np.pi) * 0.5 + 0.5) * 255

    pano_rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    img = Image.fromarray(pano_rgb)
    img.save(output_path, quality=95)
    print(f"Sample equirectangular panorama generated at: {output_path}")


if __name__ == "__main__":
    generate_sample_image(Path("examples/data/sample_room.jpg"))
    generate_sample_panorama(Path("examples/data/sample_panorama.jpg"))
