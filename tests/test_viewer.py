"""
Tests for the PanoGS interactive WebGL 3D Gaussian Splatting viewer and server.
"""

from pathlib import Path
import threading
import time
import urllib.request
import numpy as np
import pytest

from panogs.apps.cli import main
from panogs.apps.viewer.server import prepare_splat_bytes, start_viewer_server
from panogs.core.gaussian.model import GaussianModel
from panogs.io.gaussian_ply import save_gaussian_ply, save_gaussian_splat


@pytest.fixture
def sample_splat_file(tmp_path: Path) -> Path:
    xyz = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]], dtype=np.float32)
    scales = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], dtype=np.float32)
    quats = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    opacities = np.array([0.9, 0.8], dtype=np.float32)

    model = GaussianModel.from_raw(xyz, scales, quats, colors, opacities)
    splat_path = tmp_path / "scene.splat"
    save_gaussian_splat(splat_path, model)
    return splat_path


@pytest.fixture
def sample_ply_file(tmp_path: Path) -> Path:
    xyz = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]], dtype=np.float32)
    scales = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], dtype=np.float32)
    quats = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    opacities = np.array([0.9, 0.8], dtype=np.float32)

    model = GaussianModel.from_raw(xyz, scales, quats, colors, opacities)
    ply_path = tmp_path / "scene.ply"
    save_gaussian_ply(ply_path, model)
    return ply_path


def test_prepare_splat_bytes(sample_splat_file: Path, sample_ply_file: Path):
    """Test reading .splat and dynamically converting .ply."""
    splat_bytes = prepare_splat_bytes(sample_splat_file)
    assert len(splat_bytes) == 2 * 32  # 2 Gaussians * 32 bytes

    ply_splat_bytes = prepare_splat_bytes(sample_ply_file)
    assert len(ply_splat_bytes) == 2 * 32


def test_viewer_http_endpoints(sample_splat_file: Path):
    """Test HTTP server serving index.html and /api/scene.splat."""
    port = 8999
    server = start_viewer_server(
        model_path=sample_splat_file,
        port=port,
        open_browser=False,
        block=False,
    )

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)

    try:
        # Test index.html endpoint
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            assert resp.status == 200
            content = resp.read().decode("utf-8")
            assert "PanoGS Explorer" in content
            assert "WebGL" in content

        # Test splat streaming endpoint
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/scene.splat") as resp:
            assert resp.status == 200
            data = resp.read()
            assert len(data) == 64
    finally:
        server.shutdown()
        server.server_close()


def test_cli_view_help(capsys):
    """Test panogs view --help output."""
    with pytest.raises(SystemExit) as exc:
        main(["view", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "WebGL" in captured.out
    assert "model_path" in captured.out
