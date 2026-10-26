"""
Tests for the PanoGS interactive WebGL 3D Gaussian Splatting viewer and server.
"""

from pathlib import Path
import io
import json
import numpy as np
import pytest

from panogs.apps.cli import main
from panogs.apps.viewer.server import prepare_splat_bytes, start_viewer_server
from panogs.core.gaussian.model import GaussianModel
from panogs.io.gaussian_ply import is_supersplat_compatible_ply, save_gaussian_ply, save_gaussian_splat


class _InMemorySocket:
    """In-memory socket adapter for testing ThreadingHTTPServer handlers."""

    def __init__(self, raw_request: bytes):
        self._rfile = io.BytesIO(raw_request)
        self._wfile = io.BytesIO()

    def makefile(self, mode: str, *args, **kwargs):
        return self._rfile if "r" in mode else self._wfile

    def sendall(self, data: bytes):
        self._wfile.write(data)

    def getvalue(self) -> bytes:
        return self._wfile.getvalue()


def _http_request(server, method: str, path: str, body: bytes = b"") -> tuple[int, bytes]:
    port = server.server_address[1]
    raw = (
        f"{method} {path} HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("ascii") + body
    sock = _InMemorySocket(raw)
    server.RequestHandlerClass(sock, ("127.0.0.1", port), server)
    response_bytes = sock.getvalue()
    header_blob, payload = response_bytes.split(b"\r\n\r\n", 1)
    status_line = header_blob.splitlines()[0].decode("ascii")
    status_code = int(status_line.split()[1])
    return status_code, payload


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
    server = start_viewer_server(
        model_path=sample_splat_file,
        port=0,
        open_browser=False,
        block=False,
    )

    try:
        status, body = _http_request(server, "GET", "/")
        assert status == 200
        content = body.decode("utf-8")
        assert "PanoGS Explorer" in content
        assert "WebGL" in content

        status, data = _http_request(server, "GET", "/api/scene.splat")
        assert status == 200
        assert len(data) == 64
    finally:
        server.server_close()


def test_studio_open_ply_normalizes_for_supersplat(sample_ply_file: Path, tmp_path: Path, monkeypatch):
    """Test Studio /api/open-ply converts legacy PanoGS PLY into SuperSplat-ready format."""
    monkeypatch.chdir(tmp_path)
    server = start_viewer_server(
        model_path=None,
        port=0,
        open_browser=False,
        block=False,
    )
    try:
        status, studio_bytes = _http_request(server, "GET", "/")
        assert status == 200
        assert "PanoGS Studio" in studio_bytes.decode("utf-8")

        raw_ply = sample_ply_file.read_bytes()
        assert not is_supersplat_compatible_ply(sample_ply_file)

        status, resp_bytes = _http_request(server, "POST", "/api/open-ply?filename=scene.ply", body=raw_ply)
        assert status == 200
        payload = json.loads(resp_bytes.decode("utf-8"))
        assert payload["editor_url"].startswith("/supersplat/?content=")
        ply_url = payload["ply_url"]

        status, served_ply = _http_request(server, "GET", ply_url)
        assert status == 200
        assert b"coord=colmap_z180_rh; color_space=srgb-display; geometry=watertight-v4" in served_ply[:512]
    finally:
        server.server_close()


def test_cli_view_help(capsys):
    """Test panogs view --help output."""
    with pytest.raises(SystemExit) as exc:
        main(["view", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "WebGL" in captured.out
    assert "model_path" in captured.out


def test_viewer_html_collision_sliding_features():
    """Verify index.html includes density-filtered obstacle grid, spawn clearance, and tangent sliding."""
    html_path = Path(__file__).parent.parent / "panogs" / "apps" / "viewer" / "index.html"
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")

    assert "SPAWN_CLEAR_RADIUS" in content
    assert "MIN_POINTS_FOR_SOLID" in content
    assert "findNearestObstacle" in content
    assert "toggleCollision" in content
    assert "resolveMovement" in content
