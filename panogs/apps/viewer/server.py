"""
Local HTTP streaming server for the PanoGS interactive 3D WebGL viewer.
Serves the web application and streams 3DGS binary .splat scene data.
"""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
from pathlib import Path
import threading
from typing import Optional, Union
import webbrowser

from panogs.core.logging import get_logger
from panogs.io.gaussian_ply import load_gaussian_ply, save_gaussian_splat


class SplatViewerHandler(SimpleHTTPRequestHandler):
    """HTTP Request Handler serving viewer assets and binary splat stream."""

    model_bytes: bytes = b""
    html_content: str = ""

    def do_GET(self):
        logger = get_logger("viewer.server")

        if self.path == "/" or self.path.startswith("/index.html"):
            html_path = Path(__file__).parent / "index.html"
            if html_path.exists():
                with open(html_path, "r", encoding="utf-8") as f:
                    content = f.read().encode("utf-8")
            else:
                content = self.html_content.encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(content)

        elif self.path == "/api/scene.splat":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(self.model_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(self.model_bytes)

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Quiet standard HTTP access logs
        pass


def prepare_splat_bytes(model_path: Union[str, Path]) -> bytes:
    """Read a .splat file or convert a .ply 3DGS model into binary .splat bytes in memory."""
    path = Path(model_path)
    logger = get_logger("viewer.server")

    if path.suffix.lower() == ".splat":
        logger.info(f"Loading WebGL .splat scene: {path}")
        with open(path, "rb") as f:
            return f.read()

    elif path.suffix.lower() == ".ply":
        logger.info(f"Converting 3DGS PLY {path} to WebGL .splat in memory...")
        model = load_gaussian_ply(path)
        temp_splat = path.with_suffix(".tmp.splat")
        save_gaussian_splat(temp_splat, model)
        with open(temp_splat, "rb") as f:
            data = f.read()
        try:
            temp_splat.unlink()
        except OSError:
            pass
        return data

    else:
        raise ValueError(f"Unsupported 3DGS format: {path.suffix} (expected .splat or .ply)")


def start_viewer_server(
    model_path: Union[str, Path],
    port: int = 8080,
    open_browser: bool = True,
    block: bool = True,
) -> ThreadingHTTPServer:
    """
    Start the interactive WebGL 3DGS viewer HTTP server.

    Args:
        model_path: Path to .splat or .ply file.
        port: Port to serve on (default: 8080).
        open_browser: Whether to open default web browser automatically.
        block: Whether to block calling thread with serve_forever().

    Returns:
        ThreadingHTTPServer instance.
    """
    logger = get_logger("viewer.server")
    html_path = Path(__file__).parent / "index.html"

    with open(html_path, "r", encoding="utf-8") as f:
        html_str = f.read()

    splat_data = prepare_splat_bytes(model_path)
    num_gaussians = len(splat_data) // 32

    SplatViewerHandler.html_content = html_str
    SplatViewerHandler.model_bytes = splat_data

    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, SplatViewerHandler)

    url = f"http://127.0.0.1:{port}"
    logger.info(f"Serving 3D Gaussian Splatting Explorer at {url} ({num_gaussians:,} Gaussians)")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    if block:
        try:
            print(f"\n=======================================================")
            print(f"  PanoGS 3D Gaussian Splatting Interactive Viewer")
            print(f"  Running at:  {url}")
            print(f"  Gaussians:   {num_gaussians:,}")
            print(f"  Press Ctrl+C in terminal to stop.")
            print(f"=======================================================\n")
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Viewer server stopped.")
            httpd.server_close()

    return httpd
