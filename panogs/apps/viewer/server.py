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

        elif self.path == "/api/scenes" or self.path.startswith("/api/scenes"):
            import json
            output_dir = Path("output")
            scenes = []

            # Priority curated scenes explicitly aligned with branches and presentations
            curated_map = [
                {
                    "id": "comfy_cafe_main.splat",
                    "aliases": ["comfy_cafe_main.splat", "cafe_inpainted_gaussians.splat"],
                    "name": "☕ Comfy Cafe",
                    "branch": "main",
                    "badge": "Main Branch",
                    "desc": "Original spherical layout from main branch",
                },
                {
                    "id": "modern_bathroom_ver2.splat",
                    "aliases": ["modern_bathroom_ver2.splat", "modern_bathroom_v2_sharp.splat"],
                    "name": "🛁 Modern Bathroom",
                    "branch": "Ver-2",
                    "badge": "Ver-2 Sharp",
                    "desc": "Sharp Manhattan room model from Ver-2",
                },
                {
                    "id": "glasshouse_interior_ver3.splat",
                    "aliases": ["glasshouse_interior_ver3.splat", "glasshouse_interior_2k.splat"],
                    "name": "🌿 Glasshouse Interior",
                    "branch": "Ver-3",
                    "badge": "Ver-3 True 3D",
                    "desc": "True 3D depth geometry with solid backing inpaint",
                },
                {
                    "id": "mirrored_hall_ver3.splat",
                    "aliases": ["mirrored_hall_ver3.splat", "mirrored_hall_2k.splat"],
                    "name": "🏛️ Mirrored Hall",
                    "branch": "Ver-3",
                    "badge": "Ver-3 True 3D",
                    "desc": "True 3D depth geometry with solid backing inpaint",
                },
                {
                    "id": "comfy_cafe_2k.splat",
                    "aliases": ["comfy_cafe_2k.splat"],
                    "name": "☕ Comfy Cafe (V3 High-Density)",
                    "branch": "Ver-3",
                    "badge": "2.40M Splats",
                    "desc": "Dense true 3D V3 model",
                },
                {
                    "id": "modern_bathroom_v3_true3d.splat",
                    "aliases": ["modern_bathroom_v3_true3d.splat"],
                    "name": "🛁 Modern Bathroom (V3 True 3D)",
                    "branch": "Ver-3",
                    "badge": "2.36M Splats",
                    "desc": "Full unflattened 3D bathroom geometry",
                },
                {
                    "id": "seaview_suite_gaussians.splat",
                    "aliases": ["seaview_suite_gaussians.splat"],
                    "name": "🌊 Seaview Suite",
                    "branch": "Ver-1",
                    "badge": "Suite",
                    "desc": "Classic panorama model",
                },
            ]

            seen_ids = set()
            for item in curated_map:
                target_file = None
                for alias in item["aliases"]:
                    p = output_dir / alias
                    if p.exists():
                        target_file = p
                        break
                if target_file:
                    splat_cnt = target_file.stat().st_size // 32
                    size_mb = target_file.stat().st_size / (1024 * 1024)
                    scenes.append({
                        "id": target_file.name,
                        "name": item["name"],
                        "branch": item["branch"],
                        "badge": item["badge"],
                        "desc": item["desc"],
                        "gaussians": splat_cnt,
                        "size_mb": round(size_mb, 1),
                        "url": f"/?scene={target_file.name}",
                        "api_url": f"/api/scene.splat?scene={target_file.name}",
                    })
                    for a in item["aliases"]:
                        seen_ids.add(a)

            # Also discover any other .splat in output
            if output_dir.exists():
                for p in sorted(output_dir.glob("*.splat")):
                    if p.name not in seen_ids and not p.name.endswith(".tmp.splat"):
                        splat_cnt = p.stat().st_size // 32
                        size_mb = p.stat().st_size / (1024 * 1024)
                        clean_name = p.stem.replace("_", " ").title()
                        scenes.append({
                            "id": p.name,
                            "name": clean_name,
                            "branch": "Local",
                            "badge": f"{size_mb:.0f} MB",
                            "desc": f"Local model ({splat_cnt:,} splats)",
                            "gaussians": splat_cnt,
                            "size_mb": round(size_mb, 1),
                            "url": f"/?scene={p.name}",
                            "api_url": f"/api/scene.splat?scene={p.name}",
                        })
                        seen_ids.add(p.name)

            resp_body = json.dumps({"scenes": scenes}, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp_body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(resp_body)

        elif self.path.startswith("/api/scene.splat"):
            import urllib.parse
            parsed = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed.query)

            data = self.model_bytes
            if "scene" in query_params:
                scene_name = query_params["scene"][0]
                candidates = [
                    Path("output") / scene_name,
                    Path("output") / f"{scene_name}.splat",
                    Path(scene_name),
                ]
                for cand in candidates:
                    if cand.exists():
                        try:
                            data = prepare_splat_bytes(cand)
                            break
                        except Exception as e:
                            logger.error(f"Failed to load dynamic scene {cand}: {e}")

            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(data)

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
