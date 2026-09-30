"""
Local HTTP streaming server for the PanoGS interactive 3D WebGL viewer.
Serves the web application and streams standard or HDR PanoGS splat scene data.
"""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
from typing import Optional, Union
import webbrowser

from panogs.core.logging import get_logger
from panogs.io.gaussian_ply import load_gaussian_ply, save_gaussian_hdr_splat, save_gaussian_splat


class SplatViewerHandler(SimpleHTTPRequestHandler):
    """HTTP Request Handler serving viewer assets and binary splat stream."""

    model_bytes: bytes = b""
    model_format: str = "standard32"
    html_content: str = ""
    default_scene_id: str = "__current_model__"
    default_scene_name: str = "Loaded model"
    default_scene_gaussians: int = 0
    default_scene_size_mb: float = 0.0
    default_model_path: Optional[Path] = None
    supersplat_dist: Path = Path(__file__).parent / "supersplat_dist"
    jobs: dict = {}
    jobs_lock = threading.Lock()
    active_reconstruction: Optional[str] = None
    uploaded_plys: dict = {}
    studio_mode: bool = False
    output_dir: Path = Path("output").resolve()
    max_upload_bytes = 300 * 1024 * 1024
    max_ply_upload_bytes = 1024 * 1024 * 1024

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_supersplat(self, request_path: str, head_only: bool = False):
        root = self.supersplat_dist.resolve()
        relative = urllib.parse.unquote(request_path.removeprefix("/supersplat/")).lstrip("/") or "index.html"
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            self.send_error(404)
            return
        if not target.is_file():
            self.send_error(404)
            return

        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix == ".wasm":
            content_type = "application/wasm"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(target.stat().st_size))
        self.send_header("Cache-Control", "no-cache" if target.name == "index.html" else "public, max-age=3600")
        self.end_headers()
        if not head_only:
            with target.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    self.wfile.write(chunk)

    def _serve_job_ply(self, job_id: str, head_only: bool = False):
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            ply_path = Path(job["gaussian_ply"]) if job and job.get("status") == "complete" else None
        if not ply_path or not ply_path.is_file():
            self.send_error(404, "Generated Gaussian PLY is not ready")
            return

        size = ply_path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match or (not match.group(1) and not match.group(2)):
                self.send_error(416, "Invalid byte range")
                return
            if match.group(1):
                start = int(match.group(1))
                end = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
            else:
                start = max(0, size - int(match.group(2)))
            if start >= size or end < start:
                self.send_error(416, "Range not satisfiable")
                return

        length = end - start + 1
        self.send_response(206 if range_header else 200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not head_only:
            with ply_path.open("rb") as source:
                source.seek(start)
                remaining = length
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    def _serve_uploaded_ply(self, upload_id: str, head_only: bool = False):
        with self.jobs_lock:
            ply_path = self.uploaded_plys.get(upload_id)
        if not ply_path or not ply_path.is_file():
            self.send_error(404, "Uploaded PLY not found")
            return

        size = ply_path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match or (not match.group(1) and not match.group(2)):
                self.send_error(416, "Invalid byte range")
                return
            if match.group(1):
                start = int(match.group(1))
                end = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
            else:
                start = max(0, size - int(match.group(2)))
            if start >= size or end < start:
                self.send_error(416, "Range not satisfiable")
                return

        length = end - start + 1
        self.send_response(206 if range_header else 200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not head_only:
            with ply_path.open("rb") as source:
                source.seek(start)
                remaining = length
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)

    def do_HEAD(self):
        path = urllib.parse.urlsplit(self.path).path
        match = re.fullmatch(r"/api/reconstruction/([a-f0-9]{32})/gaussian\.ply", path)
        if match:
            self._serve_job_ply(match.group(1), head_only=True)
        elif re.fullmatch(r"/api/ply/[a-f0-9]{32}\.ply", path):
            self._serve_uploaded_ply(path.rsplit("/", 1)[-1][:-4], head_only=True)
        elif path.startswith("/supersplat/"):
            self._serve_supersplat(path, head_only=True)
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urllib.parse.urlsplit(self.path)
        is_ply_upload = parsed.path == "/api/open-ply"
        if parsed.path != "/api/reconstruct" and not is_ply_upload:
            self.send_error(404)
            return

        if is_ply_upload:
            filename = urllib.parse.parse_qs(parsed.query).get("filename", [""])[0]
            if Path(filename).suffix.lower() != ".ply":
                self._send_json(400, {"error": "Choose a Gaussian .ply file."})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 0
            if content_length <= 0 or content_length > self.max_ply_upload_bytes:
                self._send_json(413, {"error": "PLY upload must be between 1 byte and 1 GiB."})
                return

            upload_id = uuid.uuid4().hex
            upload_dir = self.output_dir / "panogs_uploads"
            ply_path = upload_dir / f"{upload_id}.ply"
            remaining = content_length
            try:
                upload_dir.mkdir(parents=True, exist_ok=True)
                with ply_path.open("wb") as destination:
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise OSError("Upload ended before all bytes arrived.")
                        destination.write(chunk)
                        remaining -= len(chunk)
                with ply_path.open("rb") as source:
                    if source.read(3) != b"ply":
                        raise ValueError("The selected file does not have a valid PLY header.")
            except (OSError, ValueError) as exc:
                ply_path.unlink(missing_ok=True)
                self._send_json(400, {"error": str(exc)})
                return

            with self.jobs_lock:
                self.uploaded_plys[upload_id] = ply_path
            editor_url = f"/supersplat/?content={urllib.parse.quote(f'/api/ply/{upload_id}.ply', safe='/')}"
            self._send_json(200, {"editor_url": editor_url})
            return

        filename = urllib.parse.parse_qs(parsed.query).get("filename", [""])[0]
        suffix = Path(filename).suffix.lower()
        if suffix not in {".hdr", ".exr"}:
            self._send_json(400, {"error": "Choose an .hdr or .exr panorama."})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > self.max_upload_bytes:
            self._send_json(413, {"error": "Upload must be between 1 byte and 300 MiB."})
            return

        with self.jobs_lock:
            if self.active_reconstruction is not None:
                self._send_json(409, {"error": "A panorama reconstruction is already running."})
                return
            job_id = uuid.uuid4().hex
            self.active_reconstruction = job_id
            self.jobs[job_id] = {
                "id": job_id, "status": "queued", "message": "Upload received; preparing reconstruction.",
                "created_at": time.time(), "source_name": Path(filename).name, "gaussian_ply": None, "log": "",
            }

        upload_dir = self.output_dir / "panogs_uploads"
        source_path = upload_dir / f"{job_id}{suffix}"
        remaining = content_length
        try:
            upload_dir.mkdir(parents=True, exist_ok=True)
            with source_path.open("wb") as destination:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise OSError("Upload ended before all bytes arrived.")
                    destination.write(chunk)
                    remaining -= len(chunk)
        except OSError as exc:
            with self.jobs_lock:
                self.jobs[job_id].update(status="failed", message=f"Could not save upload: {exc}")
                self.active_reconstruction = None
            self._send_json(500, {"error": "Could not save the uploaded panorama."})
            return

        output_path = self.output_dir / f"panorama_{job_id}_reconstruction.ply"
        threading.Thread(
            target=self._run_reconstruction,
            args=(job_id, source_path, output_path),
            daemon=True,
            name=f"panogs-reconstruction-{job_id[:8]}",
        ).start()
        self._send_json(202, {"job_id": job_id, "status_url": f"/api/reconstruction/{job_id}"})

    @classmethod
    def _run_reconstruction(cls, job_id: str, source_path: Path, output_path: Path):
        command = [
            sys.executable, "-m", "panogs.apps.cli", "reconstruct", str(source_path),
            "--output", str(output_path), "--model", "cubemap_depth_anything",
            "--shape", "hybrid", "--sharpness", "0.75", "--scale-factor", "0.8", "--opacity", "0.85",
        ]
        with cls.jobs_lock:
            cls.jobs[job_id].update(status="running", message="Estimating depth and building the Gaussian PLY.")
        try:
            process = subprocess.Popen(
                command, cwd=Path.cwd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            log_tail = ""
            assert process.stdout is not None
            for line in process.stdout:
                log_tail = (log_tail + line)[-6000:]
                with cls.jobs_lock:
                    cls.jobs[job_id]["log"] = log_tail
            return_code = process.wait()
            gaussian_ply = output_path.with_name(f"{output_path.stem}_gaussians.ply")
            if return_code == 0 and gaussian_ply.is_file():
                with cls.jobs_lock:
                    cls.jobs[job_id].update(
                        status="complete",
                        message="Reconstruction complete. The Gaussian PLY is ready to open in SuperSplat.",
                        gaussian_ply=str(gaussian_ply),
                    )
            else:
                with cls.jobs_lock:
                    cls.jobs[job_id].update(status="failed", message="Reconstruction failed; see the CLI log for details.")
        except Exception as exc:
            with cls.jobs_lock:
                cls.jobs[job_id].update(status="failed", message=f"Could not start reconstruction: {exc}")
        finally:
            with cls.jobs_lock:
                if cls.active_reconstruction == job_id:
                    cls.active_reconstruction = None

    def do_GET(self):
        logger = get_logger("viewer.server")
        request_path = urllib.parse.urlsplit(self.path).path

        if request_path == "/supersplat":
            self.send_response(301)
            self.send_header("Location", "/supersplat/")
            self.end_headers()
            return
        if request_path.startswith("/supersplat/"):
            self._serve_supersplat(request_path)
            return
        path_match = re.fullmatch(r"/api/reconstruction/([a-f0-9]{32})/gaussian\.ply", request_path)
        if path_match:
            self._serve_job_ply(path_match.group(1))
            return
        ply_match = re.fullmatch(r"/api/ply/([a-f0-9]{32})\.ply", request_path)
        if ply_match:
            self._serve_uploaded_ply(ply_match.group(1))
            return
        job_match = re.fullmatch(r"/api/reconstruction/([a-f0-9]{32})", request_path)
        if job_match:
            with self.jobs_lock:
                job = self.jobs.get(job_match.group(1))
                payload = dict(job) if job else None
            if payload is None:
                self._send_json(404, {"error": "Reconstruction job not found."})
            else:
                payload.pop("gaussian_ply", None)
                self._send_json(200, payload)
            return

        if request_path == "/viewer":
            self.send_response(302)
            self.send_header("Location", "/viewer/")
            self.end_headers()
            return
        if request_path == "/viewer/":
            html_path = Path(__file__).parent / "index.html"
            content = html_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(content)
            return

        if request_path == "/" or request_path.startswith("/index.html"):
            html_path = Path(__file__).parent / ("studio.html" if self.studio_mode else "index.html")
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

        elif request_path == "/api/config":
            self._send_json(200, {
                "default_scene_id": SplatViewerHandler.default_scene_id,
                "has_model": SplatViewerHandler.default_model_path is not None,
            })

        elif request_path == "/api/scenes" or request_path.startswith("/api/scenes"):
            output_dir = SplatViewerHandler.output_dir
            scenes = [{
                "id": SplatViewerHandler.default_scene_id,
                "name": SplatViewerHandler.default_scene_name,
                "branch": "Loaded file",
                "badge": "Current model",
                "desc": "The file passed to the viewer when it was launched",
                "gaussians": SplatViewerHandler.default_scene_gaussians,
                "size_mb": SplatViewerHandler.default_scene_size_mb,
                "url": f"/?scene={SplatViewerHandler.default_scene_id}",
                "api_url": f"/api/scene.splat?scene={SplatViewerHandler.default_scene_id}",
            }] if SplatViewerHandler.default_model_path is not None else []

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
                    hdr_candidate = output_dir / f"{Path(alias).stem}.hdrsplat"
                    if hdr_candidate.exists():
                        target_file = hdr_candidate
                        break
                    p = output_dir / alias
                    if p.exists():
                        target_file = p
                        break
                if target_file:
                    bytes_per_splat = 44 if target_file.suffix.lower() == ".hdrsplat" else 32
                    splat_cnt = target_file.stat().st_size // bytes_per_splat
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
                    seen_ids.add(target_file.name)

            # Also discover any other standard or HDR viewer splat in output
            if output_dir.exists():
                hdr_stems = {p.stem for p in output_dir.glob("*.hdrsplat")}
                local_splats = [*output_dir.glob("*.splat"), *output_dir.glob("*.hdrsplat")]
                for p in sorted(local_splats):
                    bytes_per_splat = 44 if p.suffix.lower() == ".hdrsplat" else 32
                    if p.suffix.lower() == ".splat" and p.stem in hdr_stems:
                        continue
                    if p.name not in seen_ids and not p.name.endswith((".tmp.splat", ".tmp.hdrsplat")):
                        splat_cnt = p.stat().st_size // bytes_per_splat
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

            self._send_json(200, {"scenes": scenes})

        elif request_path == "/api/scene.splat":
            parsed = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed.query)

            data = self.model_bytes
            scene_format = self.model_format
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
                            data, scene_format = prepare_viewer_payload(cand)
                            break
                        except Exception as e:
                            logger.error(f"Failed to load dynamic scene {cand}: {e}")

            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-PanoGS-Format", scene_format)
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


def prepare_viewer_payload(model_path: Union[str, Path]) -> tuple[bytes, str]:
    """Prepare bytes and format tag for the viewer, retaining float PLY radiance."""
    path = Path(model_path)
    if path.suffix.lower() == ".hdrsplat":
        return path.read_bytes(), "hdr44"
    if path.suffix.lower() == ".splat":
        return path.read_bytes(), "standard32"
    if path.suffix.lower() != ".ply":
        raise ValueError(f"Unsupported 3DGS format: {path.suffix} (expected .ply, .splat, or .hdrsplat)")

    model = load_gaussian_ply(path)
    with tempfile.TemporaryDirectory(prefix="panogs_hdr_") as temp_dir:
        temp_path = Path(temp_dir) / "scene.hdrsplat"
        save_gaussian_hdr_splat(temp_path, model)
        return temp_path.read_bytes(), "hdr44"


def start_viewer_server(
    model_path: Optional[Union[str, Path]] = None,
    port: int = 8080,
    open_browser: bool = True,
    block: bool = True,
    host: str = "127.0.0.1",
) -> ThreadingHTTPServer:
    """
    Start the interactive WebGL 3DGS viewer HTTP server.

    Args:
        model_path: Path to .splat, .hdrsplat, or Gaussian .ply file.
        port: Port to serve on (default: 8080).
        host: Interface to bind to (default: 127.0.0.1; use 0.0.0.0 for hosted apps).
        open_browser: Whether to open default web browser automatically.
        block: Whether to block calling thread with serve_forever().

    Returns:
        ThreadingHTTPServer instance.
    """
    logger = get_logger("viewer.server")
    html_path = Path(__file__).parent / "index.html"

    with open(html_path, "r", encoding="utf-8") as f:
        html_str = f.read()

    splat_data, model_format = prepare_viewer_payload(model_path) if model_path else (b"", "standard32")
    bytes_per_gaussian = 44 if model_format == "hdr44" else 32
    num_gaussians = len(splat_data) // bytes_per_gaussian

    SplatViewerHandler.html_content = html_str
    SplatViewerHandler.model_bytes = splat_data
    SplatViewerHandler.model_format = model_format
    SplatViewerHandler.default_scene_id = "__current_model__"
    SplatViewerHandler.default_model_path = Path(model_path).resolve() if model_path else None
    SplatViewerHandler.default_scene_name = Path(model_path).stem.replace("_", " ").title() if model_path else "Upload a panorama"
    SplatViewerHandler.default_scene_gaussians = num_gaussians
    SplatViewerHandler.default_scene_size_mb = round(len(splat_data) / (1024 * 1024), 1)
    output_dir = Path(os.environ.get("PANOGS_OUTPUT_DIR", str(Path.cwd() / "output"))).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    SplatViewerHandler.output_dir = output_dir.resolve()
    SplatViewerHandler.jobs = {}
    SplatViewerHandler.active_reconstruction = None
    SplatViewerHandler.uploaded_plys = {}
    SplatViewerHandler.studio_mode = model_path is None

    if not SplatViewerHandler.supersplat_dist.joinpath("index.html").is_file():
        raise FileNotFoundError("Bundled SuperSplat build is missing from panogs/apps/viewer/supersplat_dist")

    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, SplatViewerHandler)

    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{display_host}:{port}"
    page_mode = "lightweight reconstruction menu" if SplatViewerHandler.studio_mode else f"{num_gaussians:,} Gaussians loaded"
    logger.info(f"Serving PanoGS at {url} ({page_mode})")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    if block:
        try:
            print(f"\n=======================================================")
            print(f"  PanoGS Studio · {page_mode}")
            print(f"  Running at:  {url}")
            if not SplatViewerHandler.studio_mode:
                print(f"  Gaussians:   {num_gaussians:,}")
            print(f"  Press Ctrl+C in terminal to stop.")
            print(f"=======================================================\n")
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Viewer server stopped.")
            httpd.server_close()

    return httpd
