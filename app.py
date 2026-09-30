"""Gradio Space app for the PanoGS panorama-to-Gaussian pipeline."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Iterator

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parent
VIEWER_DIR = ROOT / "panogs" / "apps" / "viewer"
OUTPUT_DIR = Path(os.environ.get("PANOGS_OUTPUT_DIR", str(ROOT / "output"))).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS: dict[str, Path] = {}
RESULTS_LOCK = threading.Lock()


def editor_link(job_id: str) -> str:
    content_url = f"/api/generated/{job_id}.ply"
    return (
        '<a class="result-link" target="_blank" rel="noopener" '
        f'href="/supersplat/?content={content_url}">Open result in SuperSplat ↗</a>'
        f'<a class="result-link secondary" href="{content_url}" download>Download Gaussian PLY</a>'
    )


def reconstruct(panorama_path: str | None) -> Iterator[tuple[str, str, str]]:
    if not panorama_path:
        yield "Choose an HDR panorama first.", "", ""
        return

    source = Path(panorama_path)
    if source.suffix.lower() not in {".hdr", ".exr"}:
        yield "Please upload an .hdr or .exr panorama.", "", ""
        return

    job_id = uuid.uuid4().hex
    pointcloud_path = OUTPUT_DIR / f"panorama_{job_id}_reconstruction.ply"
    gaussian_path = pointcloud_path.with_name(f"{pointcloud_path.stem}_gaussians.ply")
    command = [
        sys.executable, "-m", "panogs.apps.cli", "reconstruct", str(source),
        "--output", str(pointcloud_path), "--model", "cubemap_depth_anything",
        "--shape", "hybrid", "--sharpness", "0.75", "--scale-factor", "0.8", "--opacity", "0.85",
    ]

    yield "Loading the depth model and starting reconstruction…", "", ""
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        log_tail = ""
        assert process.stdout is not None
        for line in process.stdout:
            log_tail = (log_tail + line)[-6000:]
            yield "PanoGS is estimating depth and building the Gaussian scene…", log_tail, ""
        return_code = process.wait()
        if return_code != 0 or not gaussian_path.is_file():
            yield "Reconstruction failed. See the run log below.", log_tail, ""
            return

        with RESULTS_LOCK:
            RESULTS[job_id] = gaussian_path
        yield "Reconstruction complete.", log_tail, editor_link(job_id)
    except Exception as exc:
        yield f"Could not run reconstruction: {exc}", "", ""


def open_existing_ply(ply_path: str | None) -> tuple[str, str]:
    if not ply_path:
        return "Choose a Gaussian PLY file first.", ""
    source = Path(ply_path)
    if source.suffix.lower() != ".ply" or not source.is_file():
        return "Please upload a valid .ply file.", ""
    job_id = uuid.uuid4().hex
    stored_path = OUTPUT_DIR / f"uploaded_{job_id}.ply"
    shutil.copyfile(source, stored_path)
    with RESULTS_LOCK:
        RESULTS[job_id] = stored_path
    return "PLY ready to open.", editor_link(job_id)


with gr.Blocks(title="PanoGS Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🌐 PanoGS Studio\n"
        "Upload a 360° HDR panorama to build a Gaussian scene, then open it in the bundled SuperSplat editor."
    )
    with gr.Tab("Create a scene"):
        panorama = gr.File(label="HDR panorama", file_types=[".hdr", ".exr"], type="filepath")
        run_button = gr.Button("Reconstruct panorama", variant="primary")
        status = gr.Markdown("Ready when you are.")
        editor_actions = gr.HTML()
        logs = gr.Textbox(label="Pipeline log", lines=8, max_lines=16, interactive=False)
        run_button.click(
            reconstruct,
            inputs=panorama,
            outputs=[status, logs, editor_actions],
            concurrency_limit=1,
        )
    with gr.Tab("Open a Gaussian PLY"):
        ply_file = gr.File(label="Gaussian PLY", file_types=[".ply"], type="filepath")
        open_button = gr.Button("Open in SuperSplat")
        ply_status = gr.Markdown("Choose an existing Gaussian PLY to edit it.")
        ply_actions = gr.HTML()
        open_button.click(open_existing_ply, inputs=ply_file, outputs=[ply_status, ply_actions])
    gr.Markdown(
        "The depth model downloads on first use. Reconstruction runs on this Space; "
        "the editor renders in your browser and requires WebGPU support."
    )


app = FastAPI()


@app.get("/api/generated/{filename}")
def get_generated_file(filename: str) -> FileResponse:
    match = re.fullmatch(r"([a-f0-9]{32})\.ply", filename)
    if not match:
        raise HTTPException(status_code=404, detail="Scene not found")
    with RESULTS_LOCK:
        path = RESULTS.get(match.group(1))
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Scene not found or expired")
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


app.mount("/supersplat", StaticFiles(directory=VIEWER_DIR / "supersplat_dist", html=True), name="supersplat")
app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
