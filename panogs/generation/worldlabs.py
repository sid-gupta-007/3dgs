"""World Labs Marble panorama-to-world generation client.

The provider receives an LDR image. HDR inputs are preserved locally and a
separate, gently tone-mapped preview is uploaded; generated output is an
inferred scene, not a radiometrically faithful replacement for the HDR source.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image

API_BASE = "https://api.worldlabs.ai/marble/v1"


def _request_json(
    method: str,
    url: str,
    api_key: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"WLT-Api-Key": api_key, "Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"World Labs API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach World Labs API: {exc.reason}") from exc
    if not body:
        return {}
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("World Labs API returned an invalid JSON response") from exc


def _make_hdr_preview(source: Path, output_dir: Path, max_dimension: int) -> Path:
    import numpy as np

    from panogs.io.images import linear_to_srgb, load_hdr_radiance

    radiance = load_hdr_radiance(source, max_resolution=max_dimension)
    # Reinhard mapping is confined to this disposable upload preview. The
    # original HDR file remains untouched and is copied alongside the result.
    display = linear_to_srgb(radiance / (1.0 + radiance))
    pixels = np.clip(np.round(display * 255.0), 0, 255).astype(np.uint8)
    preview = output_dir / f"{source.stem}_ai_preview.jpg"
    Image.fromarray(pixels, mode="RGB").save(preview, quality=95, subsampling=0)
    return preview


def _prepare_input(source: Path, output_dir: Path, max_dimension: int) -> Tuple[Path, bool]:
    suffix = source.suffix.lower()
    if suffix in {".hdr", ".exr"}:
        return _make_hdr_preview(source, output_dir, max_dimension), True
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError("Input must be JPG, JPEG, PNG, WebP, HDR, or EXR")
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        preview = output_dir / f"{source.stem}_ai_preview.jpg"
        image.save(preview, quality=95, subsampling=0)
    return preview, False


def _upload_image(preview: Path, api_key: str) -> str:
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}[preview.suffix.lower()]
    prepared = _request_json(
        "POST",
        f"{API_BASE}/media-assets:prepare_upload",
        api_key,
        {"file_name": preview.name, "kind": "image", "extension": preview.suffix.lstrip(".").lower()},
    )
    # Normalize provider field name differences while retaining strict checks.
    info = prepared.get("media_asset", prepared)
    asset_id = info.get("media_asset_id") or info.get("id")
    upload = prepared.get("upload_info", {})
    upload_url = upload.get("upload_url") or prepared.get("upload_url")
    if not asset_id or not upload_url:
        raise RuntimeError(f"Unexpected prepare-upload response: {json.dumps(prepared)[:1200]}")
    required = upload.get("required_headers", {})
    headers = {str(k): str(v) for k, v in required.items()}
    headers.setdefault("Content-Type", mime)
    request = urllib.request.Request(upload_url, data=preview.read_bytes(), headers=headers, method="PUT")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            if response.status not in (200, 201, 204):
                raise RuntimeError(f"Image upload returned HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Image upload returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not upload panorama: {exc.reason}") from exc
    return str(asset_id)


def generate_world(
    panorama_path: str | Path,
    output_dir: str | Path,
    *,
    prompt: Optional[str] = None,
    display_name: Optional[str] = None,
    model: str = "marble-1.1",
    resolution: str = "500k",
    max_dimension: int = 2048,
    poll_interval: float = 5.0,
    timeout: float = 1800.0,
) -> Dict[str, Any]:
    """Generate a navigable world and download its Gaussian PLY export."""
    api_key = os.environ.get("WLT_API_KEY") or os.environ.get("WORLDLABS_API_KEY")
    if not api_key:
        raise RuntimeError("Set WLT_API_KEY (or WORLDLABS_API_KEY) to your World Labs API key")
    source = Path(panorama_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Panorama not found: {source}")
    if max_dimension < 256:
        raise ValueError("--max-dimension must be at least 256")
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    preview, is_hdr = _prepare_input(source, out, max_dimension)
    local_source = out / f"original{source.suffix.lower()}"
    if source != local_source:
        shutil.copy2(source, local_source)

    print(f"Preparing panorama preview: {preview}")
    asset_id = _upload_image(preview, api_key)
    world_prompt: Dict[str, Any] = {
        "type": "image",
        "image_prompt": {"source": "media_asset", "media_asset_id": asset_id},
        "is_pano": True,
    }
    if prompt:
        world_prompt["text_prompt"] = prompt
    generation = _request_json(
        "POST",
        f"{API_BASE}/worlds:generate",
        api_key,
        {
            "display_name": display_name or source.stem,
            "model": model,
            "world_prompt": world_prompt,
        },
    )
    operation_id = generation.get("operation_id")
    if not operation_id:
        raise RuntimeError(f"Generation response did not contain operation_id: {json.dumps(generation)[:1200]}")

    deadline = time.monotonic() + timeout
    while True:
        operation = _request_json("GET", f"{API_BASE}/operations/{urllib.parse.quote(str(operation_id))}", api_key)
        metadata = operation.get("metadata", {})
        progress = metadata.get("progress", {})
        description = progress.get("description") or progress.get("status") or operation.get("status") or "working"
        print(f"World generation: {description}")
        if operation.get("error"):
            raise RuntimeError(f"World generation failed: {json.dumps(operation['error'])[:2000]}")
        if operation.get("done"):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"World generation did not finish within {timeout:.0f} seconds (operation {operation_id})")
        time.sleep(max(1.0, poll_interval))

    world = operation.get("response") or {}
    world_id = world.get("id") or world.get("world_id") or metadata.get("world_id")
    if not world_id:
        raise RuntimeError(f"Completed generation did not contain a world id: {json.dumps(operation)[:1600]}")
    export = _request_json(
        "POST",
        f"{API_BASE}/worlds/{urllib.parse.quote(str(world_id))}:export",
        api_key,
        {"asset_type": "splats", "format": "ply", "resolution": resolution},
    )
    export_operation_id = export.get("operation_id")
    if export_operation_id:
        export_deadline = time.monotonic() + timeout
        while True:
            export = _request_json(
                "GET", f"{API_BASE}/operations/{urllib.parse.quote(str(export_operation_id))}", api_key
            )
            if export.get("error"):
                raise RuntimeError(f"PLY export failed: {json.dumps(export['error'])[:2000]}")
            if export.get("done"):
                export = export.get("response") or {}
                break
            if time.monotonic() >= export_deadline:
                raise TimeoutError(f"PLY export did not finish within {timeout:.0f} seconds")
            time.sleep(max(1.0, poll_interval))
    download_url = export.get("url") or (export.get("asset") or {}).get("url")
    if not download_url:
        raise RuntimeError(f"PLY export response did not contain a download URL: {json.dumps(export)[:1600]}")
    model_path = out / "ai_world.ply"
    try:
        with urllib.request.urlopen(download_url, timeout=300) as response, model_path.open("wb") as target:
            shutil.copyfileobj(response, target)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"Could not download generated PLY: {exc}") from exc

    manifest = {
        "provider": "World Labs Marble API",
        "model": model,
        "world_id": world_id,
        "world_url": world.get("world_marble_url") or world.get("url"),
        "source_file": str(source),
        "source_hdr": is_hdr,
        "preserved_source_copy": str(local_source),
        "uploaded_preview": str(preview),
        "generated_ply": str(model_path),
        "export_resolution": resolution,
        "note": "Generated scene geometry and occluded regions are AI-inferred; this is not an exact reconstruction or HDR radiance replacement.",
    }
    manifest_path = out / "world_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
