# AI-assisted panorama to world

PanoGS can hand a panorama to the World Labs Marble API, download its generated Gaussian PLY, and open that PLY in the existing local viewer. The API call uploads the image and uses provider credits. The generated scene is an AI interpretation: geometry behind objects and outside the panorama's observed viewpoint is inferred and may not match the real location.

## Run it

Create an API key and ensure the World Labs account has credits. In PowerShell, set the key for the current terminal session:

```powershell
$env:WLT_API_KEY = "your-api-key"
```

Then run from the PanoGS repository (activate its virtual environment first):

```powershell
panogs ai-world "path\to\panorama.hdr" --prompt "A realistic cafe interior; preserve the visible room layout" --open-viewer
```

The command also accepts JPG, JPEG, PNG, and WebP. Use `panogs ai-world --help` for model, output-size, and preview controls. The default model is `marble-1.1`; the default Gaussian export is `500k` for a practical initial viewer load.

## HDR behavior

The provider accepts display images, not HDR radiance. For `.hdr` and `.exr`, PanoGS makes a separate 2048-pixel-long-edge Reinhard-mapped JPEG preview for upload. It does not change the original HDR file or treat generated colors as HDR radiance. The original is copied into the output directory, and `world_manifest.json` records the source, preview, model, world ID, and generated PLY paths. Do not use the AI-generated PLY as a replacement for the original HDR capture when exact color or radiance matters.

Output files are written to `output/ai_world` by default. If you omit `--open-viewer`, view the result later with:

```powershell
panogs view output/ai_world/ai_world.ply
```

