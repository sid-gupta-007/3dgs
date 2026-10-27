"""
Standard 3D Gaussian Splatting PLY and SPLAT serialization module.
Outputs format compatible with SuperSplat, SIBR, PlayCanvas, and Nerfstudio.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np

from panogs.core.gaussian.model import GaussianModel, SH_C0, logit, rgb_to_sh0, sh0_to_rgb, sigmoid
from panogs.core.logging import get_logger
from panogs.io.images import srgb_to_linear, tonemap_hdr_to_srgb


def save_gaussian_ply(
    file_path: Union[str, Path],
    model: GaussianModel,
    *,
    supersplat_compat: bool = False,
) -> Path:
    """
    Save a GaussianModel to standard 3DGS binary PLY format.

    Properties exported:
        - x, y, z: Position
        - nx, ny, nz: Normals (default 0)
        - f_dc_0, f_dc_1, f_dc_2: Zeroth-order spherical harmonics (SH0)
        - opacity: Logit opacity
        - scale_0, scale_1, scale_2: Log scale
        - rot_0, rot_1, rot_2, rot_3: Quaternion (qw, qx, qy, qz)

    Args:
        file_path: Output file path.
        model: GaussianModel instance.
        supersplat_compat: If True, writes display-referred sRGB SH0 colors (with
            HDR tone mapping) and applies the standard 3DGS/COLMAP Z-180 coordinate
            convention so SuperSplat, PlayCanvas, and SIBR load the scene upright
            and properly exposed.

    Returns:
        Path: Output file path.
    """
    logger = get_logger("io.gaussian_ply")
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    N = model.num_gaussians
    xyz = model.get_xyz().copy()
    normals = np.zeros((N, 3), dtype=np.float32)
    opacity = model._opacity_logits
    scales = model._scaling_log
    rotations = model.get_rotation_quats().copy()

    if supersplat_compat:
        # SuperSplat applies Transform.PLY = fromEulers(0, 0, 180) on load
        # ((x, y, z) -> (-x, -y, z)) and interprets f_dc directly as display sRGB.
        # Negating Y only (reflection D = diag(1, -1, 1)) makes SuperSplat's Z-180
        # rotation produce (-x, +y, z): upright (+Y up) AND true un-mirrored
        # right-handed orientation matching the source panorama.
        xyz[:, 1] = -xyz[:, 1]
        qw, qx, qy, qz = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
        rotations = np.stack([qw, -qx, qy, -qz], axis=-1).astype(np.float32)
        srgb = tonemap_hdr_to_srgb(model.get_rgb())
        f_dc = rgb_to_sh0(srgb).astype(np.float32)
        comment_line = "comment PanoGS 3D Gaussian Splatting Exporter; coord=colmap_z180_rh; color_space=srgb-display; geometry=watertight-v4"
    else:
        f_dc = model.get_features_dc()
        comment_line = "comment PanoGS 3D Gaussian Splatting Exporter; color_space=linear-radiance"

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        comment_line,
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header\n",
    ]
    header = "\n".join(header_lines).encode("ascii")

    # Structured array definition (all float32)
    dtype_fields = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ]
    data = np.empty(N, dtype=dtype_fields)

    data["x"] = xyz[:, 0]
    data["y"] = xyz[:, 1]
    data["z"] = xyz[:, 2]
    data["nx"] = normals[:, 0]
    data["ny"] = normals[:, 1]
    data["nz"] = normals[:, 2]
    data["f_dc_0"] = f_dc[:, 0]
    data["f_dc_1"] = f_dc[:, 1]
    data["f_dc_2"] = f_dc[:, 2]
    data["opacity"] = opacity[:, 0]
    data["scale_0"] = scales[:, 0]
    data["scale_1"] = scales[:, 1]
    data["scale_2"] = scales[:, 2]
    data["rot_0"] = rotations[:, 0]
    data["rot_1"] = rotations[:, 1]
    data["rot_2"] = rotations[:, 2]
    data["rot_3"] = rotations[:, 3]

    with open(target, "wb") as f:
        f.write(header)
        data.tofile(f)

    logger.info(f"Saved {N:,} 3D Gaussians to {target} ({target.stat().st_size / (1024*1024):.2f} MB)")
    return target


def load_gaussian_ply(file_path: Union[str, Path]) -> GaussianModel:
    """
    Load a 3D Gaussian Splatting scene model from a standard 3DGS binary PLY file.
    """
    target = Path(file_path)
    if not target.exists():
        raise FileNotFoundError(f"Gaussian PLY not found: {target}")

    with open(target, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        num_vertices = 0
        has_linear_radiance = False
        has_colmap_z180 = False
        has_colmap_z180_rh = False
        for line in header_lines:
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[2])
            if "color_space=linear-radiance" in line:
                has_linear_radiance = True
            if "coord=colmap_z180_rh" in line:
                has_colmap_z180_rh = True
            elif "coord=colmap_z180" in line:
                has_colmap_z180 = True

        if num_vertices == 0:
            raise ValueError(f"Empty PLY file: {target}")

        dtype_fields = [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
            ("opacity", "<f4"),
            ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
            ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
        ]

        data = np.fromfile(f, dtype=dtype_fields, count=num_vertices)

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=-1)
    f_dc = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=-1)
    if not has_linear_radiance:
        # Older PanoGS, SuperSplat-compat, and common external PLYs store display-encoded sRGB in SH0.
        srgb = np.clip(f_dc * SH_C0 + 0.5, 0.0, 1.0)
        linear = srgb_to_linear(srgb)
        f_dc = rgb_to_sh0(linear)
    opacity = data["opacity"][:, np.newaxis]
    scales = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=-1)
    rotations = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=-1)

    if has_colmap_z180_rh:
        xyz[:, 1] = -xyz[:, 1]
        r0, r1, r2, r3 = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
        rotations = np.stack([r0, -r1, r2, -r3], axis=-1).astype(np.float32)
    elif has_colmap_z180:
        xyz[:, 0] = -xyz[:, 0]
        xyz[:, 1] = -xyz[:, 1]
        r0, r1, r2, r3 = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
        rotations = np.stack([r3, r2, -r1, -r0], axis=-1).astype(np.float32)

    return GaussianModel(
        xyz=xyz,
        scaling_log=scales,
        rotation_quats=rotations,
        opacity_logits=opacity,
        features_dc=f_dc,
    )


def is_supersplat_compatible_ply(file_path: Union[str, Path]) -> bool:
    """Return True if a PLY file is a watertight 3DGS PLY ready for SuperSplat."""
    src = Path(file_path)
    if not src.is_file():
        return False
    header_lines = []
    with src.open("rb") as f:
        for _ in range(128):
            raw = f.readline()
            if not raw:
                break
            line = raw.decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break
    header_text = "\n".join(header_lines)
    if "coord=colmap_z180_rh" in header_text and "geometry=watertight-v4" in header_text:
        return True
    has_f_dc = "property float f_dc_0" in header_text
    is_panogs = "comment PanoGS" in header_text
    return has_f_dc and not is_panogs


def repair_panoramic_gaussians(model: GaussianModel) -> GaussianModel:
    """Repair panoramic 3D Gaussian models to eliminate floating ribbons, nadir floor holes, and occlusion voids.

    Applies to existing or newly loaded panoramic Gaussian models (N >= 2048):
      1. Regularizes ceiling light depth dips and seals narrow lattice partition/window panes.
      2. Strips mid-air floating transition ribbons along steep depth boundaries.
      3. Restores any previously pruned holes (e.g. inside grid partitions or nadir poles)
         and synthesizes missing inpainted background wall/floor splats behind foreground occluders.
      4. Boosts slanted-surface and polar splat scales so surfaces are 100% watertight in SuperSplat.
    """
    N = model.num_gaussians
    if N < 2048:
        return model

    import cv2
    from scipy.ndimage import distance_transform_edt
    from panogs.core.camera.spherical import equirectangular_rays, spherical_to_ray
    from panogs.core.gaussian.initialization import rotation_matrix_to_quaternion
    from panogs.reconstruction.inpainting import (
        inpaint_background_depth,
        inpaint_background_texture,
        segment_foreground_objects,
    )
    from panogs.reconstruction.panorama.layout import detect_depth_edges, regularize_panoramic_depth

    xyz = model.get_xyz().copy()
    scales_log = model._scaling_log.copy()
    scales = np.exp(scales_log).astype(np.float32)
    quats = model.get_rotation_quats().copy()
    opacity_logits = model._opacity_logits.copy()
    f_dc = model.get_features_dc().copy()
    rgb_linear = model.get_rgb()
    rgb_u8 = model.get_rgb_uint8()

    r = np.maximum(np.linalg.norm(xyz, axis=-1), 1e-4).astype(np.float32)
    rays = xyz / r[:, np.newaxis]
    dx, dy, dz = rays[:, 0], rays[:, 1], rays[:, 2]

    # Snap H_g to standard equirectangular power-of-two / common heights when close
    raw_h = int(round(np.sqrt(N / 2.0)))
    for std_h in (256, 512, 768, 1024):
        if abs(raw_h - std_h) <= std_h * 0.06:
            raw_h = std_h
            break
    H_g = int(np.clip(raw_h, 64, 1024))
    W_g = 2 * H_g

    theta = np.arctan2(dx, dz)
    phi = np.arcsin(np.clip(dy, -1.0, 1.0))
    u = np.clip(((theta / (2.0 * np.pi) + 0.5) * W_g).astype(np.int32), 0, W_g - 1)
    v = np.clip(((0.5 - phi / np.pi) * H_g).astype(np.int32), 0, H_g - 1)

    front_idx = np.full((H_g, W_g), -1, dtype=np.int32)
    back_r = np.zeros((H_g, W_g), dtype=np.float32)
    order_desc = np.argsort(r)[::-1]
    order_asc = order_desc[::-1]
    front_idx[v[order_desc], u[order_desc]] = order_desc
    back_r[v[order_asc], u[order_asc]] = r[order_asc]

    valid_cells = front_idx >= 0
    if not np.any(valid_cells):
        return model
    missing_cells = ~valid_cells
    if np.any(missing_cells):
        _, nearest = distance_transform_edt(missing_cells, return_indices=True)
        filled_front_idx = front_idx[nearest[0], nearest[1]]
        back_r = np.where(valid_cells, back_r, r[filled_front_idx])
    else:
        filled_front_idx = front_idx

    d_grid_raw = r[filled_front_idx].astype(np.float32)
    rgb_grid = rgb_u8[filled_front_idx].astype(np.uint8)
    grid_rays = equirectangular_rays(H_g, W_g)

    # 1. Regularize ceiling light dips, narrow grid partition panes, and edge ramps
    d_grid = regularize_panoramic_depth(d_grid_raw, grid_rays)

    # Pull front-surface Gaussians whose depth spiked inside narrow lattice panes or ceiling lights
    # onto the regularized surface so grid partitions and ceiling fixtures become solid and flat
    d_reg_pt = d_grid[v, u]
    d_raw_pt = d_grid_raw[v, u]
    was_regularized = (np.abs(r - d_raw_pt) < 0.15) & (np.abs(d_reg_pt - d_raw_pt) > 0.25)
    if np.any(was_regularized):
        r[was_regularized] = d_reg_pt[was_regularized]
        xyz[was_regularized] = rays[was_regularized] * r[was_regularized, np.newaxis]

    # 2. Snap remaining mid-air floating ribbon splats along steep depth jumps onto the regularized surface
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    d_min = cv2.erode(d_grid, kernel)
    d_max = cv2.dilate(d_grid, kernel)
    span = d_max - d_min
    is_steep = ((span / np.maximum(d_min, 0.05)) > 0.14) & (span > 0.22)
    flying_mask = detect_depth_edges(d_grid, rel_threshold=0.08, abs_threshold=0.18)

    d_lo_pt = d_min[v, u]
    d_hi_pt = d_max[v, u]
    span_pt = d_hi_pt - d_lo_pt
    steep_pt = is_steep[v, u] & (np.abs(dy) <= 0.92)
    in_mid_ramp = steep_pt & (r > d_lo_pt + 0.12 * span_pt) & (r < d_hi_pt - 0.12 * span_pt)
    if np.any(in_mid_ramp):
        # Snap mid-ramp splats to background depth or regularized grid depth so zero empty holes are created
        snap_r = np.where((r - d_lo_pt) < 0.45 * span_pt, d_lo_pt, d_hi_pt)
        r[in_mid_ramp] = snap_r[in_mid_ramp]
        xyz[in_mid_ramp] = rays[in_mid_ramp] * r[in_mid_ramp, np.newaxis]

    v_k = v
    u_k = u

    # 3. Preserve compact detail scales on walls/furniture while bounding outliers and polar rings
    angular_pixel = np.pi / float(H_g)
    s_lat = r * angular_pixel
    scales[:, 0] = np.minimum(scales[:, 0], s_lat * 1.40)
    scales[:, 1] = np.minimum(scales[:, 1], s_lat * 1.40)
    pole_pts = np.abs(dy) > 0.85
    if np.any(pole_pts):
        scales[pole_pts, 0] = np.maximum(scales[pole_pts, 0], s_lat[pole_pts] * 1.05)
        scales[pole_pts, 1] = np.maximum(scales[pole_pts, 1], s_lat[pole_pts] * 1.05)
    scales[:, 2] = np.minimum(scales[:, 2], s_lat * 0.35)

    min_logit = float(logit(np.array([0.92]))[0])
    opacity_logits = np.maximum(opacity_logits, min_logit)

    extra_xyz = []
    extra_scales = []
    extra_quats = []
    extra_opacities = []
    extra_fdc = []

    def _make_camera_facing_splats(pts_arr: np.ndarray, rays_arr: np.ndarray, s_arr: np.ndarray, fdc_arr: np.ndarray, opac_val: float = 0.96):
        normals_arr = -rays_arr
        nx_a, ny_a, nz_a = normals_arr[:, 0], normals_arr[:, 1], normals_arr[:, 2]
        sign_a = np.where(nz_a >= 0.0, 1.0, -1.0).astype(np.float32)
        a_a = -1.0 / (sign_a + nz_a + 1e-7)
        b_a = nx_a * ny_a * a_a
        t1_a = np.stack([1.0 + sign_a * nx_a * nx_a * a_a, sign_a * b_a, -sign_a * nx_a], axis=-1)
        t2_a = np.stack([b_a, sign_a + ny_a * ny_a * a_a, -ny_a], axis=-1)
        R_a = np.stack([t1_a, t2_a, normals_arr], axis=-1).astype(np.float32)
        q_a = rotation_matrix_to_quaternion(R_a)
        sc_a = np.stack([s_arr, s_arr, s_arr * 0.14], axis=-1).astype(np.float32)
        op_a = np.full((len(pts_arr), 1), float(logit(np.array([opac_val]))[0]), dtype=np.float32)
        return pts_arr.astype(np.float32), sc_a, q_a, op_a, fdc_arr.astype(np.float32)

    # 3b. Restore any missing grid cells (where old flying_mask deleted partition bars, ceiling lights, or nadir rows)
    if np.any(missing_cells):
        mv, mu = np.nonzero(missing_cells)
        rays_m = grid_rays[mv, mu]
        d_m = d_grid[mv, mu]
        pts_m = rays_m * d_m[:, np.newaxis]
        cols_m = srgb_to_linear(rgb_grid[mv, mu].astype(np.float32) / 255.0)
        fdc_m = rgb_to_sh0(cols_m)
        s_m = d_m * angular_pixel * 1.22
        p_m, sc_m, q_m, op_m, f_m = _make_camera_facing_splats(pts_m, rays_m, s_m, fdc_m, opac_val=0.96)
        extra_xyz.append(p_m)
        extra_scales.append(sc_m)
        extra_quats.append(q_m)
        extra_opacities.append(op_m)
        extra_fdc.append(f_m)

    # 4. Patch nadir floor cap if missing or black tripod hole exists
    floor_band = (dy < -0.70) & (dy > -0.92)
    if np.count_nonzero(floor_band) >= 16:
        y_floor = float(np.median(xyz[floor_band, 1]))
        floor_rgb_candidates = rgb_linear[floor_band]
        floor_luma = 0.2126 * floor_rgb_candidates[:, 0] + 0.7152 * floor_rgb_candidates[:, 1] + 0.0722 * floor_rgb_candidates[:, 2]
        bright_floor = floor_rgb_candidates[floor_luma > 0.04]
        floor_color = np.median(bright_floor if len(bright_floor) > 0 else floor_rgb_candidates, axis=0).astype(np.float32)

        nadir_dark = (dy <= -0.90) & ((0.2126 * rgb_linear[:, 0] + 0.7152 * rgb_linear[:, 1] + 0.0722 * rgb_linear[:, 2]) < 0.03)
        if np.any(nadir_dark):
            f_dc[nadir_dark] = rgb_to_sh0(np.tile(floor_color, (np.count_nonzero(nadir_dark), 1)))

        cap_radius = abs(y_floor) * 0.28
        grid_1d = np.linspace(-cap_radius, cap_radius, 15, dtype=np.float32)
        gx, gz = np.meshgrid(grid_1d, grid_1d, indexing="xy")
        inside_disk = (gx * gx + gz * gz) <= (cap_radius * cap_radius)
        cx_pts = gx[inside_disk]
        cz_pts = gz[inside_disk]
        cy_pts = np.full_like(cx_pts, y_floor)
        cap_pts = np.stack([cx_pts, cy_pts, cz_pts], axis=-1).astype(np.float32)
        cap_n = len(cap_pts)
        cap_s = max(cap_radius / 5.5, 0.025)
        cap_scales = np.tile(np.array([cap_s, cap_s, cap_s * 0.10], dtype=np.float32), (cap_n, 1))
        R_floor = np.tile(np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]], dtype=np.float32), (cap_n, 1, 1))
        cap_quats = rotation_matrix_to_quaternion(R_floor)
        cap_opac = np.full((cap_n, 1), float(logit(np.array([0.96]))[0]), dtype=np.float32)
        cap_fdc = rgb_to_sh0(np.tile(floor_color, (cap_n, 1))).astype(np.float32)

        extra_xyz.append(cap_pts)
        extra_scales.append(cap_scales)
        extra_quats.append(cap_quats)
        extra_opacities.append(cap_opac)
        extra_fdc.append(cap_fdc)

    # 5. Synthesize missing background shell behind foreground occluders when absent
    fg_mask = segment_foreground_objects(d_grid, rel_depth_threshold=0.18, min_size_pixels=80)
    inpaint_mask = (fg_mask > 0).astype(np.uint8) * 255
    if np.any(inpaint_mask > 0):
        bg_rgb = inpaint_background_texture(rgb_grid, inpaint_mask, inpaint_radius=7, method="telea")
        d_bg = inpaint_background_depth(d_grid, inpaint_mask, smooth_sigma=2.5)
        need_bg = (inpaint_mask > 0) & (d_bg > d_grid + 0.25) & (back_r < d_bg - 0.25) & (d_bg < 25.0)
        if np.any(need_bg):
            vv, uu = np.nonzero(need_bg)
            rays_bg = grid_rays[vv, uu]
            d_vals = d_bg[vv, uu]
            pts_bg = rays_bg * d_vals[:, np.newaxis]
            cols_bg = srgb_to_linear(bg_rgb[vv, uu].astype(np.float32) / 255.0)
            fdc_bg = rgb_to_sh0(cols_bg)
            s_bg = d_vals * angular_pixel * 1.15
            p_b, sc_b, q_b, op_b, f_b = _make_camera_facing_splats(pts_bg, rays_bg, s_bg, fdc_bg, opac_val=0.94)
            extra_xyz.append(p_b)
            extra_scales.append(sc_b)
            extra_quats.append(q_b)
            extra_opacities.append(op_b)
            extra_fdc.append(f_b)

    if extra_xyz:
        xyz = np.vstack([xyz, *extra_xyz]).astype(np.float32)
        scales = np.vstack([scales, *extra_scales]).astype(np.float32)
        quats = np.vstack([quats, *extra_quats]).astype(np.float32)
        opacity_logits = np.vstack([opacity_logits, *extra_opacities]).astype(np.float32)
        f_dc = np.vstack([f_dc, *extra_fdc]).astype(np.float32)

    scaling_log = np.log(np.clip(scales, 1e-6, 100.0)).astype(np.float32)
    return GaussianModel(
        xyz=xyz,
        scaling_log=scaling_log,
        rotation_quats=quats,
        opacity_logits=opacity_logits,
        features_dc=f_dc,
    )


def ensure_supersplat_ply(
    source_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Ensure a PLY file is in upright, display-sRGB, watertight 3DGS format for SuperSplat.

    Handles:
      1. Already watertight SuperSplat PanoGS PLYs (`coord=colmap_z180` and `geometry=watertight-v4`) -> returned as-is.
      2. Earlier PanoGS Gaussian PLYs (missing `geometry=watertight-v4` or `coord=colmap_z180`)
         -> repaired via `repair_panoramic_gaussians()`, tone-mapped to sRGB, and rotated to COLMAP Z-180.
      3. PanoGS Point Cloud PLYs (missing `f_dc_0`, has `red`/`green`/`blue`)
         -> initializes 3D Gaussians automatically, repairs occlusions/poles, and exports SuperSplat PLY.
      4. External 3DGS PLYs (World Labs, InfiniSplat, COLMAP) -> returned as-is.
    """
    src = Path(source_path)
    if not src.is_file():
        raise FileNotFoundError(f"PLY file not found: {src}")

    header_lines = []
    with src.open("rb") as f:
        for _ in range(128):
            raw = f.readline()
            if not raw:
                break
            line = raw.decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

    header_text = "\n".join(header_lines)
    if "coord=colmap_z180_rh" in header_text and "geometry=watertight-v4" in header_text:
        return src

    has_f_dc = "property float f_dc_0" in header_text
    is_panogs = "comment PanoGS" in header_text

    if has_f_dc and not is_panogs:
        # External 3DGS PLY (already in standard COLMAP/sRGB convention)
        return src

    dst = Path(output_path) if output_path is not None else src.with_name(f"{src.stem}_supersplat.ply")
    if has_f_dc and is_panogs:
        model = load_gaussian_ply(src)
        model = repair_panoramic_gaussians(model)
        return save_gaussian_ply(dst, model, supersplat_compat=True)

    # Point-cloud PLY (e.g. output/scene.ply instead of output/scene_gaussians.ply):
    # check if a sibling *_gaussians.ply already exists first.
    sibling_gauss = src.with_name(f"{src.stem}_gaussians.ply")
    if sibling_gauss.is_file() and sibling_gauss != src:
        return ensure_supersplat_ply(sibling_gauss, output_path=output_path)

    from panogs.core.gaussian.initialization import initialize_from_pointcloud
    from panogs.io.ply import read_pointcloud
    from panogs.reconstruction.processing import process_point_cloud

    pc = read_pointcloud(src)
    if pc.normals is None or len(pc.normals) != pc.num_points:
        pc = process_point_cloud(pc, voxel_size=None, remove_outliers=False, compute_normals=True)
    model = initialize_from_pointcloud(pc, splat_shape="hybrid", sharpness=0.55, scale_multiplier=1.10, default_opacity=0.95)
    model = repair_panoramic_gaussians(model)
    return save_gaussian_ply(dst, model, supersplat_compat=True)


def save_gaussian_splat(
    file_path: Union[str, Path],
    model: GaussianModel,
) -> Path:
    """
    Save Gaussians in the 32-byte .splat binary format:
        - Position: 3 x float32 (12 bytes)
        - Scale: 3 x float32 (12 bytes)
        - Color RGBA: 4 x uint8 (4 bytes)
        - Rotation: 4 x uint8 (4 bytes, mapped from [-1, 1] to [0, 255])
    """
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    N = model.num_gaussians
    xyz = model.get_xyz().astype(np.float32)
    scales = model.get_scaling().astype(np.float32)
    rgb = model.get_rgb_uint8()
    alpha = (model.get_opacity().flatten() * 255.0).astype(np.uint8)[:, np.newaxis]
    rgba = np.hstack([rgb, alpha])

    quats = model.get_rotation_quats()
    # Pack [-1.0, 1.0] quaternion into [0, 255] uint8
    quats_u8 = np.clip(np.round((quats * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)

    splat_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("sx", "<f4"), ("sy", "<f4"), ("sz", "<f4"),
        ("rgba", "u1", (4,)),
        ("rot", "u1", (4,)),
    ])
    splat_data = np.empty(N, dtype=splat_dtype)
    splat_data["x"] = xyz[:, 0]
    splat_data["y"] = xyz[:, 1]
    splat_data["z"] = xyz[:, 2]
    splat_data["sx"] = scales[:, 0]
    splat_data["sy"] = scales[:, 1]
    splat_data["sz"] = scales[:, 2]
    splat_data["rgba"] = rgba
    splat_data["rot"] = quats_u8

    with open(target, "wb") as f:
        splat_data.tofile(f)

    return target


def save_gaussian_hdr_splat(
    file_path: Union[str, Path],
    model: GaussianModel,
) -> Path:
    """Save the viewer's 44-byte float-radiance splat format.

    Record layout: xyz and scale (6 float32), linear RGB radiance and opacity
    (4 float32), followed by a normalized quaternion packed as 4 uint8 values.
    Unlike the interoperable 32-byte ``.splat`` format, radiance is not clipped.
    """
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = model.num_gaussians
    data = np.empty(count, dtype=np.dtype([
        ("xyz_scale", "<f4", (6,)),
        ("radiance_opacity", "<f4", (4,)),
        ("rotation", "u1", (4,)),
    ]))
    data["xyz_scale"][:, :3] = model.get_xyz()
    data["xyz_scale"][:, 3:] = model.get_scaling()
    data["radiance_opacity"][:, :3] = model.get_rgb()
    data["radiance_opacity"][:, 3] = model.get_opacity().reshape(-1)
    quats = model.get_rotation_quats()
    data["rotation"] = np.clip(np.round((quats * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
    with open(target, "wb") as f:
        data.tofile(f)
    return target
