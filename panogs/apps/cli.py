"""
Command Line Interface for PanoGS.
Provides extensible subcommands for inspection, panorama processing, depth estimation,
3D reconstruction, point cloud processing, Gaussian initialization, rendering, and viewing.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image

import panogs
from panogs.core.config import load_config
from panogs.core.gaussian import GaussianModel, initialize_from_pointcloud
from panogs.core.logging import get_logger, setup_logging
from panogs.io.depth import save_depth_result
from panogs.io.gaussian_ply import load_gaussian_ply, save_gaussian_ply, save_gaussian_splat
from panogs.io.images import inspect_image
from panogs.io.ply import read_point_cloud_ply, read_pointcloud, write_pointcloud
from panogs.reconstruction.depth import get_depth_estimator
from panogs.reconstruction.panorama import generate_synthetic_sphere, reconstruct_layout_panorama
from panogs.reconstruction.pointcloud import PointCloud, reconstruct_from_image
from panogs.reconstruction.processing import process_point_cloud
from panogs.rendering.camera import create_orbit_camera
from panogs.rendering.cpu.rasterizer import render_gaussians_cpu
from panogs.training.trainer import TrainingConfig, train_gaussians
from panogs.apps.viewer import start_viewer_server


def build_parser() -> argparse.ArgumentParser:
    """Build root CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="panogs",
        description="PanoGS: Unified Image/Panorama/Video to 3D Gaussian Splatting Reconstruction Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"PanoGS v{panogs.__version__}",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose / debug log output",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available subcommands",
        help="Run 'panogs <command> --help' for details on a specific command",
    )

    # ─────────────────────────────────────────────────────────────
    # Subcommand: inspect
    # ─────────────────────────────────────────────────────────────
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect an image file and print its format, resolution, and memory footprint",
        description="Inspect an input image (JPG, PNG) and report metadata, dimensions, and memory usage.",
    )
    inspect_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file to inspect (JPG, JPEG, PNG)",
    )

    # ─────────────────────────────────────────────────────────────
    # Subcommand: panorama
    # ─────────────────────────────────────────────────────────────
    panorama_parser = subparsers.add_parser(
        "panorama",
        help="Convert an equirectangular panorama into a synthetic 3D spherical shell PLY point cloud",
        description="Generate 3D spherical ray fields and export a synthetic spherical shell point cloud (.ply).",
    )
    panorama_parser.add_argument(
        "image_path",
        type=str,
        help="Path to equirectangular panorama image",
    )
    panorama_parser.add_argument(
        "-o", "--output",
        type=str,
        default="output/sphere.ply",
        help="Output path for the generated point cloud PLY file (default: output/sphere.ply)",
    )
    panorama_parser.add_argument(
        "-r", "--radius",
        type=float,
        default=1.0,
        help="Radius of the synthetic sphere in world coordinates (default: 1.0)",
    )
    panorama_parser.add_argument(
        "--max-res",
        type=int,
        default=None,
        help="Optional maximum image dimension for downscaling",
    )

    # ─────────────────────────────────────────────────────────────
    # Subcommand: depth
    # ─────────────────────────────────────────────────────────────
    depth_parser = subparsers.add_parser(
        "depth",
        help="Estimate monocular depth map from an image and save raw .npy and visualization .png",
        description="Run monocular depth estimation on an image and save raw float32 depth map and colored visualization.",
    )
    depth_parser.add_argument(
        "image_path",
        type=str,
        help="Path to input image (JPG, PNG)",
    )
    depth_parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="output/depth",
        help="Directory to save depth.npy, depth_vis.png, and metadata.json (default: output/depth)",
    )
    depth_parser.add_argument(
        "-m", "--model",
        type=str,
        default="midas_small",
        help="Depth model name ('midas_small', 'synthetic_room', 'synthetic_gradient') (default: midas_small)",
    )
    depth_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device for depth inference ('cpu') (default: cpu)",
    )

    # ─────────────────────────────────────────────────────────────
    # Subcommand: reconstruct
    # ─────────────────────────────────────────────────────────────
    recon_parser = subparsers.add_parser(
        "reconstruct",
        help="Reconstruct a true 3D scene point cloud from an image/panorama using depth backprojection",
        description="Combine depth estimation and camera ray geometry to reconstruct a 3D scene point cloud (.ply).",
    )
    recon_parser.add_argument(
        "image_path",
        type=str,
        help="Path to input image/panorama (JPG, PNG)",
    )
    recon_parser.add_argument(
        "-o", "--output",
        type=str,
        default="output/reconstructed.ply",
        help="Output path for the reconstructed .ply point cloud (default: output/reconstructed.ply)",
    )
    recon_parser.add_argument(
        "-m", "--model",
        type=str,
        default="cubemap_depth_anything",
        help="Depth model ('cubemap_depth_anything', 'cubemap_midas', 'depth_anything_v2', 'midas_small', 'synthetic_room') (default: cubemap_depth_anything)",
    )
    recon_parser.add_argument(
        "--camera",
        type=str,
        default="spherical",
        choices=["spherical"],
        help="Camera projection model ('spherical') (default: spherical)",
    )
    recon_parser.add_argument(
        "--min-depth",
        type=float,
        default=0.5,
        help="Minimum scene depth bound in meters (default: 0.5)",
    )
    recon_parser.add_argument(
        "--max-depth",
        type=float,
        default=8.0,
        help="Maximum scene depth bound in meters (default: 8.0)",
    )
    recon_parser.add_argument(
        "--layout",
        action="store_true",
        default=True,
        help="Use layout-guided Manhattan room architecture for true cuboidal indoor geometry (default: True)",
    )
    recon_parser.add_argument(
        "--no-layout",
        dest="layout",
        action="store_false",
        help="Disable layout-guided cuboidal geometry",
    )
    recon_parser.add_argument(
        "--floor-height",
        type=float,
        default=1.5,
        help="Height of camera above floor in meters (default: 1.5)",
    )
    recon_parser.add_argument(
        "--ceiling-height",
        type=float,
        default=1.8,
        help="Height of ceiling above camera in meters (default: 1.8)",
    )
    recon_parser.add_argument(
        "--room-depth",
        type=float,
        default=4.5,
        help="Distance from camera to front/back walls in meters (default: 4.5)",
    )
    recon_parser.add_argument(
        "--room-width",
        type=float,
        default=4.0,
        help="Distance from camera to side walls in meters (default: 4.0)",
    )
    recon_parser.add_argument(
        "--max-res",
        type=int,
        default=None,
        help="Optional maximum image dimension for downscaling",
    )

    # ─────────────────────────────────────────────────────────────
    # Subcommand: video
    # ─────────────────────────────────────────────────────────────
    video_parser = subparsers.add_parser(
        "video",
        help="Reconstruct a full 3D scene from walking video (.mp4/.mov) using multi-view tracking & depth",
        description="Extract sharp keyframes from video, track camera trajectory, and fuse multi-view depth into solid 3D point cloud & 3DGS splat.",
    )
    video_parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file (MP4, MOV, AVI)",
    )
    video_parser.add_argument(
        "-o", "--output",
        type=str,
        default="output/video_scene.ply",
        help="Output path for reconstructed point cloud PLY (default: output/video_scene.ply)",
    )
    video_parser.add_argument(
        "-m", "--model",
        type=str,
        default="depth_anything_v2",
        help="Depth model ('depth_anything_v2', 'midas_small') (default: depth_anything_v2)",
    )
    video_parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Keyframe extraction sampling frequency in FPS (default: 2.0)",
    )
    video_parser.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help="Maximum keyframes to extract (default: 60)",
    )
    video_parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.03,
        help="Voxel grid downsampling size in meters (default: 0.03m)",
    )
    video_parser.add_argument(
        "--splat",
        action="store_true",
        default=True,
        help="Also initialize 3D Gaussians and export WebGL .splat file (default: True)",
    )

    # ─────────────────────────────────────────────────────────────
    # Subcommand: process
    # ─────────────────────────────────────────────────────────────
    process_parser = subparsers.add_parser(
        "process",
        help="Process, downsample, denoise, and estimate surface normals on a 3D point cloud PLY",
        description="Voxel downsampling, statistical outlier removal, and local PCA normal estimation.",
    )
    process_parser.add_argument(
        "input_ply",
        type=str,
        help="Path to input .ply point cloud file",
    )
    process_parser.add_argument(
        "-o", "--output",
        type=str,
        default="output/processed.ply",
        help="Output path for cleaned point cloud PLY (default: output/processed.ply)",
    )
    process_parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.04,
        help="Voxel grid size in meters for downsampling (default: 0.04m, 0 to disable)",
    )
    process_parser.add_argument(
        "--no-outliers",
        action="store_true",
        help="Disable statistical outlier removal",
    )
    process_parser.add_argument(
        "--no-normals",
        action="store_true",
        help="Disable surface normal estimation",
    )

    # ─────────────────────────────────────────────────────────────
    # Subcommand: init-gaussians
    # ─────────────────────────────────────────────────────────────
    gauss_parser = subparsers.add_parser(
        "init-gaussians",
        help="Initialize 3D Gaussians from a processed point cloud and export standard 3DGS .ply/.splat",
        description="Initialize 3D Gaussians with adaptive k-NN scales, unit quaternions, logit opacity, and SH0 colors.",
    )
    gauss_parser.add_argument(
        "input_ply",
        type=str,
        help="Path to input point cloud .ply file",
    )
    gauss_parser.add_argument(
        "-o", "--output",
        type=str,
        default="output/gaussians.ply",
        help="Output path for 3DGS PLY file (default: output/gaussians.ply)",
    )
    gauss_parser.add_argument(
        "--opacity",
        type=float,
        default=0.8,
        help="Initial opacity in (0, 1) (default: 0.8)",
    )
    gauss_parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Multiplier on adaptive k-NN spacing (default: 1.0)",
    )
    gauss_parser.add_argument(
        "--splat",
        action="store_true",
        help="Also export a fast WebGL .splat binary file",
    )

    # ─────────────────────────────────────────────────────────────
    # Subcommand: render
    # ─────────────────────────────────────────────────────────────
    render_parser = subparsers.add_parser(
        "render",
        help="Render a 2D image viewpoint from a 3D Gaussian Splatting scene model on CPU",
        description="Project 3D Gaussians into 2D screen space with Jacobian covariance and front-to-back alpha blending.",
    )
    render_parser.add_argument(
        "gaussians_ply",
        type=str,
        help="Path to 3DGS .ply file",
    )
    render_parser.add_argument(
        "-o", "--output",
        type=str,
        default="output/rendered.png",
        help="Output image path (default: output/rendered.png)",
    )
    render_parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Rendered image width in pixels (default: 512)",
    )
    render_parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Rendered image height in pixels (default: 512)",
    )
    render_parser.add_argument(
        "--distance",
        type=float,
        default=2.0,
        help="Camera orbit distance in meters (default: 2.0)",
    )
    render_parser.add_argument(
        "--elevation",
        type=float,
        default=0.0,
        help="Camera elevation angle in degrees (-89 to +89) (default: 0.0)",
    )
    render_parser.add_argument(
        "--azimuth",
        type=float,
        default=0.0,
        help="Camera azimuth angle in degrees (default: 0.0)",
    )
    render_parser.add_argument(
        "--fov",
        type=float,
        default=60.0,
        help="Camera field of view in degrees (default: 60.0)",
    )
    render_parser.add_argument(
        "--bg",
        type=str,
        default="white",
        choices=["white", "black"],
        help="Background color ('white' or 'black') (default: white)",
    )

    # ─────────────────────────────────────────────────────────────
    # Subcommand: train
    # ─────────────────────────────────────────────────────────────
    train_parser = subparsers.add_parser(
        "train",
        help="Optimize 3D Gaussian Splats against multi-view perspective views with densification",
        description="Run differentiable 3DGS optimization with L1+SSIM loss, adaptive cloning, splitting, and pruning.",
    )
    train_parser.add_argument(
        "image_path",
        type=str,
        help="Path to input panorama or image",
    )
    train_parser.add_argument(
        "--init-ply",
        type=str,
        required=True,
        help="Path to initial 3DGS PLY file",
    )
    train_parser.add_argument(
        "-o", "--output",
        type=str,
        default="output/optimized_gaussians.ply",
        help="Output path for optimized 3DGS PLY (default: output/optimized_gaussians.ply)",
    )
    train_parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of optimization iterations (default: 100)",
    )
    train_parser.add_argument(
        "--views",
        type=int,
        default=8,
        help="Number of perspective training views generated from panorama (default: 8)",
    )
    train_parser.add_argument(
        "--res",
        type=int,
        default=128,
        help="Perspective training view resolution (default: 128)",
    )
    train_parser.add_argument(
        "--splat",
        action="store_true",
        default=True,
        help="Also export optimized WebGL .splat file (default: True)",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device ('cpu') (default: cpu)",
    )

    # ─────────────────────────────────────────────────────────────
    # Subcommand: view
    # ─────────────────────────────────────────────────────────────
    view_parser = subparsers.add_parser(
        "view",
        help="Launch real-time interactive 3D WebGL Gaussian Splatting scene explorer in browser",
        description="Serve and interactively explore 3DGS .splat or .ply models at 60 FPS in WebGL.",
    )
    view_parser.add_argument(
        "model_path",
        type=str,
        help="Path to 3DGS scene file (.splat or .ply)",
    )
    view_parser.add_argument(
        "-p", "--port",
        type=int,
        default=8080,
        help="Local HTTP server port (default: 8080)",
    )
    view_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the web browser",
    )

    return parser


def handle_inspect(args: argparse.Namespace) -> int:
    """Handler for 'panogs inspect' subcommand."""
    logger = get_logger("cli.inspect")
    image_path = Path(args.image_path)

    try:
        metadata = inspect_image(image_path)
        print("\n" + metadata.formatted_summary() + "\n")
        return 0
    except Exception as e:
        logger.error(f"Failed to inspect image '{image_path}': {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def handle_panorama(args: argparse.Namespace) -> int:
    """Handler for 'panogs panorama' subcommand."""
    logger = get_logger("cli.panorama")
    image_path = Path(args.image_path)
    output_path = Path(args.output)

    try:
        points, colors, saved_path = generate_synthetic_sphere(
            image_path=image_path,
            radius=args.radius,
            max_resolution=args.max_res,
            output_ply=output_path,
        )
        print(f"\nSuccessfully generated spherical point cloud:")
        print(f"  Input:    {image_path}")
        print(f"  Vertices: {points.shape[0]:,}")
        print(f"  Radius:   {args.radius}")
        print(f"  Output:   {saved_path}\n")
        return 0
    except Exception as e:
        logger.error(f"Failed to generate panorama point cloud: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def handle_depth(args: argparse.Namespace) -> int:
    """Handler for 'panogs depth' subcommand."""
    logger = get_logger("cli.depth")
    image_path = Path(args.image_path)
    output_dir = Path(args.output_dir)

    try:
        logger.info(f"Instantiating depth estimator '{args.model}' on '{args.device}'...")
        estimator = get_depth_estimator(args.model, device=args.device)

        logger.info(f"Estimating depth for '{image_path}'...")
        result = estimator.estimate(image_path)

        npy_path, vis_path, meta_path = save_depth_result(result, output_dir=output_dir)

        print("\nDepth Estimation Complete:")
        print(f"  Input Image:     {image_path}")
        print(f"  Model:           {result.model_name}")
        print(f"  Metric Depth:    {result.is_metric}")
        print(f"  Resolution:      {result.width} x {result.height} (W x H)")
        print(f"  Depth Range:     [{result.min_depth:.4f}, {result.max_depth:.4f}] (mean: {result.mean_depth:.4f})")
        print(f"  Raw Array:       {npy_path}")
        print(f"  Visualization:   {vis_path}")
        print(f"  Metadata:        {meta_path}\n")
        return 0
    except Exception as e:
        logger.error(f"Depth estimation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def handle_reconstruct(args: argparse.Namespace) -> int:
    """Handler for 'panogs reconstruct' subcommand."""
    logger = get_logger("cli.reconstruct")
    image_path = Path(args.image_path)
    output_path = Path(args.output)

    try:
        estimator = get_depth_estimator(args.model)
        
        if getattr(args, "layout", True) and args.camera == "spherical" and not args.model.startswith("synthetic"):
            logger.info("Using layout-guided Manhattan room architecture for true cuboidal 3D reconstruction...")
            pc = reconstruct_layout_panorama(
                image_path=image_path,
                depth_estimator=estimator,
                h_floor=args.floor_height,
                h_ceiling=args.ceiling_height,
                room_depth=args.room_depth,
                room_width=args.room_width,
                max_resolution=args.max_res,
                output_ply=output_path,
            )
        else:
            pc = reconstruct_from_image(
                image_path=image_path,
                depth_estimator=estimator,
                camera_type=args.camera,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                depth_edge_threshold=getattr(args, "edge_threshold", 0.0),
                max_resolution=args.max_res,
                output_ply=output_path,
            )

        min_b, max_b = pc.bounds
        print("\n3D Reconstruction Complete:")
        print(f"  Input Image:     {image_path}")
        print(f"  Points:          {pc.num_points:,}")
        print(f"  Bounding Box:    [{min_b[0]:.2f}, {min_b[1]:.2f}, {min_b[2]:.2f}] to [{max_b[0]:.2f}, {max_b[1]:.2f}, {max_b[2]:.2f}]")
        print(f"  Output PLY:      {output_path}\n")
        return 0
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def handle_video(args: argparse.Namespace) -> int:
    """Handler for 'panogs video' subcommand."""
    from panogs.reconstruction.video.video_pipeline import reconstruct_from_video
    from panogs.core.gaussian.initialization import initialize_from_pointcloud
    from panogs.io.gaussian_ply import save_gaussian_ply, save_gaussian_splat

    logger = get_logger("cli.video")
    video_path = Path(args.video_path)
    output_path = Path(args.output)

    try:
        estimator = get_depth_estimator(args.model)
        logger.info(f"Reconstructing multi-view 3D scene from video '{video_path}'...")

        pc = reconstruct_from_video(
            video_path=video_path,
            depth_estimator=estimator,
            target_fps=args.fps,
            max_frames=args.max_frames,
            voxel_size=args.voxel_size,
            output_ply=output_path,
        )

        min_b, max_b = pc.bounds
        print("\nMulti-View Video 3D Reconstruction Complete:")
        print(f"  Input Video:     {video_path}")
        print(f"  3D Points:       {pc.num_points:,}")
        print(f"  Bounding Box:    [{min_b[0]:.2f}, {min_b[1]:.2f}, {min_b[2]:.2f}] to [{max_b[0]:.2f}, {max_b[1]:.2f}, {max_b[2]:.2f}]")
        print(f"  Output PLY:      {output_path}")

        if getattr(args, "splat", True):
            splat_path = output_path.with_suffix(".splat")
            gauss_path = output_path.with_name(f"{output_path.stem}_gaussians.ply")
            logger.info("Initializing 3D Gaussians from video point cloud...")
            model = initialize_from_pointcloud(pc)
            save_gaussian_ply(gauss_path, model)
            save_gaussian_splat(splat_path, model)
            print(f"  3DGS PLY:        {gauss_path} ({model.num_gaussians:,} Gaussians)")
            print(f"  WebGL Splat:     {splat_path}\n")

        return 0
    except Exception as e:
        logger.error(f"Video reconstruction failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def handle_process(args: argparse.Namespace) -> int:
    """Handler for 'panogs process' subcommand."""
    logger = get_logger("cli.process")
    input_path = Path(args.input_ply)
    output_path = Path(args.output)

    try:
        logger.info(f"Loading input point cloud: {input_path}")
        points, colors = read_point_cloud_ply(input_path)
        pc = PointCloud(points=points, colors=colors)

        voxel_sz = args.voxel_size if args.voxel_size > 0 else None
        cleaned_pc = process_point_cloud(
            point_cloud=pc,
            voxel_size=voxel_sz,
            remove_outliers=not args.no_outliers,
            compute_normals=not args.no_normals,
        )

        logger.info(f"Saving processed point cloud to {output_path}...")
        write_pointcloud(output_path, cleaned_pc, binary=True)

        min_b, max_b = cleaned_pc.bounds
        print("\nPoint Cloud Processing Complete:")
        print(f"  Input PLY:       {input_path} ({pc.num_points:,} points)")
        print(f"  Output PLY:      {output_path} ({cleaned_pc.num_points:,} points)")
        print(f"  Reduction Ratio: {pc.num_points / max(1, cleaned_pc.num_points):.1f}x compression")
        print(f"  Bounding Box:    [{min_b[0]:.2f}, {min_b[1]:.2f}, {min_b[2]:.2f}] to [{max_b[0]:.2f}, {max_b[1]:.2f}, {max_b[2]:.2f}]")
        print(f"  Normals Present: {cleaned_pc.normals is not None}\n")
        return 0
    except Exception as e:
        logger.error(f"Point cloud processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def handle_init_gaussians(args: argparse.Namespace) -> int:
    """Handler for 'panogs init-gaussians' subcommand."""
    logger = get_logger("cli.init_gaussians")
    input_path = Path(args.input_ply)
    output_path = Path(args.output)

    try:
        logger.info(f"Loading point cloud: {input_path}")
        pc = read_pointcloud(input_path)

        if pc.normals is None or len(pc.normals) != pc.num_points:
            logger.info("Computing local surface normals for anisotropic surface-aligned splats...")
            pc = process_point_cloud(
                point_cloud=pc,
                voxel_size=None,
                remove_outliers=False,
                compute_normals=True,
            )

        model = initialize_from_pointcloud(
            point_cloud=pc,
            default_opacity=args.opacity,
            scale_multiplier=args.scale_factor,
        )

        save_gaussian_ply(output_path, model)

        if args.splat:
            splat_path = output_path.with_suffix(".splat")
            save_gaussian_splat(splat_path, model)
            logger.info(f"Saved .splat binary file to {splat_path}")

        scales = model.get_scaling()
        print("\n3D Gaussian Initialization Complete:")
        print(f"  Input Point Cloud:  {input_path} ({pc.num_points:,} points)")
        print(f"  Total Gaussians:    {model.num_gaussians:,}")
        print(f"  Scale Range:        [{np.min(scales):.4f}, {np.max(scales):.4f}] (mean: {np.mean(scales):.4f}m)")
        print(f"  Opacity:            {args.opacity:.2f}")
        print(f"  Normals Aligned:    {pc.normals is not None}")
        print(f"  Output 3DGS PLY:    {output_path}\n")
        return 0
    except Exception as e:
        logger.error(f"Gaussian initialization failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def handle_render(args: argparse.Namespace) -> int:
    """Handler for 'panogs render' subcommand."""
    logger = get_logger("cli.render")
    input_path = Path(args.gaussians_ply)
    output_path = Path(args.output)

    try:
        logger.info(f"Loading 3D Gaussians from {input_path}...")
        model = load_gaussian_ply(input_path)

        # Compute scene centroid to aim orbital camera
        target_center = tuple(np.mean(model.get_xyz(), axis=0))

        camera = create_orbit_camera(
            target=target_center,
            distance=args.distance,
            azimuth_deg=args.azimuth,
            elevation_deg=args.elevation,
            width=args.width,
            height=args.height,
            fov=args.fov,
        )

        bg_color = (1.0, 1.0, 1.0) if args.bg == "white" else (0.0, 0.0, 0.0)

        logger.info(f"Rendering {model.num_gaussians:,} Gaussians on CPU ({args.width}x{args.height})...")
        rendered_rgb = render_gaussians_cpu(
            model=model,
            camera=camera,
            bg_color=bg_color,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rendered_rgb).save(output_path)

        print("\nGaussian Splatting Render Complete:")
        print(f"  Input 3DGS PLY:  {input_path} ({model.num_gaussians:,} Gaussians)")
        print(f"  Resolution:      {args.width} x {args.height}")
        print(f"  Camera Orbit:    Distance {args.distance}m, Azimuth {args.azimuth} deg, Elevation {args.elevation} deg")
        print(f"  Output Image:    {output_path}\n")
        return 0
    except Exception as e:
        logger.error(f"Rendering failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def handle_train(args: argparse.Namespace) -> int:
    """Handler for 'panogs train' subcommand."""
    logger = get_logger("cli.train")
    image_path = Path(args.image_path)
    init_ply_path = Path(args.init_ply)
    output_path = Path(args.output)

    try:
        config = TrainingConfig(
            iterations=args.iterations,
            num_views=args.views,
            view_resolution=args.res,
            device=args.device,
        )

        model, history = train_gaussians(
            image_path=image_path,
            init_ply_path=init_ply_path,
            output_ply_path=output_path,
            config=config,
            save_splat=args.splat,
        )

        initial_loss = history[0]["loss"]
        final_loss = history[-1]["loss"]
        final_psnr = history[-1]["psnr"]

        print("\n3D Gaussian Splatting Optimization Complete:")
        print(f"  Input Panorama:     {image_path}")
        print(f"  Initial PLY:        {init_ply_path}")
        print(f"  Optimized PLY:      {output_path}")
        print(f"  Total Iterations:   {config.iterations}")
        print(f"  Initial Loss:       {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
        print(f"  Final PSNR:         {final_psnr:.2f} dB")
        print(f"  Final Gaussians:    {model.num_gaussians:,}\n")
        return 0
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def handle_view(args: argparse.Namespace) -> int:
    """Handler for 'panogs view' subcommand."""
    logger = get_logger("cli.view")
    model_path = Path(args.model_path)

    if not model_path.exists():
        logger.error(f"Scene model file not found: {model_path}")
        return 1

    try:
        start_viewer_server(
            model_path=model_path,
            port=args.port,
            open_browser=not args.no_browser,
            block=True,
        )
        return 0
    except Exception as e:
        logger.error(f"Viewer failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    # Configure logging level
    log_level = "DEBUG" if parsed_args.verbose else "INFO"
    setup_logging(level=log_level)
    logger = get_logger("cli")

    # Load configuration
    try:
        config = load_config(parsed_args.config)
        logger.debug(f"Loaded configuration for device: {config.hardware.device}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    if parsed_args.command is None:
        parser.print_help()
        return 0

    if parsed_args.command == "inspect":
        return handle_inspect(parsed_args)
    elif parsed_args.command == "panorama":
        return handle_panorama(parsed_args)
    elif parsed_args.command == "depth":
        return handle_depth(parsed_args)
    elif parsed_args.command == "reconstruct":
        return handle_reconstruct(parsed_args)
    elif parsed_args.command == "video":
        return handle_video(parsed_args)
    elif parsed_args.command == "process":
        return handle_process(parsed_args)
    elif parsed_args.command == "init-gaussians":
        return handle_init_gaussians(parsed_args)
    elif parsed_args.command == "render":
        return handle_render(parsed_args)
    elif parsed_args.command == "train":
        return handle_train(parsed_args)
    elif parsed_args.command == "view":
        return handle_view(parsed_args)

    logger.error(f"Unknown command: {parsed_args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
