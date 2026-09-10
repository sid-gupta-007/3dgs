"""Tests for CLI entry points and commands."""

from pathlib import Path
import pytest
from panogs.apps.cli import main


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "PanoGS" in captured.out
    assert "inspect" in captured.out
    assert "panorama" in captured.out
    assert "depth" in captured.out
    assert "reconstruct" in captured.out
    assert "process" in captured.out
    assert "init-gaussians" in captured.out


def test_cli_no_args(capsys):
    ret = main([])
    assert ret == 0
    captured = capsys.readouterr()
    assert "PanoGS" in captured.out


def test_cli_version(capsys):
    try:
        main(["--version"])
    except SystemExit as e:
        assert e.code == 0


def test_cli_inspect_success(temp_image_dir: Path, capsys):
    img_path = str(temp_image_dir / "sample_rgb.png")
    ret = main(["inspect", img_path])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Image Inspection" in captured.out
    assert "100 x 80" in captured.out
    assert "RGB" in captured.out


def test_cli_inspect_nonexistent(capsys):
    ret = main(["inspect", "nonexistent_file_9999.png"])
    assert ret == 1


def test_cli_panorama_success(temp_image_dir: Path, tmp_path: Path, capsys):
    img_path = str(temp_image_dir / "sample_rgb.png")
    out_ply = str(tmp_path / "output_sphere.ply")
    ret = main(["panorama", img_path, "-o", out_ply, "-r", "2.0"])
    assert ret == 0
    assert Path(out_ply).exists()
    captured = capsys.readouterr()
    assert "Successfully generated spherical point cloud" in captured.out
    assert "8,000" in captured.out


def test_cli_depth_synthetic(temp_image_dir: Path, tmp_path: Path, capsys):
    img_path = str(temp_image_dir / "sample_rgb.png")
    out_dir = str(tmp_path / "depth_out")
    ret = main(["depth", img_path, "-o", out_dir, "-m", "synthetic_room"])
    assert ret == 0
    assert (Path(out_dir) / "depth.npy").exists()
    assert (Path(out_dir) / "depth_vis.png").exists()
    assert (Path(out_dir) / "depth_metadata.json").exists()
    captured = capsys.readouterr()
    assert "Depth Estimation Complete" in captured.out


def test_cli_reconstruct_synthetic(temp_image_dir: Path, tmp_path: Path, capsys):
    img_path = str(temp_image_dir / "sample_rgb.png")
    out_ply = str(tmp_path / "recon_out.ply")
    ret = main(["reconstruct", img_path, "-o", out_ply, "-m", "synthetic_room"])
    assert ret == 0
    assert Path(out_ply).exists()
    captured = capsys.readouterr()
    assert "3D Reconstruction Complete" in captured.out
    assert "8,000" in captured.out


def test_cli_process_pointcloud(temp_image_dir: Path, tmp_path: Path, capsys):
    img_path = str(temp_image_dir / "sample_rgb.png")
    recon_ply = str(tmp_path / "recon.ply")
    proc_ply = str(tmp_path / "proc.ply")

    main(["reconstruct", img_path, "-o", recon_ply, "-m", "synthetic_room"])
    ret = main(["process", recon_ply, "-o", proc_ply, "--voxel-size", "0.1"])
    assert ret == 0
    assert Path(proc_ply).exists()
    captured = capsys.readouterr()
    assert "Point Cloud Processing Complete" in captured.out


def test_cli_init_gaussians(temp_image_dir: Path, tmp_path: Path, capsys):
    img_path = str(temp_image_dir / "sample_rgb.png")
    recon_ply = str(tmp_path / "recon.ply")
    gauss_ply = str(tmp_path / "gaussians.ply")

    main(["reconstruct", img_path, "-o", recon_ply, "-m", "synthetic_room"])
    ret = main(["init-gaussians", recon_ply, "-o", gauss_ply, "--splat"])
    assert ret == 0
    assert Path(gauss_ply).exists()
    assert Path(tmp_path / "gaussians.splat").exists()
    captured = capsys.readouterr()
    assert "3D Gaussian Initialization Complete" in captured.out


def test_cli_render(temp_image_dir: Path, tmp_path: Path, capsys):
    img_path = str(temp_image_dir / "sample_rgb.png")
    recon_ply = str(tmp_path / "recon.ply")
    gauss_ply = str(tmp_path / "gaussians.ply")
    render_png = str(tmp_path / "rendered.png")

    main(["reconstruct", img_path, "-o", recon_ply, "-m", "synthetic_room"])
    main(["init-gaussians", recon_ply, "-o", gauss_ply])
    ret = main(["render", gauss_ply, "-o", render_png, "--width", "64", "--height", "64"])
    assert ret == 0
    assert Path(render_png).exists()
    captured = capsys.readouterr()
    assert "Gaussian Splatting Render Complete" in captured.out


def test_cli_train(temp_image_dir: Path, tmp_path: Path, capsys):
    img_path = str(temp_image_dir / "sample_rgb.png")
    recon_ply = str(tmp_path / "recon.ply")
    gauss_ply = str(tmp_path / "gaussians.ply")
    train_ply = str(tmp_path / "opt_gaussians.ply")

    main(["reconstruct", img_path, "-o", recon_ply, "-m", "synthetic_room"])
    main(["init-gaussians", recon_ply, "-o", gauss_ply])
    ret = main([
        "train",
        img_path,
        "--init-ply", gauss_ply,
        "-o", train_ply,
        "--iterations", "5",
        "--views", "2",
        "--res", "32",
    ])
    assert ret == 0
    assert Path(train_ply).exists()
    assert Path(tmp_path / "opt_gaussians.splat").exists()
    captured = capsys.readouterr()
    assert "3D Gaussian Splatting Optimization Complete" in captured.out


