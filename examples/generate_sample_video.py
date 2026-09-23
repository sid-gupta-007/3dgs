"""
Generate a synthetic 3D indoor room walking video (.mp4) for testing multi-view 3DGS.
Renders an interactive perspective walking tour with moving camera, parallax, and 3D objects.
"""

from pathlib import Path
import cv2
import numpy as np


def generate_walking_room_video(output_path: Path, num_frames: int = 30, fps: float = 10.0):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    W, H = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    for f in range(num_frames):
        t = f / float(num_frames)
        # Camera trajectory: walking forward into the room with slight lateral pan
        cam_x = np.sin(t * np.pi) * 0.4
        cam_z = t * 1.8

        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # 1. Floor with perspective tiles
        for y in range(int(H * 0.55), H):
            depth = (H * 0.45) / max(1, y - int(H * 0.55))
            tile_u = ((np.arange(W) - W / 2) * depth * 0.05 + cam_x * 10).astype(int)
            tile_v = int((depth + cam_z) * 10)
            is_dark = ((tile_u // 10) + (tile_v // 10)) % 2 == 0
            color = np.where(is_dark[:, np.newaxis], [90, 80, 70], [160, 140, 120])
            frame[y, :] = color

        # 2. Back Wall (textured plaster)
        wall_h = int(H * 0.55)
        for y in range(wall_h):
            wall_pat = int((y + f) % 20 > 10)
            frame[y, :] = [210 - wall_pat * 15, 205 - wall_pat * 15, 195 - wall_pat * 10]

        # 3. Wall Painting with parallax
        paint_x = int(W * 0.65 - cam_x * 40 - cam_z * 15)
        paint_y = int(H * 0.20)
        paint_w = int(60 / (1.0 + cam_z * 0.2))
        paint_h = int(45 / (1.0 + cam_z * 0.2))
        if 0 <= paint_x < W - paint_w:
            cv2.rectangle(frame, (paint_x, paint_y), (paint_x + paint_w, paint_y + paint_h), (50, 40, 30), 2)
            cv2.rectangle(frame, (paint_x + 2, paint_y + 2), (paint_x + paint_w - 2, paint_y + paint_h - 2), (180, 100, 60), -1)

        # 4. Foreground Table with strong 3D Parallax
        table_world_z = 2.5
        rel_z = max(0.4, table_world_z - cam_z)
        scale = 1.0 / rel_z
        tbl_cx = int(W * 0.40 - cam_x * scale * 60)
        tbl_cy = int(H * 0.70 + scale * 10)
        tbl_w = int(70 * scale)
        tbl_h = int(25 * scale)

        # Draw Table top
        pt1 = (tbl_cx - tbl_w // 2, tbl_cy - tbl_h // 2)
        pt2 = (tbl_cx + tbl_w // 2, tbl_cy + tbl_h // 2)
        if 0 <= tbl_cx < W:
            cv2.rectangle(frame, pt1, pt2, (60, 130, 80), -1)
            cv2.rectangle(frame, pt1, pt2, (40, 90, 50), 2)

            # Table legs
            leg_h = int(35 * scale)
            cv2.line(frame, (pt1[0] + 5, pt2[1]), (pt1[0] + 5, pt2[1] + leg_h), (80, 50, 30), max(1, int(3 * scale)))
            cv2.line(frame, (pt2[0] - 5, pt2[1]), (pt2[0] - 5, pt2[1] + leg_h), (80, 50, 30), max(1, int(3 * scale)))

        # 5. Pillar on the left
        pil_world_z = 3.0
        pil_rel_z = max(0.5, pil_world_z - cam_z)
        pil_scale = 1.0 / pil_rel_z
        pil_x = int(W * 0.15 - cam_x * pil_scale * 50)
        pil_w = int(30 * pil_scale)
        if 0 <= pil_x < W - pil_w:
            cv2.rectangle(frame, (pil_x, 0), (pil_x + pil_w, int(H * 0.75)), (190, 180, 170), -1)
            cv2.rectangle(frame, (pil_x, 0), (pil_x + pil_w, int(H * 0.75)), (140, 130, 120), 1)

        out.write(frame)

    out.release()
    print(f"Generated synthetic walking room video at: {output_path} ({num_frames} frames)")


if __name__ == "__main__":
    generate_walking_room_video(Path("examples/data/sample_walk.mp4"), num_frames=30, fps=10.0)
