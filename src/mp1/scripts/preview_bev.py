#!/usr/bin/env python3
import os
import json
import glob
import argparse
import numpy as np
import cv2

def infer_bev_px_size(bev_height_m, bev_width_m, unit_conv):
    # unit_conv = (bev_height/bev_img_height, bev_width/bev_img_width)
    h_px = int(round(bev_height_m / unit_conv[0]))
    w_px = int(round(bev_width_m / unit_conv[1]))
    return h_px, w_px

def pick_image(path_hint=None):
    if path_hint and os.path.isfile(path_hint):
        return path_hint
    candidates = []
    # try common places from the MP
    candidates += glob.glob("data/dataset/test/images/*.*")
    candidates += glob.glob("data/dataset/train/images/*.*")
    candidates += glob.glob("data/raw_images/*.*")
    for p in candidates:
        if p.lower().endswith((".png", ".jpg", ".jpeg")):
            return p
    raise FileNotFoundError(
        "Could not find an input image. Pass --image or place one under "
        "data/dataset/*/images/ or data/raw_images/."
    )

def draw_src_points(img, src):
    img_vis = img.copy()
    pts = src.astype(int)
    color = (0, 255, 255)
    for (u, v) in pts:
        cv2.circle(img_vis, (int(u), int(v)), 6, color, -1, lineType=cv2.LINE_AA)
    cv2.polylines(img_vis, [pts.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 255), thickness=2)
    return img_vis

def overlay_grid(bev, step=50):
    grid = bev.copy()
    h, w = grid.shape[:2]
    for x in range(0, w, step):
        cv2.line(grid, (x, 0), (x, h), (255, 255, 255), 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(grid, (0, y), (w, y), (255, 255, 255), 1, cv2.LINE_AA)
    return grid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default=None, help="Path to a raw camera image to warp")
    ap.add_argument("--save_dir", type=str, default="debug", help="Where to save previews")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1) Load BEV config saved by run_bev_config.py
    cfg_path = "data/bev_config.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    bev_height_m, bev_width_m = cfg["bev_world_dim"]         # meters
    unit_conv = tuple(cfg["unit_conversion_factor"])          # (m/px along height, m/px along width)
    src = np.float32(cfg["src"])                              # 4×2 pixel coords in camera image

    # 2) Recover BEV image size (px) from the conversion factors
    bev_h_px, bev_w_px = infer_bev_px_size(bev_height_m, bev_width_m, unit_conv)

    # 3) Build the destination rectangle in BEV pixel space
    # Order matches your world-corner order in run_bev_config.py:
    #   0: far-left  -> (0, 0)
    #   1: near-left -> (0, bev_h-1)
    #   2: near-right-> (bev_w-1, bev_h-1)
    #   3: far-right -> (bev_w-1, 0)
    dst = np.float32([
        [0,                0],
        [0,                bev_h_px - 1],
        [bev_w_px - 1,     bev_h_px - 1],
        [bev_w_px - 1,     0]
    ])

    # Basic sanity checks
    if not np.isfinite(src).all():
        raise ValueError("Non-finite values detected in src points from bev_config.json")
    if src.shape != (4, 2):
        raise ValueError(f"Expected 4×2 src points, got {src.shape}")

    # 4) Load an input camera image
    img_path = pick_image(args.image)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    # 5) Compute homography and warp
    H = cv2.getPerspectiveTransform(src, dst)
    bev = cv2.warpPerspective(img, H, (bev_w_px, bev_h_px), flags=cv2.INTER_LINEAR)

    # 6) Visual aids
    img_src_vis = draw_src_points(img, src)
    bev_grid = overlay_grid(bev, step=50)

    # 7) Save previews
    out_src = os.path.join(args.save_dir, "camera_with_src_points.png")
    out_bev = os.path.join(args.save_dir, "bev_warp.png")
    out_bev_grid = os.path.join(args.save_dir, "bev_warp_with_grid.png")
    side_by_side = os.path.join(args.save_dir, "side_by_side.png")

    cv2.imwrite(out_src, img_src_vis)
    cv2.imwrite(out_bev, bev)
    cv2.imwrite(out_bev_grid, bev_grid)
    cv2.imwrite(side_by_side, np.hstack([
        cv2.resize(img_src_vis, (img_src_vis.shape[1], bev_h_px)),
        bev_grid
    ]))

    print(f"[OK] Saved:\n  {out_src}\n  {out_bev}\n  {out_bev_grid}\n  {side_by_side}")
    print("Tip: In a correct BEV, lane lines should look straight and parallel.")

if __name__ == "__main__":
    main()
