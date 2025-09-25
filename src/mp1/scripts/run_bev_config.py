#!/usr/bin/env python3
import os
import json
import numpy as np


def main():
    # intrinsics (K) from /front_single_camera/camera_info
    K = np.array([
        [476.7030836014194, 0.0, 400.5],
        [0.0, 476.7030836014194, 300.5],
        [0.0, 0.0, 1.0]
    ])
    # extrinsics (R,t): base_footprint -> front_single_camera_link
    R = np.array([
        [0, -1, 0],
        [ 0, 0, 1],
        [ -1, 0, 0]
    ])
    t = np.array([0.160, -0.110, 1.546])

    # bird eye view image size (px); 
    # NOTE: DO NOT CHANGE THIS
    bev_img_height, bev_img_width = 600, 800

    # HYPERPARAMETERS: BEV rectangle configuration (m)
    # NOTE: Revert this to the original values for submission
    bev_height, bev_width, bev_margin= 10, 10, 3

    # px -> m conversion factor
    unit_conversion_factor = (bev_height/bev_img_height, bev_width/bev_img_width)
    bev_world_coords = np.float32([
        [bev_height, -bev_width/2, 0],
        [0, -bev_width/2, 0],
        [0, bev_width/2, 0],
        [bev_height, bev_width/2, 0],
    ])
    bev_world_coords[:, 0] += bev_margin

    # convert the bev_world_coords into pixel coordinates
    src = []
    for pt in bev_world_coords:
        ##### YOUR CODE STARTS HERE #####
        # First step: use extrinsic matrix to convert from world frame to camera frame
        # M_ext = np.array([
        #     [R[0,0], R[0,1], R[0,2], t[0]],
        #     [R[1,0], R[1,1], R[1,2], t[1]],
        #     [R[2,0], R[2,1], R[2,2], t[2]],
        #     [0,      0,      0,       1  ]
        # ])
        # pt = np.append(pt, 1)
        # pt = pt.reshape(-1, 1) # make pt a 4*1 matrix
        # bev_camera_coords = M_ext @ pt

        # #Second step: use intrinsic matrix to convert from camera frame to pixel coordinates
        # M_int = np.array([
        #     [K[0,0], K[0,1], K[0,2], 0],
        #     [K[1,0], K[1,1], K[1,2], 0],
        #     [K[2,0], K[2,1], K[2,2], 0],
        # ])

        # # pixel_coord = M_int @ (bev_camera_coords/bev_camera_coords[2,0])
        # # u = pixel_coord[0,0]
        # # v = pixel_coord[1,0]

        # pixel_coord = M_int @ bev_camera_coords
        # u = pixel_coord[0,0]/pixel_coord[2,0]
        # v = pixel_coord[1,0]/pixel_coord[2,0]
        # src.append([u, v])
        # ##### YOUR CODE ENDS HERE #####
        a = R @ (pt - t)
        b = K @ a
        print(b)
        u = b[0]/b[2]
        v = b[1]/b[2]
        print(u, v)
        src.append([u, v])
        pass
    src = np.float32(src)

    output = {
        "bev_world_dim": (bev_height, bev_width),
        "bev_from_base_link": bev_margin,
        "unit_conversion_factor": unit_conversion_factor,
        "src": src.tolist(),
    }
    # save config to json
    save_fn = 'data/bev_config.json'
    if not os.path.isdir('data/'):
        print(f"Data directory not found. Generating...")
        os.makedirs('data/', exist_ok=False)
    if os.path.isfile(save_fn):
        if input("File already exists. Overwrite? (y/n):").lower() != 'y':
            print("Exiting...")
            import sys
            sys.exit()
    with open(save_fn, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved BEV config to {save_fn}.")


if __name__ == "__main__":
    main()
