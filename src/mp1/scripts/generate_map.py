#!/usr/bin/env python3
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'src'))

import cv2
import numpy as np
from utils.util import get_roads_info


### MAGIC NUMBERS ###
LANE_WIDTH = 4.5    # (m) from mesh analysis
LANE_THICKNESS = 4  # (px) lane thickness for map


def generate_lane_map(output_dir: str='data/lane_map', resolution: int=20):
    """
    Generate a 2D array representation of the map with lane lines marked
    """
    world_fn = os.path.join(script_dir,'..','..','gem_simulator/gem_gazebo/worlds/smaller_track.world')
    road_dict = get_roads_info(world_fn)
    os.makedirs(output_dir, exist_ok=True)
    for road in road_dict.keys():
        print(f"Processing road: {road}...")
        road_info = road_dict[road]
        save_fn = os.path.join(output_dir, f"{road_info['shape']}_{road_info['size']}m_{resolution}.png")
        if os.path.exists(save_fn):
            print("Skipping since file already exists.\n")
            continue
        (x_max, y_max), (x_min, y_min) = road_info["corners"]["local"]
        map_width, map_height = int((x_max-x_min) * resolution), int((y_max-y_min) * resolution)
        print(f"Map (width, height): {(map_width, map_height)}.")
        road_map = np.zeros((map_height, map_width), dtype=np.uint8)
        lane_width = int(LANE_WIDTH * resolution)
        if road_info['shape'] == "straight":
            center = map_height // 2
            road_map[center-LANE_THICKNESS//2:center+LANE_THICKNESS//2, :] = 255
            road_map[(center-lane_width)-LANE_THICKNESS//2:(center-lane_width)+LANE_THICKNESS//2, :] = 255
            road_map[(center+lane_width)-LANE_THICKNESS//2:(center+lane_width)+LANE_THICKNESS//2, :] = 255
        if road_info['shape'] == "curved":
            radius = road_info["size"] * resolution
            cv2.circle(road_map, (0,0), radius, color=255, thickness=LANE_THICKNESS)
            cv2.circle(road_map, (0,0), radius-lane_width, color=255, thickness=LANE_THICKNESS)
            cv2.circle(road_map, (0,0), radius+lane_width, color=255, thickness=LANE_THICKNESS)
        cv2.imwrite(save_fn, road_map)
        print(f"Map saved in file: {save_fn}.\n")


if __name__ == "__main__":
    generate_lane_map()
