#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
import numpy as np
from typing import List
from shapely.geometry import Polygon
from utils.util import get_roads_info, get_transformation_matrix


class GroundTruthGenerator:
    def __init__(self, world_fn: str, resolution: int):
        self.resolution = resolution
        self.road_info = get_roads_info(world_fn)
        self.lane_maps = load_lane_maps(self.resolution)
        # self.WIDTH, self.HEIGHT = 1280, 720
        self.WIDTH, self.HEIGHT = 800, 600


    def get_global_fov(self, vehicle_pose):
        relative_fov = [
            [2.46071108, 2.06219366], 
            [2.46071108, -2.06735559], 
            [1473.9659345, -1235.254], 
            [1473.9659345, 1238.346]
        ]
        T = get_transformation_matrix(vehicle_pose, from_world=False)
        pts = np.column_stack((relative_fov, np.ones(len(relative_fov))))
        global_fov = ((T @ pts.T)[:2].T).astype(np.float32)
        return global_fov

    def generate_ground_truth(self, vehicle_pose: List[float]):
        lane_mask = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.uint8)
        global_fov = self.get_global_fov(vehicle_pose)
        fov_polygon = Polygon(global_fov)
        for name, info in self.road_info.items():
            (x_max, y_max), (x_min, y_min) = info["corners"]["global"]
            road_polygon = Polygon([(x_max, y_max), (x_min, y_max), (x_min, y_min), (x_max, y_min)])
            # if fov intersects with the road, only then add it to lane mask
            if fov_polygon.intersects(road_polygon):
                # TODO: better organization in general
                T = get_transformation_matrix(info["pose"], from_world=True)
                A = info["lane_map_frame"]
                pts = np.column_stack((global_fov, np.ones(len(global_fov))))
                xy = (A @ T @ pts.T)[:2].T
                major_fov = (xy * self.resolution).astype(np.float32)
                # perspective transform from the fov to camera viwe
                src = np.array(major_fov)[:, [1, 0]].astype(np.float32)
                dst = np.float32([
                    [0.0, self.HEIGHT],
                    [self.WIDTH, self.HEIGHT],
                    [self.WIDTH, self.HEIGHT/2],
                    [0.0, self.HEIGHT/2],
                ])
                M = cv2.getPerspectiveTransform(src, dst)
                lane_map = self.lane_maps[info["shape"]][info["size"]].copy()
                warped_img = cv2.warpPerspective(lane_map, M, (self.WIDTH, self.HEIGHT))
                _, binary_mask = cv2.threshold(warped_img, 1, 255, cv2.THRESH_BINARY)
                lane_mask = cv2.bitwise_or(lane_mask, binary_mask)
        lane_mask[0:self.HEIGHT//2, ...] = 0
        return lane_mask


def load_lane_maps(resolution: int, lane_map_dir: str = "data/lane_map/"):
    maps = {
        "straight": {},
        "curved": {}
    }
    for f in os.listdir(lane_map_dir):
        fn = os.path.join(lane_map_dir, f)
        name, _ = os.path.splitext(f)
        shape, size, res = name.split("_")
        if int(res) == resolution:
            size = int(size[:-1])
            img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
            maps[shape][size] = img
    return maps
