#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import json
import numpy as np

from typing import List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from gazebo_msgs.srv import GetEntityState
from utils.util import quaternion_to_euler, get_roads_info, point_in_rectangle, get_transformation_matrix


class DetectionErrorCalculator(Node):
    def __init__(self, world_fn: str, config_fn: str):
        super().__init__('error_estimator')

        self.roads_info = get_roads_info(world_fn)
        self.scaled_poly = None
        self.pending_poly = False

        # Load BEV configuration
        try:
            with open(config_fn) as f:
                self.bev_config = json.load(f)
                self.scale = self.bev_config["unit_conversion_factor"]
                self.gem_x = self.bev_config["bev_world_dim"][0] / 2
                self.gem_y = self.bev_config["bev_world_dim"][1] + self.bev_config["bev_from_base_link"]
                self.get_logger().info(f"GEM origin in BEV: ({self.gem_x:.2f}, {self.gem_y:.2f})")
        except Exception as e:
            self.get_logger().error(f"Config not loaded: {e}")
            return

        # Subscribers
        self.create_subscription(Float32MultiArray, 'lane_fit/poly', self.poly_callback, 10)

        # Async service client
        self.cli = self.create_client(GetEntityState, '/get_entity_state')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('/get_entity_state service not available, waiting...')

        # Timer callback
        self.timer = self.create_timer(1.0 / 15.0, self.timer_callback)

    def poly_callback(self, msg):
        poly = list(msg.data)
        try:
            scaled_poly = poly.copy()
            scaled_poly[0] /= self.scale[0] ** 2
            scaled_poly[1] /= self.scale[0]
            scaled_poly = [c * self.scale[1] for c in scaled_poly]
            self.scaled_poly = scaled_poly
        except Exception as e:
            self.get_logger().error(f"Poly scaling error: {e}")

    def timer_callback(self):
        # Send async request to get model pose
        request = GetEntityState.Request()
        request.name = 'gem'
        request.reference_frame = 'world'

        future = self.cli.call_async(request)
        future.add_done_callback(self.handle_pose_response)

    def handle_pose_response(self, future):
        try:
            response = future.result()
            gem_pos = [response.state.pose.position.x, response.state.pose.position.y]
            q = response.state.pose.orientation
            roll, pitch, yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)

            # Compute predicted error
            ct_err, heading_err = self.compute_error(self.scaled_poly)

            # Compute ground truth error
            gt_ct_err, gt_heading_err = self.calculate_gt_error([gem_pos, yaw])

            self.get_logger().info(f"Predicted Cross Track Error: {ct_err:.4f} m")
            self.get_logger().info(f"Predicted Heading Error:     {heading_err:.4f} rad")
            self.get_logger().info(f"Ground Truth Cross Track:    {gt_ct_err:.4f} m")
            self.get_logger().info(f"Ground Truth Heading Error:  {gt_heading_err:.4f} rad\n")

        except Exception as e:
            self.get_logger().warn(f"Failed to get model state: {e}")

    def compute_error(self, scaled_poly):
        a, b, c = scaled_poly
        scaled_poly1d = np.poly1d(scaled_poly)
        roots = np.roots([
            -2 * a ** 2,
            -3 * a * b,
            2 * a * self.gem_x - b ** 2 - 2 * a * c - 1,
            b * (self.gem_x - c) + self.gem_y
        ])
        real_root = [r.real for r in roots if abs(r.imag) < 1e-9][0]  # Take first real root
        x_root = scaled_poly1d(real_root)
        root = (x_root, real_root)

        heading_error = (2 * a * real_root + b)
        cross_track_error = np.sqrt((root[0] - self.gem_x) ** 2 + (root[1] - self.gem_y) ** 2)

        heading_vec = np.array([heading_error,1,0])
        cross_track_vec = np.array([self.gem_x-root[0],self.gem_y-root[1],0])
        if np.cross(heading_vec, cross_track_vec)[2] > 0.0:
            cross_track_error *= -1
        return cross_track_error, heading_error

    def get_cur_road(self, gem_pos):
        for road, info in self.roads_info.items():
            global_corners = info["corners"]["global"]
            if point_in_rectangle(global_corners, gem_pos):
                return road
        self.get_logger().warn("GEM is not on any road!")
        return None

    def calculate_gt_error(self, gem_pose: List[float]):
        gem_pos, gem_yaw = gem_pose
        print(gem_pos)
        road = self.get_cur_road(gem_pos)
        if not road:
            return 0.0, 0.0  # fallback

        road_info = self.roads_info[road]
        T = get_transformation_matrix(road_info["pose"], from_world=True)
        gem_pos_hom = np.array(gem_pos + [1]).reshape(3, 1)
        rel_gem_pos = (T @ gem_pos_hom)[:2].flatten().tolist()
        road_yaw = road_info["pose"][-1]

        if road_info["shape"] == "straight":
            local_corner = road_info["corners"]["local"]
            center_line = (local_corner[0][1] + local_corner[1][1]) / 2
            lane_width = 4.5
            lanes_centers = [center_line + lane_width / 2, center_line - lane_width / 2]
            lane_deltas = [lane_center - rel_gem_pos[1] for lane_center in lanes_centers]
        elif road_info["shape"] == "curved":
            radius = road_info["size"] + 11 / 2
            quarter_coord = [rel_gem_pos[0] + radius / 2, rel_gem_pos[1] - radius / 2]
            gem_radius_delta = math.sqrt(quarter_coord[0] ** 2 + quarter_coord[1] ** 2)
            road_yaw = math.atan(-quarter_coord[0] / quarter_coord[1]) + road_info["pose"][-1]
            road_center = road_info["size"]
            lane_width = 4.5
            lanes_centers = (road_center - lane_width / 2, road_center + lane_width / 2)
            lane_deltas = [gem_radius_delta - lane_center for lane_center in lanes_centers]
        else:
            return 0.0, 0.0

        abs_heading_delta = np.pi - abs(abs(road_yaw - gem_yaw) - np.pi)
        if abs_heading_delta > np.pi / 2:
            lane_deltas = [-x for x in lane_deltas]
            lane_deltas.reverse()
        cur_lane_idx = 0 if abs(lane_deltas[0]) < abs(lane_deltas[1]) else 1
        tilted = "right" if lane_deltas[cur_lane_idx] > 0 else "left"
        cross_track_error = lane_deltas[cur_lane_idx]
        heading_error = (road_yaw - gem_yaw + np.pi / 2) % np.pi - np.pi / 2
        # print(f"GEM is on the {tilted} side of lane {cur_lane_idx} with distance from the lane center: {lane_deltas[cur_lane_idx]} (GAZEBO unit).")
        return cross_track_error, heading_error



def main(args=None):
    rclpy.init(args=args)
    
    # Import here to avoid issues if ament_index_python is not available
    try:
        from ament_index_python.packages import get_package_share_directory
        import os
        
        # Get gem_gazebo package directory
        gem_gazebo_share = get_package_share_directory('gem_gazebo')
        world_fn = os.path.join(gem_gazebo_share, 'worlds', 'smaller_track.world')
        
        # Check if the world file exists, fallback to other world files if needed
        if not os.path.exists(world_fn):
            # Try alternative world file
            world_fn = os.path.join(gem_gazebo_share, 'worlds', 'smaller_track_with_starting_point_new.world')
            if not os.path.exists(world_fn):
                print(f"Warning: Could not find world files in {gem_gazebo_share}/worlds/")
                print("Available world files:")
                worlds_dir = os.path.join(gem_gazebo_share, 'worlds')
                if os.path.exists(worlds_dir):
                    for f in os.listdir(worlds_dir):
                        if f.endswith('.world'):
                            print(f"  - {f}")
                    # Use the first available world file
                    world_files = [f for f in os.listdir(worlds_dir) if f.endswith('.world')]
                    if world_files:
                        world_fn = os.path.join(worlds_dir, world_files[0])
                        print(f"Using: {world_files[0]}")
                    else:
                        world_fn = None
                else:
                    world_fn = None
        
        # Config file - try to find it relative to the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_fn = os.path.join(script_dir, '../data/bev_config.json')
        print(script_dir)
        print(config_fn)
        if not os.path.exists(config_fn):
            # Try in parent directory
            config_fn = os.path.join(os.path.dirname(script_dir), 'bev_coords.json')
            if not os.path.exists(config_fn):
                config_fn = 'bev_config.json'  # fallback to current directory
        
    except ImportError:
        print("Warning: ament_index_python not available, using fallback paths")
        world_fn = '/opt/ros/humble/share/gem_gazebo/worlds/smaller_track.world'
        config_fn = 'bev_pixel_coords.json'
    except Exception as e:
        print(f"Error finding gem_gazebo package: {e}")
        print("Using fallback paths")
        world_fn = '/opt/ros/humble/share/gem_gazebo/worlds/smaller_track.world'
        config_fn = 'bev_pixel_coords.json'
    
    try:
        if world_fn is None:
            print("Error: Could not find a valid world file. Please check gem_gazebo package installation.")
            return
            
        print(f"Using world file: {world_fn}")
        print(f"Using config file: {config_fn}")
        
        error_calculator = DetectionErrorCalculator(world_fn, config_fn)
        rclpy.spin(error_calculator)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'error_calculator' in locals():
            error_calculator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
