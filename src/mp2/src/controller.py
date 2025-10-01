#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetEntityState
from gazebo_msgs.msg import EntityState
from ackermann_msgs.msg import AckermannDrive
import numpy as np
from std_msgs.msg import Float32MultiArray
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from util import euler_to_quaternion, quaternion_to_euler
import time


class vehicleController():

    def __init__(self, node=None):
        self.node = Node('vehicle_controller')
        self.own_node = True
            
        self.controlPub = self.node.create_publisher(AckermannDrive, "/ackermann_cmd", 1)
        self.prev_vel = 0
        self.L = 1.75 
        self.log_acceleration = False

       


    # Tasks 1: Read the documentation https://docs.ros.org/en/ros2_packages/humble/api/simulation_interfaces/srv/GetEntityState.html 
    #       and https://docs.ros.org/en/ros2_packages/humble/api/gazebo_msgs/msg/EntityState.html
    #       and extract yaw, velocity, vehicle_position_x, vehicle_position_y
    # Hint: you may use the the helper function(quaternion_to_euler()) we provide to convert from quaternion to euler
     
    def extract_vehicle_info(self, currentPose):

        ####################### TODO: Your TASK 1 code starts Here #######################
        pos_x, pos_y, vel, yaw = 0, 0, 0, 0
        if currentPose.success:
            print("gem found")
    
            # get pos_x and pos_y
            point = currentPose.state.pose.position
            pos_x = point.x
            pos_y = point.y
    
            # get yaw
            orientation = currentPose.state.pose.orientation
            quat = [orientation.x, orientation.y, orientation.z, orientation.w]
            euler = quaternion_to_euler(quat)
            yaw = euler[2]
    
            #get vel
            twist = currentPose.state.twist
            vel_x = twist.linear.x
            vel_y = twist.linear.y
            # vel_z = twist.linear.z
            vel = (vel_x ** 2 + vel_y ** 2) ** 0.5
        else:
            print("error detected")
    
        ####################### TODO: Your Task 1 code ends Here #######################
    
        return pos_x, pos_y, vel, yaw # note that yaw is in radians


    # Task 2: Longtitudal Controller
    # Based on all unreached waypoints, and your current vehicle state, decide your velocity

    def longititudal_controller(self, curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints):

        ####################### TODO: Your TASK 2 code starts Here #######################
        target_velocity = 10


        ####################### TODO: Your TASK 2 code ends Here #######################
        return target_velocity

        
        

    # Task 3: Lateral Controller (Pure Pursuit)
    def pure_pursuit_lateral_controller(self, curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints):
       
        ####################### TODO: Your TASK 3 code starts Here #######################
        # Pure Pursuit Algorithm Implementation
        # Using method 2: Set a constant lookahead distance and interpolate between future waypoints
        
        # Lookahead distance (can be tuned for better performance)
        lookahead_distance = 8.0
        
        # Find the lookahead point
        lookahead_point = self._find_lookahead_point(curr_x, curr_y, target_point, future_unreached_waypoints, lookahead_distance)
        
        if lookahead_point is None:
            # Fallback to target point if no lookahead point found
            lookahead_point = target_point
        
        # Calculate the angle between vehicle heading and lookahead line
        dx = lookahead_point[0] - curr_x
        dy = lookahead_point[1] - curr_y
        
        # Calculate the angle from vehicle to lookahead point
        alpha = math.atan2(dy, dx) - curr_yaw
        
        # Normalize angle to [-pi, pi]
        while alpha > math.pi:
            alpha -= 2 * math.pi
        while alpha < -math.pi:
            alpha += 2 * math.pi
        
        # Calculate lookahead distance
        ld = math.sqrt(dx**2 + dy**2)
        
        # Pure Pursuit formula: δ = arctan(2L*sin(α)/ld)
        if ld > 0.1:  # Avoid division by zero
            target_steering = math.atan(2 * self.L * math.sin(alpha) / ld)
        else:
            target_steering = 0.0
        
        # Limit steering angle to reasonable bounds (typically ±30 degrees)
        max_steering = math.radians(30)
        target_steering = max(-max_steering, min(max_steering, target_steering))

        ####################### TODO: Your TASK 3 code ends Here #######################
        return target_steering
    
    def _find_lookahead_point(self, curr_x, curr_y, target_point, future_waypoints, lookahead_distance):
        """
        Find the lookahead point using constant lookahead distance.
        Interpolates between waypoints to find the exact point at lookahead_distance.
        """
        # Start with the target point and future waypoints
        all_waypoints = [target_point] + future_waypoints
        
        if len(all_waypoints) < 2:
            return target_point
        
        # Find the segment that contains the lookahead point
        cumulative_distance = 0.0
        prev_point = [curr_x, curr_y]
        
        for i, waypoint in enumerate(all_waypoints):
            # Calculate distance to current waypoint
            dx = waypoint[0] - prev_point[0]
            dy = waypoint[1] - prev_point[1]
            segment_distance = math.sqrt(dx**2 + dy**2)
            
            # Check if lookahead point is in this segment
            if cumulative_distance + segment_distance >= lookahead_distance:
                # Interpolate within this segment
                remaining_distance = lookahead_distance - cumulative_distance
                if segment_distance > 0:
                    ratio = remaining_distance / segment_distance
                    lookahead_x = prev_point[0] + ratio * dx
                    lookahead_y = prev_point[1] + ratio * dy
                    return [lookahead_x, lookahead_y]
                else:
                    return waypoint
            
            cumulative_distance += segment_distance
            prev_point = waypoint
        
        # If we've gone through all waypoints, return the last one
        return all_waypoints[-1]





    def execute(self, currentPose, target_point, future_unreached_waypoints):
        # Compute the control input to the vehicle according to the
        # current and reference pose of the vehicle
        # Input:
        #   currentPose: GetEntityState response, the current state of the vehicle
        #   target_point: [target_x, target_y]
        #   future_unreached_waypoints: a list of future waypoints[[target_x, target_y]]
        # Output: None

        
        if currentPose is None:
            print("Warning: No current pose data")
            return
            
        if len(future_unreached_waypoints) == 0:
            print("Warning: No waypoints available")
            return

        curr_x, curr_y, curr_vel, curr_yaw = self.extract_vehicle_info(currentPose)

        if self.log_acceleration:
            acceleration = (curr_vel - self.prev_vel) * 100  # Since we are running at 100Hz

        target_velocity = self.longititudal_controller(curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints)
        target_steering = self.pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints)

        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = float(target_velocity)
        newAckermannCmd.steering_angle = float(target_steering)

        self.controlPub.publish(newAckermannCmd)
        
        # Store current velocity for next iteration
        self.prev_vel = curr_vel

    def stop(self):
        """Stop the vehicle by setting speed to 0 and steering to 0"""
        try:
            newAckermannCmd = AckermannDrive()
            newAckermannCmd.speed = 0.0
            newAckermannCmd.steering_angle = 0.0
            self.controlPub.publish(newAckermannCmd)
            print("Controller: Stop command sent")
        except Exception as e:
            print(f"Controller: Error sending stop command: {e}")
        
    def destroy(self):
        if self.own_node:

            self.node.destroy_node()

