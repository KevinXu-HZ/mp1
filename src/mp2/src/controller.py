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
        target_steering = 0

        ####################### TODO: Your TASK 3 code starts Here #######################
        return target_steering
       





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