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

# Add the current directory to the Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from util import euler_to_quaternion, quaternion_to_euler
import time
def calculate_curvature(x1, y1, x2, y2,curr_yaw):
        # Calculate the curvature using three consecutive points
        # Points: (x1, y1), (x2, y2), (x3, y3)
        dx1 = x1
        dy1 = y1
        dx2 = x2
        dy2 = y2
        
        angle1 = math.atan2(dy1, dx1)
        angle2 = math.atan2(dy2, dx2)
        
        # The curvature is proportional to the change in angle
        
        return angle1,angle2
class vehicleController():

    def __init__(self, node=None):
        print("Controller: Initializing vehicleController...")
        self.node = node
        if self.node is None:
            # If no node provided, create one
            # if not rclpy.ok():
            #     rclpy.init()
            self.node = Node('vehicle_controller')
            self.own_node = True
        else:
            self.own_node = False
        print("Controller: vehicleController initialization complete")
            
        # Publisher to publish the control input to the vehicle model
        self.controlPub = self.node.create_publisher(AckermannDrive, "/ackermann_cmd", 1)
        self.prev_vel = 0
        self.L = 1.75 # Wheelbase, can be get from gem_control.py
        self.log_acceleration = False

        # Service calls are now handled by the main node to avoid executor conflicts
       


    # Tasks 1: Read the documentation https://docs.ros.org/en/fuerte/api/gazebo/html/msg/ModelState.html
    #       and extract yaw, velocity, vehicle_position_x, vehicle_position_y
    # Hint: you may use the the helper function(quaternion_to_euler()) we provide to convert from quaternion to euler
     
    def extract_vehicle_info(self, currentPose):
        """Extract vehicle position, velocity, and yaw from EntityState"""
        
        return pos_x, pos_y, vel, yaw



    def longititudal_controller(self, curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints):


        ####################### TODO: Your TASK 2 code starts Here #######################
        
        
        


    def pure_pursuit_lateral_controller(self, curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints):


        ####################### TODO: Your TASK 3 code starts Here #######################
       





    def execute(self, currentPose, target_point, future_unreached_waypoints):
        """Execute the control loop with longitudinal and lateral controllers"""
        
        # Check for valid inputs
        if currentPose is None:
            print("Warning: No current pose data")
            return
            
        if len(future_unreached_waypoints) == 0:
            print("Warning: No waypoints available")
            return

        # Extract vehicle state
        curr_x, curr_y, curr_vel, curr_yaw = self.extract_vehicle_info(currentPose)

        # Acceleration Profile (optional logging)
        if self.log_acceleration:
            acceleration = (curr_vel - self.prev_vel) * 100  # Since we are running at 100Hz
            print(f"Acceleration: {acceleration:.2f} m/s²")

        # Compute control commands
        target_velocity = self.longititudal_controller(curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints)
        target_steering = self.pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints)

        # Create and publish Ackermann command
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = float(target_velocity)
        newAckermannCmd.steering_angle = float(target_steering)

        # Publish the control command
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
            # if rclpy.ok():
            #     rclpy.shutdown() 