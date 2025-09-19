#!/usr/bin/env python3

#================================================================
# File name: gem_sensor_info.py                                                                  
# Description: show sensor info in Rviz                                                              
# Author: Hang Cui
# Email: hangcui3@illinois.edu                                                                     
# Date created: 06/10/2021                                                                
# Date last modified: 07/02/2021                                                          
# Version: 0.1                                                                    
# Usage: ros2 run gem_gazebo gem_sensor_info.py                                                                     
# Python version: 3.8                                                             
#================================================================

# Python Headers
import math
import random
import numpy as np

# ROS Headers
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import Twist, Vector3, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Float32, Header
from tf2_ros import Buffer, TransformListener
from transforms3d.euler import quat2euler  # You may need to install this package

# Gazebo Headers
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState


class GEMSensorInfo(Node):

    def __init__(self):
        super().__init__('gem_sensor_info')
          
        self.sensor_info_pub = self.create_publisher(MarkerArray, "/gem/sensor_info", 1)
        self.gps_sub = self.create_subscription(NavSatFix, "/gps/fix", self.gps_callback, 10)
        self.imu_sub = self.create_subscription(Imu, "/imu", self.imu_callback, 10)
        self.timer = self.create_timer(0.1, self.update_info)  # 10Hz timer

        self.sensor_marker_array = MarkerArray()
        self.init_marker()
        self.sensor_info_update = False

        self.lat = 0.0
        self.lon = 0.0
        self.alt = 0.0
        self.imu_yaw = 0.0

        self.x = 0.0
        self.y = 0.0
        self.x_dot = 0.0
        self.y_dot = 0.0
        self.gazebo_yaw = 0.0

        # Create a client for the GetModelState service
        self.get_model_state_client = self.create_client(GetModelState, '/gazebo/get_model_state')
        # while not self.get_model_state_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('/gazebo/get_model_state service not available, waiting...')


    def gps_callback(self, msg):
        self.lat = round(msg.latitude, 6)
        self.lon = round(msg.longitude, 6)
        self.alt = round(msg.altitude, 6)


    def imu_callback(self, msg):
        orientation_q = msg.orientation
        angular_velocity = msg.angular_velocity
        linear_accel = msg.linear_acceleration
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = quat2euler(orientation_list, 'sxyz')
        self.imu_yaw = round(yaw, 6)   


    def get_gem_state(self):
        request = GetModelState.Request()
        request.model_name = 'gem'
        
        future = self.get_model_state_client.call_async(request)
        # Wait for the result
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            model_state = future.result()
            x = model_state.pose.position.x
            y = model_state.pose.position.y

            orientation_q = model_state.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]

            # roll: x-axis, pitch: y-axis, yaw: z-axis
            (roll, pitch, yaw) = quat2euler(orientation_list, 'sxyz')

            x_dot = model_state.twist.linear.x
            y_dot = model_state.twist.linear.y

            return round(x, 3), round(y, 3), round(yaw, 3), round(x_dot, 3), round(y_dot, 3)
        else:
            self.get_logger().error('Failed to get model state')
            return 0.0, 0.0, 0.0, 0.0, 0.0

    def init_marker(self):
        marker = Marker()
        marker.header.frame_id = "base_footprint"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Position the text 2 meters in front of the vehicle
        marker.pose.position.x = 2.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 2.0
        marker.pose.orientation.w = 1.0
        
        # Set the scale of the text
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        
        # Set color (blue)
        marker.color.r = 25.0 / 255.0
        marker.color.g = 1.0
        marker.color.b = 240.0 / 255.0
        marker.color.a = 1.0
        
        # Set text
        marker.text = "Initializing sensor data..."
        
        # Add to marker array
        self.sensor_marker_array.markers.append(marker)

    def update_sensor_text(self, lat=0.0, lon=0.0, alt=0.0, imu_yaw=0.0, x=0.0, y=0.0, gazebo_yaw=0.0, x_dot=0.0, y_dot=0.0, f_vel=0.0):
        text = f"""----------------------
Sensor (Measurement):
Lat = {lat}
Lon = {lon}
Alt = {alt}
Yaw = {imu_yaw}
----------------------
Gazebo (Ground Truth):
X     = {x}
Y     = {y}
X_dot = {x_dot}
Y_dot = {y_dot}
F_vel = {f_vel}
Yaw   = {gazebo_yaw}
----------------------"""
        return text

    def update_info(self):
        self.x, self.y, self.gazebo_yaw, self.x_dot, self.y_dot = self.get_gem_state()
        f_vel = round(np.sqrt(self.x_dot**2 + self.y_dot**2), 3)

        # Update marker text
        sensor_text = self.update_sensor_text(
            lat=self.lat, lon=self.lon, alt=self.alt, imu_yaw=self.imu_yaw,
            x=self.x, y=self.y, gazebo_yaw=self.gazebo_yaw, 
            x_dot=self.x_dot, y_dot=self.y_dot, f_vel=f_vel
        )
        
        # Update timestamp
        self.sensor_marker_array.markers[0].header.stamp = self.get_clock().now().to_msg()
        self.sensor_marker_array.markers[0].text = sensor_text
        
        # Publish the marker array
        self.sensor_info_pub.publish(self.sensor_marker_array)
  
  
def main(args=None):
    rclpy.init(args=args)
    
    gem_sensor_info_node = GEMSensorInfo()
    
    try:
        rclpy.spin(gem_sensor_info_node)
    except KeyboardInterrupt:
        pass
    finally:
        gem_sensor_info_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()