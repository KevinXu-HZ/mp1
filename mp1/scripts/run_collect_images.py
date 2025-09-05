#!/usr/bin/env python3
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'src'))

import cv2
import time
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from ackermann_msgs.msg import AckermannDrive
from sensor_msgs.msg import Image
from datetime import datetime
from pynput import keyboard
from PIL import Image as PILImage
from PIL.PngImagePlugin import PngInfo
from utils.util import quaternion_to_euler

from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelStates


class AckermannKeyboardControl(Node):
    def __init__(self):
        super().__init__('ackermann_keyboard_teleop')

        # Create directory for saving images if it doesn't exist
        self.image_save_dir = os.path.join(script_dir, '..', 'data', 'raw_images')        
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)
            self.get_logger().info(f"Created directory for saving images: {self.image_save_dir}")

        # Configure max speed and steering
        self.declare_parameter('max_speed', 5.0)
        self.declare_parameter('max_steering_angle', 0.8)
        self.declare_parameter('speed_increment', 0.5)
        self.declare_parameter('steering_increment', 0.2)

        self.max_speed = self.get_parameter('max_speed').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.speed_increment = self.get_parameter('speed_increment').value
        self.steering_increment = self.get_parameter('steering_increment').value

        # Initialize current speed and steering angle
        self.current_speed = 0.0
        self.current_steering_angle = 0.0
        
        # Track pressed keys
        self.keys_pressed = set()

        # CV Bridge for converting ROS images to OpenCV format
        self.bridge = CvBridge()
        self.latest_image = None
        
        # ROS subscribers
        self.camera_sub = self.create_subscription(Image, '/front_camera/image_raw', self.camera_callback, 10)
        self.gem_sub = self.create_subscription(ModelStates, '/model_states', self.gem_pose_callback, 10)
        
        # ROS publisher
        self.drive_pub = self.create_publisher(AckermannDrive, 'ackermann_cmd', 10)
        
        # Initialize keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        # Create timer for control updates
        self.timer = self.create_timer(0.02, self.timer_callback)  # 50 Hz

        # Keyboard instructions
        self.print_instructions()


    def camera_callback(self, msg):
        """Store the latest image from the camera"""
        self.latest_image = msg

    def gem_pose_callback(self, msg):
        try:
            index = msg.name.index('gem')
            pose = msg.pose[index]
            x = pose.position.x
            y = pose.position.y
            q = pose.orientation
            _, _, yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)
            self.cur_gem_pose = [x, y], yaw
        except ValueError:
            self.get_logger().info('Model not found.')

    def publish_drive_msg(self):
        """Publish the current speed and steering angle as an AckermannDrive message"""
        msg = AckermannDrive()
        msg.speed = self.current_speed
        msg.steering_angle = self.current_steering_angle
        msg.steering_angle_velocity = 0.0
        msg.acceleration = 0.0
        self.drive_pub.publish(msg)
    
    def on_press(self, key):
        """Callback for key press events"""
        try:
            if key.char == 'w':
                self.keys_pressed.add('w')
            elif key.char == 's':
                self.keys_pressed.add('s')
            elif key.char == 'a':
                self.keys_pressed.add('a')
            elif key.char == 'd':
                self.keys_pressed.add('d')
            elif key.char == 'c':
                self.save_image()
            elif key.char == 'q':
                self.listener.stop()
                rclpy.shutdown()
        except AttributeError:
            # handle special keys
            if key == keyboard.Key.up:
                self.keys_pressed.add('w')
            elif key == keyboard.Key.down:
                self.keys_pressed.add('s')
            elif key == keyboard.Key.left:
                self.keys_pressed.add('a')
            elif key == keyboard.Key.right:
                self.keys_pressed.add('d')
            elif key == keyboard.Key.space:
                self.keys_pressed.clear()
                self.current_speed = 0.0
                self.current_steering_angle = 0.0
    
    def on_release(self, key):
        """Callback for key release events"""
        try:
            if key.char == 'w':
                self.keys_pressed.discard('w')
            elif key.char == 's':
                self.keys_pressed.discard('s')
            elif key.char == 'a':
                self.keys_pressed.discard('a')
            elif key.char == 'd':
                self.keys_pressed.discard('d')
        except AttributeError:
            # handle special keys
            if key == keyboard.Key.up:
                self.keys_pressed.discard('w')
            elif key == keyboard.Key.down:
                self.keys_pressed.discard('s')
            elif key == keyboard.Key.left:
                self.keys_pressed.discard('a')
            elif key == keyboard.Key.right:
                self.keys_pressed.discard('d')

    def update_controls(self):
        """Update speed and steering based on currently pressed keys"""
        self.current_speed = 0.0
        self.current_steering_angle = 0.0
        if 'w' in self.keys_pressed:
            self.current_speed = self.max_speed
        elif 's' in self.keys_pressed:
            self.current_speed = -self.max_speed
        if 'a' in self.keys_pressed:
            self.current_steering_angle = self.max_steering_angle
        elif 'd' in self.keys_pressed:
            self.current_steering_angle = -self.max_steering_angle
    
    def save_image(self):
        """Save the current camera image"""
        if self.latest_image is None:
            self.get_logger().warn("No camera image available to save")
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(cv_image_rgb)
            [vehicle_x, vehicle_y], vehicle_yaw = self.cur_gem_pose
            # save image with vehicle pose as metadata
            metadata = PngInfo()
            metadata.add_text("vehicle_x", str(vehicle_x))
            metadata.add_text("vehicle_y", str(vehicle_y))
            metadata.add_text("vehicle_yaw", str(vehicle_yaw))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.image_save_dir, f"ackermann_image_{timestamp}.png")
            pil_image.save(filename, "PNG", pnginfo=metadata)
            self.get_logger().info(f"Image saved to: {filename}")
        except Exception as e:
            self.get_logger().error(f"Error saving image: {e}")
    
    def timer_callback(self):
        """Callback for the timer that updates and publishes controls"""
        self.update_controls()
        self.publish_drive_msg()

    def print_instructions(self):
        """Print control instructions"""
        instructions = """
Ackermann Drive Keyboard Teleop Control
----------------------------------------
Control Keys:
  w/↑ - Move forward (hold)
  s/↓ - Move backward (hold)
  a/← - Steer left (hold)
  d/→ - Steer right (hold)
  space - Emergency stop
  q - Quit
  c - Capture and save current camera image
  
Current Settings:
  Max Speed: {0} m/s
  Max Steering Angle: {1} rad
  Speed Increment: {2} m/s
  Steering Increment: {3} rad
        """.format(self.max_speed, self.max_steering_angle, self.speed_increment, self.steering_increment)
        print(instructions)

    def shutdown(self):
        """Clean shutdown function"""
        self.keys_pressed.clear()
        self.current_speed = 0.0
        self.current_steering_angle = 0.0
        self.publish_drive_msg()
        self.get_logger().info("Stopping vehicle and exiting...")
        self.listener.stop()


def main(args=None):
    rclpy.init(args=args)
    controller = AckermannKeyboardControl()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.shutdown()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
