#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from ament_index_python.packages import get_package_share_directory

class SpawnCamera(Node):

    def __init__(self):
        super().__init__('spawn_camera')
        self.client = self.create_client(SpawnEntity, '/spawn_entity')
        
        # Wait for the service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
        
        # Get the path to the model
        package_share_dir = get_package_share_directory('gem_gazebo')
        model_path = os.path.join(package_share_dir, 'models', 'simple_camera', 'model.sdf')
        
        self.get_logger().info(f'Model path: {model_path}')
        
        # Read the model file
        try:
            with open(model_path, 'r') as f:
                model_xml = f.read()
        except Exception as e:
            self.get_logger().error(f'Error reading model file: {e}')
            return
            
        # Create the request
        self.req = SpawnEntity.Request()
        self.req.name = 'simple_camera'
        self.req.xml = model_xml
        self.req.robot_namespace = ''
        self.req.reference_frame = 'world'
        
        # Call the service
        self.future = self.client.call_async(self.req)
        self.future.add_done_callback(self.callback)
    
    def callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Successfully spawned camera')
            else:
                self.get_logger().error(f'Failed to spawn camera: {response.status_message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = SpawnCamera()
    try:
        rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    
    # Check if the service call succeeded
    if node.future.done():
        try:
            response = node.future.result()
            if response.success:
                print("Camera spawned successfully!")
            else:
                print(f"Failed to spawn camera: {response.status_message}")
        except Exception as e:
            print(f"Service call failed: {e}")
    else:
        print("Service call did not complete")
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 