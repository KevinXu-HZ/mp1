#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import argparse
import sys
import os
import signal
import atexit

# Add the current directory to the Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import GetEntityState
from controller import vehicleController
import time
from waypoint_list import WayPoints
from util import euler_to_quaternion, quaternion_to_euler

class MP2Node(Node):
    def __init__(self):
        super().__init__('mp2_node')
        print("MP2Node: Initializing...")
        
        # Create service client for getting entity state
        self.get_entity_state_client = self.create_client(GetEntityState, '/get_entity_state')
        print("MP2Node: Service client created")
        
        # Pass this node to the controller so it can use the same node
        self.controller = vehicleController(node=self)
        print("MP2Node: Controller created")
        
        # Create direct publisher for emergency stop
        from ackermann_msgs.msg import AckermannDrive
        self.emergency_stop_pub = self.create_publisher(AckermannDrive, "/ackermann_cmd", 1)
        
        # Flag to track if we're shutting down
        self.shutting_down = False

        waypoints = WayPoints()
        self.pos_list = waypoints.getWayPoints()
        self.pos_idx = 1

        self.target_x, self.target_y = self.pos_list[self.pos_idx]

        # Set up timer to run main loop at 100 Hz
        self.timer = self.create_timer(0.01, self.run_loop)
        self.start_time = self.get_clock().now()
        self.prev_wp_time = self.start_time
    
    def get_entity_state(self):
        """Get entity state using this node's service client with callback"""
        
        # Wait for service to be available
        while not self.get_entity_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for get_entity_state service...')
        
        # Create request
        request = GetEntityState.Request()
        request.name = 'gem'  # Name of the vehicle entity in Gazebo
        request.reference_frame = 'world'
        
        # Print the request
        
        # Call service with callback
        self.future = self.get_entity_state_client.call_async(request)
        self.future.add_done_callback(self.entity_state_callback)
        
        # Return None for now, data will be handled in callback
        return None

    def entity_state_callback(self, future):
        """Callback for entity state service response"""
        try:
            result = future.result()
            if result.success:
                pos = result.state.pose.position
                vel = result.state.twist.linear
                # Store the result for use in run_loop
                self.current_entity_state = result
            else:
                print("Service responded with success=False")
                self.current_entity_state = None
        except Exception as e:
            self.get_logger().error(f"Exception in service callback: {e}")
            self.current_entity_state = None

    def run_loop(self):
        
        # Check if we're shutting down and send stop command
        if self.shutting_down:
            self.controller.stop()
            return
            
        # Get the current position and orientation of the vehicle using callback
        self.get_entity_state()
        
        # Check if we have entity state data from callback
        if not hasattr(self, 'current_entity_state') or self.current_entity_state is None:
            print("No entity state data available yet")
            return
            
        currState = self.current_entity_state
        
        if not currState.success:
            print("Failed to get entity state")
            return
        

        # Compute relative position between vehicle and waypoints
        distToTargetX = abs(self.target_x - currState.state.pose.position.x)
        distToTargetY = abs(self.target_y - currState.state.pose.position.y)

        cur_time = self.get_clock().now()
        time_diff = (cur_time - self.prev_wp_time).nanoseconds / 1e9
        
        if time_diff > 4:
            self.get_logger().info(f"failure to reach {self.pos_idx}-th waypoint in time")
            return

        if (distToTargetX < 2 and distToTargetY < 2):
            # If the vehicle is close to the waypoint, move to the next waypoint
            prev_pos_idx = self.pos_idx
            self.pos_idx = self.pos_idx + 1

            if self.pos_idx == len(self.pos_list): #Reached all the waypoints
                self.get_logger().info("Reached all the waypoints")
                total_time = (cur_time - self.start_time).nanoseconds / 1e9
                self.get_logger().info(f"Total time: {total_time}")
                self.controller.stop()
                rclpy.shutdown()
                return

            self.target_x, self.target_y = self.pos_list[self.pos_idx]

            time_taken = time_diff
            self.prev_wp_time = cur_time
            self.get_logger().info(f"Time Taken: {round(time_taken, 2)} reached {self.pos_list[prev_pos_idx][0]} {self.pos_list[prev_pos_idx][1]} next {self.pos_list[self.pos_idx][0]} {self.pos_list[self.pos_idx][1]}")

        self.controller.execute(currState, [self.target_x, self.target_y], self.pos_list[self.pos_idx:])

    def stop_vehicle(self):
        """Stop the vehicle and ensure it stays stopped"""
        if not self.shutting_down:
            self.shutting_down = True
            print("Stopping vehicle...")
            self.controller.stop()
            print("Vehicle stopped")

    def destroy_node(self):
        # Stop the controller before destroying the node
        self.stop_vehicle()
        super().destroy_node()


# Global variable to track the node for signal handling
global_node = None

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"Main: Received signal {signum}, shutting down gracefully...")
    global global_node
    if global_node is not None:
        global_node.stop_vehicle()
        # Force shutdown after stopping
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    
    global global_node
    node = None
    try:
        node = MP2Node()
        global_node = node  # Set global reference for signal handler
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Register cleanup function
        atexit.register(lambda: node.stop_vehicle() if node else None)
        
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Main: Keyboard interrupt received")
    except Exception as e:
        print(f"Main: Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main() 