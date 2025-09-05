# MP1 - GEM Vehicle Keyboard Control

This package provides keyboard control for the GEM vehicle in Gazebo simulation.

## Features

- Launch GEM vehicle in Gazebo simulation
- Direct keyboard control with WASD keys
- Camera image capture with vehicle pose metadata
- Real-time visualization in RViz
- Odometry and transform publishing
- Emergency stop functionality

## Building

1. Navigate to your workspace root:
   ```bash
   cd ~/Documents/UIUC-ECE-484-MP-Redesign
   ```

2. Build the workspace:
   ```bash
   colcon build --packages-select mp1
   ```

3. Source the workspace:
   ```bash
   source install/setup.bash
   ```

## Running

### Launch the complete system:

```bash
ros2 launch mp1 gem_keyboard_control.launch.py
```

This will launch:
- Gazebo with an empty world
- GEM vehicle spawned at (0, 0, 1)
- RViz for visualization
- Advanced keyboard controller with camera capture

### Control the vehicle:

When the system is running, you can control the vehicle directly with the keyboard:

- **W/↑**: Move forward (hold)
- **S/↓**: Move backward (hold)
- **A/←**: Steer left (hold)
- **D/→**: Steer right (hold)
- **Space**: Emergency stop
- **Q**: Quit
- **C**: Capture and save current camera image with vehicle pose metadata

### Manual launch (if needed):

If you prefer to launch components separately:

1. Launch Gazebo with GEM:
   ```bash
   ros2 launch gem_description gem_description.launch.py
   ros2 launch gazebo_ros gazebo.launch.py world:=src/mp1/worlds/empty.world
   ```

2. Spawn the vehicle:
   ```bash
   ros2 run gazebo_ros spawn_entity.py -entity gem -topic robot_description -x 0 -y 0 -z 1
   ```

3. Launch the keyboard controller:
   ```bash
   ros2 run mp1 keyboard_controller
   ```

4. Launch the keyboard controller:
   ```bash
   ros2 run mp1 keyboard_controller
   ```

5. Launch RViz:
   ```bash
   ros2 run rviz2 rviz2 -d src/mp1/config/mp1.rviz
   ```

## Topics

### Subscribed Topics:
- `/gem/front_single_camera/front_single_camera/image_raw` (sensor_msgs/Image): Camera feed for image capture

### Published Topics:
- `/ackermann_cmd` (ackermann_msgs/AckermannDriveStamped): Ackermann drive commands for the vehicle
- `/odom` (nav_msgs/Odometry): Vehicle odometry
- `/tf` and `/tf_static`: Transform frames

### Services:
- `/gazebo/get_model_state`: Get vehicle pose from Gazebo

## Configuration

### Vehicle Parameters:
- Max speed: 5.0 m/s (configurable)
- Max steering angle: 1.0 radians (57.3 degrees) (configurable)
- Speed increment: 0.5 m/s (configurable)
- Steering increment: 0.2 radians (configurable)

### Launch Arguments:
- `x`, `y`, `z`: Initial vehicle position
- `yaw`: Initial vehicle orientation
- `gui`: Enable/disable Gazebo GUI
- `paused`: Start Gazebo paused

Example:
```bash
ros2 launch mp1 gem_keyboard_control.launch.py x:=5.0 y:=3.0 z:=1.0 yaw:=1.57
```

## Troubleshooting

1. **Vehicle not visible in Gazebo**: Check that the GEM description package is properly built and sourced.

2. **No response to keyboard input**: Make sure the keyboard controller is running and you're pressing the correct keys. The terminal should show control instructions.

3. **Transform errors in RViz**: Ensure the odometry_publisher is running and the vehicle is spawned in Gazebo.

4. **Service not available**: Make sure Gazebo is running and the ROS 2 bridge is active.

## Dependencies

- `gem_description`: GEM vehicle description
- `gazebo_ros`: Gazebo ROS 2 bridge
- `cv_bridge`: Image conversion
- `sensor_msgs`: Sensor message types
- `python3-opencv`: OpenCV for image processing
- `python3-pil`: PIL for image saving
- `python3-pynput`: Keyboard input handling
- `rviz2`: Visualization tool
- `tf2_ros`: Transform library 