#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Launch arguments
    entity_name_arg = DeclareLaunchArgument(
        'entity_name',
        default_value='gem',
        description='Name of the entity/model to track'
    )
    
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='50.0',
        description='Publish rate in Hz'
    )
    
    use_entity_state_arg = DeclareLaunchArgument(
        'use_entity_state',
        default_value='true',
        description='Use get_entity_state service (true) or get_model_state service (false)'
    )
    
    # Odometry publisher node
    odometry_publisher = Node(
        package='gem_gazebo',
        executable='odometry_publisher.py',
        name='gem_odometry_publisher',
        parameters=[{
            'entity_name': LaunchConfiguration('entity_name'),
            'publish_rate': LaunchConfiguration('publish_rate'),
            'use_entity_state': LaunchConfiguration('use_entity_state')
        }],
        output='screen'
    )
    
    return LaunchDescription([
        entity_name_arg,
        publish_rate_arg,
        use_entity_state_arg,
        odometry_publisher,
    ]) 