from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction, DeclareLaunchArgument, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command, FindExecutable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    # Declare all launch arguments
    namespace = LaunchConfiguration('namespace')
    cmd_timeout = LaunchConfiguration('cmd_timeout')
    paused = LaunchConfiguration('paused')
    use_sim_time = LaunchConfiguration('use_sim_time')
    gui = LaunchConfiguration('gui')
    headless = LaunchConfiguration('headless')
    debug = LaunchConfiguration('debug')
    verbose = LaunchConfiguration('verbose')
    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')
    z = LaunchConfiguration('z')
    yaw = LaunchConfiguration('yaw')

    # Declare launch arguments
    declare_namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='/',
        description='Namespace'
    )
    
    declare_cmd_timeout_arg = DeclareLaunchArgument(
        'cmd_timeout',
        default_value='0.5',
        description='Command timeout'
    )
    
    declare_paused_arg = DeclareLaunchArgument(
        'paused',
        default_value='false',
        description='Start gazebo paused'
    )
    
    declare_use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    declare_gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Start gazebo gui'
    )
    
    declare_headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Start gazebo in headless mode'
    )
    
    declare_debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Start gazebo in debug mode'
    )
    
    declare_verbose_arg = DeclareLaunchArgument(
        'verbose',
        default_value='false',
        description='Enable verbose output'
    )
    
    declare_world_name_arg = DeclareLaunchArgument(
        'world_name',
        default_value='smaller_track_with_starting_point_new.world',
        description='World name'
    )
    
    declare_x_arg = DeclareLaunchArgument(
        'x',
        default_value='0.0',
        description='X position'
    )
    
    declare_y_arg = DeclareLaunchArgument(
        'y',
        default_value='-98.0',
        description='Y position'
    )
    
    declare_z_arg = DeclareLaunchArgument(
        'z',
        default_value='1.0',
        description='Z position'
    )
    
    declare_yaw_arg = DeclareLaunchArgument(
        'yaw',
        default_value='0.0',
        description='Yaw rotation'
    )

    # Get package paths
    gem_gazebo_package = FindPackageShare('gem_gazebo')
    gem_description_package = FindPackageShare('gem_description')
    
    # Set Gazebo model path and resource path
    gazebo_model_path = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=[
            os.path.join(get_package_share_directory('gem_gazebo'), 'models'),
            ':',
            os.environ.get('GAZEBO_MODEL_PATH', '')
        ]
    )
    
    gazebo_resource_path = SetEnvironmentVariable(
        name='GAZEBO_RESOURCE_PATH',
        value=[
            os.path.join(get_package_share_directory('gem_description')),
            ':',
            os.path.join(get_package_share_directory('gem_gazebo')),
            ':',
            os.environ.get('GAZEBO_RESOURCE_PATH', '')
        ]
    )
    
    # Set OGRE plugin path
    ogre_plugin_path = SetEnvironmentVariable(
        name='GAZEBO_PLUGIN_PATH',
        value=[
            os.path.join(get_package_share_directory('gem_gazebo'), 'lib'),
            ':',
            os.environ.get('GAZEBO_PLUGIN_PATH', '')
        ]
    )
    
    # Get URDF via xacro
    xacro_file = PathJoinSubstitution([gem_description_package, 'urdf', 'gem.urdf.xacro'])
    
    # Setup robot description
    robot_description_content = Command([
        FindExecutable(name='xacro'), ' ', xacro_file, ' ',
        'use_sim_time:=', use_sim_time
    ])
    
    robot_description = {'robot_description': robot_description_content}
    
    # Get config file paths
    joint_config_path = PathJoinSubstitution([gem_gazebo_package, 'config', 'gem_joint_control_params_ros2.yaml'])
    ackermann_config_path = PathJoinSubstitution([gem_gazebo_package, 'config', 'gem_ackermann_control_params_ros2.yaml'])
    
    # Robot state publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    

    # Create a group with namespace
    namespace_group = GroupAction([
        # Push ROS namespace
        PushRosNamespace(namespace),
        
        # Robot state publisher
        robot_state_publisher_node,
        
        # Spawn Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('gazebo_ros'),
                    'launch',
                    'gazebo.launch.py'
                ])
            ]),
            launch_arguments={
                'world': PathJoinSubstitution([
                    FindPackageShare('gem_gazebo'),
                    'worlds',
                    LaunchConfiguration('world_name')
                ]),
                'debug': debug,
                'gui': gui,
                'paused': paused,
                'use_sim_time': use_sim_time,
                'headless': headless,
                'verbose': verbose
            }.items()
        ),

        # Spawn the model
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='gazebo_ros',
                    executable='spawn_entity.py',
                    arguments=[
                        '-entity', 'gem',
                        '-topic', 'robot_description',
                        '-x', x,
                        '-y', y,
                        '-z', z,
                        '-Y', yaw
                    ],
                    output='screen'
                )
            ]
        ),

        # Controller spawner for joint state broadcaster
        # Node(
        #     package='controller_manager',
        #     executable='spawner',
        #     arguments=['joint_state_broadcaster'],
        #     output='screen'
        # ),
        
        # # Controller spawner for wheel controllers
        # Node(
        #     package='controller_manager',
        #     executable='spawner',
        #     arguments=[
        #         'left_steering_controller',
        #         'right_steering_controller',
        #         'left_front_wheel_controller',
        #         'right_front_wheel_controller',
        #         'left_rear_wheel_controller',
        #         'right_rear_wheel_controller'
        #     ],
        #     output='screen'
        # ),

        # Ackermann controller
        Node(
            package='gem_gazebo',
            executable='gem_control.py',
            name='ackermann_controller',
            parameters=[
                {'cmd_timeout': cmd_timeout},
                ackermann_config_path
            ],
            output='screen'
        ),

        # Load controller configurations
        # Note: ros2_control_node is not needed when using GEM simulator
        # as it's already handled by the GEM launch files
        # Node(
        #     package='controller_manager',
        #     executable='ros2_control_node',
        #     parameters=[joint_config_path],
        #     output='screen'
        # ),

        # Joint state publisher
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            parameters=[
                {'rate': 0.1},
                {'use_gui': False}
            ],
            output='screen'
        ),
        

        # GEM sensor info
        Node(
            package='gem_gazebo',
            executable='gem_sensor_info.py',
            name='gem_sensor_info',
            output='screen'
        ),

        # Include GEM transforms
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('gem_gazebo'),
                    'launch',
                    'gem_transforms.launch.py'
                ])
            ])
        ),

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=[
                '-d', PathJoinSubstitution([
                    FindPackageShare('gem_description'),
                    'config_rviz',
                    'gem_mp3.rviz'
                ])
            ],
            output='screen'
        )
    ])

    return LaunchDescription([
        declare_namespace_arg,
        declare_cmd_timeout_arg,
        declare_paused_arg,
        declare_use_sim_time_arg,
        declare_gui_arg,
        declare_headless_arg,
        declare_debug_arg,
        declare_verbose_arg,
        declare_world_name_arg,
        declare_x_arg,
        declare_y_arg,
        declare_z_arg,
        declare_yaw_arg,
        gazebo_model_path,
        gazebo_resource_path,
        ogre_plugin_path,
        namespace_group
    ]) 
