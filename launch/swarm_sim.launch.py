import os
import math
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    
    # Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    robot_count_arg = DeclareLaunchArgument('robot_count', default_value='3', description='Number of robots to spawn')
    headless_arg = DeclareLaunchArgument('headless', default_value='False', description='Run Gazebo without GUI (headless)')
    
    # World
    world_file_name = 'turtlebot3_world.world'
    world = os.path.join(pkg_turtlebot3_gazebo, 'worlds', world_file_name)

    # Gazebo Server (gzserver)
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )

    # Gazebo Client (gzclient) - Conditional launch based on 'headless' argument
    # Note: LaunchDescription conditionals are verbose, so we use a simple Python check in OpaqueFunction or just launch it unless blocked.
    # For simplicity in this script, we'll conditionally add it in the main LD construction if we could, 
    # but since LaunchConfiguration is resolved at runtime, we use a trick or just always define it and conditionally add.
    # However, to keep it simple and robust: we will use OpaqueFunction to handle the conditional logic for gzclient too if needed, 
    # OR just use the standard pattern where we let the user kill it if they want, OR providing a 'gui' arg.
    
    # Let's use the 'gui' argument pattern standard in ros launch files
    gui_arg = DeclareLaunchArgument('gui', default_value='true', description='Run Gazebo GUI')

    def launch_gzclient(context, *args, **kwargs):
        if LaunchConfiguration('gui').perform(context).lower() == 'true':
            return [IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
                )
            )]
        return []

    ld = LaunchDescription()
    ld.add_action(robot_count_arg)
    ld.add_action(gui_arg)
    ld.add_action(gzserver_cmd)
    ld.add_action(OpaqueFunction(function=launch_gzclient))
    
    # Dynamic spawning function with STAGGERED delays
    def spawn_robots(context, *args, **kwargs):
        count_str = LaunchConfiguration('robot_count').perform(context)
        count = int(count_str)
        
        actions = []
        model = os.environ.get('TURTLEBOT3_MODEL', 'burger')
        sdf_path = os.path.join(pkg_turtlebot3_gazebo, 'models', 'turtlebot3_' + model, 'model.sdf')
        
        # Grid layout settings
        spacing = 1.0
        cols = int(math.ceil(math.sqrt(count)))
        start_x = -2.0
        start_y = -2.0
        
        print(f"--- Spawning {count} robots in {cols}x{math.ceil(count/cols)} grid (Staggered) ---")

        for i in range(count):
            name = f'tb3_{i}'
            row = i // cols
            col = i % cols
            x = start_x + col * spacing
            y = start_y + row * spacing
            
            # Stagger delay: 1.5s per robot to prevent service storms
            delay = 2.0 + (i * 1.5)
            
            # Spawn Entity
            spawn_cmd = Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-entity', name,
                    '-file', sdf_path,
                    '-x', str(x),
                    '-y', str(y),
                    '-z', '0.01',
                    '-robot_namespace', name
                ],
                output='screen'
            )
            
            # Robot State Publisher
            urdf_path = os.path.join(pkg_turtlebot3_gazebo, 'urdf', 'turtlebot3_' + model + '.urdf')
            if not os.path.exists(urdf_path):
                 try:
                     pkg_turtlebot3_description = get_package_share_directory('turtlebot3_description')
                     urdf_path = os.path.join(pkg_turtlebot3_description, 'urdf', 'turtlebot3_' + model + '.urdf')
                 except:
                     pass

            if os.path.exists(urdf_path):
                rsp_cmd = Node(
                    package='robot_state_publisher',
                    executable='robot_state_publisher',
                    namespace=name,
                    output='screen',
                    parameters=[{
                        'use_sim_time': use_sim_time,
                        'robot_description': open(urdf_path, 'r').read(),
                        'publish_frequency': 5.0 # Low frequency for swarm performance
                    }],
                    remappings=[
                        ('/tf', 'tf'),
                        ('/tf_static', 'tf_static')
                    ]
                )
                actions.append(TimerAction(period=delay, actions=[spawn_cmd, rsp_cmd]))
            else:
                actions.append(TimerAction(period=delay, actions=[spawn_cmd]))
                
        return actions

    ld.add_action(OpaqueFunction(function=spawn_robots))
    return ld
