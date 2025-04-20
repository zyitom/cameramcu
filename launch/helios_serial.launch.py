import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
def get_terminal_command():
    if os.environ.get('GNOME_DESKTOP_SESSION_ID'):
        return 'gnome-terminal --'
    elif os.environ.get('TERM_PROGRAM') == 'Apple_Terminal':
        return 'open -a Terminal --args'
    else:
        return 'x-terminal-emulator -e' 

def generate_launch_description():
    # Get the package directories
    pkg_share = get_package_share_directory('helios_serial')
    
    # Define default parameters
    default_port = '/dev/ttyACM0'
    default_baud = '921600'
    default_xyz = '0.125 0 -0.035'
    default_rpy = '0 0 0'
    
    # Launch arguments
    port = LaunchConfiguration('port')
    baud = LaunchConfiguration('baud')
    xyz = LaunchConfiguration('xyz')
    rpy = LaunchConfiguration('rpy')
    
    # Declare launch arguments
    declare_port_arg = DeclareLaunchArgument(
        'port',
        default_value=default_port,
        description='Serial port for communication'
    )
    
    declare_baud_arg = DeclareLaunchArgument(
        'baud',
        default_value=default_baud,
        description='Baud rate for serial communication'
    )
    
    declare_xyz_arg = DeclareLaunchArgument(
        'xyz',
        default_value=default_xyz,
        description='Camera position relative to pitch link'
    )
    
    declare_rpy_arg = DeclareLaunchArgument(
        'rpy',
        default_value=default_rpy,
        description='Camera orientation relative to pitch link'
    )
    
    # Create the robot description parameter using properly quoted arguments
    robot_description = Command([
        'xacro ', 
        os.path.join(pkg_share, 'description', 'urdf/gimbal_description.urdf.xacro'),  # Update path if needed
        ' xyz:="', xyz, '"',
        ' rpy:="', rpy, '"'
    ])
    
    # Robot state publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'publish_frequency': 1000.0
        }]
    )
    
    # Your serial example node
    serial_example_node = Node(
        package='helios_serial',
        executable='serial_example',
        name='serial_example',
        output='screen',
        # prefix=[f"{get_terminal_command()} gdb -ex run --args"]
    )
    
    # Return launch description
    return LaunchDescription([
        declare_port_arg,
        declare_baud_arg,
        declare_xyz_arg,
        declare_rpy_arg,
        robot_state_publisher,
        serial_example_node,
    ])