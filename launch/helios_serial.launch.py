from launch import LaunchDescription
    from launch_ros.actions import Node
    from launch.substitutions import LaunchConfiguration
    from launch.actions import DeclareLaunchArgument

    def generate_launch_description():
        # 声明参数
        port_name = LaunchConfiguration('port_name')
        baud_rate = LaunchConfiguration('baud_rate')
        model_path = LaunchConfiguration('model_path')
        calibration_file = LaunchConfiguration('calibration_file')
        
        return LaunchDescription([
            # 参数定义
            DeclareLaunchArgument(
                'port_name',
                default_value='/dev/ttyACM0',
                description='Serial port name'
            ),
            DeclareLaunchArgument(
                'baud_rate',
                default_value='921600',
                description='Serial baud rate'
            ),
            DeclareLaunchArgument(
                'model_path',
                default_value='/path/to/model.onnx',
                description='Path to the model file'
            ),
            DeclareLaunchArgument(
                'calibration_file',
                default_value='/path/to/calibration.yaml',
                description='Path to the calibration file'
            ),
            
            # 节点配置
            Node(
                package='helios_serial',
                executable='helios_serial_node',
                name='helios_serial',
                output='screen',
                parameters=[{
                    'port_name': port_name,
                    'baud_rate': baud_rate,
                    'model_path': model_path,
                    'calibration_file': calibration_file
                }]
            )
        ])
    