from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolo_detector',
            executable='gpt_vision_node',
            name='gpt_vision_node',
            output='screen',
            parameters=[
                {'openai_model': 'gpt-4.1-mini'}
            ],
            remappings=[
                ('/image_input_topic', '/camera/image_raw'),
            ]
        )
    ])
