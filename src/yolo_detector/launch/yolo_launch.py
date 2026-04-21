from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolo_detector',
            executable='yolo_detector',
            name='yolo_detector',
            output='screen',
            parameters=[
                {'rate_limit': 1.0},
                {'model_file': 'yolov8n.pt'},
                {'target_roles': ['military', 'researcher', 'student', 'worker']}
            ],
            remappings=[
                ('/image_input_topic', '/camera/image_raw'),
                ('/image_output_topic', '/camera/image_annotated')
            ]
        )
    ])
