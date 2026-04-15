from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Define the YOLO Node
    yolo_node = Node(
        package='yolo_detector',
        executable='yolo_node',
        name='yolo_detector_node',
        output='screen',
        # Remapping the image input topic
        remappings=[
            ('/image_input_topic', '/camera/image_raw')
        ],
        # Setting the rate_limit parameter
        parameters=[
            {'rate_limit': 8.0}
        ]
    )

    return LaunchDescription([
        yolo_node
    ])
