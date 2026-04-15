import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from nav2_msgs.action import NavigateToPose


class MissionNode(Node):
    def __init__(self):
        super().__init__('mission_node')

        # TODO: replace these with your real 7 points
        self.candidate_locations = [
            {'id': 1, 'x': 0.0, 'y': 0.0, 'yaw': 0.0},
            {'id': 2, 'x': 1.0, 'y': 0.0, 'yaw': 0.0},
            {'id': 3, 'x': 1.0, 'y': 1.0, 'yaw': 1.57},
        ]

        self.current_index = 0
        self.latest_vision_status = 'empty'
        self.inspection_timer = None
        self.detected_results = []

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.marker_pub = self.create_publisher(MarkerArray, '/detected_markers', 10)

        self.vision_sub = self.create_subscription(
            String,
            '/vision_status',
            self.vision_callback,
            10
        )

        self.get_logger().info('Waiting for navigate_to_pose action server...')
        self.nav_client.wait_for_server()
        self.get_logger().info('Connected to Nav2 action server.')

        self.go_to_next_location()

    def vision_callback(self, msg: String) -> None:
        self.latest_vision_status = msg.data.strip().lower()

    def go_to_next_location(self) -> None:
        if self.current_index >= len(self.candidate_locations):
            self.get_logger().info('Mission complete.')
            self.publish_all_markers()
            return

        location = self.candidate_locations[self.current_index]
        self.get_logger().info(f"Navigating to location {location['id']}")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = location['x']
        goal_msg.pose.pose.position.y = location['y']

        qz, qw = self.yaw_to_quaternion(location['yaw'])
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        send_goal_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        # You can uncomment if you want lots of logs
        # self.get_logger().info('Received feedback from Nav2')
        pass

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected.')
            self.record_result('empty')
            self.current_index += 1
            self.go_to_next_location()
            return

        self.get_logger().info('Goal accepted.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        _result = future.result().result
        self.get_logger().info('Goal reached. Waiting briefly before inspection...')

        if self.inspection_timer is not None:
            self.inspection_timer.cancel()

        self.inspection_timer = self.create_timer(2.0, self.inspect_current_location_once)

    def inspect_current_location_once(self):
        if self.inspection_timer is not None:
            self.inspection_timer.cancel()
            self.inspection_timer = None

        location = self.candidate_locations[self.current_index]
        status = self.latest_vision_status

        if status not in ('authorized', 'intruder', 'empty'):
            status = 'empty'

        self.get_logger().info(
            f"Inspection at location {location['id']}: {status}"
        )

        self.record_result(status)
        self.publish_all_markers()

        self.current_index += 1
        self.go_to_next_location()

    def record_result(self, status: str) -> None:
        location = self.candidate_locations[self.current_index]
        self.detected_results.append({
            'id': location['id'],
            'x': location['x'],
            'y': location['y'],
            'status': status
        })

    def publish_all_markers(self) -> None:
        marker_array = MarkerArray()

        for i, item in enumerate(self.detected_results):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'detected_people'
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = item['x']
            marker.pose.position.y = item['y']
            marker.pose.position.z = 0.3
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.25
            marker.scale.y = 0.25
            marker.scale.z = 0.6

            if item['status'] == 'authorized':
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                marker.color.a = 1.0
            elif item['status'] == 'intruder':
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
            else:
                marker.color.r = 0.5
                marker.color.g = 0.5
                marker.color.b = 0.5
                marker.color.a = 1.0

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    @staticmethod
    def yaw_to_quaternion(yaw: float):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return qz, qw


def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
