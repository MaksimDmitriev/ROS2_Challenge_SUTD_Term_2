import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


class MissionNode(Node):
    def __init__(self):
        super().__init__('mission_node')

        self.candidate_locations = [
            {"id": 1, "x": 1.230, "y": 0.203, "z": 0.018325, "w": 0.999832},
            {"id": 2, "x": 2.298, "y": -0.610, "z": -0.663858, "w": 0.747858},
            {"id": 3, "x": 2.126, "y": 0.174, "z": 0.741676, "w": 0.670758},
            {"id": 4, "x": 2.210, "y": 0.083, "z": 0.035623, "w": 0.999365},
            {"id": 5, "x": 1.087, "y": -1.942, "z": -0.693018, "w": 0.720920},
            {"id": 6, "x": 2.438, "y": -1.839, "z": 0.749281, "w": 0.662253},
            {"id": 7, "x": 2.492, "y": -1.800, "z": 0.042288, "w": 0.999105},
            # {"id": 8, "x": 2.492, "y": -1.800, "z": 0.042288, "w": 0.999105}, TODO: fix, it's the same as id=7
        ]
        
        self.current_index = 0
        self.latest_vision_status = 'empty'
        self.inspection_timer = None
        self.detected_results = []
        self.last_completed_location_id = None
        self.current_goal_handle = None
        self.pending_stop_directive = None
        self.stop_sign_action_in_progress = False

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
        self.get_logger().info(
            f"vision_callback: {self.latest_vision_status}"
        )
        if self.latest_vision_status == 'stop_sign':
            self.handle_stop_sign_event()

    def find_location_index_by_id(self, location_id: int):
        for idx, location in enumerate(self.candidate_locations):
            if location['id'] == location_id:
                return idx
        return None

    def handle_stop_sign_event(self) -> None:
        if self.current_index >= len(self.candidate_locations):
            return

        current_target_id = self.candidate_locations[self.current_index]['id']

        if self.last_completed_location_id == 1 and current_target_id == 2:
            self.request_stop_sign_directive('skip_to_5')
        elif self.last_completed_location_id == 5 and current_target_id == 6:
            self.request_stop_sign_directive('finish_mission')

    def request_stop_sign_directive(self, directive: str) -> None:
        if self.pending_stop_directive == directive:
            return

        self.pending_stop_directive = directive
        self.get_logger().warn(f'Stop sign detected. Pending directive: {directive}')

        if self.current_goal_handle is not None and not self.stop_sign_action_in_progress:
            self.cancel_current_goal_for_stop_sign()

    def cancel_current_goal_for_stop_sign(self) -> None:
        if self.current_goal_handle is None or self.stop_sign_action_in_progress:
            return

        self.stop_sign_action_in_progress = True
        cancel_future = self.current_goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self.stop_sign_cancel_done_callback)

    def stop_sign_cancel_done_callback(self, _future) -> None:
        self.stop_sign_action_in_progress = False
        directive = self.pending_stop_directive
        self.pending_stop_directive = None
        self.current_goal_handle = None

        if directive == 'skip_to_5':
            idx = self.find_location_index_by_id(5)
            if idx is None:
                self.get_logger().error('Location id=5 not found. Ending mission.')
                self.current_index = len(self.candidate_locations)
                self.go_to_next_location()
                return

            self.current_index = idx
            self.get_logger().warn('Stop sign between 1->2. Skipping ids 2,3,4 and rerouting to id=5.')
            self.go_to_next_location()
        elif directive == 'finish_mission':
            self.get_logger().warn('Stop sign between 5->6. Finishing mission immediately.')
            self.current_index = len(self.candidate_locations)
            self.go_to_next_location()

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

        goal_msg.pose.pose.orientation.z = location['z']
        goal_msg.pose.pose.orientation.w = location['w']

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

        self.current_goal_handle = goal_handle
        self.get_logger().info('Goal accepted.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

        if self.pending_stop_directive is not None and not self.stop_sign_action_in_progress:
            self.cancel_current_goal_for_stop_sign()

    def goal_result_callback(self, future):
        result_wrapper = future.result()
        status_code = result_wrapper.status
        self.current_goal_handle = None

        if status_code == GoalStatus.STATUS_CANCELED:
            self.get_logger().info('Navigation goal canceled after stop-sign detection.')
            return

        if status_code != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().warn('Navigation did not succeed. Recording empty result and continuing.')
            self.record_result('empty')
            self.current_index += 1
            self.go_to_next_location()
            return

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
        self.last_completed_location_id = location['id']
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
            self.get_logger().info(
                f"publish_all_markers: {item['status']}"
            )
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
            self.get_logger().info(
                f"[MARKER] id={item['id']} "
                f"status={item['status']} "
                f"pos=({item['x']:.2f}, {item['y']:.2f}) "
                f"color={'BLUE' if item['status']=='authorized' else 'RED' if item['status']=='intruder' else 'GRAY'}"
            )


        self.marker_pub.publish(marker_array)


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
