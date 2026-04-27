import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from std_srvs.srv import Trigger


class MissionNode(Node):
    def __init__(self):
        super().__init__('mission_node')

        self.candidate_locations = [
            {"id": 1, "x": 1.230, "y": 0.203, "z": 0.018325, "w": 0.999832}, # TODO: need to move closer
            {"id": 2, "x": 1.1780259609222412, "y": -0.2994805574417114, "z": 0.039863363147714895, "w": 0.9992051402382562},  # 1st stop sign
            {"id": 3, "x": 2.272017240524292, "y": -0.3983581066131592, "z": -0.7151382830534804, "w": 0.6989830013035511},
            {"id": 4, "x": 2.293062448501587, "y": -0.29181909561157227, "z": 0.745175003291093, "w": 0.6668689634929186},
            {"id": 5, "x": 2.168145179748535, "y": -0.15960049629211426, "z": -0.01837860941291974, "w": 0.9998310990942657},
            {"id": 6, "x": 1.087, "y": -1.942, "z": -0.693018, "w": 0.720920},
            {"id": 7, "x": 1.475852370262146, "y": -2.1543092727661133, "z": 0.033015992097299496, "w": 0.9994548235242207},  # 2nd stop sign
            {"id": 8, "x": 2.438, "y": -1.839, "z": 0.749281, "w": 0.662253},
            {"id": 9, "x": 2.492, "y": -1.800, "z": 0.042288, "w": 0.999105},
            {"id": 10, "x": 2.438, "y": -1.816, "z": -0.683622, "w": 0.729836},
            {"id": 11, "x": 0.017, "y": 0.066, "z": 0.004746, "w": 0.999989},
        ]
        
        self.current_index = 0
        self.latest_vision_status = 'empty'
        self.inspection_timer = None
        self.detected_results = []
        self.max_detected_people = 4
        self.base_location_id = 11
        self.return_to_base_triggered = False
        self.last_completed_location_id = None
        self.current_goal_handle = None
        self.pending_stop_directive = None
        self.stop_sign_action_in_progress = False
        self.latest_scan = None
        self.latest_pose = None
        self.warned_missing_scan = False
        self.warned_missing_pose = False
        self.wall_scan_half_window_rad = math.radians(35.0)
        self.marker_wall_offset_m = 0.10

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.marker_pub = self.create_publisher(MarkerArray, '/detected_markers', 10)

        self.vision_sub = self.create_subscription(
            String,
            '/vision_status',
            self.vision_callback,
            10
        )
        self.vision_client = self.create_client(Trigger, '/classify_current_frame')
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10
        )

        self.get_logger().info('Waiting for navigate_to_pose action server...')
        self.nav_client.wait_for_server()
        self.get_logger().info('Connected to Nav2 action server.')
        self.get_logger().info('Waiting for /classify_current_frame service...')
        while not self.vision_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('/classify_current_frame not available yet, waiting...')
        self.get_logger().info('Connected to vision classification service.')

        self.go_to_next_location()

    def vision_callback(self, msg: String) -> None:
        self.latest_vision_status = msg.data.strip().lower()
        self.get_logger().info(
            f"vision_callback: {self.latest_vision_status}"
        )
        if self.latest_vision_status == 'stop_sign' and self.current_goal_handle is not None:
            self.handle_stop_sign_event()

    def scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self.latest_pose = msg.pose.pose

    def find_location_index_by_id(self, location_id: int):
        for idx, location in enumerate(self.candidate_locations):
            if location['id'] == location_id:
                return idx
        return None

    def count_detected_people(self) -> int:
        return sum(
            1 for item in self.detected_results
            if item['status'] in ('authorized', 'intruder')
        )

    def maybe_return_to_base(self) -> bool:
        if self.return_to_base_triggered:
            return False

        detected_people = self.count_detected_people()
        if detected_people < self.max_detected_people:
            return False

        base_index = self.find_location_index_by_id(self.base_location_id)
        if base_index is None:
            self.get_logger().error(f'Base waypoint id={self.base_location_id} not found.')
            return False

        if self.current_index < len(self.candidate_locations):
            next_id = self.candidate_locations[self.current_index]['id']
            if next_id == self.base_location_id:
                return False

        self.return_to_base_triggered = True
        self.current_index = base_index
        self.get_logger().warn(
            f'Detected people reached {detected_people}/{self.max_detected_people}. '
            f'Skipping remaining waypoints and returning to base id={self.base_location_id}.'
        )
        return True

    def handle_stop_sign_event(self) -> None:
        if self.current_index >= len(self.candidate_locations):
            return

        current_target_id = self.candidate_locations[self.current_index]['id']

        if self.last_completed_location_id == 2 and current_target_id == 3:
            self.request_stop_sign_directive('skip_to_6')
        elif self.last_completed_location_id == 7 and current_target_id == 8:
            self.request_stop_sign_directive('skip_to_11')

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

        if directive == 'skip_to_6':
            idx = self.find_location_index_by_id(6)
            if idx is None:
                self.get_logger().error('Location id=6 not found. Ending mission.')
                self.current_index = len(self.candidate_locations)
                self.go_to_next_location()
                return

            self.current_index = idx
            self.get_logger().warn('Stop sign at id=2 path. Skipping ids 3,4,5 and rerouting to id=6.')
            self.go_to_next_location()
        elif directive == 'skip_to_11':
            idx = self.find_location_index_by_id(11)
            if idx is None:
                self.get_logger().error('Location id=11 not found. Ending mission.')
                self.current_index = len(self.candidate_locations)
                self.go_to_next_location()
                return

            self.get_logger().warn('Stop sign at id=7 path. Skipping ids 8,9,10 and jumping to id=11.')
            self.current_index = idx
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

        location = self.candidate_locations[self.current_index]
        if location['id'] == self.base_location_id:
            self.get_logger().info(
                f"Reached base id={self.base_location_id}. Skipping inspection."
            )
            self.last_completed_location_id = location['id']
            self.current_index += 1
            self.go_to_next_location()
            return

        self.get_logger().info('Goal reached. Waiting 2 seconds before single-frame inspection...')

        if self.inspection_timer is not None:
            self.inspection_timer.cancel()

        self.inspection_timer = self.create_timer(2.0, self.inspect_current_location_once)

    def inspect_current_location_once(self):
        if self.inspection_timer is not None:
            self.inspection_timer.cancel()
            self.inspection_timer = None

        if not self.vision_client.service_is_ready():
            self.get_logger().warn('Vision service unavailable at inspection time. Marking empty.')
            self.complete_inspection('empty')
            return

        request = Trigger.Request()
        future = self.vision_client.call_async(request)
        future.add_done_callback(self.vision_result_callback)

    def vision_result_callback(self, future):
        status = 'empty'
        try:
            response = future.result()
            candidate = response.message.strip().lower()
            if candidate in ('authorized', 'intruder', 'empty', 'stop_sign'):
                status = candidate
        except Exception as e:
            self.get_logger().error(f'Vision service call failed: {e}')
            status = 'empty'

        self.latest_vision_status = status
        self.complete_inspection(status)

    def complete_inspection(self, status: str) -> None:
        location = self.candidate_locations[self.current_index]
        if location['id'] in (2, 7) and status != 'stop_sign':
            self.get_logger().info(
                f"Checkpoint at id={location['id']}: no stop sign detected; ignoring person class ({status})."
            )
            status = 'empty'

        marker_status = status if status in ('authorized', 'intruder', 'empty') else 'empty'
        marker_x, marker_y = self.estimate_marker_position(location, marker_status)
        self.get_logger().info(
            f"Inspection at location {location['id']}: {marker_status}"
        )

        self.record_result(marker_status, marker_x, marker_y)
        self.last_completed_location_id = location['id']
        self.publish_all_markers()

        self.current_index += 1
        if status == 'stop_sign':
            if location['id'] == 2:
                idx = self.find_location_index_by_id(6)
                if idx is not None:
                    self.get_logger().warn('Stop sign detected at id=2. Skipping ids 3,4,5.')
                    self.current_index = idx
            elif location['id'] == 7:
                idx = self.find_location_index_by_id(11)
                if idx is not None:
                    self.get_logger().warn('Stop sign detected at id=7. Skipping ids 8,9,10 and moving to id=11.')
                    self.current_index = idx
        self.maybe_return_to_base()
        self.go_to_next_location()

    def record_result(self, status: str, marker_x: float = None, marker_y: float = None) -> None:
        location = self.candidate_locations[self.current_index]
        if marker_x is None:
            marker_x = location['x']
        if marker_y is None:
            marker_y = location['y']
        self.detected_results.append({
            'id': location['id'],
            'x': marker_x,
            'y': marker_y,
            'status': status
        })

    @staticmethod
    def quaternion_to_yaw(z: float, w: float) -> float:
        # For planar robot orientation (x=y=0 quaternion components).
        return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)

    def estimate_marker_position(self, location: dict, status: str) -> tuple[float, float]:
        # Keep empty detections at waypoint/fallback; no marker is published for empty anyway.
        if status not in ('authorized', 'intruder'):
            return location['x'], location['y']

        if self.latest_scan is None:
            if not self.warned_missing_scan:
                self.get_logger().warn('No /scan data yet; using waypoint position for marker.')
                self.warned_missing_scan = True
            return location['x'], location['y']

        if self.latest_pose is None:
            if not self.warned_missing_pose:
                self.get_logger().warn('No /amcl_pose data yet; using waypoint position for marker.')
                self.warned_missing_pose = True
            return location['x'], location['y']

        scan = self.latest_scan
        best_range = None
        best_angle = None
        angle = scan.angle_min

        for r in scan.ranges:
            if abs(angle) <= self.wall_scan_half_window_rad:
                if math.isfinite(r) and scan.range_min < r < scan.range_max:
                    if best_range is None or r < best_range:
                        best_range = r
                        best_angle = angle
            angle += scan.angle_increment

        if best_range is None or best_angle is None:
            self.get_logger().warn('No valid frontal LiDAR wall hit; using waypoint position for marker.')
            return location['x'], location['y']

        robot_x = self.latest_pose.position.x
        robot_y = self.latest_pose.position.y
        yaw = self.quaternion_to_yaw(self.latest_pose.orientation.z, self.latest_pose.orientation.w)
        heading = yaw + best_angle
        distance = max(best_range - self.marker_wall_offset_m, 0.15)

        marker_x = robot_x + distance * math.cos(heading)
        marker_y = robot_y + distance * math.sin(heading)

        self.get_logger().info(
            f"[MARKER] lidar-placement status={status} "
            f"robot=({robot_x:.2f},{robot_y:.2f}) "
            f"range={best_range:.2f} angle={best_angle:.2f} "
            f"marker=({marker_x:.2f},{marker_y:.2f})"
        )
        return marker_x, marker_y

    def publish_all_markers(self) -> None:
        marker_array = MarkerArray()
        marker_id = 0

        for item in self.detected_results:
            if item['id'] == self.base_location_id:
                self.get_logger().info(
                    f"[MARKER] id={item['id']} is base -> no marker published"
                )
                continue

            if item['status'] == 'empty':
                self.get_logger().info(
                    f"[MARKER] id={item['id']} status=empty -> no marker published"
                )
                continue

            marker = Marker()
            marker.header.frame_id = 'map'
            self.get_logger().info(
                f"publish_all_markers: {item['status']}"
            )
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'detected_people'
            marker.id = marker_id
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
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
            elif item['status'] == 'intruder':
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0

            marker_array.markers.append(marker)
            marker_id += 1
            self.get_logger().info(
                f"[MARKER] id={item['id']} "
                f"status={item['status']} "
                f"pos=({item['x']:.2f}, {item['y']:.2f}) "
                f"color={'GREEN' if item['status']=='authorized' else 'RED'}"
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
