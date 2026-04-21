import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory


class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        self.bridge = CvBridge()

        self.publisher_image = self.create_publisher(Image, '/image_output_topic', 10)
        self.publisher_status = self.create_publisher(String, '/vision_status', 10)

        self.subscriber = self.create_subscription(
            Image,
            '/image_input_topic',
            self.image_callback,
            10
        )

        self.declare_parameter('rate_limit', 1.0)
        rate_limit = self.get_parameter('rate_limit').get_parameter_value().double_value
        if rate_limit <= 0.0:
            rate_limit = 1.0

        self.detection_interval = 1.0 / rate_limit
        self.last_detection_time = 0.0

        self.declare_parameter('model_file', 'yolov8n.pt')
        model_file = self.get_parameter('model_file').get_parameter_value().string_value
        if not model_file:
            model_file = 'yolov8n.pt'

        self.declare_parameter('target_roles', ['military', 'researcher', 'student', 'worker'])
        configured_roles = [
            role.strip().lower()
            for role in self.get_parameter('target_roles').value
            if isinstance(role, str) and role.strip()
        ]
        if not configured_roles:
            configured_roles = ['military', 'researcher', 'student', 'worker']
        self.target_roles = set(configured_roles)

        self.role_aliases = {
            'military': {'military', 'soldier', 'officer', 'military officer'},
            'researcher': {'researcher', 'scientist', 'lab coat', 'doctor'},
            'student': {'student'},
            'worker': {'worker', 'construction worker'},
        }
        self.role_aliases = {
            role: aliases
            for role, aliases in self.role_aliases.items()
            if role in self.target_roles
        }

        pkg_dir = get_package_share_directory('yolo_detector')
        model_path = model_file
        if not os.path.isabs(model_path):
            model_path = os.path.join(pkg_dir, 'models', model_path)

        self.get_logger().info(f'Loading YOLO model from: {model_path}')
        self.model = YOLO(model_path)
        self._warn_if_model_has_no_target_roles()

        self.get_logger().info('YOLO Detector Node Started')

    @staticmethod
    def _normalize_label(label: str) -> str:
        return str(label).strip().lower().replace('_', ' ').replace('-', ' ')

    def _warn_if_model_has_no_target_roles(self) -> None:
        names = self.model.names
        if isinstance(names, dict):
            model_labels = {self._normalize_label(v) for v in names.values()}
        else:
            model_labels = {self._normalize_label(v) for v in names}

        all_aliases = set()
        for aliases in self.role_aliases.values():
            all_aliases.update(aliases)

        if model_labels.isdisjoint(all_aliases):
            self.get_logger().warn(
                'Loaded model has no target-role classes. '
                'A generic model like yolov8n.pt only has "person". '
                'Use a custom-trained model with labels military/researcher/student/worker.'
            )

    def _extract_label(self, names, cls_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        if 0 <= cls_id < len(names):
            return str(names[cls_id])
        return str(cls_id)

    def _map_label_to_role(self, label: str):
        normalized = self._normalize_label(label)
        for role, aliases in self.role_aliases.items():
            if normalized in aliases:
                return role
        return None

    def image_callback(self, msg: Image) -> None:
        now = time.time()
        if now - self.last_detection_time < self.detection_interval:
            return
        self.last_detection_time = now

        try:
            frame_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            results = self.model(frame_rgb, verbose=False)

            annotated_img_rgb = results[0].plot(show=False)
            output_msg = self.bridge.cv2_to_imgmsg(annotated_img_rgb, encoding='rgb8')
            output_msg.header = msg.header
            self.publisher_image.publish(output_msg)

            status_msg = String()
            status_msg.data = self.classify_result(results[0])
            self.publisher_status.publish(status_msg)

            self.get_logger().info(f'Vision status: {status_msg.data}')

        except Exception as e:
            self.get_logger().error(f'Failed to process image: {e}')

    def classify_result(self, result) -> str:
        """
        Beginner-friendly version:
        - Reads YOLO class names from the model output
        - Maps them into: authorized / intruder / empty

        IMPORTANT:
        You MUST adapt the class-name mapping below to your custom model.
        """
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return 'empty'

        names = result.names
        detected_roles = []

        for cls_id in boxes.cls.tolist():
            cls_id = int(cls_id)
            label = self._extract_label(names, cls_id)
            role = self._map_label_to_role(label)
            if role is not None:
                detected_roles.append(role)

        self.get_logger().info(f'Detected target roles: {detected_roles}')

        if any(role in detected_roles for role in ('military', 'worker')):
            return 'intruder'
        if any(role in detected_roles for role in ('researcher', 'student')):
            return 'authorized'

        return 'empty'


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
