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

        pkg_dir = get_package_share_directory('yolo_detector')
        model_path = os.path.join(pkg_dir, 'models', 'yolov8n.pt')

        self.get_logger().info(f'Loading YOLO model from: {model_path}')
        self.model = YOLO(model_path)

        self.get_logger().info('YOLO Detector Node Started')

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
        detected_labels = []

        for cls_id in boxes.cls.tolist():
            cls_id = int(cls_id)
            label = names.get(cls_id, str(cls_id))
            detected_labels.append(label.lower())

        self.get_logger().info(f'Detected labels: {detected_labels}')

        # TODO: adjust these strings to your actual custom model class names.
        authorized_labels = {'authorized', 'student', 'young_lady', 'young lady'}
        intruder_labels = {'intruder', 'officer', 'military_officer', 'military officer'}

        for label in detected_labels:
            if label in intruder_labels:
                return 'intruder'

        for label in detected_labels:
            if label in authorized_labels:
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
