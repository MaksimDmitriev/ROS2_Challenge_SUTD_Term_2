import base64
import os

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - runtime dependency
    OpenAI = None


class GPTVisionNode(Node):
    def __init__(self):
        super().__init__('gpt_vision_node')

        self.bridge = CvBridge()
        self.latest_frame_rgb = None
        self.latest_frame_stamp = None
        self.request_counter = 0

        self.publisher_status = self.create_publisher(String, '/vision_status', 10)
        self.subscriber = self.create_subscription(
            Image,
            '/image_input_topic',
            self.image_callback,
            10
        )
        self.service = self.create_service(
            Trigger,
            '/classify_current_frame',
            self.classify_current_frame_callback
        )

        self.declare_parameter('openai_model', 'gpt-4.1-mini')
        self.declare_parameter('save_debug_images', True)
        self.declare_parameter('debug_image_dir', '/tmp/gpt_vision_requests')
        self.declare_parameter(
            'openai_prompt',
            (
                'Classify this single scene image for security. '
                'Printed photos/posters/cutouts of people count as valid people; do not label empty just because the person is on paper. '
                'Class rules: military or worker or security personnel (security guard/security officer) => intruder. '
                'Student or researcher or lab assistant => authorized. '
                'Researcher cues include lab coat, white coat, clipboard, tablet, or scientist-like attire, even if small or partially visible. '
                'If a stop sign is clearly visible anywhere, return stop_sign. '
                'Return exactly one token: intruder, authorized, empty, or stop_sign.'
            )
        )

        self.openai_model = self.get_parameter('openai_model').get_parameter_value().string_value
        self.save_debug_images = self.get_parameter('save_debug_images').get_parameter_value().bool_value
        self.debug_image_dir = self.get_parameter('debug_image_dir').get_parameter_value().string_value
        self.openai_prompt = self.get_parameter('openai_prompt').get_parameter_value().string_value

        self.openai_client = None
        api_key = os.getenv('OPENAI_API_KEY')
        if OpenAI is None:
            self.get_logger().error('openai package is not installed. Install python package `openai`.')
        elif not api_key:
            self.get_logger().error('OPENAI_API_KEY is not set in environment.')
        else:
            self.openai_client = OpenAI(api_key=api_key)
            self.get_logger().info(f'GPT vision backend ready with model: {self.openai_model}')

        self.get_logger().info('GPT Vision Node Started')

    def image_callback(self, msg: Image) -> None:
        try:
            self.latest_frame_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.latest_frame_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'Failed to cache image frame: {e}')

    def classify_current_frame_callback(self, _request, response):
        if self.latest_frame_rgb is None:
            response.success = False
            response.message = 'empty'
            return response

        if self.openai_client is None:
            self.get_logger().warn('OpenAI client unavailable, returning empty.')
            response.success = False
            response.message = 'empty'
            return response

        self.request_counter += 1
        stamp = self.latest_frame_stamp
        stamp_text = (
            f'{stamp.sec}.{stamp.nanosec:09d}'
            if stamp is not None else 'unknown'
        )
        image_path = self.save_debug_image(self.latest_frame_rgb, stamp_text)
        self.get_logger().info(
            f'Sending frame to OpenAI: request_id={self.request_counter}, '
            f'stamp={stamp_text}, saved_image={image_path}'
        )

        try:
            status = self.classify_with_openai(self.latest_frame_rgb)
        except Exception as e:
            self.get_logger().error(f'OpenAI classification failed: {e}')
            status = 'empty'

        if status not in ('intruder', 'authorized', 'empty', 'stop_sign'):
            status = 'empty'

        msg = String()
        msg.data = status
        self.publisher_status.publish(msg)

        response.success = True
        response.message = status
        self.get_logger().info(f'Vision status: {status}')
        return response

    def save_debug_image(self, frame_rgb, stamp_text: str) -> str:
        if not self.save_debug_images:
            return 'disabled'

        try:
            os.makedirs(self.debug_image_dir, exist_ok=True)
            filename = f'req_{self.request_counter:05d}_{stamp_text.replace(".", "_")}.jpg'
            output_path = os.path.join(self.debug_image_dir, filename)
            cv2.imwrite(output_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            return output_path
        except Exception as e:
            self.get_logger().warn(f'Failed to save debug image: {e}')
            return 'save_failed'

    def classify_with_openai(self, frame_rgb) -> str:
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError('Failed to encode frame as JPEG.')

        image_b64 = base64.b64encode(encoded.tobytes()).decode('utf-8')
        image_url = f'data:image/jpeg;base64,{image_b64}'

        # Prefer Responses API. Fallback to Chat Completions if needed.
        try:
            response = self.openai_client.responses.create(
                model=self.openai_model,
                input=[
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'input_text', 'text': self.openai_prompt},
                            {'type': 'input_image', 'image_url': image_url},
                        ],
                    }
                ],
                max_output_tokens=16,
            )
            text = (getattr(response, 'output_text', '') or '').strip().lower()
        except Exception:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': self.openai_prompt},
                            {'type': 'image_url', 'image_url': {'url': image_url}},
                        ],
                    }
                ],
                max_tokens=16,
            )
            text = (response.choices[0].message.content or '').strip().lower()

        if 'stop_sign' in text or 'stop sign' in text:
            return 'stop_sign'
        if 'intruder' in text:
            return 'intruder'
        if 'authorized' in text:
            return 'authorized'
        if 'empty' in text:
            return 'empty'
        return text


def main(args=None):
    rclpy.init(args=args)
    node = GPTVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
