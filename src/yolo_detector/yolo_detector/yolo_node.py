import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

from ament_index_python.packages import get_package_share_directory
import os
import time

class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
        # Publishers and Subscribers
        self.publisher_ = self.create_publisher(Image, '/image_output_topic', 10)
        self.subscriber = self.create_subscription(
            Image, 
            '/image_input_topic', 
            self.image_callback, 
            10
        )

        # Declare a ROS parameter
        self.declare_parameter('rate_limit', 1.0)
        # Get Parameter value
        rate_limit = self.get_parameter('rate_limit').get_parameter_value().double_value
        self.get_logger().info("Detection Rate Limit: %.2f"%rate_limit)
        self.last_detection_time = time.time()
        self.detection_interval = 1.0/rate_limit # minimum time between detections 
               
        # Get package installation directory
        pkg_dir = get_package_share_directory("yolo_detector")
        # Full path to model file (Within installation directory)
        model_path = os.path.join(pkg_dir, 'models', 'yolov8n.pt')
        # Initialize the YOLOv8 model
        self.get_logger().info("Loading YOLO model: " + model_path)
        self.model = YOLO(model_path)
        
        # Bridge to convert ROS <-> OpenCV
        self.bridge = CvBridge()
        
        self.get_logger().info("YOLO Detector Node Started!")

    def image_callback(self, msg):
        try:
        
            # Control detection frame rate
            if (self.detection_interval > (time.time() - self.last_detection_time)):
            # If a detection_interval has not yet passed since last detection, ignore image
                return
            self.last_detection_time = time.time()
            
            
            # 1. Convert the incoming ROS image to an RGB NumPy array
            frame_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            # 2. Run YOLO inference on this RGB image
            results = self.model(frame_rgb, verbose=False)

            # 3. results[0].plot() returns an annotated RGB image (ndarray)
            annotated_img_rgb = results[0].plot(show=False)

            # 4. Convert the annotated RGB image back to a ROS Image message
            output_msg = self.bridge.cv2_to_imgmsg(annotated_img_rgb, encoding='rgb8')

            # 5. Maintain metadata (Timestamp/Frame ID) and Publish
            output_msg.header = msg.header
            self.publisher_.publish(output_msg)
            
            # Check execution time
            if (self.detection_interval < (time.time() -self.last_detection_time)):
                # The detection process itself took a longer time than allocated detection_interval
                # The hardware is not capable enough to run detection at given rate
                self.get_logger().warn(
                    "Possible : Rate limit too high! Should be less than : %.2f Hz"%(
                        (1/(time.time()-self.last_detection_time))))
            for detection in msg.detections:
                label = detection.label
                score = detection.score
                # This prints to your terminal with a timestamp
                self.get_logger().info(f"I see a '{label}' with {score*100}% confidence")
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    yolo_detector = YOLODetector()
    
    try:
        rclpy.spin(yolo_detector)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean shutdown
        yolo_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
