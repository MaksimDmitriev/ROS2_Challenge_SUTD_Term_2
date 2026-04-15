import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class ObjectVisualizer(Node):
    def __init__(self):
        super().__init__('object_visualizer')
        # This is where your YOLO output would come in
        self.publisher = self.create_publisher(MarkerArray, '/detected_markers', 10)
        self.timer = self.create_timer(1.0, self.publish_test_markers)

    def publish_test_markers(self):
        marker_array = MarkerArray()
        
        # Example: Person (Red)
        person = self.create_marker(id=0, name="person", color=(1.0, 0.0, 0.0), pos=(2.0, 0.5))
        # Example: Stop Sign (Green)
        sign = self.create_marker(id=1, name="stop sign", color=(0.0, 1.0, 0.0), pos=(3.0, -0.5))
        
        marker_array.markers.append(person)
        marker_array.markers.append(sign)
        self.publisher.publish(marker_array)

    def create_marker(self, id, name, color, pos):
        marker = Marker()
        marker.header.frame_id = "map" # Ensure this matches your RViz fixed frame
        marker.id = id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y = pos[0], pos[1]
        marker.pose.position.z = 0.5
        marker.scale.x, marker.scale.y, marker.scale.z = 0.2, 0.2, 0.8
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color[0], color[1], color[2], 1.0
        return marker

def main():
    rclpy.init()
    node = ObjectVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
