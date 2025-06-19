import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import requests
import json_numpy
import numpy as np
import cv2

json_numpy.patch()

class OpenVLAClient(Node):
    def __init__(self):
        super().__init__('openvla_client')

        # Initialize subscribers
        self.image_sub = self.create_subscription(
            Image, '/sim/image_rgb', self.image_callback, 10)
        self.instr_sub = self.create_subscription(
            String, '/sim/instruction', self.instruction_callback, 10)

        # Publisher for robot actions
        self.action_pub = self.create_publisher(String, '/sim/action_command', 10)

        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_instruction = None

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.latest_image = cv_image
            self.try_send_request()
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

    def instruction_callback(self, msg):
        self.latest_instruction = msg.data
        self.try_send_request()

    def try_send_request(self):
        if self.latest_image is None or self.latest_instruction is None:
            return

        # Resize and preprocess image
        resized = cv2.resize(self.latest_image, (224, 224))  # depends on your VLA config
        image_np = np.array(resized).astype(np.uint8)

        # Compose payload
        payload = {
            "image": image_np,
            "instruction": self.latest_instruction,
        }

        try:
            response = requests.post(
                "http://localhost:8777/act",
                json={"encoded": json_numpy.dumps(payload)}
            )

            if response.status_code == 200:
                action = json_numpy.loads(response.text)
                self.action_pub.publish(String(data=json.dumps(action)))
                self.get_logger().info(f"Sent action: {action}")
            else:
                self.get_logger().warn(f"Server error: {response.status_code} — {response.text}")

        except Exception as e:
            self.get_logger().error(f"Request failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    client = OpenVLAClient()
    rclpy.spin(client)
    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
