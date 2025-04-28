import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import SensorOpticalFlow
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class OpticalFlow(Node):
    def __init__(self):
        super().__init__('optical_flow')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.image_subscriber = self.create_subscription(Image, '/camera/image', self.image_callback, qos_profile)
        self.camera_info_subscriber = self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info_callback, qos_profile)

        self.optical_flow_publisher = self.create_publisher(SensorOpticalFlow, '/fmu/in/sensor_optical_flow', qos_profile)

        self.cv_bridge = CvBridge()
        self.prev_gray = None
        self.camera_info = None

        self.get_logger().info('OpticalFlowNode initialized.')
    
    def camera_info_callback(self, msg):
        self.camera_info = msg
    
    def image_callback(self, msg):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return
        
        if self.prev_gray is None:
            self.prev_gray = cv_image
            return
        
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            cv_image,
            None,
            0.5,  # pyr_scale
            3,    # levels
            15,   # winsize
            3,    # iterations
            5,    # poly_n
            1.2,  # poly_sigma
            0     # flags
        )

        flow_x = flow[..., 0]
        flow_y = flow[..., 1]

        avg_flow_x = np.mean(flow_x)
        avg_flow_y = np.mean(flow_y)

        flow_msg = SensorOpticalFlow()
        flow_msg.timestamp = self.get_clock().now().nanoseconds // 1000  # ROS time in microseconds
        flow_msg.pixel_flow[1] = float(avg_flow_x)
        flow_msg.pixel_flow[0] = float(avg_flow_y)
        flow_msg.delta_angle = [0.0, 0.0, 0.0]
        flow_msg.delta_angle_available = False
        flow_msg.distance_available = False
        flow_msg.integration_timespan_us = int(1e6 / 60)  # Assuming 30 FPS camera
        flow_msg.quality = 255  # Assume perfect quality for now
        flow_msg.max_flow_rate = 5.0  # Random reasonable dpixel_flow_y_integralefault
        flow_msg.min_ground_distance = 0.5
        flow_msg.max_ground_distance = 10.0

        self.optical_flow_publisher.publish(flow_msg)

        self.prev_gray = cv_image

def main(args = None) -> None:
    print('Optical flow node started...')
    rclpy.init(args= args)
    optical_flow = OpticalFlow()

    rclpy.spin(optical_flow)

    optical_flow.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()