import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import gz.transport13
from gz.msgs10.camera_info_pb2 import CameraInfo as GZCameraInfo
from gz.msgs10.image_pb2 import Image as GZImage
from sensor_msgs.msg import CameraInfo as ROSCameraInfo
from sensor_msgs.msg import Image as ROSImage
import numpy as np
import cv2
from cv_bridge import CvBridge
import time

class ImageTranslator(Node):
    def __init__(self):
        super().__init__('image_translator')
        self.gz_node = gz.transport13.Node()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.cv_bridge = CvBridge()

        self.image_publisher = self.create_publisher(ROSImage, '/camera/image', qos_profile)
        self.camera_info_publisher = self.create_publisher(ROSCameraInfo, '/camera/camera_info', qos_profile)

        self.gz_node.subscribe(GZCameraInfo, '/world/lawn/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/camera_info', self.camera_info_callback)
        self.gz_node.subscribe(GZImage, '/world/lawn/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/image', self.image_callback)
    
    def image_callback(self, image_msg):
        # image_msg = GZImage()
        # image_msg.ParseFromString(msg)
        self.get_logger().info(f'Recieved image of format {image_msg.pixel_format_type}')
        if image_msg.pixel_format_type == 3:
            encoding = 'rgb8'
        elif image_msg.pixel_format_type == 4:
            encoding = 'rgba8'
        elif image_msg.pixel_format_type == 1:
            encoding = 'mono8'
        else:
            self.get_logger().warn('Unsupported pixel format')
            return
        
        np_arr = np.frombuffer(image_msg.data, dtype=np.uint8)
        image_np = np_arr.reshape((image_msg.height, image_msg.width, -1))

        # Convert to ROS2 Image message
        ros_image = self.cv_bridge.cv2_to_imgmsg(image_np, encoding)
        ros_image.header.stamp = self.get_clock().now().to_msg()
        ros_image.header.frame_id = 'camera_frame'

        self.image_publisher.publish(ros_image)

    def camera_info_callback(self, camera_info_msg):
        # camera_info_msg = GZCameraInfo()
        # camera_info_msg.ParseFromString(msg)
        self.get_logger().info(f'Recieved camera info at {time.time()}')
        # Convert to ROS2 CameraInfo
        ros_camera_info = ROSCameraInfo()
        ros_camera_info.header.stamp = self.get_clock().now().to_msg()
        ros_camera_info.header.frame_id = 'camera_frame'
        ros_camera_info.height = camera_info_msg.height
        ros_camera_info.width = camera_info_msg.width
        ros_camera_info.d = camera_info_msg.distortion.k
        ros_camera_info.k = camera_info_msg.intrinsics.k
        ros_camera_info.p = camera_info_msg.projection.p
        ros_camera_info.r = camera_info_msg.rectification_matrix

        self.camera_info_publisher.publish(ros_camera_info)

def main(args = None) -> None:
    print('Image translator started...')
    rclpy.init(args= args)
    image_translator = ImageTranslator()
    rclpy.spin(image_translator)
    image_translator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()