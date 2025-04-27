import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import gz.transport13
from gz.msgs10.odometry_pb2 import Odometry
from px4_msgs.msg import VehicleOdometry
import time

import numpy as np
import transforms3d

class GzRosBridge(Node):

    def __init__(self) -> None:
        super().__init__('gz_ros_bridge')
        self.gz_node = gz.transport13.Node()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.ros_publisher = self.create_publisher(
            VehicleOdometry, '/fmu/in/vehicle_visual_odometry', qos_profile
        )

        self.gz_node.subscribe(Odometry, '/model/x500_vision_0/odometry', self.gz_callback)

    def enu_to_ned_quaternion(self, q):
        # Convert ENU quaternion [x, y, z, w] to [w, x, y, z] for transforms3d
        q_enu = [q.w, q.x, q.y, q.z]

        # 180-degree rotation about X axis (ENU to NED)
        rot180_x = transforms3d.quaternions.axangle2quat([1, 0, 0], np.pi)
        q_ned = transforms3d.quaternions.qmult(rot180_x, q_enu)

        # Return as [x, y, z, w]
        return [q_ned[1], q_ned[2], q_ned[3], q_ned[0]]

    def gz_callback(self, msg) -> None:
        published_msg = VehicleOdometry()
        published_msg.timestamp = int(time.time() * 1e6)  # microseconds

        published_msg.pose_frame = VehicleOdometry.POSE_FRAME_NED
        # Position (ENU -> NED)
        published_msg.position = [
            msg.pose.position.y,         # Y_GZ -> X_PX4
            msg.pose.position.x,         # X_GZ -> Y_PX4
            -msg.pose.position.z         # Z_GZ -> -Z_PX4
        ]

        # Orientation (ENU -> NED)
        q_ned = self.enu_to_ned_quaternion(msg.pose.orientation)
        published_msg.q = q_ned

        # Linear velocity (ENU -> NED)
        published_msg.velocity = [
            msg.twist.linear.y,
            msg.twist.linear.x,
            -msg.twist.linear.z
        ]

        # Angular velocity (ENU -> NED)
        published_msg.angular_velocity = [
            msg.twist.angular.y,
            msg.twist.angular.x,
            -msg.twist.angular.z
        ]

        self.get_logger().info(f'Published VehicleOdometry at {published_msg.timestamp}')
        self.ros_publisher.publish(published_msg)

def main(args=None) -> None:
    print('Bridge started...')
    rclpy.init(args=args)
    bridge = GzRosBridge()
    rclpy.spin(bridge)
    bridge.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
