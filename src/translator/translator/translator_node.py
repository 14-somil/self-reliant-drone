#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import VehicleOdometry
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Point, Vector3

import numpy as np
from transforms3d.euler import euler2quat

class PX4OdometryConverter(Node):
    """
    ROS 2 node to convert px4_msgs/msg/VehicleOdometry to nav_msgs/msg/Odometry.
    Simplified version without covariance data.
    """
    
    def __init__(self):
        super().__init__('px4_odometry_converter')
        
        # Declare parameters
        self.declare_parameter('publish_tf', False)
        self.declare_parameter('message_queue_size', 30)
        self.declare_parameter('output_topic', 'odometry')
        self.declare_parameter('input_topic', '/fmu/out/vehicle_odometry')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link')
        
        # Get parameters
        self.publish_tf = self.get_parameter('publish_tf').value
        queue_size = self.get_parameter('message_queue_size').value
        output_topic = self.get_parameter('output_topic').value
        input_topic = self.get_parameter('input_topic').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        
        # QoS settings for PX4 communication - increased queue depth
        qos_profile_subscription = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=queue_size
        )
        
        # QoS settings for publishing - reliable with increased queue
        qos_profile_publisher = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=queue_size
        )
        
        # Create subscription to PX4 vehicle odometry
        self.subscription = self.create_subscription(
            VehicleOdometry,
            input_topic,
            self.odometry_callback,
            qos_profile_subscription
        )
        
        # Create publisher for ROS standard odometry
        self.publisher = self.create_publisher(
            Odometry,
            output_topic,
            qos_profile_publisher
        )
        
        # Optional: Add TF publisher if needed
        if self.publish_tf:
            from tf2_ros import TransformBroadcaster
            self.tf_broadcaster = TransformBroadcaster(self)
        
        # Add throttling to reduce processing load if needed
        self.last_publish_time = self.get_clock().now()
        self.publish_period = 0.02  # 50 Hz max publish rate
        
        self.get_logger().info('PX4 Odometry Converter started')
        
    def odometry_callback(self, msg: VehicleOdometry):
        """
        Convert incoming PX4 VehicleOdometry to ROS Odometry message
        """
        # Optional: Add throttling for high-frequency messages
        current_time = self.get_clock().now()
        dt = (current_time - self.last_publish_time).nanoseconds / 1e9
        
        if dt < self.publish_period:
            return  # Skip this message to reduce processing load
            
        self.last_publish_time = current_time
            
        ros_odom = Odometry()
        
        # Set header
        ros_odom.header.stamp = current_time.to_msg()
        ros_odom.header.frame_id = self.odom_frame_id
        ros_odom.child_frame_id = self.base_frame_id
        
        # Set position - explicitly convert to float
        position = Point()
        position.x = float(msg.position[0])
        position.y = float(msg.position[1])
        position.z = float(msg.position[2])
        
        # Convert quaternion if provided, otherwise convert from Euler angles
        if msg.q[0] == 0.0 and msg.q[1] == 0.0 and msg.q[2] == 0.0 and msg.q[3] == 0.0:
            # PX4 uses NED (North-East-Down) frame while ROS uses ENU (East-North-Up)
            # Convert from Euler angles considering the frame differences
            roll, pitch, yaw = float(msg.q_offset[0]), float(msg.q_offset[1]), float(msg.q_offset[2])
            # You might need to adjust these conversions based on your exact frame conventions
            qx, qy, qz, qw = euler2quat(roll, pitch, yaw, 'sxyz')
            orientation = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))
        else:
            orientation = Quaternion(
                x=float(msg.q[1]), 
                y=float(msg.q[2]), 
                z=float(msg.q[3]), 
                w=float(msg.q[0])
            )
        
        # Set pose
        ros_odom.pose.pose.position = position
        ros_odom.pose.pose.orientation = orientation
        
        # Set twist (velocity) - explicitly convert to float
        linear = Vector3()
        linear.x = float(msg.velocity[0])
        linear.y = float(msg.velocity[1])
        linear.z = float(msg.velocity[2])
        
        angular = Vector3()
        angular.x = float(msg.angular_velocity[0])
        angular.y = float(msg.angular_velocity[1])
        angular.z = float(msg.angular_velocity[2])
        
        ros_odom.twist.twist.linear = linear
        ros_odom.twist.twist.angular = angular
        
        # Covariance matrices are removed - will use default zero values
        
        # Publish the converted message
        self.publisher.publish(ros_odom)
        
        # Optionally publish the transform
        if self.publish_tf:
            from geometry_msgs.msg import TransformStamped
            
            transform = TransformStamped()
            transform.header = ros_odom.header
            transform.child_frame_id = ros_odom.child_frame_id
            transform.transform.translation.x = position.x
            transform.transform.translation.y = position.y
            transform.transform.translation.z = position.z
            transform.transform.rotation = orientation
            
            self.tf_broadcaster.sendTransform(transform)
        

def main(args=None):
    rclpy.init(args=args)
    node = PX4OdometryConverter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()