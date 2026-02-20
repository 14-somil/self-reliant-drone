#!/home/plague/pyenv_ros/bin/python3

#TODO:
# Match the subsciption and publishing frequency

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import gz.transport13
from gz.msgs10.pose_v_pb2 import Pose_V
from px4_msgs.msg import VehicleOdometry
import numpy as np
import math
from threading import Lock
import sys

class GzNode(Node):
    def __init__(self):
        super().__init__('gz_publisher')
        
        # Parameters
        self.declare_parameter('vehicle_name', 'x500_0' if len(sys.argv)<2 else f'{sys.argv[1]}_0')
        self.declare_parameter('update_rate', 50.0)
        
        self.vehicle_name = self.get_parameter('vehicle_name').value
        self.dt = 1.0 / self.get_parameter('update_rate').value
        
        # Initialize Gazebo node
        self.gz_node = gz.transport13.Node()
        
        # QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Vehicle odometry message
        self.vehicle_odom = VehicleOdometry()
        self.vehicle_odom.pose_frame = VehicleOdometry.POSE_FRAME_NED
        self.vehicle_odom.velocity_frame = VehicleOdometry.VELOCITY_FRAME_NED
        self.odom_lock = Lock()
        
        # Buffers for numerical differentiation (circular buffers)
        self.position_buffer = np.zeros((3, 3), dtype=float)
        self.euler_angle_buffer = np.zeros((3, 3), dtype=float)
        self.position_samples = 0
        self.angle_samples = 0
        
        # Subscribers and publishers
        self.gz_subscriber = self.gz_node.subscribe(
            Pose_V, '/world/default/pose/info', self.gz_callback
        )
        self.pose_publisher = self.create_publisher(
            VehicleOdometry, '/gz_position', qos_profile=qos_profile
        )
        
        # Timer
        self.create_timer(self.dt, self.timer_callback)
        self.get_logger().info(f'GZ pose publisher initialized for {self.vehicle_name}')
    
    def calculate_velocity(self):
        """Calculate linear velocity using 3-point backward difference."""
        if self.position_samples >= 2:
            # Use 3-point backward difference: f'(x) ≈ (f(x-2) - 4f(x-1) + 3f(x)) / (2Δt)
            velocity = (self.position_buffer[0] - 4.0*self.position_buffer[1] + 
                       3.0*self.position_buffer[2]) / (2.0 * self.dt)
            
            # Update buffer (circular)
            self.position_buffer = np.roll(self.position_buffer, -1, axis=0)
            self.position_buffer[-1] = self.vehicle_odom.position
        else:
            # Insufficient data
            self.position_buffer[self.position_samples] = self.vehicle_odom.position
            self.position_samples += 1
            velocity = np.zeros(3)
        
        self.vehicle_odom.velocity = velocity
    
    def quat_to_euler(self, w, x, y, z):
        """Convert quaternion to Euler angles (roll, pitch, yaw) in radians."""
        roll = math.atan2(2.0 * (x*w + y*z), (-x*x - y*y + z*z + w*w))
        pitch = math.asin(np.clip(-2.0 * (z*x - w*y), -1.0, 1.0))  # Clip for numerical stability
        yaw = math.atan2(2.0 * (x*y + w*z), (x*x - y*y - z*z + w*w))
        
        return np.array([roll, pitch, yaw])
    
    def calculate_ang_velocity(self):
        """Calculate angular velocity from Euler angles."""
        current_angles = self.quat_to_euler(*self.vehicle_odom.q)
        
        if self.angle_samples >= 2:
            velocity = (self.euler_angle_buffer[0] - 4.0*self.euler_angle_buffer[1] + 
                       3.0*self.euler_angle_buffer[2]) / (2.0 * self.dt)
            
            # Update buffer
            self.euler_angle_buffer = np.roll(self.euler_angle_buffer, -1, axis=0)
            self.euler_angle_buffer[-1] = current_angles
        else:
            self.euler_angle_buffer[self.angle_samples] = current_angles
            self.angle_samples += 1
            velocity = np.zeros(3)
        
        self.vehicle_odom.angular_velocity = velocity
    
    def timer_callback(self):
        """Publish odometry at fixed rate."""
        with self.odom_lock:
            self.vehicle_odom.timestamp = self.get_clock().now().nanoseconds // 1000
            self.pose_publisher.publish(self.vehicle_odom)
    
    def gz_callback(self, msg):
        """Callback for Gazebo pose updates."""
        try:
            for entity in msg.pose:
                if entity.name == self.vehicle_name:
                    with self.odom_lock:
                        # Convert to NED frame
                        self.vehicle_odom.position[0] = entity.position.y
                        self.vehicle_odom.position[1] = entity.position.x
                        self.vehicle_odom.position[2] = -entity.position.z
                        
                        self.vehicle_odom.q[0] = entity.orientation.w
                        self.vehicle_odom.q[1] = entity.orientation.x
                        self.vehicle_odom.q[2] = -entity.orientation.y
                        self.vehicle_odom.q[3] = -entity.orientation.z

                        self.calculate_velocity()
                        self.calculate_ang_velocity()
                    break
        except Exception as e:
            self.get_logger().error(f'Error in gz_callback: {e}')

def main(args=None):
    print('Starting GZ pose publisher...')
    if not rclpy.ok():
        rclpy.init(args=args)
    
    gz_node = GzNode()
    
    try:
        rclpy.spin(gz_node)
    except KeyboardInterrupt:
        pass
    finally:
        gz_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()