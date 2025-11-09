#!/home/plague/pyenv_ros/bin/python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import VehicleOdometry
import math
import numpy as np

class StatePublisher(Node):
    def __init__(self):
        super().__init__('state_publisher')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.state_publisher = self.create_publisher(Float32MultiArray, '/normalized_vehicle_state', qos_profile)

        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos_profile)

        self.vehicle_odom = VehicleOdometry()

        self.timer = self.create_timer(0.1, self.timer_callback)

    def odom_callback(self, msg):
        self.vehicle_odom = msg

    def _get_position(self):
        return np.array([self.vehicle_odom.position[0], 
                        self.vehicle_odom.position[1], 
                        -self.vehicle_odom.position[2]])    
    def _get_vel(self):
        return np.array([self.vehicle_odom.velocity[0],
                        self.vehicle_odom.velocity[1],
                        -self.vehicle_odom.velocity[2],
                        self.vehicle_odom.angular_velocity[0],
                        self.vehicle_odom.angular_velocity[1],
                        self.vehicle_odom.angular_velocity[2]])

    def _get_orientation(self):
        q = self.vehicle_odom.q

        w, x, y, z = q[0], q[1], q[2], q[3]

        # Roll (x-axis rotation)
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        # Pitch (y-axis rotation)
        pitch = math.asin(2.0 * (w * y - z * x))
        # Yaw (z-axis rotation) 
        yaw = -math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        return np.array([math.degrees(roll), math.degrees(pitch), math.degrees(yaw)])
    
    def _clip_and_normalize(self, state):
        MAX_LIN_VEL_XY = 10
        MAX_LIN_VEL_Z = 10
        MAX_XY = 10
        MAX_Z = 10
        MAX_PITCH_ROLL = np.pi

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[3:5], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[6:8], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[8], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)
    
        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[5] / np.pi
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_Z
        normalized_ang_vel = state[9:12] / np.linalg.norm(state[9:12]) if np.linalg.norm(state[9:12]) != 0 else state[9:12]
    
        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel]).astype(np.float32)
            
        return norm_and_clipped
    
    def _get_observation(self):
        pos = self._get_position()
        vel = self._get_vel()
        orientation = self._get_orientation()

        orientation = np.deg2rad(orientation)

        return self._clip_and_normalize(np.concatenate([pos, orientation, vel]))
    
    def timer_callback(self):
        msg = Float32MultiArray()
        msg.data = self._get_observation().flatten().tolist()

        self.state_publisher.publish(msg)
        self.get_logger().info(f'Published {msg.data}')

def main(args = None):
    print(f'Starting state_publisher...')
    if not rclpy.ok() :
        rclpy.init(args=args)
    
    state_publisher = StatePublisher()
    rclpy.spin(state_publisher)

    state_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()