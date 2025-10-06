#!/home/plague/pyenv_ros/bin/python3

#TODO:
# _get_observation
# _get_info
# _get_original_observation
# add randomization in reset

import rclpy
import csv
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, ActuatorMotors, VehicleOdometry
from datetime import datetime
import os
import time
from enum import Enum, auto
import gymnasium as gym
from gymnasium import spaces
import threading
import gz.transport13
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose
import math

class DroneNode(Node):
    def __init__(self):
        super().__init__('drone_command')

        qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
        
        self.cb_group = ReentrantCallbackGroup()
        
        self.offboard_control_mode_publisher = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.actuator_motor_publisher = self.create_publisher(ActuatorMotors, '/fmu/in/actuator_motors', qos_profile)

        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, 
            '/fmu/out/vehicle_odometry', 
            self.vehicle_odometry_callback, 
            qos_profile, 
            callback_group= self.cb_group
            )
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, 
            '/fmu/out/vehicle_status_v1', 
            self.vehicle_status_callback, 
            qos_profile,
            callback_group= self.cb_group
            )

        self.vehicle_odom = VehicleOdometry()
        self.vehicle_status = VehicleStatus()
        self.offboard_mode_counter = 0
        self.actuator_motor_control = None #[-1, 1]

        self.timer = self.create_timer(0.05, self.timer_callback, callback_group=self.cb_group)

    def vehicle_odometry_callback(self, msg):
        self.vehicle_odom = msg

    def vehicle_status_callback(self, msg):
        self.vehicle_status = msg

    def publish_offboardcontrol_heartbeat_signal(self):
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.direct_actuator = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)
    
    def publish_actuator_motors(self, control:list[float]):
        msg = ActuatorMotors()

        msg.control = list(np.interp(control, [-1, 1], [0, 1])) + [0] * 8
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        self.actuator_motor_publisher.publish(msg)

        self.get_logger().info(f'Sent control signals: {msg.control}')

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)
    
    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)

        self.get_logger().info("Arm Command sent")

    def disarm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)

        self.get_logger().info("Disarm Command sent")

    def engage_offboard_mode(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)

        self.get_logger().info("Switching to offboard mode")

    #Z axis for postion and linear velocity changed from NED to ENU frame
    #axes for rotation and orientation maintained as it is

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

    def timer_callback(self):
        self.publish_offboardcontrol_heartbeat_signal()

        if(self.offboard_mode_counter < 21):
            self.offboard_mode_counter += 1
        
        if(self.offboard_mode_counter > 20):
            if(self.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD):
                self.engage_offboard_mode()
            
            if(self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED):
                self.arm()
            
            else:
                if self.actuator_motor_control is not None: 
                    self.publish_actuator_motors(self.actuator_motor_control)

class DroneEnv(gym.Env):
    def __init__(self, is_training=True):
        super(DroneEnv, self).__init__()

        self.gz_node = gz.transport13.Node()

        rclpy.init(args=None)
        self.node = DroneNode()
        self.spin_thread = threading.Thread(target=self._spin, daemon=True)
        self.spin_thread.start()

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        observation_dim = 12  # Adjust based on the actual observation dimension
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            dtype=np.float32
        )

        self.phase1_end = 200
        self.phase2_end = 500

        self.phase_1_ranges = {
            "position_range": 0, #4
            "velocity_range": 0, #5
            "angular_velocity_range": 0, #2
            "orientation_range": 0 #np.radians(90)
        }
    
        self.phase_2_ranges = {
            "position_range": 0, #2
            "velocity_range": 0, #1
            "angular_velocity_range": 0, #1
            "orientation_range": 0 #np.radians(60)
        }
    
        self.phase_3_ranges = {
            "position_range": 0.0,
            "velocity_range": 0.0,
            "angular_velocity_range": 0.0,
            "orientation_range": 0.0
        }

        self.is_training = is_training
        self.time_on_ground = 0
        self.start_time = time.perf_counter()
        self.step_counter = 0
        self.max_steps = 500
        self.target_position = np.array([0.0, 0.0, 3.0])
        self.sway_accumulation = np.array([0.0, 0.0, 0.0])
        self.episode_counter = 0
        self.steps_beyond_terminated = 0

    def _spin(self):
        try:
            rclpy.spin(self.node)
        except Exception as e:
            self.node.get_logger().info(f'Spin excpetion : {e}')
    
    def _goto(self, position = None):
        if position is None: 
            return
        
        service_name = '/world/default/set_pose'

        request = Pose()
        request.name = 'x500_0'
        request.position.x = position[0]
        request.position.y = position[1]
        request.position.z = position[2]
        request.orientation.x = 0
        request.orientation.y = 0
        request.orientation.z = 0
        request.orientation.w = 1

        reponse = Boolean()

        timeout = 5000

        result, reponse = self.gz_node.request(service_name, request, Pose, Boolean, timeout)

        if(reponse.data == True):
            self.node.get_logger().info(f'Gazebo reset request successful')
        else:
            self.node.get_logger().info(f'Gazebo reset request failed')

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
        pos = self.node._get_position()
        vel = self.node._get_vel()
        orientation = self.node._get_orientation()

        orientation = np.deg2rad(orientation)

        return self._clip_and_normalize(np.concatenate([pos, orientation, vel]))

    def _get_original_observation(self):
        return np.concatenate([self.node._get_position(), self.node._get_orientation(), self.node._get_vel()])

    def reset(self, seed = None, options = None, initial_position = None): #position = np.array([x,y,z])

        #TODO: randomized initial position
        super().reset(seed=seed)

        if initial_position is None:
            initial_position = np.array([0.0, 0.0, 0.0])
        
        self._goto(initial_position)
        time.sleep(0.01) # wait to update the current position of drone

        self.node.actuator_motor_control = [-1.0] * 4

        self.sway_accumulation = np.array([0.0, 0.0, 0.0])
        self.last_step_time = time.perf_counter()
        self.time_on_ground = 0
        self.step_counter = 0
        self.steps_beyond_terminated = 0

        initial_observation = self._get_observation()
        info = {
            'original_observation': self._get_original_observation()
        }

        self.episode_counter += 1

        return initial_observation, info
    
    def _get_done(self):
        MAX_XY = 10
        MAX_Z = 10

        original_obs = self._get_original_observation()

        pos = original_obs[:3]
        roll, pitch, yaw = original_obs[3:6]

        if pos[2] > MAX_Z or pos[0] < -MAX_XY or pos[0] > MAX_XY or pos[1] < -MAX_XY or pos[1] > MAX_XY:
            self.node.get_logger().info(f'Drone out of limits. Terminating current step.')
            return True
        
        if self.step_counter >= self.max_steps:
            return True
        
        return False

    def _get_reward(self, terminated, action):
        state = self._get_original_observation()
        current_position = state[:3]
        roll, pitch, yaw = state[3:6]
        current_velocity = state[6:9]
        current_angular_velocity = state[9:12]

        penalty_box = 0.1

        if all(np.abs(self.target_position - current_position) < penalty_box):
            hover_reward = 1
        else:
            hover_reward = 0

        alpha_p = 1.0  # Position error weight
        alpha_v = 0.05  # Velocity error weight
        alpha_omega = 0.001  # Angular velocity error weight
        alpha_rp = 0.02  # Roll and pitch error weight
        alpha_a = 0.025  # Action penalty weight

        # Position error (e_p) - L2 norm between target position and current position
        e_p = np.linalg.norm(self.target_position - current_position)

        # Velocity error (e_v) - L2 norm between target velocity (0, 0, 0) and current velocity
        target_velocity = np.array([0.0, 0.0, 0.0])
        e_v = np.linalg.norm(target_velocity - current_velocity)

        # Angular velocity error (e_ω) - L2 norm between target angular velocity (0, 0, 0) and current angular velocity
        target_angular_velocity = np.array([0.0, 0.0, 0.0])
        e_omega = np.linalg.norm(target_angular_velocity - current_angular_velocity)

        # Roll and pitch angle errors (eξ, eφ) - L2 norm for roll and pitch
        e_rp = np.linalg.norm([roll, pitch, yaw])

        # Action penalty (L2 norm of action)
        action_penalty = np.linalg.norm(action)

        # Reward calculation
        r = (
            hover_reward  # Hover reward
            - alpha_p * e_p  # Position error penalty
            - alpha_v * e_v  # Velocity error penalty
            - alpha_omega * e_omega  # Angular velocity error penalty
            - alpha_rp * e_rp  # Roll and pitch error penalty
            - alpha_a * action_penalty  # Action penalty
        )
        
        return r
    
    def step(self, action):
        start_time = time.perf_counter()
        original_action = action
        self.node.actuator_motor_control = original_action #in range [-1, 1]

        time.sleep(0.1)
        observation = self._get_observation()
        truncated = self.step_counter >= self.max_steps
        terminated = self._get_done()
        done:bool = terminated or truncated

        reward = self._get_reward(terminated, original_action)

        info = {
            'original_action': action,
            'original_observation': self._get_original_observation(),
            'reward': reward
        }

        step_duration = time.perf_counter() - start_time
        info['step_duration'] = step_duration

        self.step_counter += 1

        return observation, reward, done, truncated, info
    
    def close(self):
        self._goto([0.0, 0.0, 0.0])
        self.node.actuator_motor_control = [-1.0] * 4

        self.node.destroy_node()
        rclpy.shutdown()
