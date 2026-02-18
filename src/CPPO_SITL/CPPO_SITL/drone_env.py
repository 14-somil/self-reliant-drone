#!/home/plague/pyenv_ros/bin/python3

#TODO:
# add randomization in reset
# speed up training
# is stabilised off kar diya again

# make all the time delays consistent
# relaunching the server everytime

import rclpy
import csv
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, ActuatorMotors, VehicleOdometry, ManualControlSetpoint
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import Pose
from datetime import datetime
import os
import time
import gymnasium as gym
from gymnasium import spaces
import threading
import gz.transport13
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose
import math
from enum import Enum, auto

class CalibrationState(Enum):
    NAVIGATION = auto()
    CALIBRATE = auto()

class OdomPublisher(Enum):
    PX4 = auto()
    GZ = auto()

class ControlMode(Enum):
    DIRECT_ACTUATOR = auto()
    MANUAL_CONTROL = auto()

class DroneNode(Node):
    def __init__(self, ):
        super().__init__('drone_command')

        qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )

        #Choose control mode
        self.control_mode = ControlMode.MANUAL_CONTROL

        self.cb_group = ReentrantCallbackGroup()
        
        self.vehicle_command_publisher = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        
        if self.control_mode == ControlMode.DIRECT_ACTUATOR:
            self.offboard_control_mode_publisher = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
            self.actuator_motor_publisher = self.create_publisher(ActuatorMotors, '/fmu/in/actuator_motors', qos_profile)

        elif self.control_mode == ControlMode.MANUAL_CONTROL:
            self.manual_control_publisher = self.create_publisher(ManualControlSetpoint, '/fmu/in/manual_control_input', qos_profile)

        self.reward_publisher = self.create_publisher(Float32, '/training_reward', qos_profile)
        self.state_publisher = self.create_publisher(Float32MultiArray, '/states', qos_profile)

        #Choose Odom publisher
        self.odom_publisher_flag = OdomPublisher.GZ
        if self.odom_publisher_flag == OdomPublisher.PX4:
            self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, 
            '/fmu/out/vehicle_odometry', 
            self.vehicle_odometry_callback, 
            qos_profile, 
            callback_group= self.cb_group
            )
        elif self.odom_publisher_flag == OdomPublisher.GZ:
            self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, 
            '/gz_position', 
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

        self.is_testing = False
        self.vehicle_odom = VehicleOdometry()
        self.vehicle_status = VehicleStatus()
        if self.control_mode == ControlMode.DIRECT_ACTUATOR:
            self.offboard_mode_counter = 0
            self.actuator_motor_control = None #[-1, 1]
        elif self.control_mode == ControlMode.MANUAL_CONTROL:
            self.manual_control = [-1.0, 0.0, 0.0, 0.0] #[Throttle, Yaw, Pitch, Roll]
        self.calibration_state = CalibrationState.NAVIGATION

        self.timer = self.create_timer(0.01, self.timer_callback, callback_group=self.cb_group)

    def publish_reward(self, reward):
        msg = Float32()
        msg.data = reward
        self.reward_publisher.publish(msg)

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

        control = np.interp(control, [-1.0, 1.0], [0.0, 1.0])
        msg.control = np.array([control[0], control[1], control[2], control[3]]).tolist() + [0] * 8
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        self.actuator_motor_publisher.publish(msg)

        # self.get_logger().info(f'Sent control signals: {msg.control}')

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
    
    def arm(self): # Forced Disarm and Arm commmands
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0) #param2 = 21196.0 for forced

        self.get_logger().info("Arm Command sent")

    def disarm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0) #param2 = 21196.0 for forced

        self.get_logger().info("Disarm Command sent")

    def engage_offboard_mode(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)

        self.get_logger().info("Switching to offboard mode")

    #Z axis for postion and linear velocity changed from NED to ENU frame
    #axes for rotation and orientation maintained as it is

    def _get_position(self):
        return np.array([self.vehicle_odom.position[1], 
                        self.vehicle_odom.position[0], 
                        -self.vehicle_odom.position[2]])    
    def _get_vel(self):
        return np.array([self.vehicle_odom.velocity[1],
                        self.vehicle_odom.velocity[0],
                        -self.vehicle_odom.velocity[2],
                        self.vehicle_odom.angular_velocity[0],
                        self.vehicle_odom.angular_velocity[1],
                        self.vehicle_odom.angular_velocity[2]]) # angular velocity in order roll pitch yaw

    def _get_orientation(self):
        q = self.vehicle_odom.q

        w, x, y, z = q[0], q[2], q[1], -q[3]

        pitch = math.atan2(2.0 * (x*w + y*z), (- x*x - y*y + z*z + w*w))
        roll = math.asin(-2.0 * (z*x - w*y))
        yaw = -math.atan2(2.0 * (x*y + w*z), (x*x - y*y - z*z + w*w))

        return np.array([math.degrees(roll), math.degrees(pitch), math.degrees(yaw)])
    
    def calibrate(self):
        self.calibration_state = CalibrationState.CALIBRATE

        if self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
            self.disarm()
        
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_PREFLIGHT_CALIBRATION, 
            param1=1.0, 
            param2=1.0, 
            param3=1.0, 
            param5=1.0, 
            param6=1.0, 
            param7=1.0
            )

        self.get_clock().sleep_for(Duration(seconds=1))
        while self.vehicle_status.calibration_enabled == True:
            self.get_clock().sleep_for(Duration(seconds=0.01))
        
        self.get_logger().info(f'Calibration completed')
        self.calibration_state = CalibrationState.NAVIGATION

    def publish_states(self):
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label='rows', size=12, stride=12),
            MultiArrayDimension(label='cols', size=1, stride=1)
        ]

        msg.data = np.concatenate([self._get_position(), self._get_orientation(), self._get_vel()]).tolist()

        self.state_publisher.publish(msg)
    
    def direct_actuator_control_mode_timer(self):
        self.publish_offboardcontrol_heartbeat_signal()

        if(self.offboard_mode_counter < 101):
            self.offboard_mode_counter += 1
        
        if(self.offboard_mode_counter > 100):
            if(self.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD):
                self.engage_offboard_mode()

            if(
                self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and
                self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED and
                self.calibration_state == CalibrationState.NAVIGATION
            ):
                self.arm()

            if(
                self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and
                self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED and
                self.calibration_state == CalibrationState.NAVIGATION
            ):
                if self.actuator_motor_control is not None:
                    self.publish_actuator_motors(self.actuator_motor_control)

    def engage_stabilized_mode(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=7.0)

        self.get_logger().info("Switching to stabilized mode")

    def publish_manual_control(self):
        msg = ManualControlSetpoint()

        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.valid = True

        msg.data_source = ManualControlSetpoint.SOURCE_MAVLINK_0
        msg.throttle, msg.yaw, msg.pitch, msg.roll = map(float, self.manual_control)
        msg.sticks_moving = True

        self.manual_control_publisher.publish(msg)

    def manual_control_mode_timer(self):
        if(self.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_STAB):
            self.engage_stabilized_mode()

        else:
            if self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED :
                self.arm()
        self.publish_manual_control()

    def timer_callback(self):
        self.publish_states()

        if self.is_testing == True: return

        if self.control_mode == ControlMode.DIRECT_ACTUATOR:
            self.direct_actuator_control_mode_timer()

        elif self.control_mode == ControlMode.MANUAL_CONTROL:
            self.manual_control_mode_timer()        

class DroneEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, is_training=True):
        super(DroneEnv, self).__init__()

        self.gz_node = gz.transport13.Node()

        try:
            if not rclpy.ok():
                rclpy.init(args=None)
        except RuntimeError:
            pass

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
        self.max_steps = 1000
        self.target_position = np.array([0.0, 0.0, 3.0])
        self.sway_accumulation = np.array([0.0, 0.0, 0.0])
        self.episode_counter = 0
        self.steps_beyond_terminated = 0
        self.total_steps = 0
        self.is_odom_stabilised = True
        self.reward_file = None
        self.csv_writer = None
        self.episode_reward = 0
        
        self.init_reward_file()

    def init_reward_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'rewards/episode_reward_{timestamp}.csv'
        self.log_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(['timestamp', 'episode', 'reward'])
        self.node.get_logger().info(f'Logging episode rewards in {filename}')

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
    
    def wait(self, initial_position):
        start_time = time.time()
        threshold = 0.75

        while(time.time() - start_time < 10):
            position = self._get_original_observation()[:3]

            if(abs(position[0]-initial_position[0]) < threshold and abs(position[1] - initial_position[1]) < threshold):
                self.is_odom_stabilised = True
                break

        self.is_odom_stabilised = False

    def reset(self, seed = None, options = None, initial_position = None): #position = np.array([x,y,z])

        #TODO: randomized initial position
        super().reset(seed=seed)

        if initial_position is None:
            initial_position = np.array([0.0, 0.0, 0.0])
        
        if self.node.control_mode == ControlMode.DIRECT_ACTUATOR:
            self.node.actuator_motor_control = 4 * [-1.0]
        elif self.node.control_mode == ControlMode.MANUAL_CONTROL:
            self.node.manual_control = [-1.0, 0.0, 0.0, 0.0]
        self._goto(initial_position)

        self.sway_accumulation = np.array([0.0, 0.0, 0.0])
        self.last_step_time = time.perf_counter()
        self.total_steps += self.step_counter
        self.time_on_ground = 0
        self.step_counter = 0
        self.steps_beyond_terminated = 0
        self.episode_reward = 0

        initial_observation = self._get_observation()
        info = {
            'original_observation': self._get_original_observation()
        }

        self.episode_counter += 1
        self.node.get_logger().info(f'Total steps completed: {self.total_steps}')
        self.node.get_logger().info(f'Starting episode: {self.episode_counter}')

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
        
        # if(roll > math.pi/3 or roll < -math.pi/3 or pitch > math.pi/3 or pitch < -math.pi/3):
        #     self.node.get_logger().info(f'Pitch or Roll too steep. Terminating step.')
        #     return True
        
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
        
        if self.node.control_mode == ControlMode.DIRECT_ACTUATOR:
            self.node.actuator_motor_control = original_action #in range [-1, 1]
        elif self.node.control_mode == ControlMode.MANUAL_CONTROL:
            self.node.manual_control = original_action

        self.node.get_clock().sleep_for(Duration(seconds=0.01))
        observation = self._get_observation()
        truncated = (self.step_counter >= self.max_steps)
        terminated = self._get_done()
        done:bool = terminated or truncated

        reward = self._get_reward(terminated, original_action)
        self.node.publish_reward(reward) # Added a reward publisher

        info = {
            'original_action': action,
            'original_observation': self._get_original_observation(),
            'reward': reward
        }

        step_duration = time.perf_counter() - start_time
        info['step_duration'] = step_duration

        self.step_counter += 1
        self.episode_reward += reward

        if done or truncated:
            self.node.get_logger().info(f'Reward of episode {self.episode_counter}: {self.episode_reward}')

            timestamp = self.node.get_clock().now().nanoseconds / 1e9

            self.csv_writer.writerow([timestamp, self.episode_counter, self.episode_reward])
        return observation, reward, done, truncated, info
    
    def close(self):
        self._goto([0.0, 0.0, 0.0])
        self.node.actuator_motor_control = [-1.0] * 4

        if self.node:
            self.node.destroy_node()
