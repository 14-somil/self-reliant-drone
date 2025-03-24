#!/usr/bin/env python3

import rclpy
import csv
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from datetime import datetime
import os
import time
from enum import Enum, auto


class MissionState(Enum):
    IDLE = auto()
    TAKEOFF = auto()
    WAYPOINT_NAVIGATION = auto()
    LAND = auto()
    COMPLETE = auto()


class WaypointNavigation(Node):
    """Node for controlling a vehicle in offboard mode with waypoint navigation."""

    def __init__(self) -> None:
        super().__init__('waypoint_navigation')

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1', self.vehicle_status_callback, qos_profile)

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.takeoff_height = -5.0  # Negative because PX4 uses NED frame
        self.mission_state = MissionState.IDLE
        self.waypoint_index = 0
        self.waypoints = []
        self.current_waypoint = None
        self.waypoint_reached_threshold = 0.5  # meters
        self.logging_rate = 10  # Hz
        self.last_log_time = 0
        self.log_file = None
        self.csv_writer = None
        
        # Load waypoints from CSV file
        self.load_waypoints('/home/plague/auto_drone_ws/src/px4_nav/px4_nav/waypoints.csv')
        
        # Initialize CSV output file
        self.init_log_file()

        # Create a timer to publish control commands (10 Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def load_waypoints(self, filename):
        """Load waypoints from a CSV file."""
        try:
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header row
                for row in reader:
                    # CSV format: x, y, z, yaw
                    # Note: z is negated since PX4 uses NED frame (negative z is up)
                    x, y, z, yaw = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    self.waypoints.append({'x': x, 'y': y, 'z': -abs(z), 'yaw': yaw})
                
            self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints from {filename}')
        except Exception as e:
            self.get_logger().error(f'Failed to load waypoints from {filename}: {e}')
            self.get_logger().info('Using default waypoint pattern instead')
            # Default square pattern if no file exists
            self.waypoints = [
                {'x': 5.0, 'y': 0.0, 'z': self.takeoff_height, 'yaw': 0.0},
                {'x': 5.0, 'y': 5.0, 'z': self.takeoff_height, 'yaw': 1.57},
                {'x': 0.0, 'y': 5.0, 'z': self.takeoff_height, 'yaw': 3.14},
                {'x': 0.0, 'y': 0.0, 'z': self.takeoff_height, 'yaw': 4.71}
            ]

    def init_log_file(self):
        """Initialize the log file for position data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'flight_log_{timestamp}.csv'
        self.log_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        # Write header with position only
        self.csv_writer.writerow(['timestamp', 'x', 'y', 'z'])
        self.get_logger().info(f'Logging flight position data to {filename}')

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position
        
        # Log position at fixed rate
        current_time = time.time()
        if current_time - self.last_log_time >= 1.0 / self.logging_rate:
            self.log_vehicle_state()
            self.last_log_time = current_time

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def log_vehicle_state(self):
        """Log the vehicle position to CSV file."""
        if self.csv_writer is None:
            return
            
        # Get timestamp in seconds
        timestamp = self.get_clock().now().nanoseconds / 1e9
        
        # Get position (NED frame)
        x = self.vehicle_local_position.x
        y = self.vehicle_local_position.y
        z = self.vehicle_local_position.z
        
        # Write only position data to CSV
        self.csv_writer.writerow([timestamp, x, y, z])

    def quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        # Extract quaternion components
        w, x, y, z = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
            
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float, yaw: float):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = yaw
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing position setpoint: x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw:.2f}")

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

    def check_waypoint_reached(self):
        """Check if current waypoint has been reached."""
        if self.current_waypoint is None:
            return False
            
        dx = self.vehicle_local_position.x - self.current_waypoint['x']
        dy = self.vehicle_local_position.y - self.current_waypoint['y']
        dz = self.vehicle_local_position.z - self.current_waypoint['z']
        
        distance = np.sqrt(dx*dx + dy*dy + dz*dz)
        return distance < self.waypoint_reached_threshold

    def timer_callback(self) -> None:
        """Callback function for the timer."""
        # Always publish offboard control heartbeat
        self.publish_offboard_control_heartbeat_signal()

        # State machine for mission execution
        if self.mission_state == MissionState.IDLE:
            # We need to send setpoints before arming
            if self.offboard_setpoint_counter < 10:
                self.publish_position_setpoint(0.0, 0.0, 0.0, 0.0)
                self.offboard_setpoint_counter += 1
            else:
                self.engage_offboard_mode()
                self.arm()
                self.mission_state = MissionState.TAKEOFF
                self.get_logger().info("Starting takeoff")
                
        elif self.mission_state == MissionState.TAKEOFF:
            # Check if we're in offboard mode and armed
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                # Send takeoff setpoint
                self.publish_position_setpoint(0.0, 0.0, self.takeoff_height, 0.0)
                
                # Check if we've reached takeoff height
                if abs(self.vehicle_local_position.z - self.takeoff_height) < 0.5:
                    self.mission_state = MissionState.WAYPOINT_NAVIGATION
                    self.waypoint_index = 0
                    if self.waypoints:
                        self.current_waypoint = self.waypoints[0]
                        self.get_logger().info("Takeoff complete, starting waypoint navigation")
                    else:
                        self.mission_state = MissionState.LAND
                        self.get_logger().info("No waypoints defined, proceeding to land")
                        
        elif self.mission_state == MissionState.WAYPOINT_NAVIGATION:
            # Ensure we have a valid current waypoint
            if self.current_waypoint is not None:
                # Send the current waypoint
                self.publish_position_setpoint(
                    self.current_waypoint['x'],
                    self.current_waypoint['y'],
                    self.current_waypoint['z'],
                    self.current_waypoint['yaw']
                )
                
                # Check if we've reached the waypoint
                if self.check_waypoint_reached():
                    self.waypoint_index += 1
                    if self.waypoint_index < len(self.waypoints):
                        # Move to next waypoint
                        self.current_waypoint = self.waypoints[self.waypoint_index]
                        self.get_logger().info(f"Waypoint {self.waypoint_index-1} reached, moving to waypoint {self.waypoint_index}")
                    else:
                        # All waypoints visited, proceed to landing
                        self.mission_state = MissionState.LAND
                        self.get_logger().info("All waypoints visited, proceeding to land")
                        
        elif self.mission_state == MissionState.LAND:
            self.land()
            self.mission_state = MissionState.COMPLETE
            self.get_logger().info("Landing initiated")
            
        elif self.mission_state == MissionState.COMPLETE:
            # Close the log file if it's still open
            if self.log_file and not self.log_file.closed:
                self.log_file.close()
                self.get_logger().info("Log file closed")
                
            # Nothing more to do, we could exit here
            pass


def main(args=None) -> None:
    print('Starting waypoint navigation node...')
    rclpy.init(args=args)
    waypoint_navigation = WaypointNavigation()
    rclpy.spin(waypoint_navigation)
    
    # Cleanup before shutdown
    if waypoint_navigation.log_file and not waypoint_navigation.log_file.closed:
        waypoint_navigation.log_file.close()
        
    waypoint_navigation.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)