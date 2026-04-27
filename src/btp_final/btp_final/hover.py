import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, ActuatorMotors, VehicleOdometry, ManualControlSetpoint
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np

class Hover(Node):
    def __init__(self):
        super().__init__('hover_node')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.cb_group = ReentrantCallbackGroup()


        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, 
            '/fmu/out/vehicle_odometry', 
            self.vehicle_odometry_callback, 
            qos_profile, 
            callback_group= self.cb_group
            )
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v4', self.vehicle_status_callback, qos_profile)

        self.vehicle_command_publisher = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.manual_control_publisher = self.create_publisher(ManualControlSetpoint, '/fmu/in/manual_control_input', qos_profile)

        self.vehicle_odom = VehicleOdometry()
        self.vehicle_status = VehicleStatus()
        self.manual_control = np.array([-1.0, 0.0, 0.0, 0.0])
        self.target_position = np.array([0.0, 0.0, -3.0])
        self.acc_z_err = 0.0
        self.last_z_err = None

        self.timer = self.create_timer(0.01, self.timer_callback, callback_group=self.cb_group)

    def vehicle_odometry_callback(self, msg):
        self.vehicle_odom = msg

    def vehicle_status_callback(self, msg):
        self.vehicle_status = msg

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
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0, param2=21196.0) #param2 = 21196.0 for forced

        self.get_logger().info("Arm Command sent")

    def disarm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0) #param2 = 21196.0 for forced

        self.get_logger().info("Disarm Command sent")

    def engage_stabilized_mode(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=7.0)

        self.get_logger().info("Switching to stabilized mode")

    def publish_manual_control(self):
        msg = ManualControlSetpoint()

        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.valid = True

        msg.data_source = ManualControlSetpoint.SOURCE_MAVLINK_0
        self.manual_control = np.clip(self.manual_control, -1, 1)
        msg.throttle, msg.yaw, msg.pitch, msg.roll = map(float, self.manual_control)
        msg.sticks_moving = True

        self.manual_control_publisher.publish(msg)

    def timer_callback(self):
        if(self.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_STAB):
            self.engage_stabilized_mode()
            return
        
        self.publish_manual_control()
        if(self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED):
            self.arm()
            return

        z_err = 1.0*(self.vehicle_odom.position[2] - self.target_position[2])
        delta_z_err = 0.0
        if self.last_z_err is not None: delta_z_err = z_err - self.last_z_err
        self.acc_z_err += z_err
        self.manual_control[0] = 0.4 * z_err + 0.00005 * self.acc_z_err + 15 * delta_z_err
        self.get_logger().info(f'p: {z_err}; i: {self.acc_z_err}; d: {delta_z_err}')
        self.last_z_err = z_err

def main(args = None):
    print('Starting Hover...')
    rclpy.init(args=args)
    hover = Hover()
    rclpy.spin(hover)
    hover.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()