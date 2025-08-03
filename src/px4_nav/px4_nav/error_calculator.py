import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import gz.transport13
from gz.msgs10.pose_v_pb2 import Pose_V
from px4_msgs.msg import VehicleOdometry
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import pandas as pd
import math 

class ErrorCalculator(Node):
    def __init__(self):
        super().__init__('error_calculator')
        self.gz_node = gz.transport13.Node()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.ros_callback, qos_profile)
        self.gz_node.subscribe(Pose_V, '/world/default/dynamic_pose/info', self.gz_callback)

        self.estimated_location = []
        self.actual_location = []
        self.logging_rate = 10  # Hz
        self.last_log_time = 0
        self.log_file = None
        self.csv_writer = None
        self.filename = None
        self.timestamp = None
        self.cummulative_error = 0
        self.counter = 0

        self.init_log_file()

        self.timer = self.create_timer(0.1, self.timer_callback)
    
    def init_log_file(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f'error_log_{self.timestamp}.csv'
        self.log_file = open(self.filename, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        # Write header with position and yaw
        self.csv_writer.writerow(['timestamp', 'x_actual', 'y_actual', 'x_estimated', 'y_estimated'])
        self.get_logger().info(f'Logging flight data to {self.filename}')
    
    def ros_callback(self, msg):
        self.estimated_location = (msg.position[0], msg.position[1])

    def gz_callback(self, msg):
        self.actual_location = (msg.pose[0].position.y, msg.pose[0].position.x)

    def timer_callback(self):
        if len(self.actual_location) != 0 and len(self.estimated_location) != 0:
            if self.csv_writer is None:
                return
                
            timestamp = self.get_clock().now().nanoseconds / 1e9

            self.csv_writer.writerow([timestamp, self.actual_location[0], self.actual_location[1], self.estimated_location[0], self.estimated_location[1]])

            self.get_logger().info(f'Logged data at {timestamp}')

            self.cummulative_error += math.sqrt((self.actual_location[0] - self.estimated_location[0]) ** 2 + (self.actual_location[1] - self.estimated_location[1]) ** 2)
            self.counter += 1
    
    def plot(self):
        df = pd.read_csv(self.filename)

        if df.empty:
            self.get_logger().warn('No data to plot.')
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))

        ax.plot(df['x_actual'], df['y_actual'], 'b-', linewidth=2, alpha=0.7, label=f'Actual Path')
        ax.plot(df['x_actual'].iloc[0], df['y_actual'].iloc[0], 'bo', markersize=8, label=f'Actual Start')
        ax.plot(df['x_actual'].iloc[-1], df['y_actual'].iloc[-1], 'bX', markersize=10, label=f'Actual End')

        ax.plot(df['x_estimated'], df['y_estimated'], 'r-', linewidth=2, alpha=0.7, label=f'Estimated Path')
        ax.plot(df['x_estimated'].iloc[0], df['y_estimated'].iloc[0], 'ro', markersize=8, label=f'Estimated Start')
        ax.plot(df['x_estimated'].iloc[-1], df['y_estimated'].iloc[-1], 'rX', markersize=10, label=f'Estimated End')

        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best', fontsize=10)
        
        # Make sure the aspect ratio is equal so the paths aren't distorted
        ax.set_aspect('equal')
        
        plt.savefig(f'plot_{self.timestamp}.png')
        # Show the plot
        plt.tight_layout()
        plt.show()

def main(args = None) -> None:
    rclpy.init(args= args)
    print('Error calculator node started...')
    error_calculator = ErrorCalculator()
    try:
        rclpy.spin(error_calculator)
    except KeyboardInterrupt:
        pass
    finally:
        error_calculator.log_file.close()
        error_calculator.get_logger().info(f"\033[94mAbsolute error is {error_calculator.cummulative_error/error_calculator.counter} \033[0m")
        error_calculator.plot()
        error_calculator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()