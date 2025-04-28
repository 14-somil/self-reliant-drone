import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import gz.transport13
from gz.msgs10.pose_v_pb2 import Pose_V
from px4_msgs.msg import VehicleOdometry
import time
import threading
import csv
import matplotlib.pyplot as plt
import datetime

class LivePlotter(Node):
    def __init__(self):
        super().__init__('live_plotter')
        self.gz_node = gz.transport13.Node()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.ros_callback, qos_profile)
        self.gz_node.subscribe(Pose_V, '/world/lawn/dynamic_pose/info', self.gz_callback)

        self.gz_positions = []
        self.ros2_positions = []

        self.running = True
        self.plot_thread = threading.Thread(target=self.live_plot)
        self.plot_thread.start()
    
    def ros_callback(self, msg):
        self.ros2_positions.append((msg.position[0], msg.position[1]))
    
    def gz_callback(self, msg):
        self.gz_positions.append((msg.pose[0].position.y, msg.pose[0].position.x))
    
    def live_plot(self):
        plt.ion()
        fig, ax = plt.subplots()
        gz_line, = ax.plot([], [], label='Gazebo', color='blue')
        ros2_line, = ax.plot([], [], label='ROS 2', color='red')

        ax.set_title("Live Odometry Plot")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend()
        ax.grid(True)
        
        while self.running:
            if self.gz_positions:
                gz_xs, gz_ys = zip(*self.gz_positions)
                gz_line.set_data(gz_xs, gz_ys)

            if self.ros2_positions:
                ros_xs, ros_ys = zip(*self.ros2_positions)
                ros2_line.set_data(ros_xs, ros_ys)

            # Dynamically adjust limits
            all_x = [x for x, _ in self.gz_positions + self.ros2_positions]
            all_y = [y for _, y in self.gz_positions + self.ros2_positions]
            if all_x and all_y:
                ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
                ax.set_ylim(min(all_y) - 1, max(all_y) + 1)

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.01)

        plt.ioff()
        plt.show()

def main(args = None) -> None:
    print('Live position plotter running...')
    rclpy.init(args= args)
    plotter = LivePlotter()
    try:
        rclpy.spin(plotter)
    except KeyboardInterrupt:
        plotter.get_logger().info("Shutting down...")
    finally:
        plotter.running = False
        plotter.plot_thread.join()
        plotter.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()