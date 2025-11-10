from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, LogInfo
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch_ros.actions import Node
import time
import rclpy
from rclpy.node import Node as ROS2Node


def wait_for_topic(topic_name, timeout=60.0):
    """
    Wait for a topic to be available.
    Returns True if topic is found, False if timeout occurs.
    """
    rclpy.init()
    temp_node = rclpy.create_node('topic_waiter')
    
    start_time = time.time()
    rate = temp_node.create_rate(2)  # Check at 2 Hz
    
    print(f"Waiting for topic: {topic_name}")
    
    while time.time() - start_time < timeout:
        topic_list = temp_node.get_topic_names_and_types()
        if any(topic_name in topic[0] for topic in topic_list):
            print(f"Topic {topic_name} found!")
            temp_node.destroy_node()
            rclpy.shutdown()
            return True
        rate.sleep()
    
    print(f"Timeout waiting for topic: {topic_name}")
    temp_node.destroy_node()
    rclpy.shutdown()
    return False


def generate_launch_description():
    """
    Generate launch description with sequential task execution.
    """
    
    # Task 1: Start PX4 SITL with Gazebo
    px4_sitl = ExecuteProcess(
        cmd=['make', 'px4_sitl', 'gz_x500'],
        output='screen',
        name='px4_sitl',
        env={'HEADLESS': '1'},
        shell=False
    )
    
    # Task 2: Start MicroXRCE Agent
    micro_xrce_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        output='screen',
        name='micro_xrce_agent',
        shell=False
    )
    
    # Handler to start MicroXRCE Agent after PX4 SITL starts
    start_agent_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=px4_sitl,
            on_start=[
                LogInfo(msg="PX4 SITL started, waiting 5 seconds..."),
                ExecuteProcess(
                    cmd=['sleep', '5'],
                    output='screen',
                    on_exit=[micro_xrce_agent]
                )
            ]
        )
    )
    
    # Task 3: Wait for /fmu/out/vehicle_odometry topic
    # This is handled by a custom ExecuteProcess that runs a Python script
    wait_for_odometry = ExecuteProcess(
        cmd=[
            'python3', '-c',
            """
import rclpy
from rclpy.node import Node
import time
import sys

rclpy.init()
node = rclpy.create_node('odometry_waiter')
timeout = 60.0
start_time = time.time()

print('Waiting for /fmu/out/vehicle_odometry topic...')
while time.time() - start_time < timeout:
    topics = node.get_topic_names_and_types()
    if any('/fmu/out/vehicle_odometry' in t[0] for t in topics):
        print('Topic /fmu/out/vehicle_odometry found!')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)
    time.sleep(0.5)

print('Timeout waiting for /fmu/out/vehicle_odometry')
node.destroy_node()
rclpy.shutdown()
sys.exit(1)
"""
        ],
        output='screen',
        name='wait_for_odometry',
        shell=False
    )
    
    # Handler to start odometry waiter after MicroXRCE Agent starts
    start_odometry_waiter = RegisterEventHandler(
        OnProcessStart(
            target_action=micro_xrce_agent,
            on_start=[
                LogInfo(msg="MicroXRCE Agent started, waiting 3 seconds before checking for odometry topic..."),
                ExecuteProcess(
                    cmd=['sleep', '3'],
                    output='screen',
                    on_exit=[wait_for_odometry]
                )
            ]
        )
    )
    
    # Task 4: Run gz_bridge node and wait for /gz_position topic
    gz_bridge_node = Node(
        package='CPPO_SITL',
        executable='gz_bridge',
        name='gz_bridge',
        output='screen'
    )
    
    wait_for_gz_position = ExecuteProcess(
        cmd=[
            'python3', '-c',
            """
import rclpy
from rclpy.node import Node
import time
import sys

rclpy.init()
node = rclpy.create_node('gz_position_waiter')
timeout = 60.0
start_time = time.time()

print('Waiting for /gz_position topic...')
while time.time() - start_time < timeout:
    topics = node.get_topic_names_and_types()
    if any('/gz_position' in t[0] for t in topics):
        print('Topic /gz_position found!')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)
    time.sleep(0.5)

print('Timeout waiting for /gz_position')
node.destroy_node()
rclpy.shutdown()
sys.exit(1)
"""
        ],
        output='screen',
        name='wait_for_gz_position',
        shell=False
    )
    
    # Handler to start gz_bridge after odometry topic is found
    start_gz_bridge = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_for_odometry,
            on_exit=[
                LogInfo(msg="Odometry topic found, starting gz_bridge node..."),
                gz_bridge_node,
                ExecuteProcess(
                    cmd=['sleep', '2'],
                    output='screen',
                    on_exit=[wait_for_gz_position]
                )
            ]
        )
    )
    
    # Task 5: Run train node
    train_node = Node(
        package='CPPO_SITL',
        executable='train',
        name='train',
        output='screen'
    )
    
    # Handler to start train node after gz_position topic is found
    start_train = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_for_gz_position,
            on_exit=[
                LogInfo(msg="gz_position topic found, starting train node..."),
                train_node
            ]
        )
    )
    
    return LaunchDescription([
        px4_sitl,
        start_agent_handler,
        start_odometry_waiter,
        start_gz_bridge,
        start_train
    ])