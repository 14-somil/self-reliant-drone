# Self Reliant Drone

## PX4 and ROS2 Setup

### PX4 Setup
* Install PX4 Ubuntu Development Environment
```sh 
git clone https://github.com/PX4/PX4-Autopilot.git --recursive 
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh
```

### ROS2 Setup
* Install ROS2 by following this [link](https://docs.ros.org/en/rolling/Installation.html).

### Micro-XRCE Agent
* Setup Micro-XRCE-DDS Agent & Client
```sh
git clone -b v2.4.3 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig /usr/local/lib/
```
### Clone and build the repo
* Install and build the ROS2 packages using colcon
```sh
git clone https://github.com/14-somil/self-reliant-drone.git
cd self-reliant-drone
colcon build --symlink-install
source ./install/setup.sh
```

## Train Model
* Edit paramenters in `./src/CPPO_SITL/CPPO_SITL/train.py`.
* Run PX4-Gazebo
```sh
make px4_sitl gz_x500 # To launch with Gazebo GUI
HEADLESS=1 make px4_sitl gz_x500 # To Launch without GUI
```
* Run Micro-XRCE Agent
```sh
MicroXRCEAgent udp4 -p 8888
```
* Run the training script
```sh
ros2 run CPPO_SITL train # To train with new parameters
ros2 run CPPO_SITL train <file_name> # To train using a pre-trained model
```

## Getting inference
* Edit the filename in `./src/CPPO_SITL/CPPO_SITL/inference.py`.
* Run the inference script
```sh
ros2 run CPPO_SITL inference
```

## Notes
* PX4's EKF4 results in a drift from actual position. To avoid this use Gazebo's topic by changing the enum `odom_publisher_flag` to `OdomPublisher.GZ` and then run 
```sh
ros2 run CPPO_SITL gz_bridge
``` 