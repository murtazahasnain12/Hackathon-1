# NVIDIA Isaac Code Examples

This directory contains example code for the AI Perception & Learning with NVIDIA Isaac chapter, demonstrating key concepts for implementing AI perception systems using Isaac's GPU-accelerated tools.

## Examples Included

### 1. Isaac Perception Example (`isaac_perception_example.py`)
- Basic perception pipeline using Isaac ROS
- Object detection and pose estimation
- GPU-accelerated processing concepts
- Integration with ROS 2 message types

### 2. Isaac SLAM Example (`isaac_slam_example.py`)
- Visual SLAM pipeline using Isaac concepts
- IMU and visual sensor fusion
- Occupancy grid mapping
- GPU-accelerated SLAM processing

## Running the Examples

### Prerequisites
- ROS 2 (Humble Hawksbill or later recommended)
- NVIDIA GPU with CUDA support
- Isaac ROS packages installed
- Python 3.8 or later
- Required Python packages: `rclpy`, `cv2`, `cv_bridge`, `numpy`

### Setting up Isaac ROS

Before running these examples, you need to install Isaac ROS packages:

```bash
# Install Isaac ROS common packages
sudo apt update
sudo apt install ros-humble-isaac-ros-common

# Install Isaac ROS perception packages
sudo apt install ros-humble-isaac-ros-perception

# Install Isaac ROS navigation packages
sudo apt install ros-humble-isaac-ros-navigation
```

### Running the Examples

```bash
# Terminal 1: Launch camera simulation (if needed)
# ros2 launch realsense2_camera rs_launch.py

# Terminal 2: Run the perception example
python3 isaac_perception_example.py

# Terminal 3: Run the SLAM example
python3 isaac_slam_example.py
```

## Key Concepts Demonstrated

- Isaac ROS integration with GPU acceleration
- Perception pipeline design
- Sensor fusion for robust perception
- Real-time processing with GPU acceleration
- ROS 2 message types for perception
- Isaac-specific optimization techniques

These examples provide a foundation for understanding how to implement AI perception systems using NVIDIA Isaac's GPU-accelerated tools and frameworks.