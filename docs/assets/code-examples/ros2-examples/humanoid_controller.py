#!/usr/bin/env python3

"""
Humanoid Robot Controller Example
This example demonstrates a basic controller for humanoid robots
using ROS 2 and the ros2_control framework.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
import numpy as np
import math


class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Joint names for a simple humanoid (left leg)
        self.joint_names = [
            'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll'
        ]

        # Publishers
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/position_trajectory_controller/joint_trajectory',
            10
        )

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.02, self.control_loop)  # 50 Hz

        # Initialize joint states
        self.current_joint_positions = {name: 0.0 for name in self.joint_names}
        self.target_joint_positions = {name: 0.0 for name in self.joint_names}

        self.get_logger().info('Humanoid Controller initialized')

    def joint_state_callback(self, msg):
        """Callback for joint state updates"""
        for i, name in enumerate(msg.name):
            if name in self.current_joint_positions:
                self.current_joint_positions[name] = msg.position[i]

    def control_loop(self):
        """Main control loop"""
        # Example: Simple walking gait pattern
        t = self.get_clock().now().nanoseconds / 1e9  # Time in seconds

        # Generate a simple walking pattern for the left leg
        step_phase = (t * 2) % (2 * math.pi)  # 2 rad/s walking frequency

        # Hip movement
        self.target_joint_positions['left_hip_pitch'] = 0.1 * math.sin(step_phase)
        self.target_joint_positions['left_hip_roll'] = 0.05 * math.sin(step_phase * 2)

        # Knee movement
        self.target_joint_positions['left_knee'] = 0.2 * math.sin(step_phase + math.pi/2)

        # Ankle movement for balance
        self.target_joint_positions['left_ankle_pitch'] = -0.05 * math.sin(step_phase)

        # Publish the trajectory
        self.publish_trajectory()

    def publish_trajectory(self):
        """Publish joint trajectory commands"""
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = [self.target_joint_positions[name] for name in self.joint_names]

        # Set velocities to 0 (for simplicity)
        point.velocities = [0.0] * len(self.joint_names)

        # Set acceleration to 0 (for simplicity)
        point.accelerations = [0.0] * len(self.joint_names)

        # Set time from start (0.1 seconds for quick response)
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 100000000  # 0.1 seconds

        trajectory_msg.points.append(point)
        self.trajectory_pub.publish(trajectory_msg)


def main(args=None):
    rclpy.init(args=args)
    humanoid_controller = HumanoidController()

    try:
        rclpy.spin(humanoid_controller)
    except KeyboardInterrupt:
        pass
    finally:
        humanoid_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()