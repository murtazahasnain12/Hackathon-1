#!/usr/bin/env python3

"""
NVIDIA Isaac SLAM Example
This example demonstrates a basic SLAM pipeline using Isaac ROS
for mapping and localization.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Header
import numpy as np


class IsaacSLAMNode(Node):
    def __init__(self):
        super().__init__('isaac_slam_node')

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Create publishers
        self.odom_pub = self.create_publisher(
            Odometry,
            '/isaac_slam/odometry',
            10
        )

        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/isaac_slam/map',
            10
        )

        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/isaac_slam/pose',
            10
        )

        # Initialize SLAM parameters
        self.slam_system = self.initialize_slam_system()
        self.robot_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.map_resolution = 0.05  # meters per cell
        self.map_width = 200  # cells
        self.map_height = 200  # cells
        self.map_origin_x = -5.0  # meters
        self.map_origin_y = -5.0  # meters

        # Initialize occupancy grid
        self.occupancy_grid = np.zeros((self.map_height, self.map_width), dtype=np.int8)

        self.get_logger().info('Isaac SLAM Node initialized')

    def initialize_slam_system(self):
        """
        Initialize the SLAM system (in a real implementation, this would
        initialize Isaac's GPU-accelerated SLAM components)
        """
        # This is a placeholder - in real Isaac implementation,
        # this would initialize the visual SLAM system
        self.get_logger().info('SLAM system initialized')
        return {"initialized": True, "type": "Visual SLAM"}

    def image_callback(self, msg):
        """
        Process incoming image for visual SLAM
        """
        try:
            # In a real Isaac implementation, this would feed to the visual SLAM pipeline
            # For this example, we'll simulate feature extraction and pose estimation
            self.process_visual_features(msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image for SLAM: {str(e)}')

    def imu_callback(self, msg):
        """
        Process IMU data for pose estimation
        """
        try:
            # Process IMU data to improve pose estimation
            # In real Isaac implementation, this would be fused with visual data
            self.process_imu_data(msg)

        except Exception as e:
            self.get_logger().error(f'Error processing IMU data: {str(e)}')

    def process_visual_features(self, image_msg):
        """
        Process visual features for SLAM (simulated)
        In a real Isaac implementation, this would use GPU-accelerated feature detection
        """
        # Simulate visual SLAM processing
        # In real implementation, this would call Isaac's visual SLAM nodes

        # Update robot pose based on simulated movement
        dt = 0.1  # 10Hz simulation
        linear_vel = 0.1  # m/s
        angular_vel = 0.05  # rad/s

        # Update pose using simple motion model
        self.robot_pose[0] += linear_vel * np.cos(self.robot_pose[2]) * dt
        self.robot_pose[1] += linear_vel * np.sin(self.robot_pose[2]) * dt
        self.robot_pose[2] += angular_vel * dt

        # Keep angle in [-pi, pi]
        self.robot_pose[2] = ((self.robot_pose[2] + np.pi) % (2 * np.pi)) - np.pi

        # Update occupancy grid with simulated sensor data
        self.update_occupancy_grid()

        # Publish results
        self.publish_slam_results(image_msg.header)

    def process_imu_data(self, imu_msg):
        """
        Process IMU data to improve pose estimation
        """
        # In a real Isaac implementation, this would be fused with visual data
        # For simulation, we'll just log the IMU data
        self.get_logger().debug(f'IMU data received: angular_velocity={imu_msg.angular_velocity}')

    def update_occupancy_grid(self):
        """
        Update the occupancy grid with simulated sensor data
        """
        # Simulate adding obstacles to the map
        robot_x, robot_y, robot_theta = self.robot_pose

        # Convert robot coordinates to grid coordinates
        grid_x = int((robot_x - self.map_origin_x) / self.map_resolution)
        grid_y = int((robot_y - self.map_origin_y) / self.map_resolution)

        # Add some simulated obstacles near the robot
        if 0 <= grid_x < self.map_width and 0 <= grid_y < self.map_height:
            # Add obstacles in a pattern around the robot
            obstacle_offsets = [
                (-1, 0), (1, 0), (0, -1), (0, 1),  # Cross pattern
                (-1, -1), (1, -1), (-1, 1), (1, 1)  # Diagonals
            ]

            for dx, dy in obstacle_offsets:
                obs_x = grid_x + dx
                obs_y = grid_y + dy
                if 0 <= obs_x < self.map_width and 0 <= obs_y < self.map_height:
                    # Set cell as occupied (100 = definitely occupied)
                    self.occupancy_grid[obs_y, obs_x] = 100

    def publish_slam_results(self, header):
        """
        Publish SLAM results (odometry, map, pose)
        """
        # Publish odometry
        odom_msg = Odometry()
        odom_msg.header = header
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = "base_link"

        # Set position
        odom_msg.pose.pose.position.x = float(self.robot_pose[0])
        odom_msg.pose.pose.position.y = float(self.robot_pose[1])
        odom_msg.pose.pose.position.z = 0.0

        # Convert angle to quaternion
        from mathutils import angle_to_quaternion
        quat = angle_to_quaternion(self.robot_pose[2])
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]

        # Set velocity (simulated)
        odom_msg.twist.twist.linear.x = 0.1  # m/s
        odom_msg.twist.twist.angular.z = 0.05  # rad/s

        self.odom_pub.publish(odom_msg)

        # Publish pose
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = "map"
        pose_msg.pose = odom_msg.pose.pose

        self.pose_pub.publish(pose_msg)

        # Publish map
        self.publish_map(header)

    def publish_map(self, header):
        """
        Publish occupancy grid map
        """
        map_msg = OccupancyGrid()
        map_msg.header = header
        map_msg.header.frame_id = "map"

        # Set map metadata
        map_msg.info.resolution = self.map_resolution
        map_msg.info.width = self.map_width
        map_msg.info.height = self.map_height
        map_msg.info.origin.position.x = self.map_origin_x
        map_msg.info.origin.position.y = self.map_origin_y
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0

        # Flatten the occupancy grid for the message
        # Note: ROS occupancy grid is row-major order (row by row)
        flat_grid = self.occupancy_grid.flatten()
        map_msg.data = [int(occ) for occ in flat_grid]

        self.map_pub.publish(map_msg)


def angle_to_quaternion(angle):
    """
    Convert angle (in radians) to quaternion representation
    """
    # For rotation around Z axis only
    cy = np.cos(angle * 0.5)
    sy = np.sin(angle * 0.5)
    return [0, 0, sy, cy]  # x, y, z, w


def main(args=None):
    rclpy.init(args=args)

    slam_node = IsaacSLAMNode()

    try:
        rclpy.spin(slam_node)
    except KeyboardInterrupt:
        pass
    finally:
        slam_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()