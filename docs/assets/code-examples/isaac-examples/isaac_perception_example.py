#!/usr/bin/env python3

"""
NVIDIA Isaac Perception Example
This example demonstrates a basic perception pipeline using Isaac ROS
for object detection and pose estimation.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import numpy as np
import cv2
from cv_bridge import CvBridge


class IsaacPerceptionNode(Node):
    def __init__(self):
        super().__init__('isaac_perception_node')

        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Create publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/isaac_perception/detections',
            10
        )

        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/isaac_perception/target_pose',
            10
        )

        # Initialize perception parameters
        self.confidence_threshold = 0.5
        self.detection_model = self.initialize_detection_model()

        self.get_logger().info('Isaac Perception Node initialized')

    def initialize_detection_model(self):
        """
        Initialize the detection model (in a real implementation, this would
        load a TensorRT optimized model)
        """
        # This is a placeholder - in real Isaac implementation,
        # this would load a TensorRT optimized model
        self.get_logger().info('Detection model initialized')
        return {"initialized": True}

    def image_callback(self, msg):
        """
        Process incoming image and perform perception
        """
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Perform object detection (placeholder implementation)
            detections = self.perform_detection(cv_image)

            # Publish detections
            self.publish_detections(detections, msg.header)

            # Estimate target pose if objects detected
            if detections.detections:
                target_pose = self.estimate_pose(detections, cv_image.shape)
                self.publish_pose(target_pose, msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def perform_detection(self, image):
        """
        Perform object detection using Isaac-optimized methods
        This is a simplified example - real implementation would use
        Isaac's GPU-accelerated detection nodes
        """
        from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose

        # In a real Isaac implementation, this would call the Isaac ROS detection node
        # For this example, we'll simulate detections
        detections_msg = Detection2DArray()

        # Simulate some detections (in real implementation, these would come from the model)
        height, width = image.shape[:2]

        # Example: Detect a person in the center
        if np.random.random() > 0.3:  # 70% chance of detection for demo
            detection = Detection2D()

            # Set bounding box (centered with some random offset)
            bbox_width = int(width * 0.2)
            bbox_height = int(height * 0.4)
            center_x = width // 2 + np.random.randint(-width//10, width//10)
            center_y = height // 2 + np.random.randint(-height//10, height//10)

            detection.bbox.center.x = float(center_x)
            detection.bbox.center.y = float(center_y)
            detection.bbox.size_x = float(bbox_width)
            detection.bbox.size_y = float(bbox_height)

            # Set detection hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = "person"
            hypothesis.hypothesis.score = 0.85
            detection.results = [hypothesis]

            detections_msg.detections.append(detection)

        return detections_msg

    def estimate_pose(self, detections, image_shape):
        """
        Estimate 3D pose from 2D detections
        In a real implementation, this would use Isaac's pose estimation capabilities
        """
        from geometry_msgs.msg import PoseStamped, Pose

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "camera_frame"  # This would be set appropriately

        # Simple pose estimation based on detection position
        # In real implementation, this would use depth information and Isaac's 3D capabilities
        if detections.detections:
            detection = detections.detections[0]  # Use first detection
            center_x = detection.bbox.center.x
            center_y = detection.bbox.center.y
            img_width, img_height = image_shape[1], image_shape[0]

            # Convert 2D position to relative 3D position (simplified)
            pose_msg.pose.position.x = 1.0  # Fixed distance for demo
            pose_msg.pose.position.y = (center_x - img_width/2) / img_width * 2.0  # Normalize to [-1, 1]
            pose_msg.pose.position.z = -(center_y - img_height/2) / img_height * 2.0  # Normalize to [-1, 1]

            # Set orientation (identity for now)
            pose_msg.pose.orientation.w = 1.0

        return pose_msg

    def publish_detections(self, detections, header):
        """
        Publish detection results
        """
        detections.header = header
        self.detection_pub.publish(detections)

    def publish_pose(self, pose, header):
        """
        Publish estimated pose
        """
        pose.header = header
        self.pose_pub.publish(pose)


def main(args=None):
    rclpy.init(args=args)

    perception_node = IsaacPerceptionNode()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        pass
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()