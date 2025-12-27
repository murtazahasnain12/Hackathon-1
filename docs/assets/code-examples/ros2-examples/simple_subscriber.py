#!/usr/bin/env python3

"""
Simple ROS 2 Subscriber Example
This example demonstrates a basic subscriber node that subscribes
to messages from a topic in ROS 2.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimpleSubscriber(Node):
    def __init__(self):
        super().__init__('simple_subscriber')

        # Create a subscription
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    rclpy.init(args=args)

    simple_subscriber = SimpleSubscriber()

    try:
        rclpy.spin(simple_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        simple_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()