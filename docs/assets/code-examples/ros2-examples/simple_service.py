#!/usr/bin/env python3

"""
Simple ROS 2 Service Example
This example demonstrates a basic service server and client in ROS 2.
"""

import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class SimpleServiceServer(Node):
    def __init__(self):
        super().__init__('simple_service_server')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning: {request.a} + {request.b} = {response.sum}')
        return response


class SimpleServiceClient(Node):
    def __init__(self):
        super().__init__('simple_service_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def server_main(args=None):
    rclpy.init(args=args)
    simple_service_server = SimpleServiceServer()

    try:
        rclpy.spin(simple_service_server)
    except KeyboardInterrupt:
        pass
    finally:
        simple_service_server.destroy_node()
        rclpy.shutdown()


def client_main(args=None):
    rclpy.init(args=args)
    simple_service_client = SimpleServiceClient()

    try:
        response = simple_service_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
        print(f'Result of add_two_ints: {response.sum}')
    except KeyboardInterrupt:
        pass
    finally:
        simple_service_client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # Run as server
        server_main()
    elif len(sys.argv) == 3:
        # Run as client
        client_main()
    else:
        print("Usage: python simple_service.py [arg1] [arg2] for client, or no args for server")