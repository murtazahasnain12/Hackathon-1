# ROS 2 Code Examples

This directory contains example code for the ROS 2 Foundations chapter, demonstrating key concepts for humanoid robotics.

## Examples Included

### 1. Simple Publisher (`simple_publisher.py`)
- Basic ROS 2 publisher node
- Publishes "Hello World" messages to a topic
- Demonstrates node creation and publishing

### 2. Simple Subscriber (`simple_subscriber.py`)
- Basic ROS 2 subscriber node
- Subscribes to messages from a topic
- Demonstrates node creation and subscription

### 3. Simple Service (`simple_service.py`)
- Basic ROS 2 service server and client
- Demonstrates request/response communication
- Shows how to handle service calls

### 4. Humanoid Controller (`humanoid_controller.py`)
- Advanced example for humanoid robot control
- Implements a simple walking gait pattern
- Demonstrates trajectory control for humanoid joints

## Running the Examples

### Prerequisites
- ROS 2 (Humble Hawksbill or later recommended)
- Python 3.8 or later
- `rclpy` and standard ROS 2 message packages

### Running Python Nodes

```bash
# Terminal 1: Start the publisher
python3 simple_publisher.py

# Terminal 2: Start the subscriber
python3 simple_subscriber.py
```

### Running the Service Example

```bash
# Terminal 1: Start the service server
python3 simple_service.py

# Terminal 2: Call the service (with two integer arguments)
python3 simple_service.py 10 20
```

### Running the Humanoid Controller

```bash
# Terminal 1: Start the controller
python3 humanoid_controller.py

# Make sure you have a joint trajectory controller running:
# ros2 run joint_state_publisher joint_state_publisher
# ros2 run robot_state_publisher robot_state_publisher
```

## Key Concepts Demonstrated

- Node creation and lifecycle
- Publisher/subscriber communication patterns
- Service/client request/response patterns
- Joint trajectory control for humanoid robots
- Real-time control considerations
- ROS 2 parameter and configuration handling

These examples provide a foundation for understanding ROS 2 concepts as they apply to humanoid robotics systems.