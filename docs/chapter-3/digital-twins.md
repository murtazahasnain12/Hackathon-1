# Digital Twins with Gazebo/Unity

## Introduction to Digital Twins in Robotics

A digital twin is a virtual representation of a physical system that enables real-time monitoring, simulation, and analysis. In robotics, digital twins serve as:
- **Virtual Prototyping**: Test robot designs before physical construction
- **Algorithm Development**: Develop and refine control algorithms in simulation
- **System Monitoring**: Monitor and predict real robot behavior
- **Predictive Maintenance**: Anticipate system failures and maintenance needs
- **Optimization**: Improve robot performance through virtual experimentation

For humanoid robotics, digital twins are particularly valuable due to the complexity and cost of physical humanoid platforms. They enable:
- **Safe Development**: Test complex behaviors without risk of hardware damage
- **Rapid Iteration**: Quickly test different control strategies
- **Cost-Effective Training**: Train AI models without expensive hardware
- **Scenario Testing**: Evaluate performance across diverse conditions

## Gazebo: The Robot Simulation Standard

### Overview

Gazebo is the most widely adopted simulation environment in robotics, particularly for ROS-based systems. It provides:
- **Realistic Physics**: Accurate simulation of rigid body dynamics
- **Sensor Simulation**: Realistic simulation of cameras, IMUs, LiDARs
- **ROS Integration**: Native support for ROS/ROS 2 message passing
- **Extensible Architecture**: Plugin system for custom sensors and controllers
- **Large Model Database**: Extensive library of robot and environment models

### Gazebo Architecture

#### Core Components
- **Gazebo Server**: Runs the physics simulation and sensor updates
- **Gazebo Client**: Provides visualization and user interface
- **Plugins**: Extend functionality for sensors, controllers, and models
- **World Files**: Define environments and simulation parameters

#### Physics Engine Integration
Gazebo supports multiple physics engines:
- **ODE (Open Dynamics Engine)**: Default engine, optimized for robotics
- **Bullet**: Robust collision detection and response
- **Simbody**: Multi-body dynamics simulation
- **DART**: Advanced articulated rigid body simulation

### Setting Up Gazebo for Humanoid Robots

#### World File Configuration
World files define the simulation environment:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_world">
    <!-- Include ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include sky -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Custom environment elements -->
    <model name="obstacle">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </visual>
      </link>
    </model>

    <!-- Physics parameters -->
    <physics name="default_physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>
  </world>
</sdf>
```

#### Robot Model Integration
Humanoid robots in Gazebo use URDF/SDF models:

```xml
<?xml version="1.0" ?>
<robot name="humanoid_robot">
  <!-- Links definition -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.5"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.5"/>
      <geometry>
        <box size="0.5 0.5 1.0"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.5"/>
      <geometry>
        <box size="0.5 0.5 1.0"/>
      </geometry>
    </collision>
  </link>

  <!-- Joint definition -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
  </joint>

  <link name="torso">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0.3"/>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.5"/>
    </inertial>
  </link>

  <!-- Additional joints and links for arms, legs, etc. -->
</robot>
```

### Gazebo Plugins for Humanoid Robotics

#### Sensor Plugins
Gazebo provides plugins for various sensors:

```xml
<!-- Camera sensor -->
<gazebo reference="head_camera">
  <sensor type="camera" name="head_camera_sensor">
    <update_rate>30.0</update_rate>
    <camera name="head_camera">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>/humanoid_robot</namespace>
        <remapping>~/image_raw:=/camera/image_raw</remapping>
        <remapping>~/camera_info:=/camera/camera_info</remapping>
      </ros>
    </plugin>
  </sensor>
</gazebo>

<!-- IMU sensor -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <plugin filename="libgazebo_ros_imu.so" name="imu_plugin">
      <ros>
        <namespace>/humanoid_robot</namespace>
        <remapping>~/out:=/imu/data</remapping>
      </ros>
      <initial_orientation_as_reference>false</initial_orientation_as_reference>
    </plugin>
  </sensor>
</gazebo>
```

#### Control Plugins
Gazebo interfaces with ROS 2 controllers:

```xml
<!-- ros2_control plugin -->
<gazebo>
  <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
    <parameters>$(find my_robot_description)/config/humanoid_control.yaml</parameters>
  </plugin>
</gazebo>
```

### Gazebo ROS 2 Integration

#### Launch Files
Launch Gazebo with ROS 2:

```python
# launch/gazebo_simulation.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    world_file = PathJoinSubstitution([
        FindPackageShare('my_robot_gazebo'),
        'worlds',
        'humanoid_world.sdf'
    ])

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': world_file,
            'verbose': 'true'
        }.items()
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher
    ])
```

#### Controller Configuration
Configure controllers for the simulated robot:

```yaml
# config/humanoid_control.yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz
    use_sim_time: true

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    left_leg_controller:
      type: position_controllers/JointGroupPositionController

    right_leg_controller:
      type: position_controllers/JointGroupPositionController

left_leg_controller:
  ros__parameters:
    joints:
      - left_hip_pitch
      - left_hip_roll
      - left_hip_yaw
      - left_knee
      - left_ankle_pitch
      - left_ankle_roll

right_leg_controller:
  ros__parameters:
    joints:
      - right_hip_pitch
      - right_hip_roll
      - right_hip_yaw
      - right_knee
      - right_ankle_pitch
      - right_ankle_roll
```

## Unity: Advanced Simulation for Humanoid Robotics

### Overview

Unity provides advanced 3D simulation capabilities with high-quality graphics and physics. For humanoid robotics, Unity offers:
- **Photorealistic Rendering**: High-quality visual simulation
- **Advanced Physics**: Sophisticated physics simulation
- **XR Integration**: Virtual and augmented reality support
- **Large Environment Support**: Complex, large-scale environments
- **Multi-platform Deployment**: Export to various platforms

### Unity Robotics Simulation Framework

#### Unity Robotics Hub
The Unity Robotics Hub provides:
- **Robot Framework**: Components for robot simulation
- **Sensor Toolkit**: Various sensor implementations
- **ROS/ROS 2 Bridge**: Communication with ROS systems
- **Example Scenes**: Pre-built environments for testing

#### ROS/Unity Bridge
The ROS/Unity bridge enables communication:

```csharp
// Example ROS publisher in Unity
using UnityEngine;
using RosMessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector;

public class UnityROSPublisher : MonoBehaviour
{
    ROSConnection ros;
    string topicName = "/unity_sensor_data";

    // Start is called before the first frame update
    void Start()
    {
        ros = ROSConnection.instance;
    }

    void Update()
    {
        // Create and publish sensor message
        var sensorMsg = new ImageMsg();
        sensorMsg.header.stamp = new TimeStamp(0, (uint)(Time.time * 1e9));
        sensorMsg.header.frame_id = "unity_camera";

        ros.Publish(topicName, sensorMsg);
    }
}
```

### Unity for Humanoid Robot Simulation

#### Humanoid Avatar Setup
Unity's humanoid avatar system provides:
- **Automatic Rigging**: Simplified humanoid model setup
- **Animation Retargeting**: Transfer animations between models
- **Balance Control**: Built-in balance and physics simulation
- **IK Solvers**: Inverse kinematics for natural movement

#### Physics Configuration
Configure Unity's physics for humanoid simulation:

```csharp
// Physics configuration for humanoid
public class HumanoidPhysics : MonoBehaviour
{
    public float gravityScale = 1.0f;
    public float massScale = 1.0f;
    public float balanceThreshold = 0.1f;

    private Rigidbody[] rigidbodies;
    private ConfigurableJoint[] joints;

    void Start()
    {
        SetupPhysics();
    }

    void SetupPhysics()
    {
        // Configure rigid bodies
        rigidbodies = GetComponentsInChildren<Rigidbody>();
        foreach (var rb in rigidbodies)
        {
            rb.mass *= massScale;
            rb.interpolation = RigidbodyInterpolation.Interpolate;
        }

        // Configure joints
        joints = GetComponentsInChildren<ConfigurableJoint>();
        ConfigureJoints();
    }

    void ConfigureJoints()
    {
        foreach (var joint in joints)
        {
            // Set joint limits and constraints
            joint.xMotion = ConfigurableJointMotion.Limited;
            joint.yMotion = ConfigurableJointMotion.Limited;
            joint.zMotion = ConfigurableJointMotion.Limited;
        }
    }
}
```

### Unity ML-Agents Integration

Unity ML-Agents enables reinforcement learning in simulation:

```python
# Python example using ML-Agents
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channels.engine_configuration_channel import EngineConfig, EngineConfigurationChannel

# Initialize Unity environment
engine_config_channel = EngineConfigurationChannel()
env = UnityEnvironment(
    file_name=None,  # Use in-editor simulation
    side_channels=[engine_config_channel]
)

engine_config_channel.set_configuration(EngineConfig(training=True, time_scale=1.0))

# Get behavior names
behavior_names = list(env.behavior_specs.keys())
print(f"Behavior names: {behavior_names}")

# Reset environment
env.reset()

# Get first behavior
behavior_name = behavior_names[0]
decision_steps, terminal_steps = env.get_steps(behavior_name)

# Run simulation loop
for episode in range(1000):
    env.reset()
    while True:
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # Process terminal steps (done episodes)
        if len(terminal_steps) > 0:
            break

        # Process decision steps
        if len(decision_steps) > 0:
            # Get actions from decision steps
            action = decision_steps.action_mask  # Use trained model here
            env.set_actions(behavior_name, action)
            env.step()

env.close()
```

## Digital Twin Architecture for Humanoid Robots

### Twin-to-Physical Mapping

The digital twin maintains synchronization with the physical robot:

```
Physical Robot → Data Collection → Digital Twin → Algorithm Testing → Physical Robot
     ↑                                                                 ↓
     ←—————————————— Feedback and Validation ←———————————————
```

#### Real-Time Synchronization
- **State Synchronization**: Joint positions, sensor readings
- **Environmental Synchronization**: Object positions, lighting conditions
- **Behavior Synchronization**: Active control strategies
- **Performance Synchronization**: Energy consumption, computation load

### Twin-to-Twin Collaboration

Multiple digital twins can collaborate:
- **Multi-Robot Simulation**: Coordinate multiple robot behaviors
- **Environment Sharing**: Shared understanding of the environment
- **Learning Transfer**: Share learned behaviors between twins
- **Load Distribution**: Distribute computation across multiple twins

## Simulation-to-Reality Transfer

### Domain Randomization

Domain randomization helps bridge the sim-to-real gap:

```python
# Example domain randomization
import numpy as np

class DomainRandomizer:
    def __init__(self):
        self.param_ranges = {
            'friction': (0.4, 0.8),
            'mass_multiplier': (0.9, 1.1),
            'gravity_multiplier': (0.95, 1.05),
            'sensor_noise': (0.0, 0.01)
        }

    def randomize_environment(self):
        """Randomize physics parameters each episode"""
        randomized_params = {}
        for param, (min_val, max_val) in self.param_ranges.items():
            randomized_params[param] = np.random.uniform(min_val, max_val)
        return randomized_params

    def apply_randomization(self, gazebo_env, params):
        """Apply randomization to Gazebo environment"""
        # Set friction coefficients
        # Adjust masses
        # Modify gravity
        # Add sensor noise
        pass
```

### System Identification

Improve simulation accuracy through system identification:

```python
# System identification for humanoid robot
import scipy.optimize as opt

class SystemIdentifier:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.collected_data = []

    def collect_data(self, real_robot, simulation):
        """Collect data from both real and simulated robots"""
        real_data = real_robot.get_sensor_data()
        sim_data = simulation.get_sensor_data()
        self.collected_data.append((real_data, sim_data))

    def identify_parameters(self):
        """Identify parameters that minimize sim-to-real gap"""
        def objective(params):
            # Set simulation parameters
            self.set_simulation_params(params)

            # Calculate error between real and simulated data
            total_error = 0
            for real_data, sim_data in self.collected_data:
                error = np.sum((real_data - sim_data) ** 2)
                total_error += error

            return total_error

        # Optimize parameters
        result = opt.minimize(objective, initial_guess)
        return result.x
```

## Best Practices for Digital Twins

### Model Accuracy

#### Physics Parameter Tuning
- **Friction coefficients**: Calibrate through physical experiments
- **Inertial properties**: Measure actual robot components
- **Actuator dynamics**: Model motor and transmission characteristics
- **Sensor noise**: Characterize actual sensor properties

#### Validation Procedures
- **Kinematic validation**: Verify joint limit and range accuracy
- **Dynamic validation**: Compare motion between sim and real
- **Sensor validation**: Verify sensor data similarity
- **Control validation**: Test identical controllers on both systems

### Performance Optimization

#### Simulation Speed
- **Simplified collision meshes**: Use simpler shapes for collision detection
- **Adaptive time stepping**: Adjust time step based on system complexity
- **Level of detail**: Reduce complexity for distant objects
- **Parallel processing**: Utilize multiple CPU cores

#### Memory Management
- **Resource pooling**: Reuse simulation objects
- **Lazy loading**: Load models only when needed
- **Caching**: Cache frequently computed values
- **Garbage collection**: Optimize memory allocation patterns

### Debugging and Visualization

#### Real-time Monitoring
- **State visualization**: Show robot state in simulation
- **Force visualization**: Display contact forces and torques
- **Trajectory visualization**: Show planned vs actual trajectories
- **Error visualization**: Highlight discrepancies

#### Logging and Analysis
- **Comprehensive logging**: Record all relevant simulation data
- **Reproducible experiments**: Ensure consistent results
- **Performance metrics**: Track simulation efficiency
- **Validation metrics**: Measure sim-to-real similarity

## Advanced Digital Twin Techniques

### Neural Simulation

Neural networks can enhance simulation:

```python
import torch
import torch.nn as nn

class NeuralDynamicsModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralDynamicsModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, state, action):
        # Combine state and action
        x = torch.cat([state, action], dim=-1)
        # Predict next state
        next_state = self.network(x)
        return next_state

# Use neural model to enhance physics simulation
class EnhancedSimulator:
    def __init__(self):
        self.physics_model = TraditionalPhysicsModel()
        self.neural_model = NeuralDynamicsModel(12, 64, 6)  # Example dimensions
        self.blend_factor = 0.1  # How much to trust neural model

    def step(self, state, action):
        # Traditional physics prediction
        physics_prediction = self.physics_model.predict(state, action)

        # Neural prediction
        neural_prediction = self.neural_model(state, action)

        # Blend predictions
        blended_prediction = (
            (1 - self.blend_factor) * physics_prediction +
            self.blend_factor * neural_prediction
        )

        return blended_prediction
```

### Adaptive Simulation

Simulation that adapts to improve accuracy:

```python
class AdaptiveSimulator:
    def __init__(self):
        self.error_threshold = 0.01
        self.adaptation_rate = 0.001
        self.simulation_params = self.initialize_params()

    def update_simulation(self, real_data, sim_data):
        """Update simulation parameters based on real-world data"""
        error = np.mean(np.abs(real_data - sim_data))

        if error > self.error_threshold:
            # Adapt parameters to reduce error
            self.adapt_parameters(real_data, sim_data)

    def adapt_parameters(self, real_data, sim_data):
        """Adjust simulation parameters based on error"""
        # Example: adjust friction coefficients
        for i, (real_val, sim_val) in enumerate(zip(real_data, sim_data)):
            error = real_val - sim_val
            # Update parameter based on error
            self.simulation_params[i] += self.adaptation_rate * error
```

## Integration with NVIDIA Isaac

### Isaac Sim Overview

NVIDIA Isaac Sim provides:
- **Photorealistic Simulation**: High-quality rendering for perception
- **PhysX Integration**: Advanced NVIDIA PhysX physics engine
- **AI Training Environment**: Optimized for deep learning
- **Isaac ROS Integration**: Native ROS 2 support

### Isaac Sim Configuration

```python
# Isaac Sim example configuration
from omni.isaac.kit import SimulationApp
import omni.isaac.core.utils.carb as carb_utils

# Launch Isaac Sim
config = {
    "headless": False,
    "rendering": True,
    "simulation_dt": 1.0/60.0
}
simulation_app = SimulationApp(config)

# Import robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

# Add robot to stage
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb_utils.acquire carb
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    exit()

# Load humanoid robot
robot_path = assets_root_path + "/Isaac/Robots/Humanoid/humanoid.usd"
add_reference_to_stage(robot_path, "/World/Humanoid")

# Start simulation
simulation_app.update()
```

## Challenges and Considerations

### Computational Requirements

Digital twins for humanoid robots require significant computational resources:
- **Real-time simulation**: Maintaining simulation speed
- **High-fidelity graphics**: For perception tasks
- **Complex physics**: Accurate multi-body dynamics
- **Large environments**: Detailed world models

### Model Complexity

Balancing accuracy and performance:
- **Level of detail**: How detailed should models be?
- **Physics parameters**: How many parameters to tune?
- **Sensor models**: How accurately to model sensors?
- **Environmental complexity**: How complex should environments be?

### Validation Challenges

Ensuring simulation validity:
- **Ground truth**: Obtaining accurate real-world measurements
- **Parameter sensitivity**: Small changes causing large differences
- **Emergent behaviors**: Unexpected behaviors in complex systems
- **Transfer validation**: Ensuring sim-to-real transfer effectiveness

## Summary

Digital twins using Gazebo and Unity provide powerful tools for developing humanoid robotics systems. Gazebo offers robust physics simulation with excellent ROS integration, while Unity provides high-quality graphics and advanced simulation capabilities. The integration of these tools with NVIDIA Isaac and ML-Agents creates comprehensive environments for developing, testing, and validating Physical AI systems.

Effective digital twin implementation requires careful attention to physics parameter tuning, validation procedures, and performance optimization. As simulation technology advances, we can expect increasingly realistic and useful digital twins that further bridge the gap between virtual and physical robotic systems.

The next section will explore the comprehensive references supporting these digital twin and simulation concepts.

## Navigation Links

- **Previous**: [Physics Simulation Fundamentals](./physics-simulation.md)
- **Next**: [Chapter 3 References](./references.md)
- **Up**: [Chapter 3](./index.md)

## Next Steps

Explore the comprehensive reference list for this chapter to understand the academic foundations of digital twin technology and physics simulation in robotics.