run  model on native + px4 and ros2 docker + sitl body native

Component	Version	Why
Windows	11 (22H2 or newer)	WSLg GPU passthrough required
WSL2	1.2.5+	Latest GPU/CUDA support
Ubuntu	22.04 LTS (Jammy)	ROS 2 Humble's target OS, PX4 officially supports it
ROS 2	Humble Hawksbill	Long-term support until 2027, matches Ubuntu 22.04
PX4 Autopilot	v1.15.0 (or latest main)	Stable, supports uXRCE-DDS, Gazebo Ignition
Gazebo	Ignition Fortress (gz-sim)	New standard, GPU accelerated, camera/LIDAR support
Python	3.10 (system) + 3.11 (Windows)	Ubuntu 22.04 ships 3.10; your Windows model can use 3.11
CUDA	12.x (Windows host)	Your RL training runs on Windows GPU
