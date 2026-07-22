# AGERE ROS 2 bridge image
# Pins ROS 2 Humble + the Gazebo Harmonic bridge + Micro XRCE-DDS Agent
# together so the whole ROS 2 side is reproducible, instead of relying on
# whatever apt resolves at "docker run" time.

FROM ros:humble-ros-base

# --- ROS 2 <-> Gazebo Harmonic bridge packages ---
# Matches the Gazebo Harmonic version bundled in px4io/px4-sitl-gazebo.
#
# ros-humble-ros-gzharmonic is NOT in the default ROS 2 apt repo — Humble's
# default ros-humble-ros-gz targets Gazebo Fortress. The Harmonic bridge
# is hosted on the separate OSRF/Gazebo apt repo, which must be added
# (with its own signing key) before apt can see the package.
RUN apt-get update && apt-get install -y --no-install-recommends \
    lsb-release wget gnupg curl \
    && curl https://packages.osrfoundation.org/gazebo.gpg \
       --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
       | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null \
    && apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-gzharmonic \
    && rm -rf /var/lib/apt/lists/*

# --- Micro XRCE-DDS Agent (bridges PX4 uORB <-> ROS 2 topics) ---
# Pinned to a specific tag rather than a moving branch, for reproducibility.
ARG XRCE_DDS_AGENT_VERSION=v2.4.2
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake \
    && git clone -b ${XRCE_DDS_AGENT_VERSION} --depth 1 \
       https://github.com/eProsima/Micro-XRCE-DDS-Agent.git /opt/Micro-XRCE-DDS-Agent \
    && cd /opt/Micro-XRCE-DDS-Agent \
    && mkdir build && cd build \
    && cmake .. && make -j"$(nproc)" \
    && make install && ldconfig /usr/local/lib/ \
    && apt-get purge -y git build-essential cmake \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /opt/Micro-XRCE-DDS-Agent

# px4_msgs is needed for ROS 2 nodes to interpret PX4 topics correctly.
# Cloned + built here (not via apt) since it isn't packaged for Humble.
ARG PX4_MSGS_BRANCH=main
RUN mkdir -p /ros2_ws/src && cd /ros2_ws/src \
    && git clone -b ${PX4_MSGS_BRANCH} --depth 1 \
       https://github.com/PX4/px4_msgs.git \
    && cd /ros2_ws \
    && . /opt/ros/humble/setup.sh \
    && colcon build --merge-install

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
