#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

# Start the Micro XRCE-DDS Agent in the background unless disabled.
# This is what PX4's uxrce_dds_client connects to on udp4:8888 by default.
if [ "${SKIP_XRCE_AGENT:-0}" != "1" ]; then
    MicroXRCEAgent udp4 -p "${XRCE_AGENT_PORT:-8888}" &
fi

exec "$@"
