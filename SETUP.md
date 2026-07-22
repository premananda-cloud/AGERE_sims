# AGERE Simulation Stack — Setup

## Why this stack

Earlier plan: RL model native on Windows (GPU), PX4 + ROS 2 in Docker on
Windows, with a to-be-decided simulator to close the loop.

That plan had one purpose for Docker: keep PX4/ROS 2 isolated while a
*separate*, Windows-native simulator (with a GUI) talked to them. Once
Gazebo-in-Docker-on-Windows was ruled out (OpenGL/X11 forwarding through
Docker Desktop's Windows integration is unreliable and slow) and
Windows-native simulator options were evaluated and rejected (see below),
the whole stack moved inside a single **WSL2 Ubuntu instance**, where
WSLg provides GPU-accelerated GUI rendering natively.

**Options considered and rejected for the simulator:**

- **AirSim / Colosseum** — genuinely native Windows GUI, but the
  underlying codebase is dead: Microsoft archived the original AirSim
  with no further updates, and the community continuation (Colosseum)
  was itself archived (read-only) as of July 2026.
- **Native Windows Gazebo** — Windows is not a fully officially supported
  Gazebo platform; no official binaries, and source builds can't even
  run server + GUI in one command.
- **FlightGear** — actively maintained, but fixed-wing; poor fit for
  quadrotor RL.
- **jMAVSim** — native and zero-friction, but no camera/LiDAR simulation.

**Build-from-source vs. image, for PX4:** initially planned to build PX4
from source inside WSL2. Tested both — the official prebuilt image
(`px4io/px4-sitl-gazebo`) starts in seconds versus a multi-minute build,
and is only worth giving up if you need to patch PX4 itself, which isn't
the case here. Switched to the image.

**ROS 2 also moved into Docker**, for the same reproducibility reason —
pin exact package versions in a Dockerfile rather than whatever `apt`
resolves at setup time on a given machine.

**Micro XRCE-DDS Agent: prebuilt image, not built from source.**
Initially built it inside the ROS 2 image (pinned to `v2.4.2`). That
build failed: the Agent's CMake superbuild fetches Fast DDS via a moving
branch reference (`2.12.x`), and that branch no longer exists upstream —
so a "pinned" Agent version silently depends on an unpinned, and now
broken, upstream ref. Switched to a prebuilt image so the Agent runs as
its own container instead of rebuilding from source. Originally used
`eprosima/micro-xrce-dds-agent:v2.4.2`, but that image no longer exists
on Docker Hub — pull fails with "repository does not exist." The
maintained prebuilt image lives under the micro-ROS org instead
(`microros/micro-ros-agent`, same maintainer as eProsima's own Agent
releases), tagged by ROS 2 distro rather than Agent version, so this
project pins `microros/micro-ros-agent:humble` to match the rest of the
stack. One less thing to rebuild when upstream build inputs shift again.

**Docker now runs natively inside the WSL2 Ubuntu distro** (`apt install
docker.io` or Docker's install script) — *not* via Docker Desktop's
Windows/WSL2 integration. This matters for two things covered below:
host networking and GUI passthrough both behave like ordinary Linux
Docker once you're not routing through Docker Desktop.

```
Windows (native)                    WSL2 Ubuntu 22.04
┌─────────────────┐                 ┌───────────────────────────────────┐
│  RL Model (GPU)  │ ── network ──▶ │  Docker (host networking)          │
│                  │ ◀────────────  │  ├── px4-sitl-gazebo container     │
└─────────────────┘                 │  │   (PX4 + Gazebo Harmonic,       │
                                     │  │    GUI via WSLg X11 socket)     │
                                     │  ├── dds-agent container           │
                                     │  │   (eProsima prebuilt image)     │
                                     │  └── ros2-bridge container         │
                                     │      (ROS 2 Humble + px4_msgs)     │
                                     └───────────────────────────────────┘
```

### Why host networking, not the Docker default

ROS 2's default DDS discovery uses UDP multicast, which Docker's default
bridge network does not forward cleanly — the exact category of flaky
cross-boundary networking problem this whole setup has been trying to
avoid. Using `network_mode: host` on both containers sidesteps it
entirely, by having them share WSL2's network stack directly instead of
sitting behind Docker's NAT. This only works simply because Docker is
running on real Linux (WSL2) — it would not be a good idea to rely on
through Docker Desktop's Windows integration.

### Why Gazebo's GUI works fine in a container here

The official `px4io/px4-sitl-gazebo` image needs X11/Wayland forwarding
for its 3D GUI. Normally that's the kind of GUI-passthrough pain this
project has deliberately avoided — but the difference here is that WSLg
already provides a real, working X11 socket inside the WSL2 filesystem
(`/tmp/.X11-unix`). Mounting that socket into the container is the
standard, well-trodden Linux Docker-GUI pattern — not the broken
Windows-Docker-Desktop path that caused problems earlier in this project.

**Worth verifying, not assuming:** whether the container is getting
hardware-accelerated rendering or falling back to slow software
rendering (llvmpipe) depends on `/dev/dri` (the GPU render node) being
mounted through as well — see the compose file. If Gazebo is already
running smoothly, that's a good sign it's already accelerated; if it's
laggy, check that mount first.

## Version stack

| Component | Version | Notes |
|---|---|---|
| OS | Windows 11 (stable) | Required for WSLg GUI + GPU passthrough. |
| WSL | WSL2 | Not WSL1. |
| Linux distro | Ubuntu 22.04 (Jammy) | Matches PX4's recommended ROS 2 platform. |
| Container runtime | Docker Engine, installed natively inside WSL2 | Not Docker Desktop's Windows integration — see rationale above. |
| PX4 + Simulator | `px4io/px4-sitl-gazebo:latest` | Official prebuilt image; PX4 + Gazebo Harmonic + full sensor suite (camera, LiDAR, depth). |
| ROS 2 | Humble Hawksbill (LTS), containerized | Built from `ros:humble-ros-base` in `docker/ros2/Dockerfile`. |
| ROS 2 ↔ Gazebo bridge | `ros-humble-ros-gzharmonic` | Matches the Gazebo Harmonic version in the PX4 image. |
| Middleware bridge | `microros/micro-ros-agent:humble` | Prebuilt image, tagged by ROS 2 distro (`eprosima/micro-xrce-dds-agent` no longer exists on Docker Hub) — see note below on why this isn't built from source. |
| Ground control | QGroundControl (Windows build) | Runs natively on Windows; connects over the network — not inside WSL2. |

## Prerequisites

- Windows 11, fully updated.
- Virtualization enabled in BIOS ("Intel VT-x" / "AMD-V").
- Up-to-date GPU driver **on the Windows side** — WSL2 GPU acceleration
  comes from the Windows driver, not one installed inside Ubuntu.

## Setup steps

### 1. Install WSL2 + Ubuntu 22.04

```powershell
wsl --install -d Ubuntu-22.04
```

### 2. Verify GPU-accelerated GUI works before touching anything else

```bash
sudo apt update && sudo apt install -y mesa-utils
glxgears
```

A window should appear on your Windows desktop and spin smoothly. Fix
the Windows-side GPU driver before continuing if not.

### 3. Install Docker Engine natively inside WSL2

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
```

Close and reopen the WSL2 shell for the group change to take effect.
Confirm it's the native engine, not Docker Desktop's integration:

```bash
docker context show   # should say "default", not "desktop-linux"
```

### 4. Clone `AGERE_sims` and bring up the stack

```bash
git clone https://github.com/premananda-cloud/AGERE_sims.git
cd AGERE_sims/docker
docker compose up --build
```

This pulls `px4io/px4-sitl-gazebo` and `eprosima/micro-xrce-dds-agent`,
builds the `ros2-bridge` image from `docker/ros2/Dockerfile`, and starts
all three with host networking. The Gazebo GUI should appear on your
Windows desktop via WSLg.

### 5. Confirm the loop is closed

In a separate WSL2 terminal:

```bash
docker exec -it agere-ros2-bridge bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
ros2 topic list | grep fmu
```

`/fmu/...` topics confirm PX4 ↔ DDS Agent ↔ ROS 2 are all talking. Camera
sanity check:

```bash
ros2 run ros_gz_image image_bridge /camera
```

### 6. Connect QGroundControl from Windows

Install QGroundControl natively on Windows. With host networking and
WSL2's mirrored networking mode, it should auto-detect the vehicle
without extra configuration.

## Open item

The RL model (native Windows) → WSL2/Docker network path hasn't been
exercised yet. This covers getting the simulator loop itself running;
wiring the RL model in is the next step — see the main repo's planning
docs for sequencing.
