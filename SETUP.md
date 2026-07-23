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

### GPU-accelerated rendering: NVIDIA + WSL2 needs a different path than /dev/dri

**This section was rewritten after hands-on debugging on 2026-07-23** —
the original assumption below (Intel/AMD Mesa via `/dev/dri`) is wrong
for a machine with an NVIDIA GPU, which is what this project actually
runs on. Documented here so the mistake isn't repeated.

If the machine has an **Intel or AMD GPU**, the earlier assumption
holds: mount `/dev/dri` (the Mesa/DRM render node) into the container
and hardware acceleration works the same way it would on native Linux.

If the machine has an **NVIDIA GPU** (as this one does — an RTX 4070),
`/dev/dri` **does not exist in WSL2** and mounting it fails outright
with `error gathering device information while adding custom device
"/dev/dri": not a device node`. NVIDIA GPU passthrough into WSL2 uses a
completely different mechanism, and it splits into two separate paths
that must both be set up:

1. **CUDA/compute** — works via the NVIDIA Container Toolkit installed
   on the WSL2 host (`nvidia-ctk runtime configure --runtime=docker`)
   plus a GPU reservation in the compose file. This alone is enough to
   make `nvidia-smi` work inside the container — but it does **not**
   give you OpenGL/GUI rendering. These are separate paths, and having
   one work is not evidence the other works.
2. **OpenGL/GUI rendering** — WSLg does not use native NVIDIA GLX
   drivers at all. It renders GPU-accelerated GUI apps through **Mesa's
   D3D12 gallium driver**, talking to the `/dev/dxg` device (WSL2's
   DirectX-on-Linux device node). Getting this working end-to-end
   requires, all at once:
   - Mounting `/usr/lib/wsl` (WSL's Mesa/D3D12 library stack) into the
     container, and pointing the dynamic linker at it:
     `LD_LIBRARY_PATH=/usr/lib/wsl/lib`.
   - Mounting the `/dev/dxg` device itself — the library mount alone
     gives the container the code to speak D3D12 but nothing to talk
     to.
   - `GALLIUM_DRIVER=d3d12` — without this, Mesa's driver-selection
     logic still silently picks `llvmpipe` (software rendering), even
     with the libraries and device present and reachable.
   - **On laptops with a dual GPU (integrated + discrete)**:
     `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`. Without this, WSLg's
     D3D12 adapter enumeration defaults to the first-listed display
     adapter — which is usually the integrated GPU. Confirmed on this
     machine: without the var, `glxinfo` reported
     `D3D12 (Intel(R) UHD Graphics 770)`; with it,
     `D3D12 (NVIDIA GeForce RTX 4070 SUPER)`.

**How to verify which path you're actually on, in order:**
```bash
nvidia-smi                                  # confirms CUDA/compute path
docker exec -it agere-px4-sitl bash
apt update && apt install -y mesa-utils
glxinfo | grep "OpenGL renderer"            # confirms OpenGL/GUI path
```
Expect `D3D12 (NVIDIA GeForce ...)` — that string **is** hardware
acceleration under WSLg, not a fallback. It will never say bare
`NVIDIA GeForce ...` the way native Linux does. `llvmpipe` is the only
"still not working" signal, and a D3D12 string naming the *wrong* GPU
(e.g. the integrated one) means the adapter-name env var above is
missing.

## Version stack

| Component | Version | Notes |
|---|---|---|
| OS | Windows 11 (stable) | Required for WSLg GUI + GPU passthrough. |
| WSL | WSL2 | Not WSL1. |
| Linux distro | Ubuntu 24.04 (Noble) | Actual distro in use as of 2026-07-23; earlier plan targeted 22.04 (Jammy) for PX4's recommended ROS 2 platform — revisit compatibility if/when ROS 2 is reintroduced. |
| GPU | NVIDIA GeForce RTX 4070 SUPER (laptop, dual-GPU with Intel UHD 770) | Confirmed via `nvidia-smi` (CUDA path) and `glxinfo` (OpenGL/D3D12 path) on 2026-07-23. See GPU rendering section above — this is NOT an Intel/AMD `/dev/dri` setup. |
| Container runtime | Docker Engine, installed natively inside WSL2 | Not Docker Desktop's Windows integration — see rationale above. |
| GPU container support | NVIDIA Container Toolkit | Installed on the WSL2 host; required for both the CUDA/compute reservation and (combined with the D3D12 env vars/mounts above) OpenGL rendering. |
| PX4 + Simulator | `px4io/px4-sitl-gazebo:latest` | Official prebuilt image; PX4 + Gazebo Harmonic + full sensor suite (camera, LiDAR, depth). |
| ROS 2 | Humble Hawksbill (LTS), containerized | **Temporarily removed from `docker-compose.yml`** as of 2026-07-23 to isolate and confirm GPU rendering. Was built from `ros:humble-ros-base` in `docker/ros2/Dockerfile` — reintroduce once GPU work is stable. |
| ROS 2 ↔ Gazebo bridge | `ros-humble-ros-gzharmonic` | Matches the Gazebo Harmonic version in the PX4 image. Not currently running (see above). |
| Middleware bridge | `microros/micro-ros-agent:humble` | **Temporarily removed from `docker-compose.yml`** as of 2026-07-23, same reason as ROS 2. Prebuilt image, tagged by ROS 2 distro (`eprosima/micro-xrce-dds-agent` no longer exists on Docker Hub). |
| Ground control | QGroundControl (Windows build) | **Parked as of 2026-07-23** — see Troubleshooting log below. Runs natively on Windows; connects over the network — not inside WSL2. |

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

## Current checkpoint (2026-07-23)

**Working:**
- PX4 SITL + Gazebo (default world) running with confirmed GPU-accelerated
  rendering (`D3D12 (NVIDIA GeForce RTX 4070 SUPER)`).
- Console control confirmed: `docker attach agere-px4-sitl` → `pxh>` shell
  → `commander arm -f` → `commander takeoff` works.
- `docker-compose.yml` is currently minimal — only the `px4-sitl-gazebo`
  service. ROS 2 bridge and the DDS agent are removed for now, not broken;
  see full details and next steps in `2026-07-23.md`.

**Open / parked:**
- Python scripting (MAVSDK) over `udpin://0.0.0.0:14540` — connection
  string fixed, but not yet confirmed receiving packets. Suspect the
  `partner IP: 192.168.1.103` shown in `mavlink status` (instead of
  `127.0.0.1`) is relevant.
- QGroundControl-from-Windows networking — dead-ended on the
  `MAV_i_BROADCAST` param approach (this image starts its 4 MAVLink
  streams via direct `mavlink start` shell commands, not the numbered
  param-config system, so the params don't do anything). Would need the
  real `mavlink start --help` flags from inside the container to pursue
  further.
- ROS 2 / DDS agent reintroduction, once GPU + Python control are solid.

See `2026-07-23.md` for the full session log.
