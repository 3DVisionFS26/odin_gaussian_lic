# Gaussian-LIC — Ubuntu 24.04 Native Install

Real-time photo-realistic SLAM using 3D Gaussian Splatting with a Manifold Tech **Odin1**
LiDAR-Inertial-Camera sensor.

This branch is for a **native (non-Docker) install on Ubuntu 24.04** with a desktop or
workstation GPU (RTX 3080/3090/4090 or similar). No Docker is required.

> **Other deployment targets:**
> - Docker on a desktop/server → [`master`](../../tree/master)
> - NVIDIA Jetson Orin (ARM64, JetPack) → [`jetson-orin`](../../tree/jetson-orin)

Additions over the original Gaussian-LIC: Odin1 sensor integration, ROS2 migration, live web monitor dashboard, and a parameter sweep utility for offline hyperparameter comparison (`scripts/sweep.py`).

See [CHANGES.md](CHANGES.md) for a full comparison with the upstream Gaussian-LIC
repositories.

---

## Hardware requirements

| Component | Minimum | Tested |
|-----------|---------|--------|
| GPU | NVIDIA RTX (Ampere or newer, sm_86+) | RTX 3090 (sm_86) |
| VRAM | 16 GB | 24 GB |
| RAM | 32 GB | 64 GB |
| CPU | 8-core x86_64 | AMD Ryzen 9 |
| OS | Ubuntu 24.04 LTS | Ubuntu 24.04 |

**CUDA architecture:** The build system auto-detects x86_64 and sets `CMAKE_CUDA_ARCHITECTURES=86`
by default. Override at build time for other cards:

```bash
# RTX 4090 / Ada Lovelace
colcon build ... --cmake-args -DCMAKE_CUDA_ARCHITECTURES=89

# A100
colcon build ... --cmake-args -DCMAKE_CUDA_ARCHITECTURES=80
```

---

## Full install guide

Follow [INSTALL.md](INSTALL.md) step by step. It covers:

1. System packages (cmake, Eigen, PCL, yaml-cpp)
2. ROS2 Jazzy
3. CUDA 12.4
4. cuDNN 9.x
5. OpenCV 4.10.0 built from source with CUDA
6. LibTorch 2.4.0 (pre-built, CUDA 12.4)
7. TensorRT 10.x
8. SPNet depth completion (ONNX export + TRT engine compilation)
9. Build Gaussian-LIC with colcon
10. Run

---

## Quick-start (after completing INSTALL.md)

### With a ROS2 bag

```bash
# Terminal 1 — mapping node
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch gaussian_lic odin1.launch.py \
    config:=/path/to/gaussian_lic/config/odin1.yaml \
    result:=/path/to/result

# Terminal 2 — bag playback
source /opt/ros/jazzy/setup.bash
ros2 bag play /path/to/bag --clock -r 1.0
```

### With the live Odin1 sensor

```bash
# Terminal 1 — Odin1 driver
ros2 launch odin_ros_driver odin1.launch.py

# Terminal 2 — mapping node
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch gaussian_lic odin1.launch.py \
    config:=/path/to/gaussian_lic/config/odin1.yaml \
    result:=/path/to/result
```

### With the optional web monitor

```bash
ros2 launch gaussian_lic odin1.launch.py \
    config:=/path/to/odin1.yaml \
    result:=/path/to/result \
    web_monitor:=true
```

Open `http://localhost:8765` in any browser to see the live 3D Gaussian cloud,
input stream, rendered view, and statistics.

---

## Output

```
result/
    point_cloud.ply       ← 3DGS map (open in any Gaussian splatting viewer)
    render/               ← rendered training views
    render_depth/         ← rendered depth maps
    gt/                   ← ground-truth training images
```

---

## Config reference (`config/odin1.yaml`)

| Key | Description |
|-----|-------------|
| `cloud_topic` / `odom_topic` / `image_topic` | ROS2 topic names |
| `odin1_width/height/fx/fy/cx/cy/skew` | Odin1 fisheye camera intrinsics |
| `k2`–`k7` | FishPoly distortion coefficients |
| `T_cl` | 4×4 camera-from-LiDAR extrinsic (row-major) |
| `width` / `height` | Output resolution (must match the TRT engine: 640×512 or 640×480) |
| `select_every_k_frame` | Keyframe decimation (higher = fewer keyframes, faster) |
| `depth_completion` | Enable SPNet depth densification |
| `max_depth` | Maximum LiDAR range in metres |
| `sh_degree` | Spherical harmonic degree (0–3) |
| `max_iters` | Optimisation iterations per keyframe |

To adapt to a different sensor, copy and edit `config/odin1.yaml`, then pass at launch:

```bash
ros2 launch gaussian_lic odin1.launch.py config:=/path/to/my_sensor.yaml
```

---

## Parameter sweep (`scripts/sweep.py`)

Automates running `gs_mapping` with different hyperparameter combinations over a
recorded bag. Each combination writes a `point_cloud.ply`, render images, and a
`stats.txt` to its own sub-directory. Use this to compare quality vs. speed
trade-offs without manually editing `odin1.yaml` between runs.

```bash
source install/setup.bash
python3 scripts/sweep.py /path/to/bag_directory
python3 scripts/sweep.py /path/to/bag_directory --rate 0.5   # slow playback
```

Results go to `$RESULTS_ROOT/sweep_<timestamp>/` (default: `/mnt/Volume/Ubuntu/3d_vision/results`; override with `RESULTS_ROOT=/your/path`):

```
sweep_2025-01-01_12-00-00/
    1/    <- point_cloud.ply, render/, gt/, stats.txt
    2/
    ...
    summary.txt   <- wall times + exit codes for all runs
```

Edit the `SWEEP` list at the top of `scripts/sweep.py` to define your own
combinations. Each dict overrides keys from `config/odin1.yaml`; unlisted keys
keep their defaults. The built-in preset:

| Run | `max_iters` | `optimize_depth` | Notes |
|-----|------------|-----------------|-------|
| 1 | 15 | off | Baseline — real-time settings |
| 2 | 30 | off | More iterations |
| 3 | 15 | on | Depth loss enabled |
| 4 | 30 | on | Depth loss + more iterations |
| 5 | 100 | on | High-quality offline |

---

## Troubleshooting

**`Cannot find CUDA`** — ensure `/usr/local/cuda` exists (created by the runfile installer),
or set `export CUDA_HOME=/usr/local/cuda-12.4`.

**`Cannot find TensorRT`** — set `export TENSORRT_ROOT=~/Software/TensorRT-10.x.x.x` before
running `colcon build`.

**`Engine file not found`** — re-run `ckpt/build_trt.sh` after upgrading TRT or changing GPU.

**`No messages received`** — verify topic names with `ros2 topic list` and
`ros2 topic hz /odin1/cloud_slam`.

---

## Citation

```bibtex
@inproceedings{lang2025gaussian,
  title={Gaussian-LIC: Real-time photo-realistic SLAM with Gaussian splatting and LiDAR-inertial-camera fusion},
  author={Lang, Xiaolei and Li, Laijian and Wu, Chenming and Zhao, Chen and Liu, Lina and Liu, Yong and Lv, Jiajun and Zuo, Xingxing},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={8500--8507},
  year={2025},
  organization={IEEE}
}
```
