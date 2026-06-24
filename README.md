# Gaussian-LIC — Jetson Orin (Bare Metal)

Real-time photo-realistic SLAM using 3D Gaussian Splatting with a Manifold Tech **Odin1**
LiDAR-Inertial-Camera sensor, running natively on NVIDIA Jetson Orin.

This branch is for a **bare-metal (non-Docker) install on Jetson Orin** with
JetPack 6.x. No Docker is required.

> **Other deployment targets:**
> - Docker on a desktop/workstation → [`master`](../../tree/master)
> - Native install on Ubuntu 24.04 workstation → [`ubuntu-2404`](../../tree/ubuntu-2404)

See [CHANGES.md](CHANGES.md) for a full comparison with the upstream Gaussian-LIC
repositories. Additions over the original include: Odin1 sensor integration, ROS2
migration, live web monitor dashboard, and a parameter sweep utility for offline
hyperparameter comparison (`scripts/sweep.py`).

---

## Hardware requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| Module | Jetson Orin NX 8 GB | AGX Orin 32/64 GB recommended for best quality |
| Unified memory | 8 GB | 16 GB+ comfortable for real-time operation |
| Storage | 64 GB fast NVMe | Results and bags can be large |
| JetPack | 6.1 (L4T R36.4) | Ubuntu 22.04 · CUDA 12.6 · TRT 10.3 · cuDNN 9.3 |

> The Orin GPU is Ampere sm_87. CMakeLists.txt auto-detects `aarch64` and sets
> `CMAKE_CUDA_ARCHITECTURES=87`.

---

## Differences from desktop branches

| Aspect | Desktop (`master` / `ubuntu-2404`) | Jetson Orin (this branch) |
|--------|------------------------------------|--------------------------|
| OS | Ubuntu 24.04 | Ubuntu 22.04 (JetPack 6.x) |
| ROS2 | Jazzy | **Humble** |
| Deployment | Docker or native | **Native (bare metal)** |
| CUDA / cuDNN / TRT | Installed manually | Pre-installed with JetPack |
| LibTorch | Pre-built zip from pytorch.org (x86) | From NVIDIA JetPack pip wheel (ARM64) |
| OpenCV | Built with CUDA | Built CPU-only (only cv:: remap/resize/imdecode used) |
| CUDA arch | 86 (RTX 3090) / auto | **87** (Orin Ampere) / auto |

---

## Install

Follow [INSTALL.md](INSTALL.md) step by step. It covers:

1. JetPack 6.1 prerequisites
2. ROS2 Humble
3. TensorRT developer headers (pre-installed by JetPack, just need dev packages)
4. OpenCV 4.10.0 from source (CPU-only build)
5. LibTorch via NVIDIA JetPack pip wheel (ARM64, CUDA 12.6)
6. Clone the repository
7. SPNet weight download
8. ONNX export + TRT engine compilation
9. Build with colcon
10. Run

---

## Quick-start (after INSTALL.md)

### With a ROS2 bag

```bash
# Terminal 1 — mapping node
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch gaussian_lic odin1.launch.py \
    config:=/path/to/gaussian_lic/config/odin1.yaml \
    result:=/path/to/result

# Terminal 2 — bag playback
source /opt/ros/humble/setup.bash
ros2 bag play /path/to/bag --clock -r 1.0
```

### With the live Odin1 sensor

```bash
# Terminal 1 — Odin1 driver
ros2 launch odin_ros_driver odin1.launch.py

# Terminal 2 — mapping node
source /opt/ros/humble/setup.bash && source install/setup.bash
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

Open `http://<jetson-ip>:8765` in a browser on the same network.

---

## Performance notes

The Orin has unified memory (CPU and GPU share DRAM). Start with these settings and tune
from there:

| Parameter in `config/odin1.yaml` | Default | Orin NX 8 GB suggestion |
|-----------------------------------|---------|--------------------------|
| `max_iters` | 100 | 50–70 |
| `select_every_k_frame` | 3 | 4–6 |
| `depth_completion` | true | try `false` first |
| `sh_degree` | 3 | 1–2 |

Before a run, set maximum performance mode:
```bash
sudo nvpmodel -m 0    # MAX-N
sudo jetson_clocks
```

Monitor with `tegrastats`.

---

## Output

```
result/
    point_cloud.ply       ← 3DGS map
    render/               ← rendered training views
    render_depth/         ← rendered depth maps
    gt/                   ← ground-truth training images
```

---

## Parameter sweep (`scripts/sweep.py`)

Automates running `gs_mapping` with different hyperparameter combinations over a
recorded bag — useful for finding the best quality/speed trade-off for your
hardware without manually editing `odin1.yaml` between runs.

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
python3 scripts/sweep.py /path/to/bag_directory
python3 scripts/sweep.py /path/to/bag_directory --rate 0.5   # reduce if frames drop
```

Results go to `/home/jetson/results/sweep_<timestamp>/` (override with
`RESULTS_ROOT=/your/path python3 scripts/sweep.py ...`):

```
sweep_2025-01-01_12-00-00/
    1/    <- point_cloud.ply, render/, gt/, stats.txt
    2/
    ...
    summary.txt   <- wall times + exit codes for all runs
```

The built-in `SWEEP` preset covers five combinations. The first is calibrated for
real-time Jetson Orin operation; run 5 is high-quality offline only:

| Run | `max_iters` | `optimize_depth` | Notes |
|-----|------------|-----------------|-------|
| 1 | 15 | off | Baseline — real-time Orin settings |
| 2 | 30 | off | More iterations |
| 3 | 15 | on | Depth loss |
| 4 | 30 | on | Depth loss + more iterations |
| 5 | 100 | on | High-quality (offline, slow on Orin NX 8 GB) |

> **Tip:** TRT engine loading takes 10–20 s on first run. `INIT_WAIT_S` at the top of
> the script is set to 20 s for Jetson; increase it if the node times out before
> printing `Publishing:`.

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
