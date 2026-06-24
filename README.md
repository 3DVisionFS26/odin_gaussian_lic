# Gaussian-LIC — Odin1 Fork

Real-time photo-realistic SLAM using 3D Gaussian Splatting with a Manifold Tech **Odin1**
LiDAR-Inertial-Camera sensor.

This is a fork of the original [Gaussian-LIC](https://github.com/APRIL-ZJU/Gaussian-LIC) /
[Gaussian-LIC2](https://github.com/APRIL-ZJU/Gaussian-LIC2) research code. The Coco-LIC
dependency is removed entirely; the node subscribes directly to the Odin1 ROS2 topics. See
[CHANGES.md](CHANGES.md) for a detailed comparison with the upstream repositories.

Additions over the original Gaussian-LIC: Odin1 sensor integration, ROS2 migration, live web monitor dashboard, and a parameter sweep utility for offline hyperparameter comparison (`scripts/sweep.py`).

---

## Branches

Three deployment targets are maintained as separate branches. Switch to the one that matches
your hardware before building.

| Branch | Target | OS / ROS2 | Primary guide |
|--------|--------|-----------|---------------|
| **`master`** (this branch) | Docker on a desktop/workstation | Ubuntu 24.04 · Jazzy (inside container) | [Quick-start below](#quick-start-docker) |
| **`ubuntu-2404`** | Native install on Ubuntu 24.04 workstation | Ubuntu 24.04 · Jazzy | [README on that branch](../../tree/ubuntu-2404) |
| **`jetson-orin`** | Native install on Jetson Orin (ARM64, JetPack 6.x) | Ubuntu 22.04 · Humble | [README on that branch](../../tree/jetson-orin) |

> **Which branch to use?**
> - Running on a desktop/server with a good NVIDIA GPU and Docker installed → stay on `master`.
> - Want to build and run natively without Docker → switch to `ubuntu-2404`.
> - Deploying on a Jetson Orin → switch to `jetson-orin` (native bare-metal install).

---

## Quick-start (Docker)

### Prerequisites

- NVIDIA GPU (tested: RTX 3090 / sm_86) with driver ≥ 525
- [Docker](https://docs.docker.com/engine/install/) with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA runtime set as the Docker default:

  ```bash
  sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
  sudo systemctl restart docker
  ```

### 1. Build the image

```bash
git clone https://github.com/3DVisionFS26/odin_gaussian_lic.git
cd gaussian_lic
./build.sh
```

`build.sh` runs two phases:

| Phase | What happens |
|-------|--------------|
| 1 — `docker build` | Clones the repo via HTTPS, runs colcon, exports ONNX models |
| 2 — `docker run` + `docker commit` | Builds TRT engines with GPU, commits as `gaussian_lic:latest` |

> **Rebuilding after a code change:**
> ```bash
> REPO_SHA=$(git rev-parse HEAD) DOCKER_BUILDKIT=1 docker build \
>     --build-arg REPO_SHA=$REPO_SHA -t gaussian_lic .
> ```
> Or just re-run `./build.sh`.

### 2. Prepare output directory

```bash
mkdir -p result
```

### 3. Run with a bag file

**Terminal 1 — start the Gaussian-LIC node:**

```bash
docker run --rm \
  -v /path/to/gaussian_lic/config/odin1.yaml:/config.yaml \
  -v /path/to/result:/results \
  -v /path/to/bag_directory:/bag \
  --network host \
  gaussian_lic:latest \
  ros2 launch gaussian_lic odin1.launch.py \
    config:=/config.yaml \
    result:=/results
```

Wait for:
```
😋 Gaussian-LIC Ready!
```

**Terminal 2 — play the bag from inside the container:**

```bash
docker exec $(docker ps -q -f ancestor=gaussian_lic:latest) \
  bash -c "source /opt/ros/jazzy/setup.sh && ros2 bag play /bag --start-paused"
```

Press **Space** to start. Playing from inside the container avoids DDS discovery delays.

### 4. Run with the live Odin1 sensor

```bash
docker run --rm \
  -v /path/to/gaussian_lic/config/odin1.yaml:/config.yaml \
  -v /path/to/result:/results \
  --network host \
  gaussian_lic:latest \
  ros2 launch gaussian_lic odin1.launch.py \
    config:=/config.yaml \
    result:=/results
```

Start `odin_ros_driver` on the host before launching Gaussian-LIC.

### 5. Stop and view results

Press **Ctrl-C**. Results are written to your mounted `result/` directory:

```
result/
    point_cloud.ply       ← 3DGS map (open in any Gaussian splatting viewer)
    render/               ← rendered training views
    render_depth/         ← rendered depth maps
    gt/                   ← ground-truth training images
```

---

## Web Monitor

An optional live browser dashboard runs on port **8765** and shows:

- 3D Gaussian point cloud (Three.js, orbit controls)
- Latest input frame and rendered Gaussian splat side-by-side
- Newly added Gaussians per keyframe highlighted in a distinct colour
- Trajectory and stats (keyframe count, Gaussian count, loss)

Enable it at launch:

```bash
docker run --rm \
  -p 8765:8765 \
  -v ... \
  gaussian_lic:latest \
  ros2 launch gaussian_lic odin1.launch.py \
    config:=/config.yaml result:=/results web_monitor:=true
```

Open `http://localhost:8765` in any browser.

---

## How it works

The node fuses three Odin1 topics to build a live 3DGS map:

| Topic | Type | Used for |
|-------|------|----------|
| `/odin1/cloud_slam` | `PointCloud2` | World-frame SLAM point cloud → depth map + coloured cloud |
| `/odin1/odometry` | `Odometry` | Sensor pose → camera pose T_wc |
| `/odin1/image/compressed` | `CompressedImage` | Fisheye JPEG → undistorted perspective image |

Fisheye undistortion uses the FishPoly polynomial model (coefficients `k2`–`k7` in
`config/odin1.yaml`). Output is a 640×512 perspective image with auto-computed pinhole
intrinsics.

---

## Where CUDA is used

| Component | File | What it does |
|-----------|------|--------------|
| Gaussian rasteriser | `src/rasterizer/cuda_rasterizer/` | Forward + backward pass for differentiable 3DGS rendering |
| Adam optimiser | `src/rasterizer/cuda_rasterizer/adam.cu` | Per-Gaussian gradient updates |
| Simple-KNN | `src/simple-knn/` | Nearest-neighbour distance for initial scale estimation |
| Fused SSIM | `src/fused-ssim/ssim.cu` | SSIM loss computation |
| SPNet depth completion | `src/depth_completer.cpp` + TensorRT | Densifies sparse LiDAR depth via a TRT engine |
| LibTorch | Throughout `gaussian.cpp` | All Gaussian tensor operations run on GPU |

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

To adapt to a different sensor, duplicate `config/odin1.yaml` and pass it at launch:

```bash
ros2 launch gaussian_lic odin1.launch.py config:=/path/to/my_sensor.yaml
```

---

## Repository layout

```
gaussian_lic/
├── src/                        C++ source and CUDA kernels
│   ├── rasterizer/             Differentiable 3DGS renderer (forward + backward)
│   ├── simple-knn/             CUDA nearest-neighbour for scale init
│   ├── fused-ssim/             CUDA SSIM loss
│   ├── lpips/                  Perceptual loss (eval only)
│   ├── mapping.cpp/.h          ROS2 node, frame pipeline, control loop
│   ├── gaussian.cpp/.h         GaussianModel, Dataset, optimize(), pruneGaussians()
│   └── depth_completer.cpp/.h  TRT-based SPNet depth completion
├── scripts/
│   ├── web_monitor_node.py     Live browser dashboard (port 8765)
│   ├── sweep.py                Parameter sweep utility
│   └── evaluate.py             Offline quality evaluation (SSIM / LPIPS)
├── web/static/                 Three.js + OrbitControls bundled for web monitor
├── config/
│   ├── odin1.yaml              Sensor calibration and Gaussian training params
│   └── carla.yaml              CARLA simulator config
├── launch/
│   ├── odin1.launch.py         Main launch file
│   └── odin1_drivers.launch.py Launch file including sensor drivers
├── ckpt/
│   ├── export_onnx_*.py        Export SPNet to ONNX
│   └── build_trt.sh            Compile ONNX → TRT engine
├── Dockerfile                  Multi-stage Docker build (OpenCV + final)
├── build.sh                    Two-phase build orchestration
├── INSTALL.md                  Native Ubuntu 24.04 install guide
├── ARCHITECTURE.md             Technical deep-dive
├── PIPELINE.md                 Operational guide (Odin bags + CARLA simulator)
└── CHANGES.md                  Full diff vs. upstream Gaussian-LIC repositories
```

---

## Parameter sweep (`scripts/sweep.py`)

Automates running `gs_mapping` with different hyperparameter combinations over a
recorded bag. Each combination writes a `point_cloud.ply`, render images, and a
`stats.txt` to its own sub-directory. Use this to compare quality vs. speed
trade-offs without manually editing `odin1.yaml` between runs.

**With Docker**, run the sweep from inside a running container with the bag and a
results volume mounted:

```bash
docker run --rm \
  -v /path/to/gaussian_lic/config/odin1.yaml:/config.yaml \
  -v /path/to/bag_directory:/bag \
  -v /path/to/results:/results \
  -e RESULTS_ROOT=/results \
  --network host \
  gaussian_lic:latest \
  bash -c "source /opt/ros/jazzy/setup.sh && \
           python3 /gaussian_lic/scripts/sweep.py /bag"
```

Results go to `$RESULTS_ROOT/sweep_<timestamp>/`:

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

```bibtex
@article{lang2025gaussian2,
  title={Gaussian-LIC2: LiDAR-Inertial-Camera Gaussian Splatting SLAM},
  author={Lang, Xiaolei and Lv, Jiajun and Tang, Kai and Li, Laijian and Huang, Jianxin and Liu, Lina and Liu, Yong and Zuo, Xingxing},
  journal={arXiv},
  year={2025}
}
```
