# Gaussian-LIC — Install Guide (Ubuntu 24.04 + ROS2 Jazzy)

Real-time photo-realistic SLAM using 3D Gaussian Splatting with a **Manifold Tech Odin1**
LiDAR-Inertial-Camera sensor. This guide covers a fresh install on Ubuntu 24.04.

---

## Hardware requirements

| Component | Minimum | Tested |
|-----------|---------|--------|
| GPU | NVIDIA RTX (Ampere or newer) | RTX 3090 |
| VRAM | 16 GB | 24 GB |
| RAM | 32 GB | 64 GB |
| CPU | 8-core | AMD Ryzen 9 |

> If your GPU is not an RTX 3090 (compute capability 8.6) update
> `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` to match your card:
> RTX 4090 → `89`, RTX 3080/3090 → `86`, A100 → `80`.

---

## Software overview

| Package | Version | Where |
|---------|---------|-------|
| Ubuntu | 24.04 LTS | — |
| ROS2 | Jazzy | apt |
| CUDA | 12.4 | NVIDIA installer |
| cuDNN | 9.x | NVIDIA installer |
| OpenCV | 4.10.0 | built from source |
| LibTorch | 2.4.0+cu124 | pre-built zip |
| TensorRT | 10.x | NVIDIA tarball |
| Eigen3 | system | apt |
| PCL | system | apt |
| yaml-cpp | system | apt |

---

## Step 0 — Clone the repository

```bash
git clone https://github.com/3DVisionFS26/odin_gaussian_lic.git -b ubuntu-2404
cd odin_gaussian_lic
```

---

## Step 1 — System packages

```bash
sudo apt update
sudo apt install -y \
    build-essential cmake git pkg-config \
    python3-pip python3-dev python3-numpy \
    libeigen3-dev libpcl-dev libyaml-cpp-dev \
    libffi-dev libssl-dev \
    wget unzip
```

---

## Step 2 — ROS2 Jazzy

Follow the official guide at https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html
or run:

```bash
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | python3 -c "import sys,json; print(json.load(sys.stdin)['tag_name'])")
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.deb"
sudo apt install /tmp/ros2-apt-source.deb
sudo apt update
sudo apt install -y ros-jazzy-desktop python3-colcon-common-extensions
```

Install the ROS2 packages the project depends on:

```bash
sudo apt install -y \
    ros-jazzy-rclcpp \
    ros-jazzy-std-msgs \
    ros-jazzy-sensor-msgs \
    ros-jazzy-geometry-msgs \
    ros-jazzy-nav-msgs \
    ros-jazzy-cv-bridge \
    ros-jazzy-image-transport \
    ros-jazzy-pcl-conversions \
    ros-jazzy-tf2 \
    ros-jazzy-tf2-eigen \
    ros-jazzy-tf2-ros \
    ros-jazzy-ament-index-cpp
```

Add to your `~/.bashrc`:

```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## Step 3 — CUDA 12.4

Download the CUDA 12.4 runfile installer from:
https://developer.nvidia.com/cuda-12-4-0-download-archive (select Linux → x86_64 → Ubuntu → 24.04 → runfile)

```bash
# Example — check NVIDIA's page for the exact URL
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run --toolkit --silent
```

After install, `/usr/local/cuda` is a symlink to `/usr/local/cuda-12.4`.

Add to `~/.bashrc`:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
nvcc --version   # should show 12.4
nvidia-smi       # should show your GPU
```

---

## Step 4 — cuDNN 9.x

Download cuDNN 9.x for CUDA 12.x from https://developer.nvidia.com/rdp/cudnn-archive
(requires free NVIDIA account). Choose the **tar** package for Linux x86_64.

```bash
tar -xf cudnn-linux-x86_64-9.x.x.x_cuda12-archive.tar.xz
cd cudnn-linux-x86_64-9.x.x.x_cuda12-archive
sudo cp include/cudnn*.h    /usr/local/cuda/include/
sudo cp lib/libcudnn*       /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

---

## Step 5 — OpenCV 4.10.0 (built from source with CUDA)

```bash
mkdir -p ~/Software/opencv && cd ~/Software/opencv

wget https://github.com/opencv/opencv/archive/refs/tags/4.10.0.tar.gz
tar -zxvf 4.10.0.tar.gz && rm 4.10.0.tar.gz

wget https://github.com/opencv/opencv_contrib/archive/refs/tags/4.10.0.tar.gz
tar -zxvf 4.10.0.tar.gz && rm 4.10.0.tar.gz

cd opencv-4.10.0 && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE \
      -DWITH_CUDA=ON -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.10.0/modules \
      -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_JPEG=ON -DWITH_FFMPEG=ON ..
make -j$(nproc)
```

> Do **not** run `make install` — CMakeLists.txt points directly to the build directory.

CMakeLists.txt auto-discovers the OpenCV build directory via an environment variable or
a glob under `~/Software/opencv/`. No manual edits are needed if you build at the default
path above. For a non-standard location, set:

```bash
export OpenCV_DIR=$HOME/Software/opencv/opencv-4.10.0/build
```

---

## Step 6 — LibTorch 2.4.0 (CUDA 12.4 pre-built)

```bash
cd ~/Software
wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.4.0+cu124.zip
rm    libtorch-cxx11-abi-shared-with-deps-2.4.0+cu124.zip
# → ~/Software/libtorch/
```

The `Torch_DIR` in CMakeLists.txt already points to `~/Software/libtorch` so no changes needed.

Add LibTorch to the runtime library path:

```bash
echo 'export LD_LIBRARY_PATH=~/Software/libtorch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 7 — TensorRT 10.x

Download the **tar.gz** package for CUDA 12.x from:
https://developer.nvidia.com/tensorrt (requires NVIDIA account, choose TensorRT 10.x → Linux x86_64 → CUDA 12.x → TAR)

```bash
cd ~/Software
tar -xf TensorRT-10.x.x.x.Linux.x86_64-gnu.cuda-12.x.tar.gz
# → ~/Software/TensorRT-10.x.x.x/
```

CMakeLists.txt auto-detects any `~/Software/TensorRT-*` directory, so no path edits are needed.

Add TensorRT to the runtime library path:

```bash
echo 'export LD_LIBRARY_PATH=~/Software/TensorRT-10.x.x.x/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

> **TRT 10 note:** `libnvparsers.so` (legacy caffe/uff parser) was removed in TRT 10.
> CMakeLists.txt handles this automatically — it links it only if found.

---

## Step 8 — SPNet depth completion (one-time setup)

SPNet densifies the sparse LiDAR depth map before it is fed to the Gaussian splatting
optimiser. This step converts the pre-trained weights into a TensorRT engine.
The engines are GPU- and TRT-version-specific so they must be built on the target machine.

### 8a — Download the pre-trained weights

Download `Large_300.pth` (~900 MB) from the SPNet Google Drive link in the original
Gaussian-LIC paper repository and place it in `ckpt/`:

```bash
# After downloading:
mv ~/Downloads/Large_300.pth  /path/to/catkin_gaussian/src/Gaussian-LIC/ckpt/
```

### 8b — Create the conda environment

```bash
conda create -n spnet python=3.10 -y
conda activate spnet
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install onnx onnxruntime
```

### 8c — Export ONNX models

```bash
cd /path/to/catkin_gaussian/src/Gaussian-LIC/ckpt
bash setup_spnet.sh   # clones SPNet repo (first time only)
bash export_onnx.sh   # produces spnet_512_640.onnx and spnet_480_640.onnx
conda deactivate
```

### 8d — Compile TensorRT engines

```bash
cd /path/to/catkin_gaussian/src/Gaussian-LIC/ckpt
bash build_trt.sh
# produces spnet_512_640.engine and spnet_480_640.engine (~450 MB each, ~10 min)
```

The engine that matches your configured output resolution (`width`/`height` in `odin1.yaml`)
is loaded automatically at runtime. The default config uses 640×512 → `spnet_512_640.engine`.

---

## Step 9 — Build Gaussian-LIC

```bash
source /opt/ros/jazzy/setup.bash
cd /path/to/catkin_gaussian
colcon build --packages-select gaussian_lic --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

If you see a CMake error about OpenCV, verify the `OpenCV_DIR` path in `CMakeLists.txt`
matches where you built it.

---

## Step 10 — Run

### With a recorded ROS2 bag

```bash
# Terminal 1 — start the mapping node
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch gaussian_lic odin1.launch.py

# Terminal 2 — play the bag
source /opt/ros/jazzy/setup.bash
ros2 bag play /path/to/datasets/Downtown1 --clock -r 1.0
```

### With a live Odin1 sensor

```bash
# Terminal 1 — start the Odin1 driver
ros2 launch odin_ros_driver odin1.launch.py

# Terminal 2 — start Gaussian-LIC
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch gaussian_lic odin1.launch.py
```

### Optional launch arguments

```bash
ros2 launch gaussian_lic odin1.launch.py \
    config:=/absolute/path/to/my_sensor.yaml \
    result:=/tmp/my_map
```

---

## Output

Results are written to `result/` (or the path passed via the `result` launch argument):

| File / folder | Contents |
|---|---|
| `point_cloud.ply` | 3DGS map — open in the Gaussian viewer |
| `render/` | Rendered training-view images |
| `render_depth/` | Rendered depth maps (jet colourmap) |
| `gt/` | Ground-truth camera frames |

---

## Topics subscribed

| Topic | Type | Source |
|---|---|---|
| `/odin1/cloud_slam` | `sensor_msgs/msg/PointCloud2` | World-frame SLAM cloud |
| `/odin1/odometry` | `nav_msgs/msg/Odometry` | Sensor pose in world |
| `/odin1/image/compressed` | `sensor_msgs/msg/CompressedImage` | Fisheye JPEG |

Topics are configurable in `config/odin1.yaml` via `cloud_topic`, `odom_topic`, `image_topic`.

---

## Config reference (`config/odin1.yaml`)

### Odin1 camera calibration

| Key | Description |
|-----|-------------|
| `odin1_width / odin1_height` | Raw fisheye image size (1600 × 1296) |
| `odin1_fx / odin1_fy` | Focal lengths A11, A22 from FishPoly model |
| `odin1_cx / odin1_cy` | Principal point u0, v0 |
| `odin1_skew` | Row-skew coefficient A12 |
| `k2` … `k7` | FishPoly polynomial distortion: r_d = r + k2·r² + … + k7·r⁷ |
| `T_cl` | 4×4 camera-from-LiDAR extrinsic (row-major) |

### Output image

| Key | Description |
|-----|-------------|
| `width / height` | Output perspective image size (default 640 × 512) |
| `fx / fy / cx / cy` | Override auto-computed pinhole intrinsics (optional) |

### Dataset

| Key | Description |
|-----|-------------|
| `select_every_k_frame` | Use every k-th aligned frame as a keyframe (default 3) |
| `depth_completion` | Enable SPNet depth completion (true/false) |
| `patch_size` | Depth propagation patch radius in pixels |
| `max_depth` | Maximum valid LiDAR depth in metres (default 50) |

### Gaussian model

| Key | Description |
|-----|-------------|
| `sh_degree` | Spherical harmonics degree (3 = colour) |
| `scaling_scale` | Initial Gaussian scale factor — larger = fewer Gaussians |
| `position_lr` | Position learning rate |
| `feature_lr` | SH feature learning rate |
| `opacity_lr` | Opacity learning rate |
| `scaling_lr` | Scale learning rate |
| `rotation_lr` | Rotation learning rate |
| `lambda_dssim` | SSIM loss weight |
| `optimize_depth` | Include depth loss in optimisation |
| `lambda_depth` | Depth loss weight |
| `iteration_decay` | Decay learning rates over time |

---

## Troubleshooting

**`Cannot find CUDA`**
Make sure `/usr/local/cuda` exists (it is a symlink created by the CUDA installer).
Alternatively set `export CUDA_HOME=/usr/local/cuda-12.4` before building.

**`Cannot find TensorRT`**
Set `export TENSORRT_ROOT=~/Software/TensorRT-10.x.x.x` before running `colcon build`.

**`libnvparsers.so not found`**
This library was removed in TensorRT 10. CMakeLists.txt handles this automatically —
make sure you pulled the latest version of the repo.

**`Engine file not found`**
The `.engine` files in `ckpt/` are built on your specific GPU + TRT version.
If you upgraded TRT or changed GPU, re-run `ckpt/build_trt.sh`.

**`No messages received`**
Check that your bag or driver publishes on exactly the topic names in `odin1.yaml`.
Use `ros2 topic list` and `ros2 topic hz /odin1/cloud_slam` to verify.

**`Slow or dropped frames`**
Reduce `viz_every_n_keyframes` in `odin1.yaml` (e.g. set to 5) to reduce GPU overhead
from the RViz2 publishing thread.
