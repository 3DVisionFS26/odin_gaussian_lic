# Gaussian-LIC — Native Install Guide (Jetson Orin, JetPack 6.x)

This guide covers a bare-metal install on a Jetson Orin module running
**JetPack 6.1** (L4T R36.4, Ubuntu 22.04, CUDA 12.6, TensorRT 10.3).

> Tested on Jetson AGX Orin 64 GB and Orin NX 16 GB.

---

## Prerequisites

Flash your Jetson with **JetPack 6.1** using NVIDIA SDK Manager or the Jetson Flash tool.
After flashing, CUDA, cuDNN, and TensorRT are already installed on the device.

Verify your JetPack version:
```bash
cat /etc/nv_tegra_release
# Should show: # R36 (release), REVISION: 4.x
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

## Step 2 — ROS2 Humble

Ubuntu 22.04 (Jammy) ships with ROS2 Humble.

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install -y curl

export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | python3 -c "import sys,json; print(json.load(sys.stdin)['tag_name'])")
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.deb"
sudo apt install /tmp/ros2-apt-source.deb
sudo apt update
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions
```

Install the ROS2 packages the project needs:

```bash
sudo apt install -y \
    ros-humble-rclcpp \
    ros-humble-std-msgs \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-nav-msgs \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-pcl-conversions \
    ros-humble-tf2 \
    ros-humble-tf2-eigen \
    ros-humble-tf2-ros \
    ros-humble-ament-index-cpp
```

Add to `~/.bashrc`:
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## Step 3 — TensorRT developer headers

JetPack includes TensorRT at the system level, but you need the developer headers:

```bash
sudo apt install -y \
    libnvinfer-dev \
    libnvinfer-headers-dev \
    libnvinfer-headers-plugin-dev \
    libnvonnxparsers-dev
```

Verify:
```bash
dpkg -l libnvinfer-dev   # should show 10.3.x or similar
```

Create the `/opt/tensorrt` layout that CMakeLists.txt expects via `TENSORRT_ROOT`:

```bash
sudo mkdir -p /opt/tensorrt
sudo ln -sf /usr/include     /opt/tensorrt/include
sudo ln -sf /usr/lib/aarch64-linux-gnu /opt/tensorrt/lib
```

Add to `~/.bashrc`:
```bash
echo 'export TENSORRT_ROOT=/opt/tensorrt' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 4 — OpenCV 4.10.0 (built from source, CPU-only)

The project only uses OpenCV for CPU operations (`cv::remap`, `cv::resize`,
`cv::imdecode`). Build without CUDA to avoid CUDA 13.x/OpenCV compatibility issues.

```bash
mkdir -p ~/Software/opencv && cd ~/Software/opencv

wget https://github.com/opencv/opencv/archive/refs/tags/4.10.0.tar.gz -O opencv-4.10.0.tar.gz
tar -zxvf opencv-4.10.0.tar.gz && rm opencv-4.10.0.tar.gz

wget https://github.com/opencv/opencv_contrib/archive/refs/tags/4.10.0.tar.gz -O opencv_contrib-4.10.0.tar.gz
tar -zxvf opencv_contrib-4.10.0.tar.gz && rm opencv_contrib-4.10.0.tar.gz

cd opencv-4.10.0 && mkdir build && cd build

cmake -DCMAKE_BUILD_TYPE=RELEASE \
      -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.10.0/modules \
      -DWITH_CUDA=OFF \
      -DWITH_CUDNN=OFF \
      -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
      -DBUILD_opencv_python3=OFF \
      -DCMAKE_CXX_STANDARD=17 ..

make -j$(nproc)
```

> **Do not run `make install`** — CMakeLists.txt points directly to the build directory.

Add to `~/.bashrc`:
```bash
echo 'export OpenCV_DIR=$HOME/Software/opencv/opencv-4.10.0/build' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 5 — LibTorch (via PyTorch for JetPack ARM64)

There is no official pre-built LibTorch zip for ARM64. Instead, install the PyTorch
Python wheel provided by NVIDIA for JetPack — it ships the C++ LibTorch headers and
shared libraries in the same package.

```bash
# JetPack 6.1 wheel (Python 3.10, CUDA 12.6, ARM64)
pip3 install --no-cache-dir \
    https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.3.0+nv24.7-cp310-cp310-linux_aarch64.whl
```

Find the cmake prefix path and add it to your environment:
```bash
TORCH_CMAKE=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
echo "export Torch_DIR=${TORCH_CMAKE}/Torch" >> ~/.bashrc

TORCH_LIB=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')")
echo "export LD_LIBRARY_PATH=${TORCH_LIB}:\$LD_LIBRARY_PATH" >> ~/.bashrc

source ~/.bashrc
```

Verify:
```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# → 2.3.0  True
```

---

## Step 6 — Python tools for ONNX export

```bash
pip3 install --no-cache-dir onnx onnxruntime gdown
```

---

## Step 7 — Clone the repository

```bash
git clone https://github.com/3DVisionFS26/odin_gaussian_lic.git -b jetson-orin
cd odin_gaussian_lic
```

---

## Step 8 — Download SPNet weights

```bash
# Option A: gdown (automatic, needs Google Drive access)
gdown "11dujPviL4pKLEXytXK0mEmPBNQDqgEak" -O ckpt/Large_300.pth

# Option B: manual
# Download Large_300.pth from the SPNet Google Drive link in the original
# Gaussian-LIC repository and place it at ckpt/Large_300.pth
```

---

## Step 9 — Export ONNX models

```bash
cd ckpt
python3 export_onnx_512_640.py   # → spnet_512_640.onnx
python3 export_onnx_480_640.py   # → spnet_480_640.onnx
cd ..
```

---

## Step 10 — Compile TRT engines

```bash
cd ckpt
bash build_trt.sh
# → spnet_512_640.engine  (~10–15 min on Orin)
# → spnet_480_640.engine
cd ..
```

The engines are GPU- and TRT-version-specific. Rebuild after any JetPack upgrade.

---

## Step 11 — Build Gaussian-LIC

```bash
# Assumes the repository was cloned into ~/catkin_ws/src/gaussian_lic or similar
# Adjust the workspace path to wherever you cloned the repo

cd /path/to/workspace   # the directory containing src/gaussian_lic

source /opt/ros/humble/setup.bash

colcon build --packages-select gaussian_lic \
    --cmake-args \
      -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=$HOME/Software/opencv/opencv-4.10.0/build \
      -DTorch_DIR=$Torch_DIR \
      -DTENSORRT_ROOT=/opt/tensorrt \
      -DCMAKE_CUDA_ARCHITECTURES=87 \
    --event-handlers console_cohesion+

source install/setup.bash
```

> CMakeLists.txt auto-detects `aarch64` and sets `CMAKE_CUDA_ARCHITECTURES=87` (Orin
> Ampere). Override with `-DCMAKE_CUDA_ARCHITECTURES=87` explicitly if needed.

---

## Step 12 — Run

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

Open `http://<jetson-ip>:8765` in any browser on the same network.

---

## Performance tuning for Orin

| Parameter | Default | Suggestion for Orin NX 8 GB |
|-----------|---------|------------------------------|
| `max_iters` | 100 | 50–70 |
| `select_every_k_frame` | 3 | 4–6 |
| `depth_completion` | true | try `false` first |
| `sh_degree` | 3 | 1–2 |

Set Orin to maximum performance before a run:
```bash
sudo nvpmodel -m 0    # MAX-N mode
sudo jetson_clocks    # lock clocks to maximum
```

Monitor GPU/CPU/memory utilisation:
```bash
tegrastats
```

---

## Troubleshooting

**`Cannot find TensorRT`** — verify the symlinks exist:
```bash
ls /opt/tensorrt/include/NvInfer.h    # should exist
ls /opt/tensorrt/lib/libnvinfer.so    # should exist
```
If missing, re-run Step 3 to create the symlinks and install the dev packages.

**`Torch_DIR not found`** — run `source ~/.bashrc` or re-run the export commands in Step 5.

**`Engine file not found`** — re-run `ckpt/build_trt.sh` after any JetPack upgrade.

**`No messages received`** — verify topic names with `ros2 topic list`.

**Very slow or dropped frames** — see performance tuning table above.
