# Gaussian-LIC — Changes vs. Upstream

This document summarises every meaningful difference between this fork and the original
[Gaussian-LIC](https://github.com/APRIL-ZJU/Gaussian-LIC) /
[Gaussian-LIC2](https://github.com/APRIL-ZJU/Gaussian-LIC2) repositories.

---

## 1. ROS1 → ROS2 Migration

The most sweeping change is a full port from ROS 1 / catkin to **ROS 2 / ament_cmake**.

| Area | Original (ROS1) | Local build (ROS2) |
|---|---|---|
| Node API | `ros::NodeHandle`, `ros::spin()` | `rclcpp::Node`, `rclcpp::spin()` |
| Messages | `sensor_msgs/Image`, `geometry_msgs/PoseStamped` | `sensor_msgs::msg::Image`, `geometry_msgs::msg::PoseStamped` |
| TF conversion | `tf::quaternionMsgToEigen`, `tf_conversions/tf_eigen.h` | Inline `Eigen::Quaterniond(w,x,y,z)` — TF dependency removed |
| cv_bridge | `#include <cv_bridge/cv_bridge.h>` | `#include <cv_bridge/cv_bridge.hpp>` (ROS 2 header) |
| Image decode | `cv_bridge::toCvCopy(msg, encoding)` | Direct pointer cast from `msg->data` + `.clone()` |
| Package path | `ros::package::getPath("gaussian_lic")` | `ament_index_cpp::get_package_share_directory(...)` |
| Build system | `catkin_package` / `include_directories` | `ament_target_dependencies` / `target_include_directories` |
| Shutdown | Process left running | `rclcpp::shutdown()` called after mapping finishes so the process exits cleanly |

**Affected files:** `CMakeLists.txt`, `package.xml`, `mapping.h`, `mapping.cpp`, `gaussian.cpp`

---

## 2. Odin1 Sensor Support

A complete integration of the **Odin1** LiDAR-camera sensor was added.

### New files
- `config/odin1.yaml` — full sensor config: Odin1 intrinsics, FishPoly distortion coefficients, LiDAR-to-camera extrinsic (`T_cl`), topic names, and tuned Gaussian model parameters.
- `config/gaussian_lic_odin1.rviz` — RViz2 layout for the Odin1.
- `launch/odin1.launch.py` — ROS 2 launch file supporting both bag replay and live sensor, with configurable `config`, `result`, and `web_monitor` arguments.

### FishPoly undistortion (`mapping.cpp`)
The Odin1 uses a polynomial fisheye model (not OpenCV's standard `fisheye`). A new `computeFishPolyUndistortMap()` function builds OpenCV `remap` maps for this model by solving the forward projection Newton-iteratively for each output pixel. Coefficients `k2`–`k7` are read from the YAML.

### Auto-computed crop + resize intrinsics (`mapping.h` — `Params`)
The original code required the user to manually specify `fx, fy, cx, cy` for the output image. The new code:
1. Reads the original Odin1 sensor resolution (`odin1_width`, `odin1_height`) and intrinsics.
2. Center-crops to the target aspect ratio.
3. Scales to the requested `width × height`.
4. Auto-computes output intrinsics from the crop/scale transform — with optional YAML override.

### Configurable topics
`cloud_topic`, `odom_topic`, `image_topic` are now YAML-configurable (defaulting to `/odin1/*` topics).

---

## 3. TensorRT 10.x Compatibility (`depth_completer.cpp/.h`)

The original code used the TensorRT 8.x binding API. The local build updates to the TRT 10.x tensor API.

| Change | Original (TRT 8) | Local build (TRT 10) |
|---|---|---|
| Binding count | `getNbBindings()` | `getNbIOTensors()` |
| Tensor shape | `getBindingDimensions(i)` | `getTensorShape(name)` |
| Inference call | `executeV2(void**)` | `enqueueV3(stream)` |
| Buffer address | Implicit via pointer array | `setTensorAddress(name, ptr)` per tensor |
| CUDA transfers | `cudaMemcpy` (synchronous) | `cudaMemcpyAsync` on a dedicated `cudaStream_t` |
| Stream lifecycle | No stream | `cudaStreamCreate` / `cudaStreamDestroy` + `cudaStreamSynchronize` |

A `mTensorNames` vector was added to store IO tensor names at allocation time, avoiding repeated string lookups during inference.

---

## 4. Gaussian Model Enhancements (`gaussian.cpp/.h`)

### Opacity-based Gaussian pruning
`GaussianModel::pruneGaussians(mask)` was added. It takes a boolean keep-mask, filters all six Gaussian parameter tensors (`xyz`, `features_dc`, `features_rest`, `opacity`, `scaling`, `rotation`), and migrates the Adam optimizer states to the pruned parameter pointers — so training momentum is preserved across prunes.

Pruning is triggered in the mapping loop every `prune_every_n_keyframes` keyframes, removing Gaussians with opacity below `prune_opacity_thresh`. Both values are YAML-configurable (default: disabled).

### Configurable optimization iterations
`max_iters` was added as a YAML/launch parameter (default: 100) and is now passed explicitly to `optimize()` rather than being a hardcoded local variable.

### Timestamp-based frame naming
Train/test frame filenames changed from zero-padded sequential integers (`train_0001.jpg`) to ROS timestamp strings (`train_1234567890_000000000.jpg`). This makes frames unambiguous when bags are replayed non-linearly.

### Safer result directory cleanup
The original `fs::remove_all(result_path)` wiped the entire output directory. The new code iterates and removes only its *contents*, preserving the directory itself.

### Early map save
`pc->saveMap(result_path)` is called immediately after optimization completes (before rendering), so the Gaussian map is on disk even if the render step crashes.

---

## 5. Live Web Monitor (`scripts/web_monitor_node.py`)

A new Python ROS 2 node serves a browser dashboard on **port 8765**.

| Endpoint | Content |
|---|---|
| `GET /` | Full dashboard HTML (Three.js 3-D viewer) |
| `GET /cloud.bin` | Downsampled Gaussian point cloud (binary: `uint32 N` + `float32 xyz[N*3]` + `uint8 rgb[N*3]`) |
| `GET /newpts.bin` | New Gaussians from the latest keyframe only |
| `GET /input.jpg` | Latest undistorted input frame (JPEG) |
| `GET /render.jpg` | Latest Gaussian-splat render (JPEG) |
| `GET /state.json` | Trajectory, stats, and cloud/newpts version counters |

The dashboard polls at 5 Hz, renders the Gaussian cloud in Three.js with orbit controls, and highlights newly added Gaussians per keyframe. It subscribes to `/gaussian_lic/cloud`, `/gaussian_lic/rendered_image`, `/gaussian_lic/input_image`, and `/odometry`.

The node is launched optionally alongside `gs_mapping` via the `web_monitor` launch argument.

---

## 6. Build System Improvements (`CMakeLists.txt`)

- **ROS 2 / ament**: full migration from catkin to `ament_cmake` with proper `install()` rules for the binary, config, launch, ckpt, lpips, and web monitor script.
- **Flexible library discovery**: `OpenCV_DIR`, `Torch_DIR`, and `TENSORRT_ROOT` are no longer hardcoded. Each is resolved in priority order: CMake cache → environment variable → auto-glob under `~/Software` and `/mnt/Volume/Ubuntu/3d_vision/Software`.
- **TensorRT 10 compatibility**: `nvparsers` (removed in TRT 10) is detected with `find_library` and linked conditionally via a generator expression.
- **CUDA architecture**: explicitly set to `86` (RTX 3090 / sm_86); CUDA resolved via `CUDA_HOME` env var or `/usr/local/cuda` symlink.

---

## 7. TRT Engine Build Script (`ckpt/build_trt.sh`)

- `TENSORRT_ROOT` is no longer hardcoded to `TensorRT-8.6.1.6`; the script auto-detects the latest `TensorRT-*` directory under `~/Software` or uses the `TENSORRT_ROOT` env var.
- The library path changed from `targets/x86_64-linux-gnu/lib` (TRT 8 layout) to `lib` (TRT 10 layout).

---

## 8. Minor Fixes

| File | Fix |
|---|---|
| `src/rasterizer/cuda_rasterizer/rasterizer_impl.h` | Added `#include <cstdint>` — required by some CUDA 13 toolchain configurations |
| `src/simple-knn/simple_knn.cu` | Added `#include <cfloat>` — `FLT_MAX` was implicitly available via CUDA headers in older toolchains but not newer ones |

---

## Files Added (not in original)

| Path | Purpose |
|---|---|
| `Dockerfile` | Reproducible Docker build environment |
| `INSTALL.md` | Step-by-step build and install guide |
| `ARCHITECTURE.md` | System architecture documentation |
| `build.sh` | Convenience wrapper for `colcon build` |
| `config/odin1.yaml` | Odin1 sensor configuration |
| `config/gaussian_lic_odin1.rviz` | RViz2 layout |
| `launch/odin1.launch.py` | ROS 2 launch file for Odin1 |
| `scripts/web_monitor_node.py` | Live browser dashboard |
