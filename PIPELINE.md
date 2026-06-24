# GaussianLIC Pipeline Guide

Two data sources are supported: the **Odin1 sensor** (real rosbags) and the **CARLA simulator** (synthetic bags). Both feed the same GaussianLIC mapping node through identical ROS2 topics.

---

## Table of Contents

1. [Quick-start: Odin rosbag](#1-quick-start-odin-rosbag)
2. [Quick-start: CARLA simulator](#2-quick-start-carla-simulator)
3. [Sensor differences: Odin vs CARLA](#3-sensor-differences-odin-vs-carla)
4. [File paths and storage layout](#4-file-paths-and-storage-layout)
5. [Testing and improving results](#5-testing-and-improving-results)
6. [Collecting log output](#6-collecting-log-output)
7. [Running the visualizers](#7-running-the-visualizers)
8. [Offline evaluation script](#8-offline-evaluation-script)

---

## 1. Quick-start: Odin rosbag

Odin bags are pre-processed — they already carry world-frame SLAM outputs, so no bridge is needed.

Results are saved to `gaussian_lic/result/<bag_name>/` — each bag gets its own
subdirectory so runs never overwrite each other.

```bash
BAG=/media/aloha/aloha_samsu/rosbags/Downtown1
BAG_NAME=$(basename "$BAG")

docker run --rm --gpus all \
    --network=host \
    -v "${BAG}:/bag:ro" \
    -v "/media/aloha/aloha_samsu/gaussian_lic/result/${BAG_NAME}:/result" \
    gaussian_lic:latest \
    bash -c "
      ros2 launch gaussian_lic odin1.launch.py result:=/result &
      sleep 12
      ros2 bag play /bag --clock -r 1.0
      wait
    "
```

**Optional launch arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `config` | `config/odin1.yaml` | Override config file |
| `result` | `<pkg>/result` | Output directory |
| `web_monitor` | `true` | Live dashboard on port 8765 |

**Web dashboard** (when running with `-p 8765:8765`): open `http://localhost:8765`.

**Live sensor** (instead of bag):
```bash
ros2 launch odin_ros_driver odin1.launch.py   # start driver first
ros2 launch gaussian_lic odin1.launch.py
```

---

## 2. Quick-start: CARLA simulator

Three-stage pipeline: CARLA sim → bridge records a bag → GaussianLIC replays the bag.

### Prerequisites

- CARLA 0.9.16 at `/media/aloha/aloha_samsu/CARLA/`
- `gaussian_lic:latest` Docker image (built via `gaussian_lic/build.sh`)
- NVIDIA Docker runtime (`nvidia-container-toolkit`)
- CARLA Python wheel at `CARLA/PythonAPI/carla/dist/` (cp312 variant)

### Step 1 — Record a bag from CARLA

```bash
/media/aloha/aloha_samsu/CARLA/launch_gaussianlic_pipeline.sh \
    --town Town03 \
    --duration 120 \
    --quality Low
```

This starts CARLA (headless), spawns an ego vehicle with LiDAR + camera via
`carla_gaussianlic_bridge.py`, and records the three GaussianLIC topics to a
timestamped bag under `/media/aloha/aloha_samsu/rosbags/carla_TIMESTAMP/`.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--town` | `Town03` | CARLA map (Town01–Town12, any installed map) |
| `--duration` | `120` | Recording length in seconds |
| `--quality` | `Low` | CARLA rendering quality: `Low` / `Medium` / `Epic` |

### Step 2 — Run GaussianLIC on the recorded bag

```bash
/media/aloha/aloha_samsu/CARLA/run_gaussianlic_on_bag.sh \
    /media/aloha/aloha_samsu/rosbags/carla_TIMESTAMP \
    [/optional/custom/result/dir]
```

The script launches GaussianLIC with `config/carla.yaml`, replays the bag at
1× speed, waits for `point_cloud.ply` to be written, then exits cleanly.

### Manual bridge (live, without recording)

```bash
# Terminal 1 — start CARLA (already running or launch manually)
cd /media/aloha/aloha_samsu/CARLA
./CarlaUE4.sh -RenderOffScreen -quality-level=Low

# Terminal 2 — run bridge + GaussianLIC together inside the container
RUN_NAME="carla_live_$(date +%Y%m%d_%H%M%S)"
docker run --rm --gpus all --network=host \
    -v /media/aloha/aloha_samsu/CARLA:/carla:ro \
    -v /media/aloha/aloha_samsu/gaussian_lic/config/carla.yaml:/carla_config.yaml:ro \
    -v "/media/aloha/aloha_samsu/gaussian_lic/result/${RUN_NAME}:/result" \
    gaussian_lic:latest \
    bash -c "
      pip install --quiet /carla/PythonAPI/carla/dist/carla-0.9.16-cp312-cp312-manylinux_2_31_x86_64.whl
      ros2 launch gaussian_lic odin1.launch.py config:=/carla_config.yaml result:=/result &
      sleep 12
      python3 /carla/carla_gaussianlic_bridge.py --town Town03 --duration 120
    "
```

---

## 3. Sensor differences: Odin vs CARLA

Both sources publish on identical topic names so GaussianLIC sees no difference at the node level. The differences are in the sensor characteristics and the config parameters that must match them.

| Property | Odin1 (real) | CARLA (simulated) |
|---|---|---|
| **Camera model** | Fisheye (FishPoly polynomial distortion) | Pinhole (zero distortion) |
| **Resolution (raw)** | 1600 × 1296 | 800 × 600 |
| **Output resolution** | 640 × 512 | 640 × 480 |
| **Focal length** | fx=733.44, fy=733.16 (FishPoly A11/A22) | fx=fy=400.0 (computed from 90° H-FOV) |
| **Distortion** | 6-term polynomial k2…k7 (non-trivial) | k2=0, k3–k7 set to Taylor-series of tan(θ) so remap is identity |
| **LiDAR type** | Physical rotating sensor (Odin hardware) | CARLA ray-cast, 32 channels |
| **LiDAR channels** | Hardware-dependent | 32 |
| **LiDAR range** | ~50 m | 50 m (`range=50` attribute) |
| **LiDAR points/scan** | Hardware-dependent | ~32 000 pts (320 000 pts/s ÷ 10 Hz) |
| **Rotation freq** | Hardware-dependent | 10 Hz (synchronous mode) |
| **FOV (vertical)** | Hardware-dependent | upper +2.0°, lower −26.8° |
| **T_cl (LiDAR→cam)** | Physical extrinsic calibration (small offset) | Pure rotation, zero translation (co-located) |
| **SLAM source** | Odin onboard SLAM → publishes world-frame cloud | Bridge transforms scan to world frame using CARLA ground-truth pose |
| **Config file** | `config/odin1.yaml` | `config/carla.yaml` |
| **SPNet depth completion** | Optional (disabled by default — was 81% of CPU) | Not applicable — no real depth sensor |
| **Noise** | Real sensor noise present | `noise_stddev=0.0` (ideal, noiseless) |
| **Exposure variation** | Present in real conditions | None (`apply_exposure: false`) |

### Key implication

Because the Odin uses a fisheye lens, the remap precomputes a correction LUT
using the polynomial coefficients. For CARLA the same code path runs but the
coefficients are chosen so the LUT is a mathematical identity — no actual
warping occurs. If you ever change the CARLA camera FOV you must recompute
`odin1_fx/fy/cx/cy` in `carla.yaml` to match.

---

## 4. File paths and storage layout

```
/media/aloha/aloha_samsu/
├── CARLA/
│   ├── CarlaUE4.sh                          # CARLA simulator binary
│   ├── PythonAPI/carla/dist/                # Python wheels (cp310/cp311/cp312)
│   ├── carla_gaussianlic_bridge.py          # ROS2 bridge node
│   ├── launch_gaussianlic_pipeline.sh       # Step 1: record from CARLA
│   └── run_gaussianlic_on_bag.sh            # Step 2: run GaussianLIC on bag
│
├── gaussian_lic/                            # This repository (git)
│   ├── config/
│   │   ├── odin1.yaml                       # Config for real Odin bags
│   │   └── carla.yaml                       # Config for CARLA bags
│   ├── launch/
│   │   └── odin1.launch.py                  # Launch file (both sources use this)
│   └── result/                              # Default output location
│       ├── point_cloud.ply                  # 3DGS map (open in Gaussian viewer)
│       ├── render/                          # Rendered training-view images
│       ├── render_depth/                    # Rendered depth maps
│       └── gt/                             # Ground-truth camera frames
│
└── rosbags/
    ├── Downtown1/                           # Real Odin bag (example)
    ├── <other Odin bags>/                   # Any bag recorded with the Odin sensor
    └── carla_YYYYMMDD_HHMMSS/              # CARLA-generated bags (timestamped)
```

**Where to store bags:**

| Bag type | Directory |
|----------|-----------|
| Real Odin bags | `/media/aloha/aloha_samsu/rosbags/<name>/` |
| CARLA-generated bags | `/media/aloha/aloha_samsu/rosbags/carla_<timestamp>/` (auto-created by the pipeline script) |

**Where to store results:**

Each run is saved to its own subdirectory named after the bag, so multiple runs
never overwrite each other.

| Run type | Default result dir |
|----------|--------------------|
| Odin bag | `/media/aloha/aloha_samsu/gaussian_lic/result/<bag_name>/` |
| CARLA bag (via script) | `/media/aloha/aloha_samsu/gaussian_lic/result/carla_<timestamp>/` |
| CARLA bag (manual) | `/media/aloha/aloha_samsu/gaussian_lic/result/carla_live_<timestamp>/` |

Override the result dir by passing `result:=/your/path` as a launch arg (Odin),
or as the second positional argument to `run_gaussianlic_on_bag.sh` (CARLA).

---

## 5. Testing and improving results

### Visualizing output

```bash
# RViz2 (live, while running)
rviz2 -d /media/aloha/aloha_samsu/gaussian_lic/config/gaussian_lic_odin1.rviz

# Offline: inspect the PLY in a Gaussian splatting viewer
# e.g. SuperSplat (browser), SIBR viewer, or Polycam
```

Web dashboard renders training-view images in real time at
`http://localhost:8765` (requires `-p 8765:8765` on the container).

---

### Gaussian initialization

Gaussians are seeded one-per-LiDAR-point at each new keyframe via `extend()`.
Scale is set proportional to depth (closer = smaller). Rotation is initialized
to identity.

**What to tune:**

| Parameter | Location | Effect |
|-----------|----------|--------|
| `scaling_scale` | `odin1.yaml` / `carla.yaml` | Multiplier on initial Gaussian scale. Increase if coverage is sparse (blurry borders); decrease if Gaussians bleed into each other. |
| `max_depth` | config yaml | Points beyond this are discarded at init. Raise if your scene has distant structure; lower to focus on close geometry. |
| `sh_degree` | config yaml | Spherical harmonics degree (0–3). Degree 3 = full view-dependent color; degree 0 = constant color. Lower degrees are faster and more stable on small datasets. |
| `skybox_points_num` | config yaml | Number of sky-sphere Gaussians added at init. Currently 0 — set to e.g. 1000 for outdoor scenes with visible sky to stop "floaters" forming above the scene. |

---

### Optimization iterations

`optimize()` runs on every keyframe and samples 100 random past keyframes for
one gradient step each.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_iters` | 10 | Gradient steps per keyframe. 10 = near real-time; 100 = high quality offline. Start here when debugging quality. |
| `iteration_decay` | true | Linearly decays learning rates toward the end of the sequence. Keeps early frames from being overwritten. |
| `lambda_dssim` | 0.2 | SSIM weight in `loss = (1−λ)·L1 + λ·SSIM`. Increase toward 0.4–0.5 for sharper structural detail; standard 3DGS uses 0.2. |
| `optimize_depth` | false | Add a depth supervision term using projected LiDAR points. Enable (`true`) + tune `lambda_depth` (0.005–0.05) if geometry is drifting. |
| `lambda_depth` | 0.005 | Weight of depth loss when `optimize_depth: true`. |

**Practical tuning sequence for CARLA:**
1. Raise `max_iters` to 50–100 and run offline to see the quality ceiling.
2. If colors are wrong → check `sh_degree`; try degree 1 first for debugging.
3. If geometry is mushy → enable `optimize_depth: true`, set `lambda_depth: 0.01`.
4. If results improve, walk `max_iters` back down until quality degrades — that's your real-time budget.

---

### Keyframe selection

| Parameter | Default | Effect |
|-----------|---------|--------|
| `select_every_k_frame` | 3 | Use every k-th aligned frame as a keyframe. Lower = more coverage, slower. Higher = faster, risk of missing fast motion. |
| `prune_every_n_keyframes` | 5 | Remove near-invisible Gaussians every N keyframes. Keeps Gaussian count bounded. |
| `prune_opacity_thresh` | 0.01 | Gaussians with opacity below this (post-sigmoid) are pruned. Standard 3DGS uses 0.005; 0.01 is more aggressive. |

---

### Synchronization

GaussianLIC aligns the three topics (`/odin1/cloud_slam`, `/odin1/odometry`,
`/odin1/image/compressed`) by timestamp with a ±50 ms window (see
`ARCHITECTURE.md`, line 475–509).

**CARLA-specific:** the bridge ticks in synchronous mode at 10 Hz and stamps
all three messages with the same `ros::Time::now()` so they align perfectly.
If you see "no aligned frame" warnings:

1. Check that `ros2 bag play` uses `--clock` — without it, ROS time doesn't
   advance and timestamps never match.
2. If running live (not bag), ensure the bridge's `_lidar_ready` / `_image_ready`
   events are both being set before the publish (check for "skipping tick" logs).
3. Widen the sync window in `mapping.cpp` (search for `50` ms) if your sensor
   has higher latency jitter.

**Odin-specific:** the bag already contains pre-aligned data from the Odin
onboard processor, so sync issues are rare. If they appear, check that the bag
was recorded with `--clock` and play it back with `--clock`.

---

### CARLA-specific improvements

| Issue | What to try |
|-------|-------------|
| Sparse point cloud | Increase `points_per_second` in the bridge (e.g. 640 000) or add channels (64-ch). Update `max_iters` to compensate for denser init. |
| Unrealistic lighting | Switch to `--quality Epic` in the pipeline script; re-record. `Medium`/`Epic` enables shadows and better material shaders. |
| Scene too simple | Use Town10HD or Town12 (complex urban) instead of Town01/03. |
| No motion blur / exposure | Enable `apply_exposure: true` + `exposure_lr: 0.001` to let the model fit per-frame exposure differences. |
| Gaussian drift at map edges | Set `skybox_points_num: 1000` + `skybox_radius: 1000` to anchor background Gaussians. |
| Noisy initialization test | Set `noise_stddev: 0.02` on the LiDAR blueprint in the bridge to simulate real sensor noise and test robustness. |

---

### Evaluating quality

After a run, the `result/` directory contains:

- `gt/` — ground-truth frames (the input images)
- `render/` — GaussianLIC's rendered views at the same poses
- `render_depth/` — rendered depth maps

LPIPS (AlexNet backbone) is computed at the end of each run and printed to
the node log. Lower is better (0 = perfect). Compare across parameter sweeps
to track progress.

For CARLA, because poses are ground-truth, any LPIPS degradation is due to the
Gaussian model itself — not SLAM drift — making it a clean signal for tuning
`max_iters`, `sh_degree`, and `lambda_dssim`.

---

## 6. Collecting log output

GaussianLIC writes everything to **stdout/stderr** of the `gs_mapping` process.
The easiest way to capture this is to `tee` the docker run output to a file on
the host.

### Tee all output to a file

```bash
BAG=/media/aloha/aloha_samsu/rosbags/Downtown1
BAG_NAME=$(basename "$BAG")
RESULT=/media/aloha/aloha_samsu/gaussian_lic/result/${BAG_NAME}
mkdir -p "$RESULT"

docker run --rm --gpus all --network=host \
    -v "${BAG}:/bag:ro" \
    -v "${RESULT}:/result" \
    gaussian_lic:latest \
    bash -c "
      ros2 launch gaussian_lic odin1.launch.py result:=/result 2>&1 &
      sleep 12
      ros2 bag play /bag --clock -r 1.0
      wait
    " 2>&1 | tee "${RESULT}/run.log"
```

The `2>&1` merges stderr into stdout before `tee` so nothing is missed.

### Enable verbose ROS2 logging

ROS2 log levels can be set per-node at launch:

```bash
ros2 launch gaussian_lic odin1.launch.py \
    --ros-args --log-level gaussianlic:=DEBUG
```

Levels in order of verbosity: `DEBUG` > `INFO` > `WARN` > `ERROR` > `FATAL`.
Default is `INFO`. Setting `DEBUG` will print every `RCLCPP_DEBUG*` call,
including the throttled sync-gap diagnostics added to `mapping.cpp`.

### What the log lines mean

| Pattern | Meaning |
|---------|---------|
| `[Waiting for aligned frame (N misses)]` | Buffers have been empty or mismatched for ~N×2 ms. Normal at start; sustained after bag finishes. If this appears mid-bag, the bag may be missing a topic or `--clock` is not set. |
| `Odom/cloud sync gap N ms` | Odometry timestamp is >50 ms ahead of the LiDAR cloud. The cloud message is dropped. If this fires repeatedly during a recording, the sensor or bridge has a timestamping offset. |
| `Image/cloud sync gap N ms` | Same as above but for the camera. |
| `[Init] Initializing Gaussian map from keyframe N` | First keyframe accepted — Gaussians are being seeded from this LiDAR scan. |
| `[Init] Done — N Gaussians seeded` | Initialization complete; optimization begins from the next keyframe. |
| `frame N (non-kf)` (overwrites on `\r`) | The current frame was not selected as a keyframe (controlled by `select_every_k_frame`). |
| `Cur Frame N, Update Xw GS/iter [add=Xms ext=Xms opt=Xms]` | Keyframe processed. `add` = undistort+project time, `ext` = extend() time, `opt` = optimize() time. |
| `[Pruned N GS, M remaining]` | Opacity-pruning step ran; N low-opacity Gaussians removed. |
| `[Training View PSNR/SSIM/LPIPS]` | Final metrics on frames seen during training (upper bound). |
| `[In-Sequence Novel View PSNR/SSIM/LPIPS]` | Final metrics on held-out test frames (true quality measure). |

### Keep a persistent log with timestamps (tee + ts)

```bash
# Requires 'moreutils' (apt install moreutils)
docker run ... gaussian_lic:latest bash -c "..." 2>&1 \
    | ts '[%Y-%m-%d %H:%M:%.S]' \
    | tee "${RESULT}/run.log"  # RESULT set as shown above
```

---

## 7. Running the visualizers

### Web dashboard (port 8765)

The web monitor is launched automatically alongside `gs_mapping` by
`odin1.launch.py` (controlled by the `web_monitor:=true/false` launch arg).

To expose port 8765 from Docker to the host, add `-p 8765:8765`:

```bash
BAG=/media/aloha/aloha_samsu/rosbags/Downtown1
BAG_NAME=$(basename "$BAG")

docker run --rm --gpus all --network=host \
    -p 8765:8765 \
    -v "${BAG}:/bag:ro" \
    -v "/media/aloha/aloha_samsu/gaussian_lic/result/${BAG_NAME}:/result" \
    gaussian_lic:latest \
    bash -c "
      ros2 launch gaussian_lic odin1.launch.py result:=/result 2>&1 &
      sleep 12
      ros2 bag play /bag --clock -r 1.0
      wait
    "
```

Then open **http://localhost:8765** in any browser on the host machine.

**Dashboard panels:**

| Panel | Source topic | Description |
|-------|-------------|-------------|
| Left: 3-D view | `/gaussian_lic/gaussians` | Interactive 3-D Gaussian splat cloud (drag/scroll/two-finger) |
| Left: Trajectory | `/odin1/odometry` | Colour-coded path (cyan → warm yellow = older → newer) |
| Right: Live sensor | `/gaussian_lic/input` | Undistorted camera frame entering the mapper (~1 Hz refresh) |
| Right: Reconstruction | `/gaussian_lic/render` | Latest Gaussian-splatted render at the current pose (~2 Hz) |
| Stats card | `/gaussian_lic/gaussians` | Odometry message count, total Gaussian count |
| Elapsed timer | — | Wall-clock time since first data arrived |

New Gaussians added at each keyframe briefly flash **yellow** before fading
into the main cloud — useful for watching the map build in real time.

**Disable the web monitor** (saves ~10 MB/s of ROS2 transport):

```bash
ros2 launch gaussian_lic odin1.launch.py web_monitor:=false
```

**Run the web monitor separately** (e.g. to reconnect to an already-running
GaussianLIC instance on the same network):

```bash
ros2 run gaussian_lic web_monitor
```

---

### RViz2

RViz2 can be run either inside the container or on the host (both work when
using `--network=host`).

**Inside the container:**

```bash
docker exec -it <container_name> \
    rviz2 -d /ros2_ws/install/gaussian_lic/share/gaussian_lic/config/gaussian_lic_odin1.rviz
```

**On the host (if ROS2 is installed):**

```bash
rviz2 -d /media/aloha/aloha_samsu/gaussian_lic/config/gaussian_lic_odin1.rviz
```

**Topics published by GaussianLIC:**

| Topic | Type | Description |
|-------|------|-------------|
| `/gaussian_lic/input` | `sensor_msgs/Image` | Undistorted, cropped camera frame |
| `/gaussian_lic/render` | `sensor_msgs/Image` | Gaussian-splatted render at current pose |
| `/gaussian_lic/depth` | `sensor_msgs/Image` | Rendered depth map (jet colormap) |
| `/gaussian_lic/cloud` | `sensor_msgs/PointCloud2` | Raw SLAM cloud from LiDAR (world frame) |
| `/gaussian_lic/gaussians` | `sensor_msgs/PointCloud2` | All current Gaussian centres with colour |
| `/gaussian_lic/new_points` | `sensor_msgs/PointCloud2` | Gaussians added at the latest keyframe |

Publish rate for `render`, `depth`, `gaussians`, and `new_points` is controlled
by `viz_every_n_keyframes` in the config YAML (default: every 5 keyframes).
Set to `1` for maximum refresh rate or `0` to disable.

**Quick topic inspection without RViz2:**

```bash
# Watch live render frames
ros2 topic echo /gaussian_lic/render --no-arr

# Count incoming keyframe data
ros2 topic hz /gaussian_lic/render

# Check all active topics
ros2 topic list | grep gaussian_lic
```

---

## 8. Offline evaluation script

After a run, `scripts/evaluate.py` computes **PSNR, SSIM, LPIPS, and MAE** for
every matched pair in `gt/` vs `render/` and writes a CSV + JSON summary.

### Usage

```bash
# Set these to match your run
BAG_NAME=Downtown1   # or carla_20260528_153000, etc.
RESULT=/media/aloha/aloha_samsu/gaussian_lic/result/${BAG_NAME}

# Inside the container (recommended — has torch, cv2, lpips_alex.pt)
docker run --rm --gpus all \
    -v /media/aloha/aloha_samsu/gaussian_lic:/ws:ro \
    gaussian_lic:latest \
    python3 /ws/scripts/evaluate.py /ws/result/${BAG_NAME}

# On the host (requires: torch, opencv-python, numpy)
python3 /media/aloha/aloha_samsu/gaussian_lic/scripts/evaluate.py \
    "${RESULT}"
```

The script auto-discovers `src/lpips/lpips_alex.pt` relative to the result
directory. Pass `--lpips-model /path/to/lpips_alex.pt` to override.

### Example output

```
Frame                        PSNR     SSIM    LPIPS      MAE
------------------------------------------------------------
test_0000.jpg               22.41   0.7823   0.1842   0.0491
test_0001.jpg               21.89   0.7701   0.1978   0.0523
...
------------------------------------------------------------
MEAN  (42 frames)           22.14   0.7762   0.1910   0.0507
```

### Output files

| File | Contents |
|------|----------|
| `<result_dir>/metrics.csv` | Per-frame PSNR, SSIM, LPIPS, MAE |
| `<result_dir>/metrics.json` | Summary means (machine-readable) |

### Metric interpretation

| Metric | Better | Typical range | What it measures |
|--------|--------|---------------|-----------------|
| PSNR (dB) | Higher | 20–30 for novel-view synthesis | Pixel-level fidelity (L2-based) |
| SSIM | Higher | 0.7–0.95 | Structural / perceptual similarity |
| LPIPS | Lower | 0.05–0.25 | Deep perceptual similarity (AlexNet) |
| MAE | Lower | 0.02–0.10 | Mean absolute pixel error |

LPIPS is the most perceptually meaningful metric — a low PSNR with low LPIPS
usually means the render is visually good but slightly offset in brightness.
PSNR is dominated by bright regions; SSIM captures texture structure.

### Comparing runs

Because each bag gets its own subdirectory, you can evaluate and compare
multiple runs with a simple loop:

```bash
RESULTS_ROOT=/media/aloha/aloha_samsu/gaussian_lic/result

for run_dir in "${RESULTS_ROOT}"/*/; do
    python3 "${RESULTS_ROOT}/../scripts/evaluate.py" "$run_dir"
done

# Side-by-side summary
for run_dir in "${RESULTS_ROOT}"/*/; do
    name=$(basename "$run_dir")
    echo -n "${name}: "
    cat "${run_dir}/metrics.json" | python3 -c \
        "import sys,json; d=json.load(sys.stdin); print(f'PSNR={d[\"mean_psnr\"]:.2f}  SSIM={d[\"mean_ssim\"]:.4f}  LPIPS={d[\"mean_lpips\"]}  MAE={d[\"mean_mae\"]:.4f}')"
done
```
