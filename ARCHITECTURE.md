# Odin Gaussian LIC Architecture

## Depth Completion

Estimate depth between sparse LiDAR points. Used to estimate depth for image patches where no LiDAR information is available.

### Relevant Files

- `ckpt/`
  - where the SPNet model is located

- `src/depth_completer.cpp`
  - runs SPNet with the provided lidar and image data

---

## SSIM

Structural Similarity Index Measure is a type of loss that focuses more on regional similarity (luminance, contrast, structure) than per pixel similarity like the L1 loss.

It contributes 20% to the loss when training the gaussians (the remaining loss is L1 loss which weird because in literature 80% SSIM is recommended Edit: the original 3DGS paper also uses 20% SSIM)

### Relevant Files

- `fused_ssim/`
  - A cuda optimized implementation of the SSIM loss

---

## LPIPS

Learned Perceptual Image Patch Similarity is a neural network based image loss which compares the latent space (embeddings at different depths of the backbone) of the original and the rendered image.

Currently only a model with alex as a backbone is saved but it also supports vgg and squeeze as a backbone.

It is only used for evaluation purposes and doesn’t influence training (because inference is computationally expensive)

### Relevant Files

- `lpips/lpipsPyTorch`
  - LPIPS implementation

- `lpips/lpips_alex.pt`
  - trained model to be loaded for inference in LibTorch in C++

---

## Rasterizer

This is where the “splatting” happens. The gaussians are splatted onto the image plane to produce the rendered image. It also returns the per pixel depth by accumulating the gaussian depths.

The rasterizer is differentiable so the loss is backpropagated to optimize the gaussian map.

### Relevant Files

- `rasterizer/cuda_rasterizer`
  - odin_gaussian_lic uses a cuda optimized per gaussian backpropagation to improve runtime

- `rasterizer/`
  - high level rasterizer implementation

---

## kNN

K-nearest neighbors is used to compute the mean squared distance of a point in the pointcloud to its closest 3 neighbours.

This can be used to initialize the size of the gaussian for that point. The higher the point density, the smaller the gaussian needs to be.

In gaussian_lic this metric is only used for the sky gaussians while the foreground gaussians are initialized based on depth (closer = smaller gaussian)

### Relevant Files

- `simple-knn/`
  - A simple cuda optimized k-nearest neighbors implementation using Morton sorting + box partitioning + local KNN search

---

# Gaussian Mapping

The gaussian map defines the number, position, rotation, scaling, opacity and color of the gaussians.

This is what we’re trying to optimize based on images of the environment to generate renders from it.

All of this happens in `gaussian.cpp` so below are the most important classes.

---

## `Dataset`

- bundles a pointcloud, a pose and an image that correspond to approximately the same timestamp into a camera/frame which is used for training (if its a keyframe) or evaluation (if it isn’t a keyframe).

---

## `GaussianModel`

- The gaussian model stores the gaussian map.
- It can be initialized by creating one gaussian per point in the pointcloud (and optional sky gaussians which are currently set to 0).
  - `scale -> depth`
  - `position -> point position`
  - `color -> point color`
  - `rotation -> identity`

---

## `extend()`

- adds gaussians to the gaussian map where the coverage isnt good enough yet (`opacity < 0.99`).
- It only adds a a gaussian if there is a point there in the pointcloud (unlike traditional gaussian splatting which splits gaussians if needed).

This is called on every new keyframe to check if the new pointcloud has any useful points.

---

## `optimize()`

- randomly picks 100 keyframes and does one training step (forward + back propagation) per camera/keyframe to optimize the gaussians that are in view.

---

## `evaluateVisualQuality()`

- saves the gaussian map and renders the test frames to evaluate quality

---

# Mapping

`mapping.cpp` orchestrates the whole control loop.

It reads the incoming topics from the odin1 sensor or the bagfiles and processes them with the modules described above.

Align the incoming data temporally and create a frame/camera. Then initialize or extend the gaussian map. Then optimize and once no more data is incoming evaluate the results.

This class was extended to handle the ROS2/Odin-specific operations.

We should probably move to separate files for better readability. One for ROS2 integration `GaussianLICNode`, one for frame processing and the original `mapping.cpp`

---

## FishPoly undistortion (lines 62–91)

- fisheye polynomial model, angle-based, requires precomputed remap to undistort image

---

## JPEG decompression (line 549)

- camera publishes compressed, need to decode

---

## Pose conversion (line 561)

- `/odin1/odometry` is LiDAR frame
- need `T_wl × T_lc⁻¹ → camera frame`

---

## Depth map generation (line 567)

- project SLAM cloud into camera
- turn point visibility into depth map to use for depth completion

---

## Colorized cloud (line 572)

- bilinear interpolate image colors onto SLAM points for ground truth

---

## ROS2 messaging (lines 475–509)

- subscribe with QoS
- buffer
- align by timestamp (`±50 ms`)

---

## Live visualization (lines 330–347)

- publish diagnostics and renders back to RViz2

---

# SLAM

In gaussian LIC the SLAM is done by Coco LIC but in our case the odin1 sensor does all the SLAM for us. Yay :3
