/*
 * Gaussian-LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-Inertial-Camera Fusion
 * Copyright (C) 2025 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include "yaml_utils.h"

#include <chrono>
#include <cmath>
#include <deque>
#include <queue>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

// ROS2 headers
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/empty.hpp>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#  include <cv_bridge/cv_bridge.hpp>
#else
#  include <cv_bridge/cv_bridge.h>
#endif

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Eigen>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>

// ============================================================
// Params – loaded from YAML config
// ============================================================
class Params
{
public:
    Params(const YAML::Node& node)
    {
        // ------- Odin1 original camera intrinsics (before undistortion / resize) -------
        odin1_width  = node["odin1_width"].as<int>(1600);
        odin1_height = node["odin1_height"].as<int>(1296);
        odin1_fx     = node["odin1_fx"].as<double>(733.44);
        odin1_fy     = node["odin1_fy"].as<double>(733.16);
        odin1_cx     = node["odin1_cx"].as<double>(813.52);
        odin1_cy     = node["odin1_cy"].as<double>(619.15);
        odin1_skew   = node["odin1_skew"].as<double>(0.0);  // A12 in cam_in_ex.txt

        // FishPoly distortion coefficients (k2..k7 → powers r^2 .. r^7)
        k2 = node["k2"].as<double>(0.0);
        k3 = node["k3"].as<double>(0.0);
        k4 = node["k4"].as<double>(0.0);
        k5 = node["k5"].as<double>(0.0);
        k6 = node["k6"].as<double>(0.0);
        k7 = node["k7"].as<double>(0.0);

        // LiDAR-to-camera extrinsic: T_cl (camera-from-LiDAR, 4×4 row-major)
        auto tcl_vec = node["T_cl"].as<std::vector<double>>();
        T_cl = Eigen::Matrix4d::Identity();
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                T_cl(i, j) = tcl_vec[i * 4 + j];

        // ------- Output image size (input to Gaussian splatting) -------
        width  = node["width"].as<int>(640);
        height = node["height"].as<int>(512);

        // Auto-compute crop+resize intrinsics from original calibration.
        // Center-crop the undistorted image to the target aspect ratio, then resize.
        double target_aspect = static_cast<double>(width) / height;
        int crop_w = odin1_width;
        int crop_h = static_cast<int>(std::round(static_cast<double>(crop_w) / target_aspect));
        if (crop_h > odin1_height) {
            crop_h = odin1_height;
            crop_w = static_cast<int>(std::round(static_cast<double>(crop_h) * target_aspect));
        }
        // Store crop offsets for use in processFrame
        crop_x = (odin1_width  - crop_w) / 2;
        crop_y = (odin1_height - crop_h) / 2;
        crop_w_out = crop_w;
        crop_h_out = crop_h;

        double scale_x = static_cast<double>(width)  / crop_w;
        double scale_y = static_cast<double>(height) / crop_h;

        // Compute output intrinsics (allow YAML override)
        fx = node["fx"] ? node["fx"].as<double>() : odin1_fx * scale_x;
        fy = node["fy"] ? node["fy"].as<double>() : odin1_fy * scale_y;
        cx = node["cx"] ? node["cx"].as<double>() : (odin1_cx - crop_x) * scale_x;
        cy = node["cy"] ? node["cy"].as<double>() : (odin1_cy - crop_y) * scale_y;

        // ------- Gaussian splatting dataset params -------
        select_every_k_frame = node["select_every_k_frame"].as<int>();
        depth_completion     = node["depth_completion"].as<bool>();
        patch_size           = node["patch_size"].as<int>();
        max_depth            = node["max_depth"].as<double>();

        // Engine path for SPNet depth completion
        std::string pkg_path = ament_index_cpp::get_package_share_directory("gaussian_lic");
        if (height == 512 && width == 640) engine_path = pkg_path + "/ckpt/spnet_512_640.engine";
        if (height == 480 && width == 640) engine_path = pkg_path + "/ckpt/spnet_480_640.engine";

        // ------- Gaussian model params -------
        sh_degree             = node["sh_degree"].as<int>();
        white_background      = node["white_background"].as<bool>();
        random_background     = node["random_background"].as<bool>();
        convert_SHs_python    = node["convert_SHs_python"].as<bool>();
        compute_cov3D_python  = node["compute_cov3D_python"].as<bool>();
        lambda_erank          = node["lambda_erank"].as<double>();
        scaling_scale         = node["scaling_scale"].as<double>();

        position_lr   = node["position_lr"].as<double>();
        feature_lr    = node["feature_lr"].as<double>();
        opacity_lr    = node["opacity_lr"].as<double>();
        scaling_lr    = node["scaling_lr"].as<double>();
        rotation_lr   = node["rotation_lr"].as<double>();
        lambda_dssim  = node["lambda_dssim"].as<double>();
        optimize_depth = node["optimize_depth"].as<bool>();
        lambda_depth  = node["lambda_depth"].as<double>();
        iteration_decay = node["iteration_decay"].as<bool>();
        max_optimize_iters = node["max_optimize_iters"] ? node["max_optimize_iters"].as<int>() : 100;

        apply_exposure    = node["apply_exposure"].as<bool>();
        exposure_lr       = node["exposure_lr"].as<double>();
        skybox_points_num = node["skybox_points_num"].as<int>();
        skybox_radius     = node["skybox_radius"].as<int>();

        // ------- Odin1 ROS2 topic names -------
        cloud_topic = node["cloud_topic"] ? node["cloud_topic"].as<std::string>() : "/odin1/cloud_slam";
        odom_topic  = node["odom_topic"]  ? node["odom_topic"].as<std::string>()  : "/odin1/odometry";
        image_topic = node["image_topic"] ? node["image_topic"].as<std::string>() : "/odin1/image/compressed";

        // How often to publish the rendered image and Gaussian cloud (every N keyframes).
        // 1 = every keyframe (max quality, more CPU/GPU overhead).
        // 0 = disable visualization publishing entirely.
        viz_every_n_keyframes    = node["viz_every_n_keyframes"] ? node["viz_every_n_keyframes"].as<int>() : 5;
        max_iters                = node["max_iters"] ? node["max_iters"].as<int>() : 100;
        prune_opacity_thresh     = node["prune_opacity_thresh"] ? node["prune_opacity_thresh"].as<double>() : 0.005;
        prune_every_n_keyframes  = node["prune_every_n_keyframes"] ? node["prune_every_n_keyframes"].as<int>() : 0;
        max_gaussians            = node["max_gaussians"] ? node["max_gaussians"].as<int>() : 0; // 0 = unlimited
    }

    // Output image size / intrinsics (for Gaussian splatting)
    int    width, height;
    double fx, fy, cx, cy;

    // Crop parameters (computed from odin1_* and width/height)
    int crop_x, crop_y, crop_w_out, crop_h_out;

    // Dataset
    int    select_every_k_frame;
    bool   depth_completion;
    int    patch_size;
    double max_depth;
    std::string engine_path;

    // Gaussian model
    int    sh_degree;
    bool   white_background, random_background;
    bool   convert_SHs_python, compute_cov3D_python;
    float  lambda_erank;
    double scaling_scale;

    double position_lr, feature_lr, opacity_lr, scaling_lr, rotation_lr;
    double lambda_dssim;
    bool   optimize_depth;
    double lambda_depth;
    bool   iteration_decay;
    int    max_optimize_iters;

    bool   apply_exposure;
    double exposure_lr;
    int    skybox_points_num, skybox_radius;

    // Optimization
    int max_iters;

    // Pruning
    double prune_opacity_thresh;
    int    prune_every_n_keyframes;

    // Hard cap on Gaussian count (0 = unlimited)
    int    max_gaussians;

    // Visualization
    int viz_every_n_keyframes;

    // Odin1 topics
    std::string cloud_topic, odom_topic, image_topic;

    // Odin1 original calibration
    int    odin1_width, odin1_height;
    double odin1_fx, odin1_fy, odin1_cx, odin1_cy, odin1_skew;

    // FishPoly distortion coefficients
    double k2, k3, k4, k5, k6, k7;

    // LiDAR-to-camera extrinsic (camera-from-LiDAR)
    Eigen::Matrix4d T_cl;
};

// ============================================================
// Frame – one synchronized multi-modal keyframe
// ============================================================
struct Frame
{
    // Colorized XYZRGB point cloud in world frame
    sensor_msgs::msg::PointCloud2::SharedPtr point_msg;
    // Camera pose in world frame (T_wc)
    geometry_msgs::msg::PoseStamped::SharedPtr pose_msg;
    // Undistorted + resized BGR8 image
    sensor_msgs::msg::Image::SharedPtr image_msg;
    // Float32 depth map (metric, same size as image)
    sensor_msgs::msg::Image::SharedPtr depth_msg;
};
