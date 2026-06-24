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

#include "mapping.h"
#include "gaussian.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <iomanip>

// ============================================================
// Global state shared between ROS2 callbacks and mapping thread
// ============================================================

std::mutex m_buf;
std::atomic<bool>   exit_flag(false);
std::atomic<bool>   recording_started(false);
std::atomic<double> last_cloud_time(0.0);
std::atomic<bool>   gaussians_initialized(false);

// Forward-declare the node so the mapping thread can access it
class GaussianLICNode;
std::shared_ptr<GaussianLICNode> g_node;

// ============================================================
// Helper: FishPoly undistortion remap  (angle-based polynomial model)
//
// The Odin1 uses PolynomialCamera (manifoldsdk/odin_ros_driver):
//
//   Forward (world→pixel):
//     θ  = acos(z / |xyz|)   – angle from optical axis
//     θ_d = θ + k2·θ² + k3·θ³ + … + k7·θ⁷
//     scaling = θ_d / r   (r = sqrt(x²+y²))
//     u_d = fx·(x·scaling) + skew·(y·scaling) + cx
//     v_d = fy·(y·scaling) + cy
//
//   Inverse (pixel→ray) uses 7-iteration Newton solve (see cam2world).
//
// For cv::remap we compute the FORWARD map: for each OUTPUT (undistorted)
// pixel (u_u, v_u) → find its source in the INPUT (distorted) image.
// We treat (u_u, v_u) as a perfect pinhole projection of an unobserved ray,
// then re-project that ray through the distortion model to get (u_d, v_d).
// ============================================================
static void computeFishPolyUndistortMap(
    int width, int height,
    double fx, double fy, double cx, double cy, double skew,
    double k2, double k3, double k4, double k5, double k6, double k7,
    cv::Mat& mapX, cv::Mat& mapY)
{
    mapX.create(height, width, CV_32FC1);
    mapY.create(height, width, CV_32FC1);

    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            // Undistorted normalised coordinates (pinhole back-projection)
            double yn = (v - cy) / fy;
            double xn = (u - cx - yn * skew) / fx;

            double r = std::sqrt(xn * xn + yn * yn);  // r = tan(θ)

            // θ = atan(r),  then thetad_from_theta (exact forward model)
            double theta  = std::atan(r);
            double th2 = theta * theta;
            double th3 = th2 * theta;
            double th4 = th3 * theta;
            double th5 = th4 * theta;
            double th6 = th5 * theta;
            double th7 = th6 * theta;
            double theta_d = theta + k2*th2 + k3*th3 + k4*th4 + k5*th5 + k6*th6 + k7*th7;

            double scale = (r > 1e-8) ? (theta_d / r) : 1.0;

            double xd = xn * scale;
            double yd = yn * scale;

            mapX.at<float>(v, u) = static_cast<float>(fx * xd + skew * yd + cx);
            mapY.at<float>(v, u) = static_cast<float>(fy * yd + cy);
        }
    }
}

// ============================================================
// Helper: Odometry → 4×4 world-from-LiDAR transform
// ============================================================
static Eigen::Matrix4d odomToMatrix(const nav_msgs::msg::Odometry& odom)
{
    const auto& q = odom.pose.pose.orientation;
    const auto& t = odom.pose.pose.position;

    Eigen::Quaterniond quat(q.w, q.x, q.y, q.z);
    quat.normalize();

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = quat.toRotationMatrix();
    T(0, 3) = t.x;
    T(1, 3) = t.y;
    T(2, 3) = t.z;
    return T;
}

// ============================================================
// Helper: project world-frame SLAM cloud into camera → depth map
// ============================================================
static cv::Mat generateDepthMap(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_w,
    const Eigen::Matrix4d& T_cw,
    int width, int height,
    double fx, double fy, double cx, double cy,
    double max_depth)
{
    cv::Mat depth = cv::Mat::zeros(height, width, CV_32FC1);

    for (const auto& pt : cloud_w->points) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;

        Eigen::Vector4d p_w(pt.x, pt.y, pt.z, 1.0);
        Eigen::Vector4d p_c = T_cw * p_w;

        double z = p_c(2);
        if (z < 0.1 || z > max_depth) continue;

        double u = fx * p_c(0) / z + cx;
        double v = fy * p_c(1) / z + cy;

        int ui = static_cast<int>(std::round(u));
        int vi = static_cast<int>(std::round(v));
        if (ui < 0 || ui >= width || vi < 0 || vi >= height) continue;

        float& cur = depth.at<float>(vi, ui);
        if (cur == 0.0f || static_cast<float>(z) < cur)
            cur = static_cast<float>(z);
    }
    return depth;
}

// ============================================================
// Helper: project world-frame SLAM cloud into camera, colorize from image
//         → XYZRGB PointCloud2 in world frame
// ============================================================
static sensor_msgs::msg::PointCloud2::SharedPtr generateColorizedCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_w,
    const cv::Mat& image_bgr,
    const Eigen::Matrix4d& T_cw,
    int width, int height,
    double fx, double fy, double cx, double cy,
    const std_msgs::msg::Header& header)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>);
    colored->reserve(cloud_w->size());

    for (const auto& pt : cloud_w->points) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;

        Eigen::Vector4d p_w(pt.x, pt.y, pt.z, 1.0);
        Eigen::Vector4d p_c = T_cw * p_w;

        double z = p_c(2);
        if (z < 0.1) continue;

        double u = fx * p_c(0) / z + cx;
        double v = fy * p_c(1) / z + cy;

        // Bilinear interpolation bounds check
        int u0 = static_cast<int>(u);
        int v0 = static_cast<int>(v);
        if (u0 < 0 || u0 + 1 >= width || v0 < 0 || v0 + 1 >= height) continue;

        double du = u - u0;
        double dv = v - v0;

        auto interp = [&](int c) -> uint8_t {
            double val =
                (1 - du) * (1 - dv) * image_bgr.at<cv::Vec3b>(v0,     u0    )[c]
              + (    du) * (1 - dv) * image_bgr.at<cv::Vec3b>(v0,     u0 + 1)[c]
              + (1 - du) * (    dv) * image_bgr.at<cv::Vec3b>(v0 + 1, u0    )[c]
              + (    du) * (    dv) * image_bgr.at<cv::Vec3b>(v0 + 1, u0 + 1)[c];
            return static_cast<uint8_t>(std::clamp(val, 0.0, 255.0));
        };

        pcl::PointXYZRGB p_rgb;
        p_rgb.x = pt.x; p_rgb.y = pt.y; p_rgb.z = pt.z;
        p_rgb.b = interp(0);   // OpenCV stores BGR
        p_rgb.g = interp(1);
        p_rgb.r = interp(2);
        colored->push_back(p_rgb);
    }

    auto cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
    pcl::toROSMsg(*colored, *cloud_msg);
    cloud_msg->header = header;
    return cloud_msg;
}

// ============================================================
// GaussianLICNode – subscribes to Odin1 topics, processes data,
// feeds synchronized frames to the Gaussian splatting thread
// ============================================================
class GaussianLICNode : public rclcpp::Node
{
public:
    // Constructor: reads config_path from ROS2 parameter server (set by launch file),
    // loads the YAML, then initialises everything.
    // NOTE: params_ is value-initialised first as a placeholder; loadConfig_() then
    // re-constructs it once the Node base class is fully alive.
    explicit GaussianLICNode()
        : Node("gaussianlic"),
          params_(initParams_())   // Node is alive here; declare_parameter works
    {
        result_path_ = get_parameter("result_path").as_string();
        lpips_path_  = get_parameter("lpips_path").as_string();

        RCLCPP_INFO(get_logger(), "Config: %s", get_parameter("config_path").as_string().c_str());

        // Precompute FishPoly undistortion map (full original resolution)
        RCLCPP_INFO(get_logger(), "Precomputing FishPoly undistortion maps (%dx%d)...",
            params_.odin1_width, params_.odin1_height);
        computeFishPolyUndistortMap(
            params_.odin1_width, params_.odin1_height,
            params_.odin1_fx, params_.odin1_fy,
            params_.odin1_cx, params_.odin1_cy, params_.odin1_skew,
            params_.k2, params_.k3, params_.k4,
            params_.k5, params_.k6, params_.k7,
            undist_map_x_, undist_map_y_);

        RCLCPP_INFO(get_logger(),
            "Crop %dx%d @ offset (%d,%d) → resize to %dx%d  |  fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
            params_.crop_w_out, params_.crop_h_out,
            params_.crop_x,     params_.crop_y,
            params_.width,      params_.height,
            params_.fx, params_.fy, params_.cx, params_.cy);

        // QoS matching Odin1 bag replay (BEST_EFFORT publishers)
        auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();

        sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            params_.cloud_topic, qos,
            [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) { cloudCallback(std::move(msg)); });

        sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
            params_.odom_topic, qos,
            [this](nav_msgs::msg::Odometry::SharedPtr msg) { odomCallback(std::move(msg)); });

        sub_image_ = create_subscription<sensor_msgs::msg::CompressedImage>(
            params_.image_topic, qos,
            [this](sensor_msgs::msg::CompressedImage::SharedPtr msg) { imageCallback(std::move(msg)); });

        RCLCPP_INFO(get_logger(),
            "Subscribed to:\n  cloud: %s\n  odom:  %s\n  image: %s",
            params_.cloud_topic.c_str(),
            params_.odom_topic.c_str(),
            params_.image_topic.c_str());

        pub_input_      = create_publisher<sensor_msgs::msg::Image>("/gaussian_lic/input",      1);
        pub_render_     = create_publisher<sensor_msgs::msg::Image>("/gaussian_lic/render",     1);
        pub_depth_      = create_publisher<sensor_msgs::msg::Image>("/gaussian_lic/depth",      1);
        pub_cloud_      = create_publisher<sensor_msgs::msg::PointCloud2>("/gaussian_lic/cloud",      1);
        pub_gaussians_  = create_publisher<sensor_msgs::msg::PointCloud2>("/gaussian_lic/gaussians",  1);
        pub_new_points_ = create_publisher<sensor_msgs::msg::PointCloud2>("/gaussian_lic/new_points", 1);

        // If wait_for_start is true, hold off processing until the web UI button is pressed.
        // Default false so bag replay / sweep runs start automatically.
        bool wait_for_start = declare_parameter<bool>("wait_for_start", false);
        if (!wait_for_start) {
            recording_started = true;
        } else {
            sub_start_ = create_subscription<std_msgs::msg::Empty>(
                "/gaussian_lic/start_recording", rclcpp::QoS(1),
                [](std_msgs::msg::Empty::SharedPtr) {
                    recording_started = true;
                    std::cout << "\n\033[1;32m  *** Recording Started! Now processing sensor data. ***\033[0m\n\n";
                });
            RCLCPP_INFO(get_logger(),
                "Wait-for-start mode: press the Record button in the web dashboard to begin.");
        }

        sub_stop_ = create_subscription<std_msgs::msg::Empty>(
            "/gaussian_lic/stop_recording", rclcpp::QoS(1),
            [](std_msgs::msg::Empty::SharedPtr) {
                std::cout << "\n\033[1;33m  *** Stop requested — saving results… ***\033[0m\n\n";
                exit_flag = true;
            });

        RCLCPP_INFO(get_logger(),
            "Publishing: /gaussian_lic/input  /gaussian_lic/render  /gaussian_lic/depth"
            "  /gaussian_lic/cloud  /gaussian_lic/gaussians  /gaussian_lic/new_points");
    }

    // Called by the mapping thread (under m_buf lock) to pop one aligned frame
    bool getAlignedFrame(Frame& out_frame)
    {
        if (cloud_buf_.empty() || odom_buf_.empty() || image_buf_.empty())
            return false;

        // Use cloud timestamp as the reference
        double frame_time = rclcpp::Time(cloud_buf_.front()->header.stamp).seconds();

        // Advance odom buffer to within ±150 ms (live sensor has ~97 ms camera/LiDAR offset)
        while (!odom_buf_.empty() &&
               rclcpp::Time(odom_buf_.front()->header.stamp).seconds() < frame_time - 0.15)
            odom_buf_.pop();
        if (odom_buf_.empty()) return false;
        if (rclcpp::Time(odom_buf_.front()->header.stamp).seconds() > frame_time + 0.15) {
            double delta_ms = (rclcpp::Time(odom_buf_.front()->header.stamp).seconds() - frame_time) * 1000.0;
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "Odom/cloud sync gap %.0f ms (odom ahead) — dropping cloud msg. "
                "Check --clock is set on bag play.", delta_ms);
            cloud_buf_.pop();
            return false;
        }

        // Advance image buffer to within ±150 ms (live sensor has ~97 ms camera/LiDAR offset)
        while (!image_buf_.empty() &&
               rclcpp::Time(image_buf_.front()->header.stamp).seconds() < frame_time - 0.15)
            image_buf_.pop();
        if (image_buf_.empty()) return false;
        if (rclcpp::Time(image_buf_.front()->header.stamp).seconds() > frame_time + 0.15) {
            double delta_ms = (rclcpp::Time(image_buf_.front()->header.stamp).seconds() - frame_time) * 1000.0;
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "Image/cloud sync gap %.0f ms (image ahead) — dropping cloud msg. "
                "Check --clock is set on bag play.", delta_ms);
            cloud_buf_.pop();
            return false;
        }

        auto cloud_msg = cloud_buf_.front(); cloud_buf_.pop();
        auto odom_msg  = odom_buf_.front();  odom_buf_.pop();
        auto image_msg = image_buf_.front(); image_buf_.pop();

        return processFrame(cloud_msg, odom_msg, image_msg, out_frame);
    }

    bool cloudBufEmpty() const { return cloud_buf_.empty(); }

    bool getLatestTwc(Eigen::Matrix4d& T_wc)
    {
        std::lock_guard<std::mutex> lk(latest_pose_mutex_);
        if (!latest_pose_valid_) return false;
        T_wc = latest_T_wc_;
        return true;
    }

    // Publish the Gaussian splat centres as a coloured PointCloud2 (world frame).
    // xyz:          [N, 3] float32 GPU — splat positions
    // features_dc:  [N, 1, 3] float32 GPU — DC spherical-harmonic colour
    void publishGaussians(const torch::Tensor& xyz, const torch::Tensor& features_dc)
    {
        const float C0 = 0.28209479177387814f;
        auto xyz_cpu = xyz.detach().contiguous().cpu();
        auto rgb_cpu = (features_dc.detach().squeeze(1) * C0 + 0.5f)
                           .clamp(0.f, 1.f).mul(255.f)
                           .to(torch::kUInt8).contiguous().cpu();

        int N = xyz_cpu.size(0);
        const float*   xp = xyz_cpu.data_ptr<float>();
        const uint8_t* cp = rgb_cpu.data_ptr<uint8_t>();

        // XYZRGB layout: x(4) y(4) z(4) rgb-packed-as-float(4) = 16 bytes/point
        sensor_msgs::msg::PointCloud2 msg;
        msg.header.stamp    = now();
        msg.header.frame_id = "world";
        msg.height   = 1;
        msg.width    = N;
        msg.is_bigendian = false;
        msg.is_dense     = true;
        msg.point_step   = 16;
        msg.row_step     = 16 * N;

        auto& F = msg.fields;
        F.resize(4);
        for (auto& f : F) f.count = 1;
        F[0].name = "x";   F[0].offset =  0; F[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
        F[1].name = "y";   F[1].offset =  4; F[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
        F[2].name = "z";   F[2].offset =  8; F[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
        F[3].name = "rgb"; F[3].offset = 12; F[3].datatype = sensor_msgs::msg::PointField::FLOAT32;

        msg.data.resize(msg.row_step);
        uint8_t* out = msg.data.data();
        for (int i = 0; i < N; ++i, out += 16) {
            memcpy(out, xp + i * 3, 12);
            uint32_t packed = ((uint32_t)cp[i*3+0] << 16) |
                              ((uint32_t)cp[i*3+1] <<  8) |
                               (uint32_t)cp[i*3+2];
            memcpy(out + 12, &packed, 4);
        }
        pub_gaussians_->publish(msg);
    }

    // Publish a slice of the Gaussian cloud (the just-added points from extend()).
    // xyz and features_dc must already be contiguous CPU tensors, or GPU tensors
    // (we call detach().contiguous().cpu() internally).
    void publishNewPoints(const torch::Tensor& xyz, const torch::Tensor& features_dc)
    {
        if (xyz.size(0) == 0) return;
        const float C0 = 0.28209479177387814f;
        auto xyz_cpu = xyz.detach().contiguous().cpu();
        auto rgb_cpu = (features_dc.detach().squeeze(1) * C0 + 0.5f)
                           .clamp(0.f, 1.f).mul(255.f)
                           .to(torch::kUInt8).contiguous().cpu();

        int N = xyz_cpu.size(0);
        const float*   xp = xyz_cpu.data_ptr<float>();
        const uint8_t* cp = rgb_cpu.data_ptr<uint8_t>();

        sensor_msgs::msg::PointCloud2 msg;
        msg.header.stamp    = now();
        msg.header.frame_id = "world";
        msg.height = 1; msg.width = N;
        msg.is_bigendian = false; msg.is_dense = true;
        msg.point_step = 16; msg.row_step = 16 * N;

        auto& F = msg.fields; F.resize(4);
        for (auto& f : F) f.count = 1;
        F[0].name="x"; F[0].offset= 0; F[0].datatype=sensor_msgs::msg::PointField::FLOAT32;
        F[1].name="y"; F[1].offset= 4; F[1].datatype=sensor_msgs::msg::PointField::FLOAT32;
        F[2].name="z"; F[2].offset= 8; F[2].datatype=sensor_msgs::msg::PointField::FLOAT32;
        F[3].name="rgb"; F[3].offset=12; F[3].datatype=sensor_msgs::msg::PointField::FLOAT32;

        msg.data.resize(msg.row_step);
        uint8_t* out = msg.data.data();
        for (int i = 0; i < N; ++i, out += 16) {
            memcpy(out, xp + i * 3, 12);
            uint32_t packed = ((uint32_t)cp[i*3+0] << 16) |
                              ((uint32_t)cp[i*3+1] <<  8) |
                               (uint32_t)cp[i*3+2];
            memcpy(out + 12, &packed, 4);
        }
        pub_new_points_->publish(msg);
    }

    // Publish the Gaussian-rendered depth as a jet-colourmap image.
    // rendered_depth: [1, H, W] float32 (metric, metres) on GPU.
    void publishDepth(const torch::Tensor& rendered_depth)
    {
        auto depth = rendered_depth.squeeze(0).contiguous().cpu();  // [H, W]
        int H = depth.size(0), W = depth.size(1);
        cv::Mat depth_f(H, W, CV_32FC1, depth.data_ptr<float>());

        // Normalize to [0, 255] using the visible range, then apply jet colormap
        double dmin, dmax;
        cv::minMaxLoc(depth_f, &dmin, &dmax);
        cv::Mat norm8;
        if (dmax > dmin)
            depth_f.convertTo(norm8, CV_8UC1, 255.0 / (dmax - dmin),
                              -255.0 * dmin / (dmax - dmin));
        else
            norm8 = cv::Mat::zeros(H, W, CV_8UC1);

        cv::Mat colored;
        cv::applyColorMap(norm8, colored, cv::COLORMAP_JET);  // BGR8

        sensor_msgs::msg::Image msg;
        msg.header.stamp    = now();
        msg.header.frame_id = "camera";
        msg.height      = H;
        msg.width       = W;
        msg.encoding    = "bgr8";
        msg.is_bigendian = 0;
        msg.step        = W * 3;
        msg.data.assign(colored.data, colored.data + colored.total() * 3);
        pub_depth_->publish(msg);
    }

    // Publish the Gaussian-rendered image (called from the mapping thread).
    // rendered_image: [3, H, W] float32 RGB in [0,1] on GPU.
    void publishRender(const torch::Tensor& rendered_image)
    {
        auto img = rendered_image.clamp(0.f, 1.f).mul(255.f)
                       .to(torch::kUInt8).permute({1, 2, 0}).contiguous().cpu();
        int H = img.size(0), W = img.size(1);
        cv::Mat rgb(H, W, CV_8UC3, img.data_ptr<uint8_t>());
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);

        sensor_msgs::msg::Image msg;
        msg.header.stamp = now();
        msg.header.frame_id = "camera";
        msg.height      = H;
        msg.width       = W;
        msg.encoding    = "bgr8";
        msg.is_bigendian = 0;
        msg.step        = W * 3;
        msg.data.assign(bgr.data, bgr.data + bgr.total() * 3);
        pub_render_->publish(msg);
    }

    const Params&      getParams()      const { return params_; }
    const std::string& getResultPath()  const { return result_path_; }
    const std::string& getLpipsPath()   const { return lpips_path_; }
    const YAML::Node&  getConfigNode()  const { return config_node_; }

private:
    // Called from member-initialiser list after Node base is constructed.
    Params initParams_()
    {
        declare_parameter("config_path", "");
        declare_parameter("result_path", "");
        declare_parameter("lpips_path",  "");

        std::string config_path = get_parameter("config_path").as_string();
        if (config_path.empty())
            throw std::runtime_error("config_path ROS2 parameter is not set!");

        config_node_ = YAML::LoadFile(config_path);
        return Params(config_node_);
    }

    YAML::Node config_node_;
    Params params_;
    std::string result_path_, lpips_path_;

    cv::Mat undist_map_x_, undist_map_y_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr  sub_cloud_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr        sub_odom_;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_image_;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr           sub_start_;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr           sub_stop_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr      pub_input_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr      pub_render_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr      pub_depth_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_gaussians_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_new_points_;

    // Latest camera pose — updated every odometry message, used for rendering
    std::mutex          latest_pose_mutex_;
    Eigen::Matrix4d     latest_T_wc_  = Eigen::Matrix4d::Identity();
    bool                latest_pose_valid_ = false;

    std::queue<sensor_msgs::msg::PointCloud2::SharedPtr>     cloud_buf_;
    std::queue<nav_msgs::msg::Odometry::SharedPtr>           odom_buf_;
    std::queue<sensor_msgs::msg::CompressedImage::SharedPtr> image_buf_;

    // ---- Callbacks ----

    void cloudCallback(sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(m_buf);
        cloud_buf_.push(std::move(msg));
        last_cloud_time = rclcpp::Clock(RCL_STEADY_TIME).now().seconds();
    }

    void odomCallback(nav_msgs::msg::Odometry::SharedPtr msg)
    {
        {
            // Cache the latest camera pose for real-time rendering
            Eigen::Matrix4d T_wl = odomToMatrix(*msg);
            Eigen::Matrix4d T_wc = T_wl * params_.T_cl.inverse();
            std::lock_guard<std::mutex> lk(latest_pose_mutex_);
            latest_T_wc_       = T_wc;
            latest_pose_valid_ = true;
        }
        std::lock_guard<std::mutex> lk(m_buf);
        odom_buf_.push(std::move(msg));
    }

    void imageCallback(sensor_msgs::msg::CompressedImage::SharedPtr msg)
    {
        // Publish raw fisheye at full camera rate, decoupled from the mapping thread.
        // Just decompress — no undistortion, keeps the callback fast (~2 ms).
        std::vector<uint8_t> buf(msg->data.begin(), msg->data.end());
        cv::Mat raw = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (!raw.empty()) {
            sensor_msgs::msg::Image out;
            out.header.stamp    = msg->header.stamp;
            out.header.frame_id = "camera";
            out.height      = raw.rows;
            out.width       = raw.cols;
            out.encoding    = "bgr8";
            out.is_bigendian = 0;
            out.step        = raw.cols * 3;
            out.data.assign(raw.data, raw.data + raw.total() * 3);
            pub_input_->publish(out);
        }

        std::lock_guard<std::mutex> lk(m_buf);
        image_buf_.push(std::move(msg));
    }

    // ---- Frame processing ----

    bool processFrame(
        const sensor_msgs::msg::PointCloud2::SharedPtr&     cloud_msg,
        const nav_msgs::msg::Odometry::SharedPtr&           odom_msg,
        const sensor_msgs::msg::CompressedImage::SharedPtr& compressed_img,
        Frame& out_frame)
    {
        // 1. Decompress JPEG image
        std::vector<uint8_t> buf(compressed_img->data.begin(), compressed_img->data.end());
        cv::Mat raw = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (raw.empty()) {
            RCLCPP_WARN(get_logger(), "Failed to decode compressed image – skipping frame");
            return false;
        }

        // 2. FishPoly undistortion → center-crop → resize to output resolution
        cv::Mat undistorted;
        cv::remap(raw, undistorted, undist_map_x_, undist_map_y_, cv::INTER_LINEAR);

        cv::Mat cropped = undistorted(
            cv::Rect(params_.crop_x, params_.crop_y,
                     params_.crop_w_out, params_.crop_h_out));
        cv::Mat resized;
        cv::resize(cropped, resized,
                   cv::Size(params_.width, params_.height), 0, 0, cv::INTER_LINEAR);

        // 3. Camera pose: T_wc = T_wl * T_lc  (T_lc = inv(T_cl))
        Eigen::Matrix4d T_wl = odomToMatrix(*odom_msg);
        Eigen::Matrix4d T_wc = T_wl * params_.T_cl.inverse();
        Eigen::Matrix4d T_cw = T_wc.inverse();

        // 4. Parse SLAM cloud (world frame, XYZ)
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_w(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*cloud_msg, *cloud_w);

        // 5. Generate depth map
        cv::Mat depth = generateDepthMap(
            cloud_w, T_cw,
            params_.width, params_.height,
            params_.fx, params_.fy, params_.cx, params_.cy,
            params_.max_depth);

        // 6. Generate colorized XYZRGB cloud in world frame
        std_msgs::msg::Header hdr;
        hdr.stamp    = cloud_msg->header.stamp;
        hdr.frame_id = "world";

        auto colored_cloud = generateColorizedCloud(
            cloud_w, resized, T_cw,
            params_.width, params_.height,
            params_.fx, params_.fy, params_.cx, params_.cy,
            hdr);

        // 7. Pack into Frame using lightweight synthetic ROS2 messages

        // BGR8 image message (data is a contiguous clone)
        cv::Mat resized_cont = resized.clone();  // ensure row-contiguous
        auto img_msg         = std::make_shared<sensor_msgs::msg::Image>();
        img_msg->header      = hdr;
        img_msg->height      = params_.height;
        img_msg->width       = params_.width;
        img_msg->encoding    = "bgr8";
        img_msg->is_bigendian = 0;
        img_msg->step        = params_.width * 3;
        img_msg->data.assign(resized_cont.data,
                             resized_cont.data + resized_cont.total() * 3);

        // Float32 depth map message
        cv::Mat depth_cont = depth.clone();
        auto dep_msg        = std::make_shared<sensor_msgs::msg::Image>();
        dep_msg->header      = hdr;
        dep_msg->height      = params_.height;
        dep_msg->width       = params_.width;
        dep_msg->encoding    = "32FC1";
        dep_msg->is_bigendian = 0;
        dep_msg->step        = params_.width * sizeof(float);
        dep_msg->data.assign(
            reinterpret_cast<uint8_t*>(depth_cont.data),
            reinterpret_cast<uint8_t*>(depth_cont.data) + depth_cont.total() * sizeof(float));

        // PoseStamped message (T_wc)
        Eigen::Quaterniond q_wc(T_wc.block<3, 3>(0, 0));
        q_wc.normalize();
        auto pose_msg                         = std::make_shared<geometry_msgs::msg::PoseStamped>();
        pose_msg->header                      = hdr;
        pose_msg->pose.orientation.w          = q_wc.w();
        pose_msg->pose.orientation.x          = q_wc.x();
        pose_msg->pose.orientation.y          = q_wc.y();
        pose_msg->pose.orientation.z          = q_wc.z();
        pose_msg->pose.position.x             = T_wc(0, 3);
        pose_msg->pose.position.y             = T_wc(1, 3);
        pose_msg->pose.position.z             = T_wc(2, 3);

        out_frame.point_msg = colored_cloud;
        out_frame.pose_msg  = pose_msg;
        out_frame.image_msg = img_msg;
        out_frame.depth_msg = dep_msg;

        // Publish SLAM cloud for live RViz2 view
        pub_cloud_->publish(*colored_cloud);

        return true;
    }
};

// ============================================================
// Mapping thread – unchanged Gaussian splatting logic
// ============================================================
static void mapping(const YAML::Node& node,
                    const std::string& result_path,
                    const std::string& lpips_path)
{
    torch::jit::setGraphExecutorOptimize(false);

    Params prm(node);
    auto gaussians = std::make_shared<GaussianModel>(prm);
    auto dataset   = std::make_shared<Dataset>(prm);

    auto t_start = std::chrono::steady_clock::now();
    auto t_end   = t_start;
    double total_mapping_time   = 0;
    double total_adding_time    = 0;
    double total_extending_time = 0;

    int viz_keyframe_count = 0;
    int keyframe_count     = 0;
    double add_window_ms   = 0.0;   // accumulated addFrame time since last keyframe
    int align_fail_streak  = 0;     // consecutive align failures (buffer empty or sync gap)
    Frame cur_frame;
    while (!exit_flag)
    {
        // Hold off until the user presses "Start Recording" in the web dashboard
        if (!recording_started) {
            m_buf.lock();
            g_node->getAlignedFrame(cur_frame);  // drain queues to prevent overflow
            m_buf.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        // [1] data alignment
        m_buf.lock();
        bool align_flag = g_node->getAlignedFrame(cur_frame);
        m_buf.unlock();
        if (!align_flag) {
            ++align_fail_streak;
            continue;
        }
        align_fail_streak = 0;

        // [2] add every frame
        t_start = std::chrono::steady_clock::now();
        dataset->addFrame(cur_frame);
        torch::cuda::synchronize();
        t_end = std::chrono::steady_clock::now();
        {
            double this_add_ms = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count() * 1000.0;
            total_adding_time += this_add_ms / 1000.0;
            add_window_ms     += this_add_ms;
        }
        if (dataset->is_keyframe_current_) {
            std::cout << "\033[1;33m     Cur Frame " << dataset->all_frame_num_ - 1 << ",\033[0m";
        } else {
            // Non-keyframe: show current frame count on the same line without spamming the log
            std::cout << "\033[0;37m  frame " << dataset->all_frame_num_ - 1
                      << " (non-kf)\033[0m\r" << std::flush;
            continue;
        }

        double extend_ms = 0.0;
        if (!gaussians->is_init_) {
            // [3] initialize map
            gaussians->is_init_   = true;
            gaussians_initialized = true;
            std::cout << "\033[1;32m  [Init] Initializing Gaussian map from keyframe "
                      << dataset->all_frame_num_ - 1 << "\033[0m\n";
            gaussians->initialize(dataset);
            gaussians->trainingSetup();
            std::cout << "\033[1;32m  [Init] Done — "
                      << gaussians->xyz_.size(0) << " Gaussians seeded\033[0m\n";
        } else {
            // [4] extend map (skip if hard Gaussian cap is reached)
            int64_t gs_now = gaussians->xyz_.size(0);
            bool cap_hit = (prm.max_gaussians > 0 && gs_now >= (int64_t)prm.max_gaussians);
            if (cap_hit) {
                std::cout << "\033[0;33m  [GS cap: " << gs_now << "/" << prm.max_gaussians
                          << " — skipping extend until pruning frees space]\033[0m\r" << std::flush;
                extend_ms = 0.0;
            } else {
                t_start = std::chrono::steady_clock::now();
                int64_t gs_before = gs_now;
                extend(dataset, gaussians);
                torch::cuda::synchronize();
                t_end = std::chrono::steady_clock::now();
                extend_ms = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count() * 1000.0;
                total_extending_time += extend_ms / 1000.0;

                int64_t gs_after = gaussians->xyz_.size(0);
            }
        }

        // [5] optimize map
        t_start = std::chrono::steady_clock::now();
        double updated_num = optimize(dataset, gaussians, prm.max_iters);
        torch::cuda::synchronize();
        t_end = std::chrono::steady_clock::now();
        double opt_ms = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count() * 1000.0;
        total_mapping_time += opt_ms / 1000.0;
        std::cout << std::fixed << std::setprecision(0)
                  << "\033[1;36m Update " << updated_num / 10000
                  << "w GS/iter"
                  << " \033[0;36m[add=" << add_window_ms << "ms ext=" << extend_ms << "ms opt=" << opt_ms << "ms]\033[0m" << std::endl;
        add_window_ms = 0.0;

        // [5b] prune low-opacity Gaussians
        ++keyframe_count;
        if (prm.prune_every_n_keyframes > 0 && keyframe_count % prm.prune_every_n_keyframes == 0) {
            torch::NoGradGuard no_grad;
            auto opacities = torch::sigmoid(gaussians->opacity_).squeeze(1);
            auto keep_mask = opacities >= static_cast<float>(prm.prune_opacity_thresh);
            int64_t before = gaussians->xyz_.size(0);
            gaussians->pruneGaussians(keep_mask);
            int64_t after = gaussians->xyz_.size(0);
            std::cout << "\033[1;35m  [Pruned " << (before - after) << " GS, "
                      << after << " remaining]\033[0m\n";
        }

        // [6] publish rendered view + Gaussian splat cloud (rate-limited by viz_every_n_keyframes)
        if (prm.viz_every_n_keyframes > 0 && ++viz_keyframe_count % prm.viz_every_n_keyframes == 0) {
            torch::NoGradGuard no_grad;
            torch::Tensor bg = gaussians->white_background_
                ? torch::ones ({3}, torch::kFloat32).cuda()
                : torch::zeros({3}, torch::kFloat32).cuda();

            // Render from the latest odometry pose so the view stays in sync
            // with the input image rather than lagging N keyframes behind.
            Eigen::Matrix4d latest_T_wc;
            auto render_cam = g_node->getLatestTwc(latest_T_wc)
                ? [&]() {
                    auto cam = std::make_shared<Camera>();
                    cam->setIntrinsic(prm.width, prm.height,
                                      prm.fx, prm.fy, prm.cx, prm.cy);
                    cam->setPose(latest_T_wc.block<3,3>(0,0),
                                 latest_T_wc.block<3,1>(0,3));
                    return cam;
                  }()
                : dataset->train_cameras_.back();  // fallback before first odom

            auto render_pkg = render(render_cam, gaussians, bg, gaussians->apply_exposure_);
            torch::cuda::synchronize();
            g_node->publishRender(std::get<0>(render_pkg));
            g_node->publishDepth(std::get<1>(render_pkg));
        }
    }

    // [6] statistics + save
    std::cout << "\n     🎉 Runtime Statistics 🎉\n";
    std::cout << std::fixed << std::setprecision(2)
              << "\n        [Total Mapping Time] "   << total_mapping_time   << "s\n"
              << "         1) Forward "             << gaussians->t_forward_ << "s\n"
              << "         2) Backward "            << gaussians->t_backward_ << "s\n"
              << "         3) Step "               << gaussians->t_step_     << "s\n"
              << "         4) CPU2GPU "             << gaussians->t_tocuda_   << "s\n"
              << "        [Total Adding Time] "     << total_adding_time      << "s\n"
              << "        [Total Extending Time] "  << total_extending_time   << "s\n";

    torch::NoGradGuard no_grad;
    try {
        evaluateVisualQuality(dataset, gaussians, result_path, lpips_path);
    } catch (const std::exception& e) {
        std::cerr << "[WARN] evaluateVisualQuality failed: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "[WARN] evaluateVisualQuality failed (unknown exception)\n";
    }
    std::cout << "\n\n😋 Gaussian-LIC Done!\n\n\n";
    exit_flag = true;
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv)
{
    std::cout << "\n\n😋 Gaussian-LIC Ready!\n\n\n";
    rclcpp::init(argc, argv);

    try {
        g_node = std::make_shared<GaussianLICNode>();
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("gaussianlic"), "%s", e.what());
        rclcpp::shutdown();
        return 1;
    }

    const YAML::Node  config_node = g_node->getConfigNode();
    const std::string result_path = g_node->getResultPath();
    const std::string lpips_path  = g_node->getLpipsPath();

    // Launch the Gaussian splatting thread
    std::thread mapping_thread(mapping, config_node, result_path, lpips_path);

    // Monitor thread: detect end-of-bag (no data for 5 s after init)
    std::thread monitor_thread([]() {
        rclcpp::Clock steady_clock(RCL_STEADY_TIME);
        while (!exit_flag) {
            double now = steady_clock.now().seconds();
            if (gaussians_initialized && (now - last_cloud_time > 5.0)) {
                std::lock_guard<std::mutex> lk(m_buf);
                if (g_node && g_node->cloudBufEmpty())
                    exit_flag = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });

    while (rclcpp::ok() && !exit_flag)
        rclcpp::spin_some(g_node);

    exit_flag = true;  // propagate SIGINT to mapping/monitor threads
    mapping_thread.join();
    monitor_thread.join();
    g_node.reset();
    rclcpp::shutdown();
    return 0;
}
