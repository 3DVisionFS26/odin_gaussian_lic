cmake_minimum_required(VERSION 3.8)
project(gaussian_lic)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse4.2 -Wall -Wno-sign-compare -Wno-unused -Wno-comment -pthread")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wno-sign-compare -Wno-unused -Wno-comment -pthread")
endif()

# CUDA – set CMAKE_CUDA_ARCHITECTURES via -DCMAKE_CUDA_ARCHITECTURES=<sm> at build time.
# Defaults: 86 (RTX 3090 / x86), 87 (Jetson Orin)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CMAKE_CUDA_ARCHITECTURES 87)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 86)
  endif()
endif()
# Resolve CUDA root: prefer CUDA_HOME env var, fall back to /usr/local/cuda symlink
if(DEFINED ENV{CUDA_HOME})
  set(CUDA_ROOT "$ENV{CUDA_HOME}")
elseif(EXISTS "/usr/local/cuda")
  set(CUDA_ROOT "/usr/local/cuda")
else()
  message(FATAL_ERROR "Cannot find CUDA installation. Set CUDA_HOME or install CUDA.")
endif()
set(CMAKE_CUDA_COMPILER "${CUDA_ROOT}/bin/nvcc")
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# ---- ROS2 dependencies ----
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(pcl_conversions REQUIRED)
# Only need PCL common headers; avoid pulling in OpenCV-dependent PCL modules
find_package(PCL REQUIRED COMPONENTS common)
find_package(tf2 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(ament_index_cpp REQUIRED)

# ---- External dependencies ----
# Each library is resolved in order:
#   1. CMake cache / -D flag (highest priority)
#   2. Environment variable
#   3. Glob over candidate install prefixes
set(_sw_prefixes "$ENV{HOME}/Software" "/mnt/Volume/Ubuntu/3d_vision/Software")

if(NOT DEFINED OpenCV_DIR OR OpenCV_DIR STREQUAL "")
  if(DEFINED ENV{OpenCV_DIR} AND NOT "$ENV{OpenCV_DIR}" STREQUAL "")
    set(OpenCV_DIR "$ENV{OpenCV_DIR}")
  else()
    foreach(_prefix ${_sw_prefixes})
      file(GLOB _cv_builds "${_prefix}/opencv/opencv-*/build")
      if(_cv_builds)
        list(SORT _cv_builds)
        list(GET _cv_builds -1 OpenCV_DIR)
        break()
      endif()
    endforeach()
  endif()
endif()
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
find_package(OpenCV 4 REQUIRED)

if(NOT DEFINED Torch_DIR OR Torch_DIR STREQUAL "")
  if(DEFINED ENV{Torch_DIR} AND NOT "$ENV{Torch_DIR}" STREQUAL "")
    set(Torch_DIR "$ENV{Torch_DIR}")
  else()
    foreach(_prefix ${_sw_prefixes})
      if(EXISTS "${_prefix}/libtorch/share/cmake/Torch/TorchConfig.cmake")
        set(Torch_DIR "${_prefix}/libtorch/share/cmake/Torch")
        break()
      endif()
    endforeach()
  endif()
endif()
message(STATUS "Torch_DIR: ${Torch_DIR}")
find_package(Torch REQUIRED)

# TensorRT – set TENSORRT_ROOT to wherever you extracted the TensorRT tarball.
# TRT 8.x:  ~/Software/TensorRT-8.6.1.6
# TRT 10.x: ~/Software/TensorRT-10.x.x.x
if(DEFINED ENV{TENSORRT_ROOT} AND NOT "$ENV{TENSORRT_ROOT}" STREQUAL "")
  set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}")
else()
  foreach(_prefix ${_sw_prefixes})
    file(GLOB _trt_dirs "${_prefix}/TensorRT-*")
    if(_trt_dirs)
      list(SORT _trt_dirs)
      list(GET _trt_dirs -1 TENSORRT_ROOT)
      break()
    endif()
  endforeach()
  if(NOT TENSORRT_ROOT)
    message(FATAL_ERROR "Cannot find TensorRT. Set the TENSORRT_ROOT environment variable.")
  endif()
endif()
message(STATUS "TensorRT root: ${TENSORRT_ROOT}")
set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
set(TENSORRT_LIB_DIR     "${TENSORRT_ROOT}/lib")

set(CUDA_TOOLKIT_ROOT_DIR "${CUDA_ROOT}")
set(CUDA_INCLUDE_DIRS     "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDA_LIB_DIR          "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
find_package(CUDA REQUIRED)

find_library(FFI_LIB ffi)

find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp>=0.5)

set(Python3_EXECUTABLE "/usr/bin/python3")
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(Eigen3 REQUIRED)

# ---- Linker search paths for non-system installs ----
link_directories(
  "${OpenCV_DIR}/lib"
  "${TENSORRT_LIB_DIR}"
)

# ---- Build target ----
add_executable(gs_mapping
  ${PROJECT_SOURCE_DIR}/src/mapping.cpp
  ${PROJECT_SOURCE_DIR}/src/gaussian.cpp
  ${PROJECT_SOURCE_DIR}/src/tinyply.cpp
  ${PROJECT_SOURCE_DIR}/src/depth_completer.cpp
  ${PROJECT_SOURCE_DIR}/src/simple-knn/spatial.cu
  ${PROJECT_SOURCE_DIR}/src/simple-knn/simple_knn.cu
  ${PROJECT_SOURCE_DIR}/src/fused-ssim/ssim.cu
  ${PROJECT_SOURCE_DIR}/src/rasterizer/renderer.cpp
  ${PROJECT_SOURCE_DIR}/src/rasterizer/rasterizer.cpp
  ${PROJECT_SOURCE_DIR}/src/rasterizer/rasterize_points.cu
  ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/adam.cu
  ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/forward.cu
  ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/backward.cu
  ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/rasterizer_impl.cu
)

set_target_properties(gs_mapping PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(gs_mapping PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${OpenCV_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}   # PCL common headers only
)

ament_target_dependencies(gs_mapping
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  cv_bridge
  image_transport
  pcl_conversions
  tf2
  tf2_eigen
  tf2_ros
  ament_index_cpp
)

find_library(TRT_INFER   nvinfer        HINTS "${TENSORRT_LIB_DIR}" REQUIRED)
find_library(TRT_ONNX    nvonnxparser   HINTS "${TENSORRT_LIB_DIR}" REQUIRED)
find_library(TRT_PLUGIN  nvinfer_plugin HINTS "${TENSORRT_LIB_DIR}" REQUIRED)
# nvparsers (caffe/uff parsers) was removed in TensorRT 10 – link only if present
find_library(TRT_PARSERS nvparsers      HINTS "${TENSORRT_LIB_DIR}")

target_link_libraries(gs_mapping
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${Python3_LIBRARIES}
  ${FFI_LIB}
  ${CUDA_LIB_DIR}/libcudart.so
  ${TRT_INFER}
  ${TRT_ONNX}
  ${TRT_PLUGIN}
  $<$<BOOL:${TRT_PARSERS}>:${TRT_PARSERS}>
)

# ---- Install ----
install(TARGETS gs_mapping
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY ckpt/
  DESTINATION share/${PROJECT_NAME}/ckpt
  OPTIONAL  # ckpt/ may not exist yet before engines are built
)

install(DIRECTORY src/lpips
  DESTINATION share/${PROJECT_NAME}/src
)

# Web monitor – Python node installed as a standalone executable
install(PROGRAMS scripts/web_monitor_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME web_monitor
)

ament_package()
