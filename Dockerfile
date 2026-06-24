# ── Stage 1: build OpenCV with CUDA ──────────────────────────────────────────
FROM nvcr.io/nvidia/tensorrt:26.04-py3 AS opencv_builder
# nvcr.io/nvidia/tensorrt:26.04-py3 = Ubuntu 24.04 + CUDA 12.8 + TensorRT 10.16

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential git \
    libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev \
    libjpeg-dev libpng-dev libtiff-dev libopenjp2-7-dev \
    libwebp-dev libtbb-dev && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv.git /opencv && \
    git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv_contrib.git /opencv_contrib

RUN cmake -B /opencv/build -S /opencv \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
    -DWITH_CUDA=ON -DCUDA_ARCH_BIN="8.6" \
    -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON \
    -DBUILD_opencv_python3=OFF \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX=/opt/opencv && \
    cmake --build /opencv/build -j$(nproc) && \
    cmake --install /opencv/build

# ── Stage 2: final image ──────────────────────────────────────────────────────
FROM nvcr.io/nvidia/tensorrt:26.04-py3

# ROS2 Jazzy (Ubuntu 24.04 Noble)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg lsb-release openssh-client && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ros-base \
    ros-jazzy-sensor-msgs ros-jazzy-nav-msgs ros-jazzy-geometry-msgs \
    ros-jazzy-cv-bridge ros-jazzy-image-transport \
    ros-jazzy-pcl-conversions ros-jazzy-tf2 ros-jazzy-tf2-ros \
    ros-jazzy-tf2-eigen ros-jazzy-ament-index-cpp \
    ros-jazzy-rmw-fastrtps-cpp \
    python3-colcon-common-extensions \
    libeigen3-dev libpcl-dev libyaml-cpp-dev libffi-dev libglm-dev && \
    rm -rf /var/lib/apt/lists/*

# OpenCV from build stage
COPY --from=opencv_builder /opt/opencv /opt/opencv
ENV OpenCV_DIR=/opt/opencv/lib/cmake/opencv4

# LibTorch — retry up to 5 times in case of network drops
RUN curl -L --retry 5 --retry-delay 15 --retry-all-errors \
    https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip \
    -o /tmp/libtorch.zip && \
    unzip -q /tmp/libtorch.zip -d /opt && \
    rm /tmp/libtorch.zip
ENV Torch_DIR=/opt/libtorch/share/cmake/Torch
ENV TENSORRT_ROOT=/opt/tensorrt
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:/opt/opencv/lib:${LD_LIBRARY_PATH}

# Python deps for ONNX export and model download (torch already in base image)
RUN pip3 install --no-cache-dir onnx onnxruntime onnxscript gdown && \
    pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Build gaussian_lic
WORKDIR /ws
ARG REPO_SHA=unknown
RUN git clone https://github.com/3DVisionFS26/odin_gaussian_lic.git src/gaussian_lic
RUN gdown "11dujPviL4pKLEXytXK0mEmPBNQDqgEak" \
    -O src/gaussian_lic/ckpt/Large_300.pth
RUN . /opt/ros/jazzy/setup.sh && \
    colcon build --packages-select gaussian_lic \
    --event-handlers console_cohesion+ \
    --cmake-args \
      -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=/opt/opencv/lib/cmake/opencv4 \
      -DTorch_DIR=/opt/libtorch/share/cmake/Torch \
      -DTENSORRT_ROOT=/opt/tensorrt \
      -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCMAKE_VERBOSE_MAKEFILE=ON \
    || { find /ws/log -name "*.log" | xargs grep -l "error:" | xargs cat 2>/dev/null; exit 1; } && \
    rm -rf build log

# Build TensorRT engines.
# BuildKit (needed for --mount=type=ssh) does not forward GPU devices to build
# containers, so trtexec cannot run inside `docker build`. build.sh handles this
# by running a second phase: `docker run` (which does get the nvidia runtime) to
# call build_trt.sh, then `docker commit` to bake the engines into the final image.
# SKIP_TRT=1 lets the Dockerfile finish without GPU so build.sh can do phase 2.
ARG SKIP_TRT=0
RUN cd /ws/install/gaussian_lic/share/gaussian_lic/ckpt && \
    env -u LD_LIBRARY_PATH python3 export_onnx_512_640.py && \
    env -u LD_LIBRARY_PATH python3 export_onnx_480_640.py && \
    if [ "$SKIP_TRT" = "0" ]; then \
        TENSORRT_ROOT=/opt/tensorrt bash build_trt.sh; \
    fi

# Runtime entrypoint
RUN printf '#!/bin/bash\n. /opt/ros/jazzy/setup.sh\n. /ws/install/setup.sh\nexec "$@"\n' \
    > /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "gaussian_lic", "odin1.launch.py"]
