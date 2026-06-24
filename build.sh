#!/usr/bin/env bash
# Build the gaussian_lic Docker image in two phases:
#   Phase 1 – docker build (HTTPS clone, colcon, ONNX export)
#   Phase 2 – docker run with GPU (trtexec), then docker commit
set -euo pipefail

IMAGE=${1:-gaussian_lic}
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_SHA=$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo "unknown")

echo "==> Phase 1: building image (HTTPS clone + colcon + ONNX export)"
DOCKER_BUILDKIT=1 docker build \
    --progress=plain \
    --build-arg REPO_SHA="${REPO_SHA}" \
    --build-arg SKIP_TRT=1 \
    -t "${IMAGE}:ready" \
    "$(dirname "$0")"

echo "==> Phase 2: building TRT engines inside container (requires GPU)"
CONTAINER=$(docker create "${IMAGE}:ready" \
    bash -c "cd /ws/install/gaussian_lic/share/gaussian_lic/ckpt && \
             TENSORRT_ROOT=/opt/tensorrt bash build_trt.sh")

docker start -a "${CONTAINER}"

echo "==> Committing TRT engines into final image ${IMAGE}:latest"
docker commit "${CONTAINER}" "${IMAGE}:latest"
docker rm "${CONTAINER}"
docker rmi "${IMAGE}:ready"

echo "==> Done. Run with:"
echo "    docker run --rm ${IMAGE}:latest"
