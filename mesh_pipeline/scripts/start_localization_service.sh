#!/bin/bash
# Start a persistent Docker container for localization service
# This container will stay running and accept multiple localization requests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONTAINER_NAME="lamar-localization-service"
DOCKER_IMAGE="lamar:lamar"

echo "=========================================="
echo "Starting LaMAR Localization Service"
echo "=========================================="

# Check if container already exists and is running
if docker ps -q -f name="${CONTAINER_NAME}" | grep -q .; then
    echo "✓ Container '${CONTAINER_NAME}' is already running"
    exit 0
fi

# Check if container exists but is stopped
if docker ps -aq -f name="${CONTAINER_NAME}" | grep -q .; then
    echo "→ Removing stopped container '${CONTAINER_NAME}'"
    docker rm "${CONTAINER_NAME}"
fi

# Get required paths
LAMAR_REPO="${PROJECT_ROOT}/mesh_pipeline/third_party/lamar-benchmark"
NAVVIS_DATA="${PROJECT_ROOT}/mesh_pipeline/data"
CACHE_DIR="${HOME}/.cache/torch"

echo ""
echo "Configuration:"
echo "  Container:  ${CONTAINER_NAME}"
echo "  Image:      ${DOCKER_IMAGE}"
echo "  LaMAR repo: ${LAMAR_REPO}"
echo "  Data dir:   ${NAVVIS_DATA}"
echo "  Cache dir:  ${CACHE_DIR}"
echo ""

# Check for GPU support
GPU_FLAG=""
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected, enabling GPU support"
    GPU_FLAG="--gpus all"
else
    echo "! No NVIDIA GPU detected, running in CPU mode"
fi

echo ""
echo "Starting container in background..."

# Start container in detached mode with sleep infinity to keep it alive
docker run -d \
    --name "${CONTAINER_NAME}" \
    --init \
    --shm-size=16g \
    ${GPU_FLAG} \
    -v "${LAMAR_REPO}:${LAMAR_REPO}" \
    -v "${NAVVIS_DATA}:${NAVVIS_DATA}" \
    -v "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    -v "${CACHE_DIR}:/root/.cache/torch" \
    -w "${LAMAR_REPO}" \
    "${DOCKER_IMAGE}" \
    sleep infinity

echo ""
echo "=========================================="
echo "✓ Localization service started!"
echo "=========================================="
echo ""
echo "Container '${CONTAINER_NAME}' is now running in the background."
echo "It will handle multiple localization requests without restart overhead."
echo ""
echo "To stop the service, run:"
echo "  ./stop_localization_service.sh"
echo ""
