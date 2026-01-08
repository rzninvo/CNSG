#!/bin/bash
# Stop the persistent localization service container

set -e

CONTAINER_NAME="lamar-localization-service"

echo "=========================================="
echo "Stopping LaMAR Localization Service"
echo "=========================================="

# Check if container exists
if ! docker ps -aq -f name="${CONTAINER_NAME}" | grep -q .; then
    echo "! Container '${CONTAINER_NAME}' not found"
    exit 0
fi

# Check if container is running
if docker ps -q -f name="${CONTAINER_NAME}" | grep -q .; then
    echo "→ Stopping container..."
    docker stop "${CONTAINER_NAME}"
fi

echo "→ Removing container..."
docker rm "${CONTAINER_NAME}"

echo ""
echo "=========================================="
echo "✓ Localization service stopped!"
echo "=========================================="
