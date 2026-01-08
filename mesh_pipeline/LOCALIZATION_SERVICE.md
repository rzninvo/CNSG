# LaMAR Localization Service - Persistent Mode

## Overview

For applications that need to process multiple localization requests, you can run the LaMAR Docker container as a persistent service. This eliminates the ~4-5 second Docker startup/shutdown overhead on every request.

## Performance Comparison

**Without persistent service** (docker run each time):
- Startup overhead: ~2-3s
- Localization: ~7-9s  
- Shutdown overhead: ~1-2s
- **Total: ~13-14s per request**

**With persistent service** (docker exec on running container):
- Startup overhead: **0s** ✓
- Localization: ~7-9s
- Shutdown overhead: **0s** ✓
- **Total: ~7-9s per request** 🚀

**Speed improvement: ~40-50% faster on multiple requests!**

## Usage

### 1. Start the Service (Once)

```bash
cd mesh_pipeline/scripts
./start_localization_service.sh
```

This starts a Docker container named `lamar-localization-service` that runs in the background.

### 2. Run Localization Requests (Multiple Times)

The localization scripts automatically detect and use the persistent container:

```bash
# Single request
./run_localization_pipeline.sh --query-image /path/to/image1.jpg --num-retrieval 3 --fast

# Another request (no Docker restart!)
./run_localization_pipeline.sh --query-image /path/to/image2.jpg --num-retrieval 3 --fast

# And another...
./run_localization_pipeline.sh --query-image /path/to/image3.jpg --num-retrieval 3 --fast
```

Each subsequent request will be **~40-50% faster** because it reuses the running container.

### 3. Stop the Service (When Done)

```bash
./stop_localization_service.sh
```

## Python API Example

If you're integrating this into a Python application:

```python
import subprocess
from pathlib import Path

# Start service once at application startup
subprocess.run(["./scripts/start_localization_service.sh"], check=True)

# Process multiple images
for image_path in image_list:
    subprocess.run([
        "./scripts/run_localization_pipeline.sh",
        "--query-image", str(image_path),
        "--num-retrieval", "3",
        "--fast"
    ], check=True)

# Stop service at application shutdown
subprocess.run(["./scripts/stop_localization_service.sh"], check=True)
```

## How It Works

1. **start_localization_service.sh**: Creates a Docker container with all mounts and keeps it alive with `sleep infinity`
2. **run_localization.py**: Detects the running container and uses `docker exec` instead of `docker run`
3. **stop_localization_service.sh**: Stops and removes the container

## Troubleshooting

### Check if service is running
```bash
docker ps | grep lamar-localization-service
```

### View service logs
```bash
docker logs lamar-localization-service
```

### Restart the service
```bash
./stop_localization_service.sh
./start_localization_service.sh
```

### Service not being used?
The scripts will fall back to `docker run` mode if the persistent service isn't running. You'll see a message:
```
ℹ  Tip: For faster execution on multiple requests, start the persistent service:
  ./scripts/start_localization_service.sh
```

## Notes

- The service container uses the same GPU, mounts, and cache as the normal mode
- All existing flags (`--num-retrieval`, `--fast`, `--no-gpu`, etc.) work the same way
- The service automatically cleans up when stopped
- If the container crashes, simply restart it with `start_localization_service.sh`
