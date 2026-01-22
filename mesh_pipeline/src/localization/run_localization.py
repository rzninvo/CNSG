#!/usr/bin/env python3
"""
Run single image localization using LaMAR Docker container.

This script automatically loads NavVis data paths from the project config
and runs the LaMAR localization pipeline in Docker.

Usage:
    python run_localization.py --query_image /path/to/image.jpg
    python run_localization.py --query_image /path/to/image.jpg --output_dir ./my_outputs
    python run_localization.py --query_image /path/to/image.jpg --fast
    python run_localization.py --query_image /path/to/image.jpg --no-gpu
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import os

# Add parent directory to path to import config utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config_utils import load_paths


def load_localization_config(config_path: Path = None) -> dict:
    """Load localization configuration from JSON file."""
    if config_path is None:
        # Default config path
        config_path = Path(__file__).resolve().parents[2] / "config" / "localization_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Localization config not found: {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)


def check_docker_image(image_name: str = "lamar:lamar") -> bool:
    """Check if the required Docker image exists."""
    try:
        result = subprocess.run(
            ["docker", "images", "-q", image_name],
            capture_output=True,
            text=True,
            check=True
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False


def check_persistent_container(container_name: str = "lamar-localization-service") -> bool:
    """Check if persistent localization service container is running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name={container_name}"],
            capture_output=True,
            text=True,
            check=True
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False


def build_docker_command(
    repo_path: Path,
    navvis_data_path: Path,
    query_image_path: Path,
    map_session_path: Path,
    output_dir: Path,
    config: dict,
    use_gpu: bool = True,
    docker_image: str = "lamar:lamar",
    use_persistent: bool = False,
    container_name: str = "lamar-localization-service"
) -> list:
    """Build the Docker command for localization.

    Args:
        use_persistent: If True, use docker exec on persistent container instead of docker run
        container_name: Name of the persistent container
    """

    # Get the directory containing the query image for volume mounting
    query_image_dir = query_image_path.parent.absolute()

    # Set up cache directory for persistent model storage
    cache_dir = Path.home() / ".cache" / "torch"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Convert config to JSON string
    config_json = json.dumps(config)

    # Build Python command arguments
    python_args = [
        "python", "localize_single_image.py",
        "--map_path", str(map_session_path),
        "--query_image", str(query_image_path.absolute()),
        "--output_dir", "./outputs",
        "--config", config_json,
    ]

    # Use persistent container if available (much faster - no startup overhead)
    if use_persistent:
        docker_cmd = ["docker", "exec", container_name] + python_args
        return docker_cmd

    # Otherwise, use traditional docker run (slower but works without persistent service)
    docker_cmd = [
        "docker", "run",
        "-it", "--rm", "--init",
        "--shm-size=16g",
    ]

    # Add GPU support if requested and nvidia-docker is available
    if use_gpu:
        try:
            subprocess.run(["docker", "info"], capture_output=True, check=True)
            # Try to detect if nvidia runtime is available
            docker_cmd.extend(["--gpus", "all"])
        except:
            pass  # GPU not available, continue without it

    docker_cmd.extend([
        # Mount the LaMAR repository
        "-v", f"{repo_path}:{repo_path}",
        # Mount the NavVis data directory
        "-v", f"{navvis_data_path}:{navvis_data_path}",
        # Mount the query image directory
        "-v", f"{query_image_dir}:{query_image_dir}",
        # Mount cache directory for persistent model storage
        "-v", f"{cache_dir}:/root/.cache/torch",
        # Set working directory
        "-w", str(repo_path),
        # Docker image
        docker_image,
    ] + python_args)

    return docker_cmd


def run_localization(query_image: Path, output_dir: Path = None, use_gpu: bool = True, fast_mode: bool = False):
    """Run the localization pipeline.

    Args:
        query_image: Path to the query image
        output_dir: Output directory for results
        use_gpu: Whether to use GPU acceleration if available (default: True)
        fast_mode: Use faster settings (lower resolution, fewer keypoints)
    """

    # Load paths from config
    print("Loading configuration...")
    try:
        paths = load_paths()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Make sure config/paths.yml is properly configured.")
        sys.exit(1)

    # Load localization config
    try:
        all_configs = load_localization_config()
    except Exception as e:
        print(f"Error loading localization config: {e}")
        sys.exit(1)

    # Select config based on fast_mode
    config_name = "fast" if fast_mode else "default"
    if config_name not in all_configs:
        print(f"Error: Config '{config_name}' not found in localization_config.json")
        sys.exit(1)

    config = all_configs[config_name]
    print(f"Using '{config_name}' configuration")

    # Validate query image exists
    if not query_image.exists():
        print(f"Error: Query image not found: {query_image}")
        sys.exit(1)

    # Validate session directory exists
    if not paths.session_dir.exists():
        print(f"Error: NavVis session directory not found: {paths.session_dir}")
        print(f"Expected session: {paths.session_dir}")
        sys.exit(1)

    # Set up paths
    repo_root = paths.root
    lamar_repo = repo_root / "third_party" / "lamar-benchmark"

    if not lamar_repo.exists():
        print(f"Error: LaMAR repository not found: {lamar_repo}")
        print("Make sure the lamar-benchmark submodule is initialized.")
        sys.exit(1)

    # Set default output directory if not provided
    if output_dir is None:
        output_dir = paths.outputs_root
    output_dir.mkdir(exist_ok=True, parents=True)

    # Check for persistent container service (much faster!)
    container_name = "lamar-localization-service"
    # Docker image name
    docker_image = "lamar:lamar"

    use_persistent = check_persistent_container(container_name)

    if use_persistent:
        print(f"\n✓ Using persistent container service '{container_name}'")
        print(f"  (No startup overhead - instant execution!)")
    else:
        # Check if Docker image exists
        print(f"\nChecking for Docker image '{docker_image}'...")
        if not check_docker_image(docker_image):
            print(f"Warning: Docker image '{docker_image}' not found.")
            print("\nTo build the image, run:")
            print(f"  cd {lamar_repo}")
            print(f"  docker build --target lamar -t {docker_image} -f Dockerfile ./")

            response = input("\nWould you like to continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)

        print("\nℹ  Tip: For faster execution on multiple requests, start the persistent service:")
        print("  ./scripts/start_localization_service.sh")

    print(f"\n{'='*60}")
    print("Running LaMAR Localization")
    print(f"{'='*60}")
    print(f"Query image:    {query_image}")
    print(f"NavVis session: {paths.session_dir}")
    print(f"Output dir:     {output_dir}")
    print(f"Config:         {config_name}")
    print(f"GPU enabled:    {use_gpu}")
    print(f"Cache dir:      {Path.home() / '.cache' / 'torch'}")
    print(f"{'='*60}\n")

    docker_cmd = build_docker_command(
        repo_path=lamar_repo,
        navvis_data_path=paths.data_root,
        query_image_path=query_image,
        map_session_path=paths.session_dir,
        output_dir=output_dir,
        config=config,
        use_gpu=use_gpu,
        docker_image=docker_image,
        use_persistent=use_persistent,
        container_name=container_name
    )

    # Print the command for debugging (truncate config for readability)
    print("Docker command:")
    cmd_display = docker_cmd.copy()
    for i, arg in enumerate(cmd_display):
        if arg == "--config" and i + 1 < len(cmd_display):
            cmd_display[i + 1] = "'<config json>'"
    print(" ".join(cmd_display))
    print()

    # Run the Docker command
    try:
        subprocess.run(docker_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Docker command failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nLocalization interrupted by user.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Localization complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Localize a single image using LaMAR and NavVis data"
    )
    parser.add_argument(
        "--query_image",
        type=Path,
        required=True,
        help="Path to the query image to localize"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for results (default: outputs/localization)"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster settings (lower resolution, fewer keypoints)"
    )

    args = parser.parse_args()

    run_localization(
        args.query_image,
        args.output_dir,
        use_gpu=not args.no_gpu,
        fast_mode=args.fast
    )


if __name__ == "__main__":
    main()
