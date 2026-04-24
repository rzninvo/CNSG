# CNSG Mesh Pipeline

End-to-end mesh and semantic processing for NavVis LaMAR sessions: Poisson meshing, depth rendering, semantic segmentation, label fusion/cleanup, and Habitat export. NavVis data comes from the LaMAR benchmark; Poisson meshing is adapted from LaMAR (see `third_party`). OpenYOLO3D is vendored for future integration.

## Environment
Use conda (Python 3.10.6 recommended):
```bash
conda create -n CNSG-meshing python=3.10.6
conda activate CNSG-meshing
pip install -r requirements.txt
```
CUDA is recommended for faster segmentation; install the appropriate PyTorch build if needed.

## Data
Download NavVis LaMAR session and processed add-ons:
```bash
./scripts/download_data.sh
```
This fetches `navvis_2022-02-06_12.55.11` into `data/` and auto-extracts provided `depth_maps` and `semantic_masks` from the Google Drive bundle.

## Pipelines
Run commands from the repo root.

### Mesh generation
**NOTE**: If you already ran `./scripts/download_data.sh`, a mesh and voxelized mesh are provided; **you can skip this part unless you want to tweak parameters and regenerate**.
```bash
./scripts/generate_mesh.sh
```

The script prompts for:
- Poisson mesh only
- Poisson + voxelized mesh
- Full pipeline (Poisson + voxelize + depth maps)

Underlying components (`src/mesh_generation`):
- `run_poisson_meshing.py` — build Poisson mesh from point cloud.
- `habitat_mesh_reduce.py` — voxel-based simplification for Habitat.
- `generate_depth.py` — raycast depth maps using trajectories.

### 3D segmentation

The legacy chain (`run_segmentation_pipeline.sh` → Mask2Former + DBSCAN + trimesh transfer + hm3d export) was replaced in Phase 3 — see [docs/report/01_architecture-lean-migration/phase3-status.md](../docs/report/01_architecture-lean-migration/phase3-status.md). Current entry point lives at the repo root:

```bash
bash ../scripts/build_hge_semantics.sh
```

What runs under the hood (in the `cnsg-seg` conda env, py3.12 + torch 2.11 cu128 + SAM 3 + flash-attn-3):

- `cnsg.segmentation.structural_ade20k` — Mask2Former-ADE20K for structural classes (wall, floor, ceiling, window, door, column, clutter).
- `cnsg.segmentation.sam3_per_frame` — SAM 3 open-vocab text-prompt instances for foreground objects.
- `cnsg.segmentation.lift_2d_to_3d` — per-frame project + depth-test + union-find → per-vertex instance + class.
- `cnsg.segmentation.hierarchy` — floor / room partitioning (HOV-SG-style watershed on BEV).
- `cnsg.segmentation.export_habitat` — HM3D-compatible `.semantic.glb` + `.semantic.txt` + `scene_dataset_config.json`.

The script writes into `data/maps/hge/` (repo root). `scripts/install.sh` Step 5 symlinks those outputs into `habitat-sim/data/scene_datasets/` so the viewer invocation in the top-level README works without manual file copies.

### Image localization

Localize phone/camera images against the NavVis map using the LaMAR framework. The pipeline uses visual features (SuperPoint + SuperGlue) to match query images and estimate camera poses in NavVis coordinates.

#### Prerequisites

1. **Build LaMAR Docker image** (first time only):

   ```bash
   cd third_party/lamar-benchmark
   docker build --target lamar -t lamar:lamar -f Dockerfile ./
   ```

#### Quick start

```bash
python src/localization/run_localization.py \
    --query_image /path/to/your/photo.jpg
```

The script automatically:

- Loads NavVis session from `config/paths.yml`
- Validates paths and Docker image
- Runs localization in Docker with proper volume mounts
- Outputs camera pose to `outputs/localization/` (or custom `--output_dir`)

#### Output format

Camera pose saved as `poses.txt`:

```
# timestamp, sensor_id, qw, qx, qy, qz, tx, ty, tz
1000000, camera, 0.707, 0.0, 0.707, 0.0, 10.5, 2.3, -5.1
```

Where:

- `qw, qx, qy, qz` — Rotation quaternion (camera to world)
- `tx, ty, tz` — Camera position in NavVis world coordinates (meters)

#### Performance notes

- **First run**: 2-4 hours to process map (features extracted and cached)
- **Subsequent runs**: 1-2 minutes per query image (reuses cached features)

Underlying components (`src/localization`):

- `run_localization.py` — automated Docker runner with config integration.

See `third_party/lamar-benchmark/LAMAR_USAGE_GUIDE.md` for advanced usage and troubleshooting.

## Configuration
`config/paths.yml` centralizes dataset and asset paths (session name, point cloud, images, depth, meshes, exports) and is loaded via `src/utils/config_utils.py`. Adjust entries if your data lives elsewhere.

## Third-Party
- LaMAR benchmark assets and adapted Poisson tools in `third_party/`.
- OpenYOLO3D (not yet used) is vendored under `third_party/OpenYOLO3D`; integration planned.

## Contact
Questions: rzendehdel@ethz.ch

## Citations
If you use this pipeline or data, please cite:
```
@inproceedings{sarlin2022lamar,
  author    = {Paul-Edouard Sarlin and
               Mihai Dusmanu and
               Johannes L. Schönberger and
               Pablo Speciale and
               Lukas Gruber and
               Viktor Larsson and
               Ondrej Miksik and
               Marc Pollefeys},
  title     = {{LaMAR: Benchmarking Localization and Mapping for Augmented Reality}},
  booktitle = {ECCV},
  year      = {2022},
}

@inproceedings{
boudjoghra2025openyolo,
title={Open-{YOLO} 3D: Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation},
author={Mohamed El Amine Boudjoghra and Angela Dai and Jean Lahoud and Hisham Cholakkal and Rao Muhammad Anwer and Salman Khan and Fahad Shahbaz Khan},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025}
}
```
