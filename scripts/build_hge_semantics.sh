#!/usr/bin/env bash
# End-to-end HGE semantic-mesh build.
#
# Activates the cnsg-seg env (py3.12 + torch 2.10 cu128 + SAM 3) and runs
# `python -m cnsg.segmentation.build_hge`. The result lands under
# `data/maps/hge/` as `HGE.semantic.glb` + `.semantic.txt` +
# `room_id_to_name_map.json` + `HGE.scene_dataset_config.json`.
#
# Usage:
#   bash scripts/build_hge_semantics.sh                # full 2408-frame run
#   bash scripts/build_hge_semantics.sh --max-frames 20  # quick smoke
#
# Prereq:
#   1. `scripts/download_data.sh` has populated mesh_pipeline/data/.
#   2. `cnsg-seg` env is built (see SETUP.md / phase3-research.md).
#   3. `huggingface-cli login` with an approved token for
#      https://huggingface.co/facebook/sam3.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"

SESSION="$ROOT_DIR/mesh_pipeline/data/navvis_2022-02-06_12.55.11"
MESH="$ROOT_DIR/mesh_pipeline/data/HGE_cut.voxelized.ply"
OUT_DIR="$ROOT_DIR/data/maps/hge"
# Cache per-frame segmentation outputs so iterating on lift / export knobs
# doesn't re-run Mask2Former + SAM 3 (~52 min). Entries self-invalidate when
# prompts or confidence change (hash-tagged). Delete the dir to force a
# full re-run.
SEG_CACHE_DIR="$ROOT_DIR/data/maps/hge/seg_cache"

mkdir -p "$OUT_DIR"

# Run inside the cnsg-seg env.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cnsg-seg

exec python -m cnsg.segmentation.build_hge \
    --navvis-session "$SESSION" \
    --mesh "$MESH" \
    --out-dir "$OUT_DIR" \
    --stem HGE \
    --seg-cache-dir "$SEG_CACHE_DIR" \
    "$@"
