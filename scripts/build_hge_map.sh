#!/usr/bin/env bash
# Phase 1 one-off: migrate LaMAR-produced localization artifacts to
# the stable data/maps/hge/ layout that cnsg.localization.Localizer reads.
#
# Run once after `scripts/download_data.sh` has populated
# mesh_pipeline/third_party/lamar-benchmark/outputs/.
#
# Default mode is `move` — after this succeeds, the LaMAR submodule can
# be removed without data loss. Pass `--copy` to keep LaMAR intact.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"

# Input: staged LaMAR-format outputs from `scripts/download_data.sh`.
LAMAR_OUT="$ROOT_DIR/data/maps/hge_staged/outputs"
OUT="$ROOT_DIR/data/maps/hge"
MODE="move"

if [[ "${1:-}" == "--copy" ]]; then
    MODE="copy"
fi

if [[ ! -d "$LAMAR_OUT" ]]; then
    echo "Error: staged inputs missing at $LAMAR_OUT"
    echo "Run scripts/download_data.sh first to fetch the LaMAR-format localization archive."
    exit 1
fi

python -m cnsg.localization.map_builder \
    --lamar-outputs "$LAMAR_OUT" \
    --out "$OUT" \
    --mode "$MODE"

echo ""
echo "Stable layout ready at: $OUT"
if [[ "$MODE" == "move" ]]; then
    echo "The lamar-benchmark submodule no longer contains needed artifacts."
    echo "You can now remove it:  git submodule deinit -f mesh_pipeline/third_party/lamar-benchmark"
fi
