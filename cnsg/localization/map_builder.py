"""One-off map-artifact migration from LaMAR outputs → stable `data/maps/hge/`.

The LaMAR benchmark wrapper produced localization artifacts under a deeply
nested, method-encoding directory layout:

    mesh_pipeline/third_party/lamar-benchmark/outputs/
      mapping/map/triangulation/superpoint/
        netvlad-10_frustum_pose-120-20-250/superglue/sfm/
      extraction/map/superpoint/features.h5
      extraction/map/netvlad/features.h5

Phase 1 of the migration (see docs/report/01_architecture-lean-migration/plan.md)
collapses that into a stable, method-agnostic schema read by
`cnsg.localization.inference.Localizer.from_settings()`:

    data/maps/hge/
      sfm/
        cameras.bin
        images.bin
        points3D.bin
      features_superpoint.h5
      features_netvlad.h5
      manifest.json

Run once via `scripts/build_hge_map.sh`. Default mode is `move` (renames the
artifacts in place so the LaMAR submodule can then be deleted without data
loss); `--copy` leaves LaMAR intact for audit.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


# Source paths inside the LaMAR outputs tree (validated in build()).
_LAMAR_RELATIVE = {
    "sfm_dir": Path(
        "mapping/map/triangulation/superpoint/"
        "netvlad-10_frustum_pose-120-20-250/superglue/sfm"
    ),
    "superpoint_h5": Path("extraction/map/superpoint/features.h5"),
    "netvlad_h5": Path("extraction/map/netvlad/features.h5"),
}


def _ensure_empty(dst: Path) -> None:
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(
            f"target {dst} already exists and is non-empty; "
            f"delete it first or run with a different --out."
        )
    dst.mkdir(parents=True, exist_ok=True)


def _move_or_copy(src: Path, dst: Path, mode: str) -> None:
    if not src.exists():
        raise FileNotFoundError(f"source artifact missing: {src}")
    if mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"unknown mode: {mode!r} (expected 'move' or 'copy')")


def build(lamar_outputs: Path, out: Path, mode: str = "move") -> None:
    """Produce the stable map artifact layout under `out` from `lamar_outputs`.

    Args:
        lamar_outputs: path to `mesh_pipeline/third_party/lamar-benchmark/outputs`.
        out: target directory (typically `data/maps/hge`).
        mode: 'move' (default) or 'copy'.
    """
    lamar_outputs = Path(lamar_outputs).resolve()
    out = Path(out).resolve()

    if not lamar_outputs.is_dir():
        raise FileNotFoundError(f"lamar outputs dir missing: {lamar_outputs}")

    srcs = {k: lamar_outputs / rel for k, rel in _LAMAR_RELATIVE.items()}
    for label, src in srcs.items():
        if not src.exists():
            raise FileNotFoundError(f"source {label}: {src} missing")

    _ensure_empty(out)

    # 1. COLMAP reconstruction.
    _move_or_copy(srcs["sfm_dir"], out / "sfm", mode)
    # 2. Per-image SuperPoint features.
    _move_or_copy(srcs["superpoint_h5"], out / "features_superpoint.h5", mode)
    # 3. Per-image NetVLAD global descriptors.
    _move_or_copy(srcs["netvlad_h5"], out / "features_netvlad.h5", mode)

    # 4. Manifest: record provenance so a future reader knows what pipeline,
    #    what matcher, what retrieval the recon was built with. Load-bearing
    #    for reproducibility (CLAUDE.md §0).
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "source": str(lamar_outputs),
        "source_pipeline": "lamar-benchmark",
        "source_layout": {k: str(v) for k, v in _LAMAR_RELATIVE.items()},
        "features": {
            "local": "superpoint (nms_radius=3, max_keypoints=2048, resize_max=1024)",
            "retrieval": "netvlad (resize_max=640)",
            "matcher_used_during_sfm": "superglue",
            "num_reference_pairs": 10,
        },
        "scene": "HGE (ETH Hauptgebäude, E floor)",
        "session": "navvis_2022-02-06_12.55.11",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[map_builder] wrote stable layout to {out} (mode={mode})")
    print("[map_builder] contents:")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            print(f"  {p.relative_to(out)} ({size / 1024 / 1024:.1f} MB)")


# --- CLI ---------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--lamar-outputs",
        type=Path,
        required=True,
        help="Path to mesh_pipeline/third_party/lamar-benchmark/outputs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Target map directory (e.g. data/maps/hge).",
    )
    parser.add_argument(
        "--mode",
        choices=("move", "copy"),
        default="move",
        help="Default 'move' — renames in place so the LaMAR submodule can then "
        "be deleted without data loss. Use 'copy' to keep LaMAR intact.",
    )
    args = parser.parse_args()

    build(args.lamar_outputs, args.out, args.mode)


if __name__ == "__main__":
    _main()
