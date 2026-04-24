"""Run PKU-EPIC/MaskClustering on an HGE seg_cache as the 2D→3D lifter.

Alternative to `cnsg.segmentation.lift_2d_to_3d` (our union-find + majority-
vote baseline). MaskClustering (CVPR'24) lifts per-frame mask images to
global 3D instances via view-consensus graph clustering — training-free,
beats union-find on ScanNet / ScanNet++.

## How to run

1. Clone the MaskClustering repo into the vendor directory (one-time):

       mkdir -p mesh_pipeline/third_party
       git clone --depth 1 https://github.com/PKU-EPIC/MaskClustering.git \\
           mesh_pipeline/third_party/MaskClustering

   The path is gitignored; we don't vendor the source.

2. Populate the seg_cache once with `scripts/build_hge_semantics.sh` (the
   build stores each frame's post-combine `(instance_mask, class_mask)` npz).

3. Invoke:

       python -m cnsg.segmentation.lift_maskclustering \\
           --session mesh_pipeline/data/navvis_2022-02-06_12.55.11 \\
           --mesh    data/maps/hge/HGE.semantic.glb \\
           --seg-cache-dir data/maps/hge/seg_cache \\
           --out-dir data/maps/hge/maskclustering_out \\
           [--stride 1]   # every frame; raise to e.g. 5 for smoke tests

## Design notes

- Our adapter (`cnsg.segmentation.maskclustering_adapter.HgeMaskClusteringDataset`)
  presents our NavVis data in the exact shape MaskClustering's `ScanNetDataset`
  does — intrinsics via `o3d.camera.PinholeCameraIntrinsic`, extrinsic as
  4×4 cam-to-world (absolute frame, alignment applied), depth in metres,
  mask image at full depth resolution. Zero modification to the upstream
  MaskClustering source.
- We skip MaskClustering's CLIP-based class-labelling pass. The cluster
  output is class-agnostic; we bolt a per-cluster majority vote over our
  structural S3DIS labels on top (cheaper + uses labels we already have
  from Mask2Former / EoMT).
- The upstream default `view_consensus_threshold=0.9` is tuned for
  ScanNet's CropFormer masks. SAM 3 produces tighter boundaries and more
  masks per frame, so 0.9 tends to over-split. We expose it as a CLI flag
  and default to 0.8 (slightly looser) as a starting point; sweep per run.

Budget: a full HGE run (2408 frames, 33 SAM 3 prompts) is estimated at
10-20 min on RTX 5090 (ScanNet++ numbers from the paper + our mask-count
scaling). Dominant cost is the `contained_mask` graph build; the iterative
clustering itself is < 1 min.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from cnsg.segmentation.maskclustering_adapter import HgeMaskClusteringDataset


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MASKCLUSTERING_DEFAULT = REPO_ROOT / "mesh_pipeline" / "third_party" / "MaskClustering"


def _require_maskclustering_on_path(clone_dir: Path) -> None:
    """Inject the vendored MaskClustering clone into sys.path or die loud.

    Also: ensures `{clone_dir}/utils/__init__.py` and
    `{clone_dir}/graph/__init__.py` exist (upstream ships them as
    namespace packages, which lose the resolution race against our repo's
    regular `utils/` package). And installs our pytorch3d shim so the
    single `pytorch3d.ops.ball_query` dependency doesn't force a source
    build on Blackwell / cu128.
    """
    if not (clone_dir / "main.py").exists():
        raise SystemExit(
            f"[FATAL] MaskClustering source not found at {clone_dir}. "
            f"Clone it first:\n\n"
            f"    mkdir -p {clone_dir.parent}\n"
            f"    git clone --depth 1 "
            f"https://github.com/PKU-EPIC/MaskClustering.git {clone_dir}\n"
        )
    # Upgrade MaskClustering's namespace packages to regular packages so
    # they beat our repo's `utils/__init__.py` in import resolution.
    for sub in ("utils", "graph", "evaluation"):
        init = clone_dir / sub / "__init__.py"
        if not init.exists():
            init.touch()
    sys.path.insert(0, str(clone_dir))
    # Evict any previously-cached `utils` module that might point at our
    # repo's shadow package — forces re-resolution against the new path.
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            del sys.modules[name]
    # Patch away the single pytorch3d dependency before upstream imports fire.
    from cnsg.segmentation._pytorch3d_shim import install_shim
    install_shim()


def _build_args_namespace(
    dataset_name: str,
    seq_name: str,
    step: int,
    view_consensus_threshold: float,
    config_name: str = "hge",
    debug: bool = False,
) -> argparse.Namespace:
    """Hand-roll the argparse.Namespace MaskClustering's main expects.

    Its own `get_args()` would insist on `--seq_name_list` / `--dataset` +
    a config.json on disk. We bypass that and pass the dict directly.
    """
    return argparse.Namespace(
        dataset=dataset_name,
        seq_name=seq_name,
        seq_name_list=seq_name,
        config=config_name,
        step=step,
        view_consensus_threshold=view_consensus_threshold,
        debug=debug,
        # Knobs read by mask_graph_construction / iterative_clustering
        # (defaults from MaskClustering/configs/scannet.json). Values are
        # the published ScanNet defaults — we can sweep later once we see
        # qualitative output.
        contained_threshold=0.8,
        coverage_threshold=0.3,
        distance_threshold=0.05,   # bumped 0.03 → 0.05 — decimated mesh has coarser verts
        undersegmentation_threshold=0.3,
        point_filter_threshold=50,
    )


def run(
    session_dir: Path,
    mesh_path: Path,
    seg_cache_dir: Path,
    out_dir: Path,
    *,
    stride: int = 1,
    view_consensus_threshold: float = 0.8,
    clone_dir: Path = MASKCLUSTERING_DEFAULT,
    max_frames: Optional[int] = None,
    debug: bool = False,
) -> dict:
    """End-to-end: build dataset adapter, run MaskClustering graph + cluster
    + post-process, write outputs, return a summary dict.
    """
    _require_maskclustering_on_path(clone_dir)

    # Now we can import upstream.
    import torch
    from graph.construction import mask_graph_construction
    from graph.iterative_clustering import iterative_clustering
    # `post_process` writes into dataset.object_dict_dir; we'll re-use it.
    from utils.post_process import post_process

    dataset = HgeMaskClusteringDataset(
        session_dir=session_dir,
        mesh_path=mesh_path,
        seg_cache_dir=seg_cache_dir,
        out_dir=out_dir,
        max_frames=max_frames,
    )

    frame_list = dataset.get_frame_list(stride)
    scene_points = dataset.get_scene_points()
    print(f"[maskclust] frames={len(frame_list)}  verts={len(scene_points):,}")

    args_ns = _build_args_namespace(
        dataset_name="hge",
        seq_name=dataset.seq_name,
        step=stride,
        view_consensus_threshold=view_consensus_threshold,
        debug=debug,
    )

    t0 = time.time()
    with torch.no_grad():
        nodes, observer_num_thresholds, mask_point_clouds, point_frame_matrix = (
            mask_graph_construction(args_ns, scene_points, frame_list, dataset)
        )
        t_graph = time.time() - t0
        print(
            f"[maskclust] graph built: {len(nodes)} nodes "
            f"({t_graph:.1f}s)"
        )

        t0 = time.time()
        object_list = iterative_clustering(
            nodes, observer_num_thresholds, view_consensus_threshold, debug
        )
        t_cluster = time.time() - t0
        print(
            f"[maskclust] clustering: {len(object_list)} objects "
            f"({t_cluster:.1f}s)"
        )

        t0 = time.time()
        post_process(
            dataset, object_list, mask_point_clouds, scene_points,
            point_frame_matrix, frame_list, args_ns,
        )
        t_post = time.time() - t0
        print(f"[maskclust] post-process: ({t_post:.1f}s)")

    summary = {
        "num_frames": len(frame_list),
        "stride": stride,
        "view_consensus_threshold": view_consensus_threshold,
        "num_nodes": len(nodes),
        "num_objects": len(object_list),
        "time_graph_s": round(t_graph, 2),
        "time_cluster_s": round(t_cluster, 2),
        "time_post_s": round(t_post, 2),
        "total_time_s": round(t_graph + t_cluster + t_post, 2),
        "out_dir": str(dataset.out_dir),
        "object_dict_dir": dataset.object_dict_dir,
    }
    (out_dir / "maskclustering_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[maskclust] summary → {out_dir / 'maskclustering_summary.json'}\n"
        f"  total: {summary['total_time_s']:.1f}s; "
        f"{summary['num_objects']} objects from {summary['num_frames']} frames"
    )
    return summary


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--seg-cache-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--view-consensus-threshold", type=float, default=0.8)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument(
        "--clone-dir", type=Path, default=MASKCLUSTERING_DEFAULT,
        help="where MaskClustering's source is checked out",
    )
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run(
        session_dir=args.session,
        mesh_path=args.mesh,
        seg_cache_dir=args.seg_cache_dir,
        out_dir=args.out_dir,
        stride=args.stride,
        view_consensus_threshold=args.view_consensus_threshold,
        clone_dir=args.clone_dir,
        max_frames=args.max_frames,
        debug=args.debug,
    )


if __name__ == "__main__":
    _main()
