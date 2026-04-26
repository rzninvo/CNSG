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
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from cnsg.segmentation.maskclustering_adapter import HgeMaskClusteringDataset


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MASKCLUSTERING_DEFAULT = REPO_ROOT / "mesh_pipeline" / "third_party" / "MaskClustering"


def _patch_mask_backprojection() -> None:
    """Monkey-patch `utils.mask_backprojection` for two NavVis-specific bugs:

    (1) Replace `backproject` with a numpy implementation so `view_points`
        has BYTE-EXACTLY one row per `depth_mask`-passing pixel. The upstream
        `o3d.geometry.PointCloud.create_from_depth_image` silently drops a
        handful of pixels that `get_depth_mask` keeps, causing
        `view_points[valid_mask]` to raise IndexError downstream.

    (2) Bump `DISTANCE_THRESHOLD` 3 cm → 10 cm and relax `COVERAGE_THRESHOLD`
        0.30 → 0.15. Upstream's 3 cm is tuned for ScanNet's ~5 mm-spaced verts.
        Our decimated HGE mesh has 153 k verts over ~5 000 m² of surface →
        ~5-10 cm average vertex spacing. A 3 cm K=20 ball_query rarely hits
        K neighbours, coverage falls below 30 %, EVERY mask gets rejected,
        and the global frame-mask list ends up empty → `torch.stack` blows
        up at construction.py:152. Agent C's deep-dive flagged this exact
        failure mode; here's the concrete tuning.
    """
    import numpy as np
    import open3d as o3d
    import utils.mask_backprojection as mb

    DEPTH_TRUNC = mb.DEPTH_TRUNC

    def backproject_numpy(depth, intrinsics, extrinsics):
        depth_np = np.asarray(depth, dtype=np.float32)
        H, W = depth_np.shape
        valid = (depth_np > 0) & (depth_np < DEPTH_TRUNC)
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        u_v = u[valid]
        v_v = v[valid]
        z = depth_np[valid]

        K = np.asarray(intrinsics.intrinsic_matrix, dtype=np.float64)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x = (u_v.astype(np.float64) - cx) * z / fx
        y = (v_v.astype(np.float64) - cy) * z / fy
        pts_cam = np.stack([x, y, z.astype(np.float64)], axis=1)

        T = np.asarray(extrinsics, dtype=np.float64)
        pts_world = (T[:3, :3] @ pts_cam.T).T + T[:3, 3]

        pcld = o3d.geometry.PointCloud()
        pcld.points = o3d.utility.Vector3dVector(pts_world)
        return pcld

    mb.backproject = backproject_numpy
    mb.DISTANCE_THRESHOLD = 0.10
    mb.COVERAGE_THRESHOLD = 0.15
    mb.FEW_POINTS_THRESHOLD = 25  # unchanged but pinned for clarity
    print(
        f"[maskclust] patched mask_backprojection: "
        f"DISTANCE_THRESHOLD=0.10m, COVERAGE_THRESHOLD=0.15, "
        f"backproject=numpy",
        flush=True,
    )


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
    # Replace upstream's open3d-based backproject (which silently drops a few
    # pixels on NavVis depth maps) with a numpy version that matches
    # `get_depth_mask` exactly. Must run AFTER the upstream `utils` import.
    _patch_mask_backprojection()


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
    # Field names match `args.*` accesses across upstream
    # `mask_graph_construction`, `process_one_mask`, `iterative_clustering`,
    # and `post_process` — verified by `grep -oE 'args\.\w+'` over the
    # vendored MaskClustering source. Default values lifted from
    # `configs/scannet.json` (the published ScanNet recipe).
    return argparse.Namespace(
        dataset=dataset_name,
        seq_name=seq_name,
        seq_name_list=seq_name,
        config=config_name,
        step=step,
        debug=debug,
        # Graph + clustering thresholds (ScanNet defaults).
        view_consensus_threshold=view_consensus_threshold,  # 0.9
        contained_threshold=0.8,
        mask_visible_threshold=0.3,
        undersegment_filter_threshold=0.3,
        point_filter_threshold=0.5,
    )


def split_oversized_clusters_3d(
    object_dict: dict,
    scene_points: np.ndarray,
    mask_point_clouds: dict,
    *,
    max_extent_m: float = 5.0,
    eps_m: float = 1.0,
    dbscan_min_points: int = 4,
    min_cc_verts: int = 8,
    min_masks_per_subcluster: int = 2,
    mask_intersection_fraction: float = 0.2,
) -> tuple[dict, dict]:
    """Re-split clusters whose bbox spans more than `max_extent_m`.

    Why: SAM 3's open-vocab prompts can produce a single mask covering a row
    of similar instances (e.g. all 5 columns down a hallway), which MC's
    view-consensus then merges into one global cluster spanning the row.
    Upstream's 0.1m DBSCAN inside `post_process` doesn't always split these
    because back-projected scatter bridges the column gaps. We re-split at
    `eps_m` (default 0.5m) so single columns stay intact while landmark-
    scale ambiguities split apart.

    For each over-sized cluster:
      1. Open3D DBSCAN(eps=eps_m, min_points=dbscan_min_points) on its vert
         positions. `dbscan_min_points` defaults to 4 (matches upstream's
         own DBSCAN call inside `post_process`); `min_cc_verts` is the
         POST-clustering size filter and is a separate, larger threshold.
         The MC vert distribution per cluster is sparse — back-projection
         only stamps mesh verts along depth rays, so requiring 20 neighbours
         within 0.5 m starves nearly every cluster (we observed 95/97
         oversized clusters returning 0 sub-clusters with min_points=20,
         eps=0.5m). Loose core + tight post-filter handles both cases.
      2. Build vert→sub_label map; for every supporting `(frame_id, mask_id)`,
         compute intersection with each sub-cluster. The mask "supports" a
         sub-cluster if `intersection / |mask_verts| >= mask_intersection_fraction`
         — a single SAM 3 over-mask covering BOTH physical columns is
         evidence for both sub-clusters and must count for both, otherwise
         every winner-takes-all dispatch starves the smaller sub-cluster.
      3. Drop sub-clusters whose mask_list shrinks below
         `min_masks_per_subcluster` (matches upstream's < 2 filter).

    Args:
        object_dict: cluster dict as written by upstream `post_process` —
            `{cid: {"point_ids": iterable[int], "mask_list": list[(frame_id,
            mask_id, coverage)], "repre_mask_list": ...}}`. Mutation safe
            (returns a fresh dict).
        scene_points: (V, 3) mesh vertex positions.
        mask_point_clouds: in-memory `{f"{frame_id}_{mask_id}": set[int]}`
            from `mask_graph_construction`'s second return.
        max_extent_m: trigger threshold; clusters with bbox dim > this get
            re-split.
        eps_m: DBSCAN neighborhood for the re-split. 0.5m breaks columns
            (~3-5m apart) without splitting a single column (~1m wide).
        min_cc_verts: drop sub-clusters smaller than this.
        min_masks_per_subcluster: drop sub-clusters with fewer supporting
            masks than this.

    Returns:
        `(new_object_dict, stats)` — stats has keys `n_input`, `n_output`,
        `n_split`, `n_dropped_small`, `extent_before_max`, `extent_after_max`.
    """
    import open3d as o3d

    new_dict: dict = {}
    next_id = 0
    n_split = 0
    n_dropped_small = 0
    extent_before_max = 0.0
    extent_after_max = 0.0

    for c in object_dict.values():
        point_ids_arr = np.asarray(list(c["point_ids"]), dtype=np.int64)
        if len(point_ids_arr) == 0:
            continue
        pts = scene_points[point_ids_arr]
        extent = pts.max(0) - pts.min(0)
        ext_max = float(extent.max())
        extent_before_max = max(extent_before_max, ext_max)

        if ext_max <= max_extent_m:
            new_dict[next_id] = {
                "point_ids": point_ids_arr,
                "mask_list": list(c["mask_list"]),
                "repre_mask_list": list(c.get("repre_mask_list", c["mask_list"][:5])),
            }
            extent_after_max = max(extent_after_max, ext_max)
            next_id += 1
            continue

        # Re-split. Use open3d DBSCAN to match upstream tooling. Loose
        # core-point criterion (min_points=dbscan_min_points, default 4)
        # plus tight post-filter (min_cc_verts) keeps sparse-but-real
        # landmark sub-clusters and drops noise.
        pcld = o3d.geometry.PointCloud()
        pcld.points = o3d.utility.Vector3dVector(pts)
        labels = np.asarray(
            pcld.cluster_dbscan(eps=eps_m, min_points=dbscan_min_points),
            dtype=np.int64,
        )
        unique_labels = sorted(int(l) for l in np.unique(labels) if l >= 0)

        if len(unique_labels) <= 1:
            # Either DBSCAN found one big component (the whole over-sized
            # cluster IS connected at 0.5 m) or only noise. Keep as-is —
            # we'd rather over-merge than drop clusters silently.
            new_dict[next_id] = {
                "point_ids": point_ids_arr,
                "mask_list": list(c["mask_list"]),
                "repre_mask_list": list(c.get("repre_mask_list", c["mask_list"][:5])),
            }
            extent_after_max = max(extent_after_max, ext_max)
            next_id += 1
            print(
                f"[WARN] split_oversized: cluster extent {ext_max:.1f}m exceeds "
                f"max_extent_m={max_extent_m}m but DBSCAN(eps={eps_m}m) found "
                f"{len(unique_labels)} sub-cluster(s); kept as-is. "
                f"fallback=keep-original",
                flush=True,
            )
            continue

        n_split += 1

        # vert_id → sub_label, used to reassign each mask.
        vert_to_label = {int(v): int(l) for v, l in zip(point_ids_arr, labels) if l >= 0}

        # For each (frame_id, mask_id, coverage) in the parent mask_list,
        # count its verts per sub-cluster and dispatch to every sub-cluster
        # where the intersection ≥ mask_intersection_fraction of the mask's
        # in-cluster verts. A genuine over-mask (one SAM 3 mask covering
        # both physical columns) lands in BOTH sub-clusters; a mask with a
        # tiny tail in another sub-cluster only lands in the dominant one.
        sub_masks: dict[int, list] = {l: [] for l in unique_labels}
        for entry in c["mask_list"]:
            frame_id, mask_id = int(entry[0]), int(entry[1])
            key = f"{frame_id}_{mask_id}"
            mask_verts = mask_point_clouds.get(key)
            if mask_verts is None:
                continue
            counts: Counter = Counter()
            n_in_parent = 0
            for v in mask_verts:
                lbl = vert_to_label.get(int(v))
                if lbl is not None:
                    counts[lbl] += 1
                    n_in_parent += 1
            if not counts:
                continue
            threshold = max(1, int(mask_intersection_fraction * n_in_parent))
            dispatched = False
            for lbl, n in counts.items():
                if n >= threshold:
                    sub_masks[lbl].append(entry)
                    dispatched = True
            if not dispatched:
                # Mask had verts in this cluster but each sub-cluster's share
                # was below the threshold — fall back to dominant assignment
                # so we never silently drop evidence.
                sub_masks[counts.most_common(1)[0][0]].append(entry)

        for lbl in unique_labels:
            sub_idx = np.where(labels == lbl)[0]
            if len(sub_idx) < min_cc_verts:
                n_dropped_small += 1
                continue
            sub_mask_list = sub_masks[lbl]
            if len(sub_mask_list) < min_masks_per_subcluster:
                n_dropped_small += 1
                continue
            sub_point_ids = point_ids_arr[sub_idx]
            sub_pts = scene_points[sub_point_ids]
            sub_ext_max = float((sub_pts.max(0) - sub_pts.min(0)).max())
            extent_after_max = max(extent_after_max, sub_ext_max)
            sub_mask_list_sorted = sorted(
                sub_mask_list,
                key=lambda x: x[2] if len(x) > 2 else 0.0,
                reverse=True,
            )
            new_dict[next_id] = {
                "point_ids": sub_point_ids,
                "mask_list": sub_mask_list,
                "repre_mask_list": sub_mask_list_sorted[:5],
            }
            next_id += 1

    stats = {
        "n_input": len(object_dict),
        "n_output": len(new_dict),
        "n_split": n_split,
        "n_dropped_small": n_dropped_small,
        "extent_before_max_m": round(extent_before_max, 2),
        "extent_after_max_m": round(extent_after_max, 2),
        "max_extent_m": max_extent_m,
        "eps_m": eps_m,
        "dbscan_min_points": dbscan_min_points,
    }
    return new_dict, stats


def majority_phrase_per_cluster(
    object_dict: dict,
    dataset: HgeMaskClusteringDataset,
    *,
    fallback_phrase: str = "clutter",
) -> dict[int, str]:
    """Per cluster, majority-vote an open-vocab phrase from supporting masks.

    Companion to `majority_class_per_cluster` but operates on the per-frame
    SAM 3 phrase sidecar (written by `build_hge._seg_cache_save` when
    `--use-gpt5-tagger` is on). Each `(frame_id, mask_id)` in the cluster's
    `mask_list` resolves to the SAM 3 prompt phrase that GENERATED that
    instance — which, with the GPT-5.5 tagger upstream, is a heritage-quality
    description like "marble bust on plinth" rather than the S3DIS-13 label.
    Vote = sum of mask coverage areas across all supporting frames.

    Returns `fallback_phrase` for any cluster whose supporting masks have
    no phrase sidecar at all (older cache without open-vocab support); a
    [WARN] is emitted on the first such cluster so the regression is loud.

    Args:
        object_dict: post-split cluster dict.
        dataset: adapter with `get_instance_phrases(frame_id)` returning
            `{sam3_instance_id: prompt_phrase}` or None.
        fallback_phrase: assigned when no sidecar exists for any supporting
            frame of a cluster (S3DIS "clutter" by convention).

    Returns:
        `{cluster_id: phrase}`.
    """
    # Cache the phrase sidecar per frame so multiple clusters sharing a
    # frame don't re-read the same JSON file.
    phrase_cache: dict[int, Optional[dict[int, str]]] = {}

    def _get(frame_id: int) -> Optional[dict[int, str]]:
        cached = phrase_cache.get(frame_id, "MISS")
        if cached != "MISS":
            return cached  # type: ignore[return-value]
        loaded = dataset.get_instance_phrases(frame_id)
        phrase_cache[frame_id] = loaded
        return loaded

    cluster_phrases: dict[int, str] = {}
    n_warned = 0
    for cid, c in object_dict.items():
        votes: Counter = Counter()
        for entry in c["mask_list"]:
            frame_id, mask_id = int(entry[0]), int(entry[1])
            phrases = _get(frame_id)
            if phrases is None:
                continue
            phrase = phrases.get(mask_id)
            if not phrase:
                continue
            # Weight by coverage if the entry carries one (post_process emits
            # tuples of (frame_id, mask_id, coverage)); otherwise unit-weight.
            weight = float(entry[2]) if len(entry) > 2 else 1.0
            votes[phrase] += weight
        if votes:
            cluster_phrases[cid] = votes.most_common(1)[0][0]
        else:
            cluster_phrases[cid] = fallback_phrase
            n_warned += 1
    if n_warned > 0:
        print(
            f"[WARN] majority_phrase_per_cluster: expected=phrases sidecar, "
            f"got=missing for {n_warned}/{len(object_dict)} clusters, "
            f"fallback={fallback_phrase!r}. Re-run build_hge with "
            f"--use-gpt5-tagger to populate the sidecars.",
            flush=True,
        )
    return cluster_phrases


def majority_class_per_cluster(
    object_dict: dict,
    dataset: HgeMaskClusteringDataset,
    *,
    background_class: int = 0,
) -> dict[int, int]:
    """Per cluster, majority-vote an S3DIS class from supporting-mask pixels.

    MaskClustering is class-agnostic; for Habitat to render meaningful
    semantics, every cluster needs ONE class label. For each
    `(frame_id, mask_id, ...)` in `cluster.mask_list`, look at the cached
    structural class_mask (from the M2F/EoMT pass) restricted to the
    instance-mask region. Aggregate counts across all supporting masks;
    the cluster's class is the most-frequent non-zero label.

    Args:
        object_dict: post-split cluster dict.
        dataset: adapter exposing `get_segmentation` (instance_only mask)
            and `get_structural_class_map` (cached S3DIS pixel labels).
        background_class: ignore this label when voting (S3DIS 0 = unknown).

    Returns:
        `{cluster_id: class_id}`. Clusters with no labeled votes get
        `background_class`.
    """
    # Frame-level cache so we don't reload the same npz multiple times when
    # several clusters share supporting frames (the common case).
    frame_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _get(frame_id: int) -> tuple[np.ndarray, np.ndarray]:
        cached = frame_cache.get(frame_id)
        if cached is not None:
            return cached
        inst = dataset.get_segmentation(frame_id, align_with_depth=False)
        cls = dataset.get_structural_class_map(frame_id, align_with_depth=False)
        frame_cache[frame_id] = (inst, cls)
        return inst, cls

    cluster_classes: dict[int, int] = {}
    for cid, c in object_dict.items():
        votes: Counter = Counter()
        for entry in c["mask_list"]:
            frame_id, mask_id = int(entry[0]), int(entry[1])
            inst, cls = _get(frame_id)
            mask_pixels = inst == mask_id
            if not mask_pixels.any():
                continue
            ids, counts = np.unique(cls[mask_pixels], return_counts=True)
            for c_id, n in zip(ids.tolist(), counts.tolist()):
                if c_id == background_class:
                    continue
                votes[int(c_id)] += int(n)
        cluster_classes[cid] = int(votes.most_common(1)[0][0]) if votes else background_class
    return cluster_classes


def export_clusters_to_habitat_open_vocab(
    object_dict: dict,
    cluster_phrases: dict[int, str],
    mesh_path: Path,
    out_dir: Path,
    stem: str,
    *,
    external_stage_glb: Optional[Path] = None,
    min_verts_per_instance: int = 20,
    drop_phrases: tuple[str, ...] = (),
    merge_clusters_with_same_phrase: bool = True,
) -> dict:
    """Write a Habitat HM3D bundle whose category names are the open-vocab
    phrases (NOT S3DIS-13). The downstream LLM in `mr_viewer` reads these
    names directly when generating navigation directions, so heritage-quality
    phrases like "marble bust on plinth" land in user-facing instructions
    instead of S3DIS's "clutter".

    Each cluster becomes one Habitat semantic instance via
    `region_id = cluster_id + 1`, identical layout to
    `export_clusters_to_habitat` but with the class taxonomy expanded to
    "every distinct phrase ever voted by any cluster" — synthesised on
    the fly so we don't need to maintain a static heritage class list.

    Args:
        object_dict: post-split cluster dict.
        cluster_phrases: `{cluster_id: phrase}` from
            `majority_phrase_per_cluster`.
        mesh_path: path to the SAME mesh whose vert indices the cluster
            point_ids reference.
        out_dir, stem, external_stage_glb, min_verts_per_instance:
            identical to `export_clusters_to_habitat`.
        drop_phrases: phrases to skip emitting (e.g. ("clutter",) when the
            caller wants only meaningful-named clusters in the bundle).

    Returns:
        Summary dict — counts and the phrase histogram.
    """
    import trimesh
    from cnsg.segmentation.export_habitat import export_habitat

    mesh = trimesh.load(mesh_path, force="mesh")
    n_verts = len(mesh.vertices)

    # Build a synthetic class taxonomy from the phrase set: assign each
    # distinct phrase a unique class_id starting at 1. Class 0 stays
    # reserved for "unknown / drop" per export_habitat's contract.
    drop_set = {p.strip().lower() for p in drop_phrases}
    distinct_phrases: list[str] = []
    phrase_to_class_id: dict[str, int] = {}
    for cid in object_dict:
        phrase = cluster_phrases.get(cid, "")
        key = phrase.strip().lower()
        if not key or key in drop_set:
            continue
        if phrase not in phrase_to_class_id:
            phrase_to_class_id[phrase] = len(distinct_phrases) + 1
            distinct_phrases.append(phrase)
    class_id_to_name = {cid: name for name, cid in phrase_to_class_id.items()}

    per_vertex_class = np.zeros(n_verts, dtype=np.int64)
    per_vertex_region = np.zeros(n_verts, dtype=np.int64)
    n_emitted = 0
    phrase_histogram: Counter = Counter()

    for cid, c in object_dict.items():
        phrase = cluster_phrases.get(cid, "")
        cls_id = phrase_to_class_id.get(phrase)
        if cls_id is None:
            continue
        point_ids = np.asarray(list(c["point_ids"]), dtype=np.int64)
        if len(point_ids) < min_verts_per_instance:
            continue
        valid = (point_ids >= 0) & (point_ids < n_verts)
        if not valid.any():
            continue
        pids = point_ids[valid]
        per_vertex_class[pids] = cls_id
        # When merge_clusters_with_same_phrase is True (default for UF
        # output where the same phrase like "pillar" gets voted by 192
        # separate UF clusters), every cluster sharing a phrase lands in
        # the same (cls, reg) bucket — producing ONE Habitat instance
        # per phrase rather than 192. Otherwise, region_id = cluster_id+1
        # so every UF root becomes its own Habitat instance (the original
        # MC behaviour where view-consensus already enforces uniqueness).
        per_vertex_region[pids] = cls_id if merge_clusters_with_same_phrase else (cid + 1)
        n_emitted += 1
        phrase_histogram[phrase] += 1

    if merge_clusters_with_same_phrase:
        # One region per phrase; name = the phrase itself (region_id == cls_id).
        region_name_map = {
            cls_id: name
            for name, cls_id in phrase_to_class_id.items()
        }
    else:
        # One region per cluster; preserves the prior (per-MC-cluster) layout.
        region_name_map = {
            cid + 1: f"cluster_{cid:04d}_{cluster_phrases.get(cid, 'unknown').replace(' ', '_')}"
            for cid in object_dict
        }

    manifest = export_habitat(
        mesh=mesh,
        per_vertex_class_id=per_vertex_class,
        per_vertex_region_id=per_vertex_region,
        class_id_to_name=class_id_to_name,
        out_dir=out_dir,
        stem=stem,
        region_id_to_name=region_name_map,
        group_per_class_region=True,
        min_verts_per_instance=min_verts_per_instance,
        external_stage_glb=external_stage_glb,
    )
    return {
        "num_clusters_in": len(object_dict),
        "num_clusters_emitted": n_emitted,
        "num_distinct_phrases": len(distinct_phrases),
        "num_instances": int(manifest.num_instances),
        "num_regions": int(manifest.num_regions),
        "stem": stem,
        "out_dir": str(out_dir),
        "phrase_histogram_top10": dict(phrase_histogram.most_common(10)),
    }


def export_clusters_to_habitat(
    object_dict: dict,
    cluster_classes: dict[int, int],
    mesh_path: Path,
    out_dir: Path,
    stem: str,
    *,
    external_stage_glb: Optional[Path] = None,
    min_verts_per_instance: int = 20,
) -> dict:
    """Bake MC clusters into HM3D-compatible files via `export_habitat`.

    Each cluster becomes one Habitat semantic instance. We achieve this
    one-instance-per-cluster mapping by passing
    `region_id = cluster_id + 1` (region 0 is reserved for "unknown"),
    `class_id = voted S3DIS class`, with `group_per_class_region=True`. The
    exporter then makes one (class, region) → instance pair per cluster.

    The cost: `room_id_to_name_map.json` ends up with one entry per cluster
    rather than per physical room. mr_viewer's floor-height heuristic still
    works (it picks the median of region centroids), but the per-room name
    map is now a per-cluster name map. That's intentional — for a
    landmark-navigation pipeline the per-cluster identity is what matters.

    Args:
        object_dict: post-split cluster dict.
        cluster_classes: `{cluster_id: s3dis_class_id}` from
            `majority_class_per_cluster`.
        mesh_path: path to the SAME mesh whose vert indices the cluster
            point_ids reference.
        out_dir: destination for the exported bundle.
        stem: shared basename for the emitted files.
        external_stage_glb: optional photorealistic stage GLB
            (e.g. `HGE.basis.glb`). Same semantics as `export_habitat`.
        min_verts_per_instance: drop clusters smaller than this; matches
            our existing `build_hge` setting and keeps Habitat's CC-bbox
            pass within memory budget.

    Returns:
        Summary dict — counts and the manifest.
    """
    import trimesh
    from cnsg.segmentation.export_habitat import export_habitat
    from cnsg.segmentation.taxonomy import S3DIS_CLASSES

    mesh = trimesh.load(mesh_path, force="mesh")
    n_verts = len(mesh.vertices)

    per_vertex_class = np.zeros(n_verts, dtype=np.int64)
    per_vertex_region = np.zeros(n_verts, dtype=np.int64)
    n_emitted = 0

    for cid, c in object_dict.items():
        cls_id = int(cluster_classes.get(cid, 0))
        if cls_id == 0:
            continue
        point_ids = np.asarray(list(c["point_ids"]), dtype=np.int64)
        if len(point_ids) < min_verts_per_instance:
            continue
        valid = (point_ids >= 0) & (point_ids < n_verts)
        if not valid.any():
            continue
        pids = point_ids[valid]
        per_vertex_class[pids] = cls_id
        per_vertex_region[pids] = cid + 1  # region 0 reserved for "unknown"
        n_emitted += 1

    class_name_lookup = {c.id: c.name for c in S3DIS_CLASSES}
    region_name_map = {
        cid + 1: f"cluster_{cid:04d}_{class_name_lookup.get(int(cluster_classes.get(cid, 0)), 'unknown')}"
        for cid in object_dict
    }

    manifest = export_habitat(
        mesh=mesh,
        per_vertex_class_id=per_vertex_class,
        per_vertex_region_id=per_vertex_region,
        class_id_to_name=class_name_lookup,
        out_dir=out_dir,
        stem=stem,
        region_id_to_name=region_name_map,
        group_per_class_region=True,
        min_verts_per_instance=min_verts_per_instance,
        external_stage_glb=external_stage_glb,
    )
    return {
        "num_clusters_in": len(object_dict),
        "num_clusters_emitted": n_emitted,
        "num_instances": int(manifest.num_instances),
        "num_regions": int(manifest.num_regions),
        "stem": stem,
        "out_dir": str(out_dir),
    }


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
    split_max_extent_m: float = 5.0,
    split_eps_m: float = 0.5,
    export_habitat_stem: Optional[str] = None,
    external_stage_glb: Optional[Path] = None,
    open_vocab_labels: bool = False,
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

    # ---- spatial-extent post-filter -----------------------------------------
    # Upstream's `post_process` writes object_dict.npy at
    # `{object_dict_dir}/{config}/object_dict.npy`. We re-load it, split any
    # over-sized clusters in 3D, and overwrite in place. See
    # `split_oversized_clusters_3d` for the why.
    object_dict_path = (
        Path(dataset.object_dict_dir) / args_ns.config / "object_dict.npy"
    )
    if not object_dict_path.exists():
        raise RuntimeError(
            f"[FATAL] post_process didn't write object_dict to {object_dict_path}; "
            f"can't run extent-split."
        )

    t0 = time.time()
    object_dict_pre = np.load(object_dict_path, allow_pickle=True).item()
    object_dict, split_stats = split_oversized_clusters_3d(
        object_dict_pre,
        scene_points,
        mask_point_clouds,
        max_extent_m=split_max_extent_m,
        eps_m=split_eps_m,
    )
    np.save(object_dict_path, object_dict, allow_pickle=True)
    t_split = time.time() - t0
    print(
        f"[maskclust] extent-split: {split_stats['n_input']} → "
        f"{split_stats['n_output']} clusters "
        f"(split {split_stats['n_split']}, dropped-small {split_stats['n_dropped_small']}; "
        f"max-extent {split_stats['extent_before_max_m']}m → "
        f"{split_stats['extent_after_max_m']}m); "
        f"({t_split:.1f}s)"
    )

    # ---- per-cluster majority class vote + Habitat export -------------------
    # Optional: if `export_habitat_stem` is provided, vote a class per
    # cluster and write the HM3D bundle so SemanticSensor / mr_viewer can
    # consume MC's clusters in place of the union-find ones.
    habitat_export_stats: Optional[dict] = None
    class_vote_stats: Optional[dict] = None
    t_vote = 0.0
    t_export = 0.0
    if export_habitat_stem is not None:
        if open_vocab_labels:
            t0 = time.time()
            cluster_phrases = majority_phrase_per_cluster(object_dict, dataset)
            t_vote = time.time() - t0
            phrase_hist: Counter = Counter(cluster_phrases.values())
            class_vote_stats = {
                "mode": "open_vocab",
                "num_clusters": len(cluster_phrases),
                "num_distinct_phrases": len(phrase_hist),
                "phrase_histogram_top10": dict(phrase_hist.most_common(10)),
            }
            print(
                f"[maskclust] open-vocab phrase-vote: "
                f"{class_vote_stats['num_clusters']} clusters, "
                f"{class_vote_stats['num_distinct_phrases']} distinct phrases "
                f"({t_vote:.1f}s)"
            )

            t0 = time.time()
            habitat_export_stats = export_clusters_to_habitat_open_vocab(
                object_dict=object_dict,
                cluster_phrases=cluster_phrases,
                mesh_path=mesh_path,
                out_dir=out_dir,
                stem=export_habitat_stem,
                external_stage_glb=external_stage_glb,
            )
            t_export = time.time() - t0
            print(
                f"[maskclust] habitat-export (open-vocab): "
                f"{habitat_export_stats['num_clusters_emitted']}/"
                f"{habitat_export_stats['num_clusters_in']} clusters → "
                f"{habitat_export_stats['num_instances']} instances, "
                f"{habitat_export_stats['num_distinct_phrases']} classes "
                f"({t_export:.1f}s)"
            )
        else:
            t0 = time.time()
            cluster_classes = majority_class_per_cluster(object_dict, dataset)
            t_vote = time.time() - t0
            class_hist: Counter = Counter(cluster_classes.values())
            class_vote_stats = {
                "mode": "s3dis_13",
                "num_clusters": len(cluster_classes),
                "num_unlabeled": int(class_hist.get(0, 0)),
                "class_histogram": {int(k): int(v) for k, v in class_hist.items()},
            }
            print(
                f"[maskclust] class-vote (S3DIS-13): {class_vote_stats['num_clusters']} clusters, "
                f"{class_vote_stats['num_unlabeled']} unlabeled "
                f"({t_vote:.1f}s)"
            )

            t0 = time.time()
            habitat_export_stats = export_clusters_to_habitat(
                object_dict=object_dict,
                cluster_classes=cluster_classes,
                mesh_path=mesh_path,
                out_dir=out_dir,
                stem=export_habitat_stem,
                external_stage_glb=external_stage_glb,
            )
            t_export = time.time() - t0
            print(
                f"[maskclust] habitat-export (S3DIS-13): "
                f"{habitat_export_stats['num_clusters_emitted']}/"
                f"{habitat_export_stats['num_clusters_in']} clusters → "
                f"{habitat_export_stats['num_instances']} instances "
                f"({t_export:.1f}s)"
            )

    summary = {
        "num_frames": len(frame_list),
        "stride": stride,
        "view_consensus_threshold": view_consensus_threshold,
        "num_nodes": len(nodes),
        "num_objects": len(object_list),
        "num_clusters_post_split": split_stats["n_output"],
        "split_stats": split_stats,
        "class_vote_stats": class_vote_stats,
        "habitat_export_stats": habitat_export_stats,
        "time_graph_s": round(t_graph, 2),
        "time_cluster_s": round(t_cluster, 2),
        "time_post_s": round(t_post, 2),
        "time_split_s": round(t_split, 2),
        "time_vote_s": round(t_vote, 2),
        "time_export_s": round(t_export, 2),
        "total_time_s": round(
            t_graph + t_cluster + t_post + t_split + t_vote + t_export, 2
        ),
        "out_dir": str(dataset.out_dir),
        "object_dict_dir": dataset.object_dict_dir,
    }
    (out_dir / "maskclustering_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[maskclust] summary → {out_dir / 'maskclustering_summary.json'}\n"
        f"  total: {summary['total_time_s']:.1f}s; "
        f"{summary['num_objects']} raw → "
        f"{summary['num_clusters_post_split']} after extent-split, "
        f"from {summary['num_frames']} frames"
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
    p.add_argument(
        "--split-max-extent-m", type=float, default=5.0,
        help="re-split clusters whose 3D bbox dim exceeds this (metres)",
    )
    p.add_argument(
        "--split-eps-m", type=float, default=0.5,
        help="DBSCAN eps (metres) for the extent-split re-cluster pass",
    )
    p.add_argument(
        "--export-habitat-stem", type=str, default=None,
        help="if set, write a Habitat .semantic.glb bundle under out_dir/<stem>.*",
    )
    p.add_argument(
        "--external-stage-glb", type=Path, default=None,
        help="optional photorealistic stage GLB referenced by the exported config",
    )
    p.add_argument(
        "--open-vocab-labels", action="store_true",
        help="Emit Habitat semantic.txt with open-vocab phrases voted from the "
             "GPT-5.5 / SAM 3 prompt sidecars (e.g. 'marble bust on plinth') "
             "instead of S3DIS-13 buckets. Requires seg_cache populated by "
             "build_hge --use-gpt5-tagger so phrases.json sidecars exist.",
    )
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
        split_max_extent_m=args.split_max_extent_m,
        split_eps_m=args.split_eps_m,
        export_habitat_stem=args.export_habitat_stem,
        external_stage_glb=args.external_stage_glb,
        open_vocab_labels=args.open_vocab_labels,
    )


if __name__ == "__main__":
    _main()
