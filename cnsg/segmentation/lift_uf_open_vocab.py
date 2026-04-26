"""Lift the existing seg_cache onto an arbitrary mesh + emit open-vocab labels.

Sister of `lift_maskclustering.py` and `build_hge.py`'s internal lift, but:

  - Re-uses the seg_cache populated by a prior `build_hge --use-gpt5-tagger`
    run (no fresh segmentation — pure lift step).
  - Lifts onto WHATEVER mesh you point it at — typically the photorealistic
    basis mesh (~4 M verts) so the SemanticSensor reads from the same
    geometry the RGB camera renders. No two-mesh mismatch.
  - Uses union-find voting (`lift_2d_to_3d.lift_masks_to_3d`) for high
    coverage (≈ 57 % on HGE) instead of MaskClustering's view-consensus
    filter (≈ 10 %).
  - Class voter is the open-vocab phrase voter from `lift_maskclustering`,
    fed `instance_to_mask_list` (newly exposed on `LiftResult`) so each
    cluster gets a heritage phrase (e.g. "marble bust on plinth") rather
    than the S3DIS-13 collapse.

## Why a separate module

`build_hge.py` does (segmentation + lift + export) in one process. Splitting
the lift out lets us re-target the SAME segmentation cache onto a different
mesh without re-running ~92 minutes of SAM 3 + EoMT GPU work. This is
essential when iterating on the mesh choice (decimated → basis → variant
decimations).

## Usage

    python -m cnsg.segmentation.lift_uf_open_vocab \\
        --session   mesh_pipeline/data/navvis_2022-02-06_12.55.11 \\
        --mesh      data/maps/hge/HGE.basis.glb        \\
        --seg-cache-dir data/maps/hge/seg_cache         \\
        --out-dir   data/maps/hge/builds/G_uf_basis_openvocab \\
        --stem      HGE \\
        --external-stage-glb data/maps/hge/HGE.basis.glb

Output: an HM3D-compatible Habitat bundle with the basis mesh as both the
visible stage AND the semantic asset, vertex-coloured by our open-vocab
cluster labels.

## Cost

Lift on a 4 M-vert basis mesh: ~10-30 min wall (depth-test scales with
n_verts × n_frames). Export + class vote: <30 s. No GPU/API spend
(seg_cache + gpt5_cache already on disk).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np


def run(
    *,
    session_dir: Path,
    mesh_path: Path,
    seg_cache_dir: Path,
    out_dir: Path,
    stem: str = "HGE",
    external_stage_glb: Optional[Path] = None,
    depth_tolerance_m: float = 0.15,
    # Default lowered 0.5 → 0.2 because the basis mesh's vertex density
    # makes per-frame projections of the same physical landmark only
    # share ~30-40 % of verts (occluded backs differ across viewpoints).
    # 0.5 starves the UF merge step → 192 separate clusters all called
    # "pillar". 0.2 keeps the same-physical-object UF instances merged
    # while still rejecting accidental cross-object collisions. Verified
    # with the H_uf_basis_openvocab visualisation regression.
    overlap_threshold: float = 0.2,
    max_frames: Optional[int] = None,
    fallback_phrase: str = "clutter",
    drop_phrases: tuple[str, ...] = (),
    min_verts_per_instance: int = 20,
    # When True (default), all UF clusters that vote the same phrase
    # collapse to ONE Habitat semantic instance instead of N — fixes the
    # rainbow-stripe rendering symptom.
    merge_clusters_with_same_phrase: bool = True,
) -> dict:
    """End-to-end: load mesh, stream seg_cache, UF-lift, vote phrases, export.

    Args:
        session_dir: NavVis Capture session — needed for poses + intrinsics.
        mesh_path: target mesh to label. The basis mesh works here as long
            as Habitat's CC pass survives it (use the patched Geo.h that
            replaces the recursive DFS with an iterative one).
        seg_cache_dir: directory of `frame_*.npz` (instance/class masks)
            and `frame_*.phrases.json` (per-instance prompt phrases) that
            `build_hge --use-gpt5-tagger` writes.
        out_dir: where to drop the Habitat bundle + summary.
        stem: file stem (`<stem>.semantic.glb`, etc.).
        external_stage_glb: optional photorealistic stage (typically the
            same mesh_path when labeling the basis directly). If None, the
            palette-coloured fallback `<stem>.glb` is the rendered stage.
        depth_tolerance_m: depth-test tolerance for visibility test.
            0.15 m is the validated HGE setting (see report 02).
        overlap_threshold: UF merge threshold (fraction of smaller mask
            covered by intersection).
        max_frames: optional cap (smoke runs); None = all frames.
        fallback_phrase: passed to the open-vocab voter for clusters with
            no surviving phrase votes.
        drop_phrases: phrases to skip emitting (e.g. ("clutter",)).
        min_verts_per_instance: drop instances smaller than this; matches
            the existing build_hge / Habitat CC-bbox safety setting.

    Returns:
        Summary dict (also written to `out_dir/lift_uf_open_vocab_summary.json`).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "lift_uf_open_vocab_summary.json"

    # Heavy imports lazy-loaded so `--help` doesn't pull torch.
    import trimesh
    from cnsg.segmentation.lift_2d_to_3d import (
        Frame,
        LiftResult,
        lift_masks_to_3d,
    )
    from cnsg.segmentation.lift_maskclustering import (
        export_clusters_to_habitat_open_vocab,
        majority_phrase_per_cluster,
    )
    from cnsg.segmentation.maskclustering_adapter import HgeMaskClusteringDataset

    t_all = time.time()
    print(f"[lift_uf] session: {session_dir}")
    print(f"[lift_uf] mesh:    {mesh_path}")
    print(f"[lift_uf] cache:   {seg_cache_dir}")
    print(f"[lift_uf] out:     {out_dir}")

    # ---- 1. Load target mesh.
    t0 = time.time()
    mesh = trimesh.load(mesh_path, force="mesh")
    n_verts = len(mesh.vertices)
    n_faces = len(mesh.faces)
    print(f"[lift_uf] mesh loaded: {n_verts:,} verts, {n_faces:,} faces ({time.time() - t0:.1f}s)")
    verts_np = np.asarray(mesh.vertices, dtype=np.float32)

    # ---- 2. Build the dataset adapter (poses, intrinsics, depth, seg_cache).
    # We re-use `HgeMaskClusteringDataset` because it already does NavVis frame
    # parsing + alignment + seg_cache reading — even though we're not running
    # MaskClustering, the per-frame I/O contract is identical.
    dataset = HgeMaskClusteringDataset(
        session_dir=session_dir,
        mesh_path=mesh_path,
        seg_cache_dir=seg_cache_dir,
        out_dir=out_dir,
        max_frames=max_frames,
    )
    frame_ids = dataset.get_frame_list(stride=1)
    print(f"[lift_uf] frames available: {len(frame_ids)}")

    # ---- 3. Stream Frame objects from the cache into lift_masks_to_3d.
    # One frame in memory at a time → peak RSS ~one frame's masks (~25 MB),
    # not 2 408× that.
    def _frame_stream():
        for fid in frame_ids:
            depth = dataset.get_depth(fid)
            inst_mask = dataset.get_segmentation(fid, align_with_depth=False)
            cls_mask = dataset.get_structural_class_map(fid, align_with_depth=False)
            T_world_cam = dataset.get_extrinsic(fid)
            intr = dataset.get_intrinsics(fid)
            K = np.asarray(intr.intrinsic_matrix, dtype=np.float64)
            yield Frame(
                frame_id=int(fid),
                depth=depth.astype(np.float32),
                instance_mask=inst_mask.astype(np.int32),
                class_mask=cls_mask.astype(np.int16),
                T_world_cam=T_world_cam,
                fx=float(K[0, 0]), fy=float(K[1, 1]),
                cx=float(K[0, 2]), cy=float(K[1, 2]),
            )

    # ---- 4. UF lift.
    t0 = time.time()
    lift_result: LiftResult = lift_masks_to_3d(
        verts_np,
        _frame_stream(),
        depth_tolerance=depth_tolerance_m,
        overlap_threshold=overlap_threshold,
    )
    t_lift = time.time() - t0
    n_labeled = int((lift_result.instance_ids > 0).sum())
    coverage = n_labeled / max(n_verts, 1)
    print(
        f"[lift_uf] UF lift: {lift_result.num_instances} instances, "
        f"{n_labeled:,}/{n_verts:,} verts ({coverage * 100:.2f}%) "
        f"({t_lift:.1f}s)"
    )

    # ---- 5. Build a `cluster_dict` shape compatible with majority_phrase_per_cluster.
    # The voter consumes `{cluster_id: {"point_ids": [...], "mask_list": [...]}}`,
    # which mirrors MaskClustering's output. We synthesise it from the
    # per-vertex instance assignments + the new instance_to_mask_list field.
    cluster_dict: dict[int, dict] = {}
    for iid, mask_pairs in lift_result.instance_to_mask_list.items():
        if iid == 0:
            continue  # 0 = unassigned
        # mask_list entries are (frame_id, mask_id, coverage); the voter
        # treats coverage as a weight. UF doesn't expose per-mask coverage
        # directly so we use unit weight. Identical to phrase_vote behaviour
        # on MC clusters where coverage is mostly close to 1.0 anyway.
        mask_list_with_coverage = [
            (int(fid), int(mid), 1.0) for fid, mid in mask_pairs
        ]
        cluster_dict[int(iid)] = {
            "point_ids": np.where(lift_result.instance_ids == iid)[0],
            "mask_list": mask_list_with_coverage,
            "repre_mask_list": mask_list_with_coverage[:5],
        }
    print(f"[lift_uf] synthesised {len(cluster_dict)} cluster_dict entries from UF instances")

    # ---- 6. Open-vocab phrase vote per cluster.
    t0 = time.time()
    cluster_phrases = majority_phrase_per_cluster(
        cluster_dict, dataset, fallback_phrase=fallback_phrase
    )
    t_vote = time.time() - t0
    phrase_hist = Counter(cluster_phrases.values())
    print(
        f"[lift_uf] phrase-vote: {len(cluster_phrases)} clusters, "
        f"{len(phrase_hist)} distinct phrases "
        f"(top 5: {[p for p, _ in phrase_hist.most_common(5)]}) "
        f"({t_vote:.1f}s)"
    )

    # ---- 7. Habitat export — open-vocab class taxonomy on the target mesh.
    t0 = time.time()
    export_stats = export_clusters_to_habitat_open_vocab(
        object_dict=cluster_dict,
        cluster_phrases=cluster_phrases,
        mesh_path=mesh_path,
        out_dir=out_dir,
        stem=stem,
        external_stage_glb=external_stage_glb,
        min_verts_per_instance=min_verts_per_instance,
        drop_phrases=drop_phrases,
        merge_clusters_with_same_phrase=merge_clusters_with_same_phrase,
    )
    t_export = time.time() - t0
    print(
        f"[lift_uf] habitat-export: {export_stats['num_clusters_emitted']}/"
        f"{export_stats['num_clusters_in']} clusters → "
        f"{export_stats['num_instances']} instances, "
        f"{export_stats['num_distinct_phrases']} classes "
        f"({t_export:.1f}s)"
    )

    summary = {
        "mesh_path": str(mesh_path),
        "out_dir": str(out_dir),
        "stem": stem,
        "n_verts": n_verts,
        "n_faces": n_faces,
        "n_frames": len(frame_ids),
        "depth_tolerance_m": depth_tolerance_m,
        "overlap_threshold": overlap_threshold,
        "uf_lift": {
            "num_instances": int(lift_result.num_instances),
            "labeled_verts": n_labeled,
            "coverage_fraction": round(coverage, 4),
            "time_s": round(t_lift, 2),
        },
        "phrase_vote": {
            "num_clusters": len(cluster_phrases),
            "num_distinct_phrases": len(phrase_hist),
            "phrase_histogram_top10": dict(phrase_hist.most_common(10)),
            "time_s": round(t_vote, 2),
        },
        "habitat_export": {**export_stats, "time_s": round(t_export, 2)},
        "total_time_s": round(time.time() - t_all, 2),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(
        f"[lift_uf] done in {summary['total_time_s']:.1f}s — "
        f"summary → {summary_path}"
    )
    return summary


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--seg-cache-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--stem", default="HGE")
    p.add_argument(
        "--external-stage-glb", type=Path, default=None,
        help="optional photorealistic stage GLB; pass --mesh's path here when "
             "labeling the basis directly so the rendered + semantic meshes match.",
    )
    p.add_argument("--depth-tolerance", type=float, default=0.15)
    p.add_argument(
        "--overlap-threshold", type=float, default=0.2,
        help="UF cross-frame merge threshold (fraction of smaller mask). "
             "Default 0.2 (lowered from upstream 0.5 because basis-mesh "
             "vertex density makes per-viewpoint projections of the same "
             "landmark share <50%% of verts).",
    )
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument(
        "--drop-phrases", nargs="*", default=[],
        help="open-vocab phrases to skip emitting (e.g. --drop-phrases clutter unknown).",
    )
    p.add_argument("--min-verts-per-instance", type=int, default=20)
    p.add_argument(
        "--no-merge-same-phrase", action="store_true",
        help="Disable the default 'collapse all clusters with the same phrase "
             "into one Habitat instance' rule. Use only if you specifically "
             "want N rainbow-coloured Habitat instances per phrase (e.g. "
             "for debugging UF cluster fragmentation).",
    )
    args = p.parse_args()

    run(
        session_dir=args.session,
        mesh_path=args.mesh,
        seg_cache_dir=args.seg_cache_dir,
        out_dir=args.out_dir,
        stem=args.stem,
        external_stage_glb=args.external_stage_glb,
        depth_tolerance_m=args.depth_tolerance,
        overlap_threshold=args.overlap_threshold,
        max_frames=args.max_frames,
        drop_phrases=tuple(args.drop_phrases),
        min_verts_per_instance=args.min_verts_per_instance,
        merge_clusters_with_same_phrase=not args.no_merge_same_phrase,
    )


if __name__ == "__main__":
    _main()
