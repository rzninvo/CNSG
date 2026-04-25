"""Unit tests for the MaskClustering post-filter helpers.

Targets the three pure-Python functions added on top of upstream's
post_process pipeline:

  - `split_oversized_clusters_3d` — re-splits >5m clusters via 3D DBSCAN
    and re-assigns each supporting mask to whichever sub-cluster contains
    the majority of its verts.
  - `majority_class_per_cluster` — votes an S3DIS class per cluster by
    aggregating the cached per-pixel structural class mask.
  - `export_clusters_to_habitat` — writes the HM3D semantic bundle so
    each MC cluster becomes its own Habitat semantic instance.

These do NOT exercise the upstream MaskClustering graph code (which needs
GPU + the vendored clone). The contracts checked here are the ones our
build relies on after MaskClustering returns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from cnsg.segmentation.lift_maskclustering import (
    export_clusters_to_habitat,
    majority_class_per_cluster,
    split_oversized_clusters_3d,
)


# ----- shared fixtures -------------------------------------------------------


def _two_columns_one_chair(n_per_blob: int = 60) -> tuple[np.ndarray, dict, dict]:
    """Build scene_points + cluster_dict + mask_point_clouds for a scene with:
      - one big "row of columns" cluster (verts spanning 8m, two blobs 6m apart)
      - one well-formed "chair" cluster (verts inside 0.8m bbox)

    The first cluster is the over-masking failure case the splitter targets:
    SAM 3 labels both columns with mask_id=1 across two frames, so MC merges
    them; we want the splitter to recover two separate clusters.
    """
    rng = np.random.default_rng(0)

    col_a = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.2, size=(n_per_blob, 3))
    col_b = rng.normal(loc=[7.0, 0.0, 0.0], scale=0.2, size=(n_per_blob, 3))
    chair = rng.normal(loc=[3.5, 5.0, 0.0], scale=0.25, size=(n_per_blob, 3))

    scene_points = np.vstack([col_a, col_b, chair]).astype(np.float64)
    n = n_per_blob

    col_a_ids = list(range(0, n))
    col_b_ids = list(range(n, 2 * n))
    chair_ids = list(range(2 * n, 3 * n))

    object_dict = {
        # Row-of-columns over-mask: one MC cluster spanning both physical columns.
        0: {
            "point_ids": col_a_ids + col_b_ids,
            "mask_list": [(10, 1, 0.8), (11, 1, 0.7), (12, 1, 0.6)],
            "repre_mask_list": [(10, 1, 0.8), (11, 1, 0.7), (12, 1, 0.6)],
        },
        # Compact, well-formed chair cluster — must NOT be split.
        1: {
            "point_ids": chair_ids,
            "mask_list": [(10, 4, 0.9), (11, 4, 0.85)],
            "repre_mask_list": [(10, 4, 0.9), (11, 4, 0.85)],
        },
    }

    # Per-frame masks: column-frames assign half their pixels to col_a verts,
    # half to col_b verts (simulating a single SAM 3 mask covering both).
    # Chair frames map to chair verts.
    mask_point_clouds: dict = {}
    for frame_id in (10, 11, 12):
        mask_point_clouds[f"{frame_id}_1"] = set(col_a_ids + col_b_ids)
    for frame_id in (10, 11):
        mask_point_clouds[f"{frame_id}_4"] = set(chair_ids)

    return scene_points, object_dict, mask_point_clouds


# ----- split_oversized_clusters_3d ------------------------------------------


def test_split_breaks_two_columns_into_two() -> None:
    scene_points, object_dict, mpc = _two_columns_one_blob = _two_columns_one_chair()
    new_dict, stats = split_oversized_clusters_3d(
        object_dict, scene_points, mpc,
        max_extent_m=5.0, eps_m=0.5, min_cc_verts=20, min_masks_per_subcluster=1,
    )
    # The 7m row should split into 2 sub-clusters; chair should remain.
    assert stats["n_input"] == 2
    assert stats["n_split"] == 1
    assert stats["n_output"] == 3
    # Every output cluster must fit inside max_extent_m.
    for c in new_dict.values():
        pts = scene_points[list(c["point_ids"])]
        ext_max = float((pts.max(0) - pts.min(0)).max())
        assert ext_max <= 5.0


def test_split_preserves_chair_when_extent_below_threshold() -> None:
    scene_points, object_dict, mpc = _two_columns_one_chair()
    new_dict, _ = split_oversized_clusters_3d(
        object_dict, scene_points, mpc,
        max_extent_m=5.0, eps_m=0.5, min_cc_verts=20, min_masks_per_subcluster=1,
    )
    # One of the output entries must keep the chair's mask_list intact.
    chair_keys = [
        cid for cid, c in new_dict.items()
        if any(m[1] == 4 for m in c["mask_list"])
    ]
    assert len(chair_keys) == 1
    chair_cluster = new_dict[chair_keys[0]]
    chair_masks = sorted(m[:2] for m in chair_cluster["mask_list"])
    assert chair_masks == [(10, 4), (11, 4)]


def test_split_dispatches_overmask_to_both_subclusters() -> None:
    """A SAM 3 mask that genuinely covers both physical columns is evidence
    for BOTH sub-clusters. Winner-takes-all would starve the loser; we
    instead dispatch to every sub-cluster the mask materially overlaps
    (≥ mask_intersection_fraction)."""
    scene_points, object_dict, mpc = _two_columns_one_chair()
    new_dict, _ = split_oversized_clusters_3d(
        object_dict, scene_points, mpc,
        max_extent_m=5.0, eps_m=0.5, min_cc_verts=20, min_masks_per_subcluster=1,
        mask_intersection_fraction=0.2,
    )
    # The 3 parent column-masks split ~50/50 across the two columns, so
    # both sub-clusters should receive all 3 frames.
    column_subclusters = [
        c for c in new_dict.values()
        if any(m[1] == 1 for m in c["mask_list"])
    ]
    assert len(column_subclusters) == 2
    for c in column_subclusters:
        mask_id_1_entries = [m for m in c["mask_list"] if m[1] == 1]
        assert len(mask_id_1_entries) == 3


def test_split_drops_subclusters_below_min_masks() -> None:
    scene_points, object_dict, mpc = _two_columns_one_chair()
    # min_masks_per_subcluster=10 should starve every column sub-cluster
    # (both got 3 masks via the over-mask dispatch, which is < 10).
    new_dict, stats = split_oversized_clusters_3d(
        object_dict, scene_points, mpc,
        max_extent_m=5.0, eps_m=0.5, min_cc_verts=20, min_masks_per_subcluster=10,
    )
    # The chair survives (only 2 masks, but stays unsplit so its mask_list
    # never goes through the threshold). Both column sub-clusters die.
    assert stats["n_dropped_small"] == 2
    assert stats["n_output"] == 1


def test_split_keeps_unsplittable_oversized_cluster() -> None:
    """If DBSCAN can't carve the over-sized cluster (eps too small), keep
    the original rather than dropping it silently."""
    scene_points, object_dict, mpc = _two_columns_one_chair()
    # eps=0.01 forbids any 0.5m-spaced points from joining → DBSCAN labels
    # almost everything as noise. We expect the cluster preserved as a fallback.
    new_dict, stats = split_oversized_clusters_3d(
        object_dict, scene_points, mpc,
        max_extent_m=5.0, eps_m=0.01, min_cc_verts=20, min_masks_per_subcluster=1,
    )
    assert stats["n_output"] == 2  # nothing emitted from a failed split
    assert stats["n_split"] == 0


def test_split_emits_extent_stats() -> None:
    scene_points, object_dict, mpc = _two_columns_one_chair()
    _, stats = split_oversized_clusters_3d(
        object_dict, scene_points, mpc,
        max_extent_m=5.0, eps_m=0.5,
    )
    assert stats["extent_before_max_m"] >= 5.0  # input had a 7m cluster
    assert stats["extent_after_max_m"] < 5.0    # all outputs under threshold


# ----- majority_class_per_cluster -------------------------------------------


class _StubDataset:
    """Minimal HgeMaskClusteringDataset stand-in for the class-voter test.

    Stores per-frame instance + structural masks, exposes them through the
    same API the voter calls.
    """

    def __init__(self, frame_masks: dict[int, tuple[np.ndarray, np.ndarray]]):
        self.frame_masks = frame_masks

    def get_segmentation(self, frame_id: int, align_with_depth: bool = False):
        return self.frame_masks[frame_id][0]

    def get_structural_class_map(self, frame_id: int, align_with_depth: bool = False):
        return self.frame_masks[frame_id][1]


def _stub_frame(mask_id: int, class_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a 32x32 (instance_mask, class_mask) pair where the
    `mask_id` region holds the given S3DIS class label."""
    inst = np.zeros((32, 32), dtype=np.int32)
    cls = np.zeros((32, 32), dtype=np.int16)
    inst[8:24, 8:24] = mask_id
    cls[8:24, 8:24] = class_id
    return inst, cls


def test_majority_class_picks_dominant_class() -> None:
    # Cluster supported by 3 frames: 2 vote class=5 (column), 1 votes class=9 (chair).
    frames = {
        100: _stub_frame(mask_id=7, class_id=5),
        101: _stub_frame(mask_id=7, class_id=5),
        102: _stub_frame(mask_id=7, class_id=9),
    }
    dataset = _StubDataset(frames)
    object_dict = {
        0: {
            "point_ids": [0, 1],
            "mask_list": [(100, 7, 0.9), (101, 7, 0.85), (102, 7, 0.8)],
            "repre_mask_list": [],
        }
    }
    classes = majority_class_per_cluster(object_dict, dataset)
    assert classes[0] == 5


def test_majority_class_returns_zero_for_unlabeled_cluster() -> None:
    # Cluster's mask_id doesn't appear in any frame's instance_mask → no votes.
    frames = {200: _stub_frame(mask_id=7, class_id=5)}
    dataset = _StubDataset(frames)
    object_dict = {
        0: {
            "point_ids": [0],
            "mask_list": [(200, 9, 1.0)],  # mask_id=9 not present in frame 200
            "repre_mask_list": [],
        }
    }
    classes = majority_class_per_cluster(object_dict, dataset)
    assert classes[0] == 0


def test_majority_class_ignores_background_class() -> None:
    # Mask region pixels are MOSTLY class=0 (background) with a few class=7 pixels.
    inst = np.zeros((32, 32), dtype=np.int32)
    cls = np.zeros((32, 32), dtype=np.int16)
    inst[0:32, 0:32] = 5
    cls[0:2, 0:2] = 7  # 4 pixels of chair
    # The other 1020 pixels are class=0 — voter must ignore them.
    dataset = _StubDataset({500: (inst, cls)})
    object_dict = {
        0: {
            "point_ids": [0],
            "mask_list": [(500, 5, 1.0)],
            "repre_mask_list": [],
        }
    }
    classes = majority_class_per_cluster(object_dict, dataset)
    assert classes[0] == 7


def test_majority_class_caches_frames_across_clusters() -> None:
    """Same frame cited by two clusters must only be loaded once."""
    load_log: list[int] = []

    class _LoggingDataset(_StubDataset):
        def get_segmentation(self, frame_id: int, align_with_depth: bool = False):
            load_log.append(frame_id)
            return super().get_segmentation(frame_id, align_with_depth)

    dataset = _LoggingDataset({
        100: _stub_frame(mask_id=1, class_id=5),
        101: _stub_frame(mask_id=2, class_id=9),
    })
    object_dict = {
        0: {"point_ids": [0], "mask_list": [(100, 1, 1.0)], "repre_mask_list": []},
        1: {"point_ids": [1], "mask_list": [(100, 1, 1.0), (101, 2, 1.0)], "repre_mask_list": []},
    }
    majority_class_per_cluster(object_dict, dataset)
    # Frame 100 appears in both clusters but the cache should only call once.
    assert load_log.count(100) == 1


# ----- export_clusters_to_habitat -------------------------------------------


def test_export_writes_one_instance_per_cluster(tmp_path: Path) -> None:
    """Each cluster with a non-zero class and ≥ min_verts_per_instance verts
    should produce exactly one entry in `<stem>.semantic.txt`."""
    # 200-vert cube mesh — plenty of verts to absorb cluster point_ids.
    mesh = trimesh.creation.icosphere(subdivisions=3)
    n_verts = len(mesh.vertices)
    assert n_verts >= 100
    mesh_path = tmp_path / "test_mesh.glb"
    mesh.export(str(mesh_path))

    object_dict = {
        0: {  # column cluster (class 5, 30 verts)
            "point_ids": list(range(0, 30)),
            "mask_list": [(0, 1, 1.0)],
            "repre_mask_list": [],
        },
        1: {  # chair cluster (class 9, 25 verts) — below 20 vert default keeps it
            "point_ids": list(range(30, 55)),
            "mask_list": [(0, 2, 1.0)],
            "repre_mask_list": [],
        },
        2: {  # tiny cluster — must be dropped by min_verts_per_instance
            "point_ids": list(range(55, 60)),  # only 5 verts
            "mask_list": [(0, 3, 1.0)],
            "repre_mask_list": [],
        },
    }
    cluster_classes = {0: 5, 1: 9, 2: 7}

    stats = export_clusters_to_habitat(
        object_dict=object_dict,
        cluster_classes=cluster_classes,
        mesh_path=mesh_path,
        out_dir=tmp_path,
        stem="MCEXPORT",
        min_verts_per_instance=20,
    )
    assert stats["num_clusters_in"] == 3
    assert stats["num_clusters_emitted"] == 2  # cluster 2 dropped (5 < 20)
    assert stats["num_instances"] == 2

    semantic_txt = (tmp_path / "MCEXPORT.semantic.txt").read_text()
    # Header line + one row per emitted instance.
    rows = [r for r in semantic_txt.splitlines() if r and not r.startswith("HM3D")]
    assert len(rows) == 2
    # Row format: "<idx>,<hex>,\"<class_name>\",<region>"
    assert any('"column"' in r for r in rows)
    assert any('"chair"' in r for r in rows)


def test_export_skips_unlabeled_clusters(tmp_path: Path) -> None:
    """Clusters voted to background (class 0) must NOT appear as instances."""
    mesh = trimesh.creation.icosphere(subdivisions=3)
    mesh_path = tmp_path / "test_mesh.glb"
    mesh.export(str(mesh_path))

    object_dict = {
        0: {"point_ids": list(range(0, 30)), "mask_list": [(0, 1, 1.0)], "repre_mask_list": []},
        1: {"point_ids": list(range(30, 60)), "mask_list": [(0, 2, 1.0)], "repre_mask_list": []},
    }
    cluster_classes = {0: 5, 1: 0}  # cluster 1 voted to "unknown"

    stats = export_clusters_to_habitat(
        object_dict=object_dict,
        cluster_classes=cluster_classes,
        mesh_path=mesh_path,
        out_dir=tmp_path,
        stem="UNLAB",
        min_verts_per_instance=20,
    )
    assert stats["num_clusters_emitted"] == 1
    assert stats["num_instances"] == 1
