"""Unit tests for HgeMaskClusteringDataset.

We can't smoke-test against the real MaskClustering pipeline here (it wants
a torch+GPU environment and a populated seg_cache), but we CAN pin the
contract the adapter promises: given a synthetic NavVis session on disk,
the dataset exposes the same call shape as ScanNetDataset and returns
well-formed objects.

Regression targets:
  - intrinsics honour the (width, height, fx, fy, cx, cy) from sensors.txt
  - extrinsic is a 4×4 float64 in the ABSOLUTE frame (alignment applied)
  - depth is metres, float32, shape matches the PNG on disk
  - get_segmentation reads from seg_cache and honours `align_with_depth`
  - get_frame_list + get_scene_points don't crash on an empty-frames setup
  - missing seg_cache entry → FileNotFoundError with an actionable message
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from cnsg.segmentation.maskclustering_adapter import HgeMaskClusteringDataset


# ----- synthetic NavVis session fixtures ------------------------------------


def _make_synthetic_session(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write a 2-frame NavVis session + mesh + seg_cache to tmp_path."""
    session = tmp_path / "session"
    session.mkdir()
    (session / "raw_data").mkdir()
    (session / "depth_maps").mkdir()
    (session / "proc").mkdir()

    # Sensors file — one center camera at 640x480 PINHOLE.
    # Real format (from mesh_pipeline/data/navvis_.../sensors.txt):
    # `sensor_id, name, camera, MODEL, W, H, fx, fy, cx, cy`
    (session / "sensors.txt").write_text(
        "# sensor_id, name, sensor_type, [sensor_params]+\n"
        "cam0_center, synthetic center cam, camera, PINHOLE, 640, 480, 400.0, 400.0, 320.0, 240.0\n"
    )

    # Trajectories — two identity-pose frames 1 ms apart.
    lines = ["# timestamp, id, qw, qx, qy, qz, tx, ty, tz\n"]
    for i, ts in enumerate([1000, 2000]):
        lines.append(f"{ts}, cam0_center, 1.0, 0.0, 0.0, 0.0, {i * 0.1}, 0.0, 0.0\n")
    (session / "trajectories.txt").write_text("".join(lines))

    # Images — two jpg stubs (empty but openable).
    raw = session / "raw_data"
    img_lines = ["# timestamp, id, relative_path\n"]
    for i, ts in enumerate([1000, 2000]):
        rel = f"frame_{i:03d}.jpg"
        cv2.imwrite(str(raw / rel), np.zeros((480, 640, 3), dtype=np.uint8))
        img_lines.append(f"{ts}, cam0_center, {rel}\n")
    (session / "images.txt").write_text("".join(img_lines))

    # Depth PNGs — uint16 mm; write at 640×480.
    dep = session / "depth_maps"
    for i in range(2):
        depth_u16 = np.full((480, 640), 2000, dtype=np.uint16)  # 2 m everywhere
        cv2.imwrite(str(dep / f"frame_{i:03d}.png"), depth_u16)

    # alignment_global.txt — identity transform (simplifies the test).
    (session / "proc" / "alignment_global.txt").write_text(
        "# label, reference_id, qw, qx, qy, qz, tx, ty, tz\n"
        "pose_graph_optimized, __absolute__, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0\n"
    )

    # Mesh — a unit cube.
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    mesh_path = tmp_path / "mesh.ply"
    mesh.export(str(mesh_path))

    # seg_cache — one npz per frame with int32 mask + int16 class.
    seg_cache = tmp_path / "seg_cache"
    seg_cache.mkdir()
    for i in range(2):
        mask = np.zeros((480, 640), dtype=np.int32)
        mask[100:200, 200:400] = 1001  # "wall" structural offset
        mask[250:350, 150:300] = 7    # SAM 3 chair instance
        cls = np.zeros((480, 640), dtype=np.int16)
        cls[100:200, 200:400] = 1  # wall
        cls[250:350, 150:300] = 7  # chair
        np.savez_compressed(
            seg_cache / f"frame_{i:06d}.tmp.npz",
            instance_mask=mask,
            class_mask=cls,
            config_hash=np.array("test"),
        )
        # Rename to drop the .tmp suffix our atomic write uses.
        (seg_cache / f"frame_{i:06d}.tmp.npz").rename(
            seg_cache / f"frame_{i:06d}.npz"
        )

    return session, mesh_path, seg_cache


# ----- tests ----------------------------------------------------------------


def test_frame_list_enumerates_all_frames(tmp_path: Path) -> None:
    session, mesh, cache = _make_synthetic_session(tmp_path)
    ds = HgeMaskClusteringDataset(session, mesh, cache)
    assert ds.get_frame_list(stride=1) == [0, 1]
    assert ds.get_frame_list(stride=2) == [0]


def test_intrinsics_match_sensors_txt(tmp_path: Path) -> None:
    session, mesh, cache = _make_synthetic_session(tmp_path)
    ds = HgeMaskClusteringDataset(session, mesh, cache)
    k = ds.get_intrinsics(0)
    m = np.asarray(k.intrinsic_matrix)
    assert m[0, 0] == 400.0 and m[1, 1] == 400.0
    assert m[0, 2] == 320.0 and m[1, 2] == 240.0
    assert k.width == 640 and k.height == 480


def test_extrinsic_shape_and_translation(tmp_path: Path) -> None:
    """With identity rotation + identity alignment, extrinsic.t matches
    trajectories.txt translation exactly (frame 0: (0,0,0), frame 1: (0.1,0,0))."""
    session, mesh, cache = _make_synthetic_session(tmp_path)
    ds = HgeMaskClusteringDataset(session, mesh, cache)
    e0 = ds.get_extrinsic(0)
    e1 = ds.get_extrinsic(1)
    assert e0.shape == (4, 4) and e0.dtype == np.float64
    np.testing.assert_allclose(e0[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(e0[:3, 3], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(e1[:3, 3], [0.1, 0.0, 0.0], atol=1e-12)
    np.testing.assert_array_equal(e0[3], [0, 0, 0, 1])


def test_depth_loaded_as_metres_float32(tmp_path: Path) -> None:
    session, mesh, cache = _make_synthetic_session(tmp_path)
    ds = HgeMaskClusteringDataset(session, mesh, cache)
    d = ds.get_depth(0)
    assert d.dtype == np.float32
    assert d.shape == (480, 640)
    # we wrote 2000 mm → 2.0 m
    np.testing.assert_allclose(d, 2.0, atol=1e-3)


def test_segmentation_filters_structural_offset_by_default(tmp_path: Path) -> None:
    """Default (`instance_only=True`) zeroes out IDs ≥ 1000 (structural-class
    offset our combiner uses) so MaskClustering only sees SAM 3 instance IDs.
    """
    session, mesh, cache = _make_synthetic_session(tmp_path)
    ds = HgeMaskClusteringDataset(session, mesh, cache)
    m = ds.get_segmentation(0, align_with_depth=False)
    assert m.dtype == np.int32
    assert m.shape == (480, 640)
    assert 1001 not in np.unique(m), "structural offset must be filtered by default"
    assert 7 in np.unique(m), "SAM 3 instance ids must be preserved"
    assert 0 in np.unique(m), "structural pixels must become background"


def test_segmentation_keeps_structural_when_instance_only_false(tmp_path: Path) -> None:
    """Explicit `instance_only=False` keeps the original combined stream
    for compatibility / debugging."""
    session, mesh, cache = _make_synthetic_session(tmp_path)
    ds = HgeMaskClusteringDataset(session, mesh, cache, instance_only=False)
    m = ds.get_segmentation(0)
    assert 1001 in np.unique(m)
    assert 7 in np.unique(m)


def test_structural_class_map_returned_separately(tmp_path: Path) -> None:
    """`get_structural_class_map` exposes the class_mask the build wrote
    next to instance_mask; class IDs are S3DIS, independent of SAM 3."""
    session, mesh, cache = _make_synthetic_session(tmp_path)
    ds = HgeMaskClusteringDataset(session, mesh, cache)
    cls = ds.get_structural_class_map(0)
    assert cls.dtype == np.int16
    assert cls.shape == (480, 640)
    # The fixture writes wall (S3DIS class 1) and chair (S3DIS class 7).
    assert 1 in np.unique(cls)
    assert 7 in np.unique(cls)


def test_segmentation_resizes_to_depth_when_requested(tmp_path: Path) -> None:
    """Our synthetic seg_cache has mask already at depth resolution, but
    exercising align_with_depth=True should still return a (H, W) == depth
    shape without errors."""
    session, mesh, cache = _make_synthetic_session(tmp_path)
    ds = HgeMaskClusteringDataset(session, mesh, cache)
    m = ds.get_segmentation(0, align_with_depth=True)
    assert m.shape == (480, 640)


def test_scene_points_from_mesh_shape_n3(tmp_path: Path) -> None:
    session, mesh, cache = _make_synthetic_session(tmp_path)
    ds = HgeMaskClusteringDataset(session, mesh, cache)
    pts = ds.get_scene_points()
    assert pts.ndim == 2 and pts.shape[1] == 3
    # unit cube has 8 verts before trimesh processing (may dedupe to 8)
    assert len(pts) >= 8


def test_missing_seg_cache_errors_loudly(tmp_path: Path) -> None:
    session, mesh, cache = _make_synthetic_session(tmp_path)
    ds = HgeMaskClusteringDataset(session, mesh, cache)
    # frame 5 has no seg_cache entry
    with pytest.raises(FileNotFoundError, match="seg_cache miss"):
        ds.get_segmentation(5)


def test_get_label_id_returns_s3dis_vocabulary(tmp_path: Path) -> None:
    session, mesh, cache = _make_synthetic_session(tmp_path)
    ds = HgeMaskClusteringDataset(session, mesh, cache)
    label2id, id2label = ds.get_label_id()
    assert "wall" in label2id and "floor" in label2id and "ceiling" in label2id
    assert id2label[label2id["wall"]] == "wall"


def test_max_frames_caps_frame_table(tmp_path: Path) -> None:
    session, mesh, cache = _make_synthetic_session(tmp_path)
    ds = HgeMaskClusteringDataset(session, mesh, cache, max_frames=1)
    assert ds.get_frame_list() == [0]
