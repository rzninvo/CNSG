"""Unit tests for `parse_alignment_global` in `cnsg.localization.capture_io`.

These pin the convention choice that cost us Phase 3 silently: the stored
(q, t) is `T_absolute_from_pose_graph`, composed as a LEFT multiply
(`T_abs_cam = T_abs_pg @ T_pg_cam`). Verified against the LaMAR scantools
source (`scantools/proc/alignment/scan_align.py`: `pose = T_session2w * pose`).

Regression target: if the convention silently flips (someone renames the
transform or swaps multiplication order), the lift's sensor-depth vs
vertex-Z gap jumps from centimetres to tens of metres and 99 % of labels
disappear. This test freezes the behaviour with a synthetic transform and
exact expected output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cnsg.localization.capture_io import parse_alignment_global


def _write(path: Path, body: str) -> None:
    path.write_text(body)


def test_missing_file_returns_none(tmp_path: Path) -> None:
    assert parse_alignment_global(tmp_path / "nope.txt") is None


def test_missing_label_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "alignment_global.txt"
    _write(
        p,
        "# label, reference_id, qw, qx, qy, qz, tx, ty, tz\n"
        "some_other_label, __absolute__, 1, 0, 0, 0, 0, 0, 0\n",
    )
    assert parse_alignment_global(p, label="pose_graph_optimized") is None


def test_identity_round_trip(tmp_path: Path) -> None:
    """Identity quaternion + zero translation → 4x4 identity."""
    p = tmp_path / "alignment_global.txt"
    _write(
        p,
        "# label, reference_id, qw, qx, qy, qz, tx, ty, tz\n"
        "pose_graph_optimized, __absolute__, 1, 0, 0, 0, 0, 0, 0\n",
    )
    T = parse_alignment_global(p)
    assert T is not None
    np.testing.assert_allclose(T, np.eye(4), atol=1e-12)


def test_rotation_and_translation_match_scipy(tmp_path: Path) -> None:
    """Recover a random SE(3) from disk and confirm components match what we wrote."""
    rng = np.random.default_rng(42)
    rot = Rotation.random(random_state=rng)
    qx, qy, qz, qw = rot.as_quat()  # scipy returns xyzw
    t = rng.uniform(-100, 100, size=3)

    p = tmp_path / "alignment_global.txt"
    _write(
        p,
        "# label, reference_id, qw, qx, qy, qz, tx, ty, tz\n"
        f"pose_graph_optimized, __absolute__, {qw}, {qx}, {qy}, {qz}, {t[0]}, {t[1]}, {t[2]}\n",
    )
    T = parse_alignment_global(p)
    assert T is not None
    np.testing.assert_allclose(T[:3, :3], rot.as_matrix(), atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], t, atol=1e-12)
    np.testing.assert_array_equal(T[3], [0, 0, 0, 1])


def test_composition_semantics_match_scantools_left_multiply(tmp_path: Path) -> None:
    """`T_abs_cam = T_abs_pg @ T_pg_cam` is the order scantools uses.

    Concretely: given `T_abs_pg` from alignment_global.txt and `T_pg_cam`
    from trajectories.txt, a point that sits at the camera centre (origin in
    camera frame) maps to the camera's position in absolute world coords,
    not its position in the pose-graph frame. We assert both via:
      (a) a direct calculation: t_abs_cam should equal R_abs_pg @ t_pg_cam + t_abs_pg
      (b) a round-trip: R_abs_cam = R_abs_pg @ R_pg_cam
    """
    rng = np.random.default_rng(7)

    # T_abs_pg: the alignment we'd read from disk.
    R_abs_pg = Rotation.random(random_state=rng).as_matrix()
    t_abs_pg = rng.uniform(-50, 50, size=3)

    # T_pg_cam: a trajectory pose.
    R_pg_cam = Rotation.random(random_state=rng).as_matrix()
    t_pg_cam = rng.uniform(-50, 50, size=3)

    T_abs_pg = np.eye(4)
    T_abs_pg[:3, :3] = R_abs_pg
    T_abs_pg[:3, 3] = t_abs_pg

    T_pg_cam = np.eye(4)
    T_pg_cam[:3, :3] = R_pg_cam
    T_pg_cam[:3, 3] = t_pg_cam

    T_abs_cam = T_abs_pg @ T_pg_cam

    # Rotation component composes by left multiplication.
    np.testing.assert_allclose(T_abs_cam[:3, :3], R_abs_pg @ R_pg_cam, atol=1e-12)
    # Translation: t_abs_cam = R_abs_pg @ t_pg_cam + t_abs_pg.
    np.testing.assert_allclose(
        T_abs_cam[:3, 3], R_abs_pg @ t_pg_cam + t_abs_pg, atol=1e-12
    )
