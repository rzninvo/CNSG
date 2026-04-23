"""Unit test for `Localizer`'s alignment-compose path.

Phase 3c follow-up to `02_hge-lift-frame-mismatch`: if the caller passes an
`alignment_global.txt`, every `LocalizationResult` must have its pose expressed
in the absolute frame (what Habitat loads), not the COLMAP/pose-graph frame
that hloc natively returns.

This test stubs hloc's `Localizer._parse_results_file` path end-to-end with a
synthetic `results.txt` so we don't need a real reconstruction or GPU — we're
testing ONE thing only: the frame-compose arithmetic.

Regression target: if someone swaps `T_abs_src @ T_src_cam` for the inverse
direction or right-multiplies instead of left, this test catches it.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def _write_alignment(path: Path, R_abs_pg: np.ndarray, t_abs_pg: np.ndarray) -> None:
    q_xyzw = Rotation.from_matrix(R_abs_pg).as_quat()
    qw = q_xyzw[3]
    qx, qy, qz = q_xyzw[0], q_xyzw[1], q_xyzw[2]
    tx, ty, tz = t_abs_pg
    path.write_text(
        "# label, reference_id, qw, qx, qy, qz, tx, ty, tz\n"
        f"pose_graph_optimized, __absolute__, {qw}, {qx}, {qy}, {qz}, {tx}, {ty}, {tz}\n"
    )


def _write_hloc_results(path: Path, query_key: str, cam_from_world_wxyz: np.ndarray, t_cw: np.ndarray) -> None:
    qw, qx, qy, qz = cam_from_world_wxyz
    tx, ty, tz = t_cw
    path.write_text(f"{query_key} {qw} {qx} {qy} {qz} {tx} {ty} {tz}\n")


def _write_hloc_logs(results_path: Path, query_key: str, num_inliers: int) -> None:
    pkl_path = Path(f"{results_path}_logs.pkl")
    with pkl_path.open("wb") as f:
        pickle.dump(
            {"loc": {query_key: {"PnP_ret": {"num_inliers": num_inliers}, "keypoint_index_to_db": (list(range(50)), None)}}},
            f,
        )


def test_compose_applies_left_multiply_to_localized_pose(tmp_path: Path) -> None:
    """Given a known T_abs_src alignment and a hloc result in the src frame,
    the returned LocalizationResult must encode T_abs_cam = T_abs_src @ T_src_cam.
    """
    try:
        from cnsg.localization.inference import Localizer
    except Exception as exc:
        pytest.skip(f"hloc not importable in this env: {exc}")

    # Construct the Localizer with the synthetic map dirs it expects. We
    # bypass __init__ because we don't need a real pycolmap.Reconstruction —
    # only the `_parse_results_file` method is exercised here.
    loc = Localizer.__new__(Localizer)
    # Known alignment: 30° rotation around z + big translation.
    rng = np.random.default_rng(123)
    R_abs_pg = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    t_abs_pg = np.array([42.0, -7.0, 0.5])

    align_path = tmp_path / "alignment_global.txt"
    _write_alignment(align_path, R_abs_pg, t_abs_pg)
    from cnsg.localization.capture_io import parse_alignment_global
    loc._T_abs_src = parse_alignment_global(align_path)
    assert loc._T_abs_src is not None

    # Settings stub with the min_inliers threshold the parse path checks.
    class _S:
        ransac_min_inliers = 20
    loc._settings = _S()

    # Known src-frame camera pose: T_src_cam with random R, known t.
    R_src_cam = Rotation.random(random_state=rng).as_matrix()
    q_src_cam_xyzw = Rotation.from_matrix(R_src_cam).as_quat()
    t_src_cam = np.array([3.0, -2.0, 1.0])

    # hloc writes (qw, qx, qy, qz) for CAM-FROM-WORLD — invert our src-frame world_from_cam.
    R_cw = R_src_cam.T
    t_cw = -R_cw @ t_src_cam
    q_cw_xyzw = Rotation.from_matrix(R_cw).as_quat()
    q_cw_wxyz = np.array([q_cw_xyzw[3], q_cw_xyzw[0], q_cw_xyzw[1], q_cw_xyzw[2]])

    results_path = tmp_path / "results.txt"
    _write_hloc_results(results_path, "query.jpg", q_cw_wxyz, t_cw)
    _write_hloc_logs(results_path, "query.jpg", num_inliers=50)

    out = loc._parse_results_file(results_path, "query.jpg", tmp_path)

    # Expected: T_abs_cam = T_abs_src @ T_src_cam
    expected_R = R_abs_pg @ R_src_cam
    expected_t = R_abs_pg @ t_src_cam + t_abs_pg

    got_R = Rotation.from_quat(
        [out.rotation_wxyz[1], out.rotation_wxyz[2], out.rotation_wxyz[3], out.rotation_wxyz[0]]
    ).as_matrix()
    np.testing.assert_allclose(got_R, expected_R, atol=1e-6)
    np.testing.assert_allclose(out.position, expected_t, atol=1e-6)
    assert out.success is True


def test_compose_is_noop_when_alignment_missing(tmp_path: Path) -> None:
    """Without an alignment path, poses come back in the native (src) frame."""
    try:
        from cnsg.localization.inference import Localizer
    except Exception as exc:
        pytest.skip(f"hloc not importable in this env: {exc}")

    loc = Localizer.__new__(Localizer)
    loc._T_abs_src = None

    class _S:
        ransac_min_inliers = 20
    loc._settings = _S()

    # Identity cam-from-world → identity world-from-cam.
    results_path = tmp_path / "results.txt"
    _write_hloc_results(results_path, "q.jpg", np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3))
    _write_hloc_logs(results_path, "q.jpg", num_inliers=30)

    out = loc._parse_results_file(results_path, "q.jpg", tmp_path)
    np.testing.assert_allclose(out.position, np.zeros(3), atol=1e-9)
    np.testing.assert_allclose(out.rotation_wxyz, [1.0, 0.0, 0.0, 0.0], atol=1e-9)
