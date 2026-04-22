"""Phase-1 smoke test for cnsg.localization.Localizer.

Exit criterion (from docs/report/01_architecture-lean-migration/plan.md §Phase 1):
  > `pytest tests/test_localization_smoke.py` passes on a saved query image;
  > pose is within 20 cm / 2° of the ground-truth pose.

Strategy:
  - Pick one of the NavVis map images as the query. Because this image is
    already registered in the COLMAP reconstruction, localization should
    recover the reconstruction's own pose for it to sub-cm precision.
  - Ground truth comes from the COLMAP reconstruction, NOT from NavVis's
    `trajectories.txt`. The two are in different world frames (the
    session-relative NavVis frame and COLMAP's internal frame differ by
    the transform in `proc/alignment_global.txt`). The Localizer returns
    poses in COLMAP's frame; downstream Habitat consumers handle frame
    conversion separately (Phase 2 concern).
  - A 20 cm / 2° gate catches catastrophic regressions (inverted quaternion
    convention, wrong PnP solver, broken hloc pipeline) while tolerating
    pycolmap's internal float rounding.

Preconditions (skipped cleanly when missing):
  - Stable map layout at `data/maps/hge/` (produced by
    `scripts/build_hge_map.sh` after `scripts/download_data.sh`).
  - NavVis session under `mesh_pipeline/data/navvis_2022-02-06_12.55.11/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import pytest

from cnsg.config import LocalizationSettings
from cnsg.localization.capture_io import parse_images, parse_sensors


# --- paths -------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSION_DIR = REPO_ROOT / "mesh_pipeline" / "data" / "navvis_2022-02-06_12.55.11"

# Stable map layout produced by `scripts/build_hge_map.sh`.
MAP_DIR = REPO_ROOT / "data" / "maps" / "hge"
SFM_DIR = MAP_DIR / "sfm"
MAP_SP = MAP_DIR / "features_superpoint.h5"
MAP_NV = MAP_DIR / "features_netvlad.h5"


# --- helpers -----------------------------------------------------------------

def _quat_angle_deg(a_wxyz: np.ndarray, b_wxyz: np.ndarray) -> float:
    """Angle in degrees between two unit quaternions (wxyz), sign-ambiguity safe."""
    dot = abs(float(np.dot(a_wxyz, b_wxyz)))
    dot = min(1.0, max(-1.0, dot))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _world_from_cam_of_colmap_image(im: pycolmap.Image) -> tuple[np.ndarray, np.ndarray]:
    """Return (position_wxyz_free, quat_wxyz) for world-from-camera of a COLMAP image."""
    cam_from_world = im.cam_from_world()           # method in pycolmap 3.13
    world_from_cam = cam_from_world.inverse()
    q_xyzw = world_from_cam.rotation.quat          # pycolmap convention (x,y,z,w)
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    t = np.asarray(world_from_cam.translation)
    return t, q_wxyz


# --- skip conditions ---------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not (SFM_DIR.exists() and MAP_SP.exists() and MAP_NV.exists() and SESSION_DIR.exists()),
    reason="LaMAR map artifacts or NavVis session missing; run scripts/download_data.sh first.",
)


# --- the test ----------------------------------------------------------------


def test_localize_known_map_image_matches_colmap_reconstruction() -> None:
    """Localize a map-registered image; assert pose matches the COLMAP recon's pose for it."""
    from cnsg.localization.inference import Localizer

    settings = LocalizationSettings(
        map_dir=MAP_DIR,
        retrieval_num_pairs=10,
        ransac_max_error_px=12.0,
        ransac_min_inliers=20,
    )
    localizer = Localizer.from_settings(settings)

    images = parse_images(SESSION_DIR / "images.txt")
    sensors = parse_sensors(SESSION_DIR / "sensors.txt")
    record = images[0]
    sensor = sensors[record.sensor_id]
    query_path = SESSION_DIR / "raw_data" / record.relative_path
    assert query_path.exists(), f"query image missing: {query_path}"

    # Ground truth in COLMAP frame: look up the same image in the reconstruction.
    recon = pycolmap.Reconstruction(str(SFM_DIR))
    expected_name = f"map/raw_data/{record.relative_path}"
    gt_images = [im for im in recon.images.values() if im.name == expected_name]
    assert gt_images, f"image {expected_name!r} is not in the COLMAP reconstruction"
    gt_pos, gt_quat_wxyz = _world_from_cam_of_colmap_image(gt_images[0])

    intrinsics = {
        "model": sensor.model,
        "width": sensor.width,
        "height": sensor.height,
        "params": list(sensor.params),
    }
    result = localizer.localize(query_path, intrinsics=intrinsics)

    assert result.success, (
        f"localization failed: num_inliers={result.num_inliers} "
        f"(min={settings.ransac_min_inliers})"
    )

    trans_err_m = float(np.linalg.norm(result.position - gt_pos))
    rot_err_deg = _quat_angle_deg(result.rotation_wxyz, gt_quat_wxyz)

    print(
        f"\n[smoke] query={record.relative_path} inliers={result.num_inliers} "
        f"trans_err={trans_err_m:.4f} m rot_err={rot_err_deg:.4f} deg"
    )

    # Phase 1 plan exit criterion: 20 cm / 2°. The image is already registered
    # in the map so we expect sub-cm; 20/2 is the regression gate.
    assert trans_err_m < 0.20, f"translation error too large: {trans_err_m:.3f} m"
    assert rot_err_deg < 2.0, f"rotation error too large: {rot_err_deg:.3f} deg"
