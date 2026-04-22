"""In-process visual localization.

Replaces the LaMAR Docker pipeline with a direct hloc + pycolmap call chain:

    query RGB  →  SuperPoint keypoints/descriptors
                →  NetVLAD global descriptor
                →  top-K retrieval against pre-built map
                →  LightGlue match to top-K map images
                →  pycolmap.absolute_pose_estimation (PnP + RANSAC)
                →  world-from-camera 6-DoF pose

Map artifacts are loaded once at construction time and kept read-only.

See `docs/report/01_architecture-lean-migration/plan.md` §Phase 1.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pycolmap
from hloc import extract_features, match_features, pairs_from_retrieval, localize_sfm
from PIL import Image

from cnsg.config import LocalizationSettings, get_settings


# -------- hloc feature configurations --------

# Must match the config used to build the map's features.h5, otherwise the
# query descriptors won't be comparable to the cached map descriptors.
SUPERPOINT_CONF = {
    "name": "superpoint",
    "model": {"name": "superpoint", "nms_radius": 3, "max_keypoints": 2048},
    "preprocessing": {"grayscale": True, "resize_max": 1024},
    "output": "feats-superpoint-n2048-r1024",
}

NETVLAD_CONF = {
    "name": "netvlad",
    "model": {"name": "netvlad"},
    "preprocessing": {"resize_max": 640},
    "output": "global-feats-netvlad",
}

# LightGlue matching for SuperPoint features — Apache-2.0, drop-in for SuperGlue.
LIGHTGLUE_CONF = match_features.confs["superpoint+lightglue"]


# -------- result type --------


@dataclass(frozen=True)
class LocalizationResult:
    """Output of `Localizer.localize()`.

    Pose is **world-from-camera** (`T_wc`): translation is the camera's
    position in world coordinates; rotation transforms camera-frame
    directions into world-frame directions. Matches NavVis Capture and
    Habitat agent conventions.
    """

    position: np.ndarray          # shape (3,), [tx, ty, tz] in world frame
    rotation_wxyz: np.ndarray     # shape (4,), [qw, qx, qy, qz] world-from-camera
    num_inliers: int
    num_matches: int
    success: bool
    query_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position.tolist(),
            "rotation": self.rotation_wxyz.tolist(),
            "num_inliers": int(self.num_inliers),
            "num_matches": int(self.num_matches),
            "success": bool(self.success),
            "query_key": self.query_key,
        }


# -------- localizer --------


class Localizer:
    """Single-process visual localizer.

    Load the map once (`Localizer.from_settings()`), then call `.localize()`
    per query image. Thread-safety: not thread-safe (hloc internals hold a
    single GPU model). For concurrent serving, wrap in a worker process
    (see Phase 4 server design in the migration plan).
    """

    def __init__(
        self,
        sfm_dir: Path,
        map_features_path: Path,
        map_retrieval_path: Path,
        settings: Optional[LocalizationSettings] = None,
    ):
        self._settings = settings or get_settings().localization
        self._sfm_dir = Path(sfm_dir)
        self._map_features_path = Path(map_features_path)
        self._map_retrieval_path = Path(map_retrieval_path)

        for p, label in [
            (self._sfm_dir, "sfm_dir"),
            (self._map_features_path, "map features.h5"),
            (self._map_retrieval_path, "map retrieval.h5"),
        ]:
            if not p.exists():
                raise FileNotFoundError(f"localizer: {label} missing at {p}")

        self._reconstruction = pycolmap.Reconstruction(str(self._sfm_dir))

    @classmethod
    def from_settings(cls, settings: Optional[LocalizationSettings] = None) -> "Localizer":
        """Build from LocalizationSettings map_dir convention."""
        s = settings or get_settings().localization
        return cls(
            sfm_dir=s.map_dir / "sfm",
            map_features_path=s.map_dir / "features_superpoint.h5",
            map_retrieval_path=s.map_dir / "features_netvlad.h5",
            settings=s,
        )

    # ---- public API ----

    def localize(
        self,
        image_path: Path,
        intrinsics: Optional[dict] = None,
    ) -> LocalizationResult:
        """Localize a single image against the pre-built map.

        Args:
            image_path: path to a JPEG/PNG on disk.
            intrinsics: optional dict with keys `{model, width, height, params}`.
                If None, falls back to a heuristic PINHOLE with
                `f = default_focal_ratio * max(W, H)` and principal point
                at image center (matches LaMAR's heuristic).

        Returns:
            LocalizationResult. `.success=False` if PnP found fewer than
            `settings.ransac_min_inliers` inliers.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"query image missing: {image_path}")

        with tempfile.TemporaryDirectory(prefix="cnsg_loc_") as tmp_str:
            tmp = Path(tmp_str)
            query_dir = tmp / "query_images"
            query_dir.mkdir()
            query_key = "query.jpg"
            # Always re-encode as JPEG so hloc's loader behavior is deterministic.
            self._copy_or_convert(image_path, query_dir / query_key)

            # 1. Extract query SuperPoint features.
            query_sp_h5 = tmp / "query_superpoint.h5"
            extract_features.main(
                SUPERPOINT_CONF,
                query_dir,
                feature_path=query_sp_h5,
                as_half=True,
                image_list=[query_key],
            )

            # 2. Extract query NetVLAD global descriptor.
            query_nv_h5 = tmp / "query_netvlad.h5"
            extract_features.main(
                NETVLAD_CONF,
                query_dir,
                feature_path=query_nv_h5,
                as_half=True,
                image_list=[query_key],
            )

            # 3. Retrieve top-K map images by NetVLAD similarity.
            pairs_path = tmp / "pairs.txt"
            pairs_from_retrieval.main(
                descriptors=query_nv_h5,
                output=pairs_path,
                num_matched=self._settings.retrieval_num_pairs,
                query_list=[query_key],
                db_model=self._sfm_dir,  # hloc reads images.bin from this path
                db_descriptors=self._map_retrieval_path,
            )

            # 4. Match query SuperPoint to each retrieved map image via LightGlue.
            matches_h5 = tmp / "matches.h5"
            match_features.main(
                LIGHTGLUE_CONF,
                pairs_path,
                features=query_sp_h5,
                features_ref=self._map_features_path,
                matches=matches_h5,
            )

            # 5. Build queries.txt (hloc format: "<name> <MODEL> <W> <H> *params").
            w, h, cam_model, cam_params = self._resolve_intrinsics(
                image_path, intrinsics
            )
            queries_txt = tmp / "queries.txt"
            queries_txt.write_text(
                f"{query_key} {cam_model} {w} {h} "
                + " ".join(f"{p:.6f}" for p in cam_params)
                + "\n"
            )

            # 6. PnP + RANSAC via pycolmap (wrapped by hloc.localize_sfm).
            results_txt = tmp / "results.txt"
            localize_sfm.main(
                reference_sfm=self._reconstruction,
                queries=queries_txt,
                retrieval=pairs_path,
                features=query_sp_h5,
                matches=matches_h5,
                results=results_txt,
                ransac_thresh=int(self._settings.ransac_max_error_px),
            )

            # 7. Parse the result file. Format written by hloc.localize_sfm.write_poses:
            #       "<basename> qw qx qy qz tx ty tz"
            #    where the pose is cam-from-world. We invert to world-from-camera.
            result = self._parse_results_file(results_txt, query_key, tmp)
            return result

    # ---- helpers ----

    def _resolve_intrinsics(
        self,
        image_path: Path,
        intrinsics: Optional[dict],
    ) -> tuple[int, int, str, list[float]]:
        """Resolve camera intrinsics. Returns (W, H, MODEL, params)."""
        if intrinsics is not None:
            w = int(intrinsics["width"])
            h = int(intrinsics["height"])
            model = str(intrinsics.get("model", "PINHOLE"))
            params = [float(x) for x in intrinsics["params"]]
            return w, h, model, params

        # Heuristic fallback (same as LaMAR's): f = ratio * max(W,H), principal at center.
        with Image.open(image_path) as im:
            w, h = im.size
        ratio = self._settings.default_focal_ratio
        f = ratio * max(w, h)
        print(
            f"[WARN] intrinsics: expected=explicit dict, got=None, "
            f"fallback=heuristic PINHOLE {w}x{h} fx=fy={f:.2f} cx={w/2:.2f} cy={h/2:.2f}",
            flush=True,
        )
        return w, h, "PINHOLE", [f, f, w / 2.0, h / 2.0]

    def _copy_or_convert(self, src: Path, dst: Path) -> None:
        """Copy image to dst; re-encode if not JPEG."""
        if src.suffix.lower() in (".jpg", ".jpeg"):
            shutil.copy(src, dst)
        else:
            with Image.open(src) as im:
                im.convert("RGB").save(dst, format="JPEG", quality=95)

    def _parse_results_file(
        self,
        path: Path,
        query_key: str,
        tmp: Path,
    ) -> LocalizationResult:
        """Parse hloc.localize_sfm output; invert cam-from-world → world-from-camera.

        Also reads `<path>_logs.pkl` for inlier count. If no line matches the
        query_key, returns a failed LocalizationResult.
        """
        if not path.exists():
            return LocalizationResult(
                position=np.zeros(3),
                rotation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                num_inliers=0,
                num_matches=0,
                success=False,
                query_key=query_key,
            )

        cam_from_world: Optional[pycolmap.Rigid3d] = None
        for line in path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            name, qw, qx, qy, qz, tx, ty, tz = parts
            if name != query_key:
                continue
            # hloc writes (qw, qx, qy, qz); pycolmap.Rigid3d expects quat xyzw.
            quat_xyzw = np.array([float(qx), float(qy), float(qz), float(qw)])
            translation = np.array([float(tx), float(ty), float(tz)])
            cam_from_world = pycolmap.Rigid3d(
                rotation=pycolmap.Rotation3d(quat_xyzw),
                translation=translation,
            )
            break

        # Inlier / match stats from the logs pickle.
        num_inliers, num_matches = self._read_stats_pkl(path, query_key)

        if cam_from_world is None:
            return LocalizationResult(
                position=np.zeros(3),
                rotation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                num_inliers=num_inliers,
                num_matches=num_matches,
                success=False,
                query_key=query_key,
            )

        world_from_cam = cam_from_world.inverse()
        q_xyzw = world_from_cam.rotation.quat
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        t_wc = np.asarray(world_from_cam.translation)

        success = num_inliers >= self._settings.ransac_min_inliers
        return LocalizationResult(
            position=t_wc,
            rotation_wxyz=q_wxyz,
            num_inliers=num_inliers,
            num_matches=num_matches,
            success=success,
            query_key=query_key,
        )

    def _read_stats_pkl(self, results_path: Path, query_key: str) -> tuple[int, int]:
        """Pull inlier/match counts from hloc's results_logs.pkl."""
        pkl_path = Path(f"{results_path}_logs.pkl")
        if not pkl_path.exists():
            return 0, 0
        try:
            import pickle
            with pkl_path.open("rb") as f:
                logs = pickle.load(f)
        except Exception as exc:
            print(
                f"[WARN] localizer stats: expected=readable {pkl_path}, "
                f"got={type(exc).__name__}: {exc}, fallback=0/0",
                flush=True,
            )
            return 0, 0
        entry = logs.get("loc", {}).get(query_key, {})
        pnp = entry.get("PnP_ret", {})
        num_inliers = int(pnp.get("num_inliers", 0))
        # Matches summed over all retrieved reference images.
        keypoints = entry.get("keypoint_index_to_db", (None, None))
        num_matches = int(len(keypoints[0])) if keypoints and keypoints[0] is not None else 0
        return num_inliers, num_matches
