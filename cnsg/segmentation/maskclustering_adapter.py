"""Adapter so PKU-EPIC/MaskClustering can lift our SAM 3 + structural masks.

MaskClustering (CVPR'24, https://github.com/PKU-EPIC/MaskClustering) replaces
our union-find majority-vote lifter with a view-consensus graph-clustering
algorithm. Key idea: mask i in frame f is "contained" in mask j in frame f'
iff, after 3-cm ball-query from depth back-projection, ≥80 % of i's points
land inside j. Build pairwise (i, j) co-containment matrix over all frames,
threshold → iterative graph clustering with a walking observer threshold.
Training-free, beats majority-vote on ScanNet/ScanNet++.

We produce a `Dataset` object that MaskClustering's `main.py` can consume
without modification. Map of what we expose:

| MaskClustering calls                            | We provide from             |
|-------------------------------------------------|-----------------------------|
| `get_scene_points()`                            | decimated HGE_cut mesh      |
| `get_frame_list(stride)`                        | NavVis frame_ids            |
| `get_intrinsics(frame_id)` → o3d.PinholeCam    | sensors.txt per sensor_id   |
| `get_extrinsic(frame_id)` → 4×4 cam-from-world  | trajectories.txt + align    |
| `get_depth(frame_id)` → (H, W) float32 meters   | NavVis depth_maps/ uint16/1000 |
| `get_segmentation(frame_id, align_with_depth)`  | seg_cache's instance_mask   |

The seg_cache key is our combined (SAM 3 + structural-offset) instance mask,
already hash-tagged by prompts + backbone (`build_hge._seg_cache_config_hash`).
MaskClustering only needs integer mask IDs — it doesn't care if they came
from CropFormer, SAM 3, or a mix. Our structural-offset convention (class-id
+ `_STRUCTURAL_ID_OFFSET=1000`) means wall/floor/ceiling regions get stable
per-class IDs across frames; SAM 3 instances get their own frame-local IDs.

The per-class semantic label (S3DIS class_id) is NOT carried by
MaskClustering — it's a class-agnostic clusterer. We bolt a class-vote on
top after clustering: for each final cluster, count the structural/offset
IDs in its constituent masks and assign the majority S3DIS class.

Runtime env: `cnsg-seg` (pytorch3d + networkx + open3d + our cnsg module).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from cnsg.localization.capture_io import (
    parse_alignment_global,
    parse_images,
    parse_sensors,
    parse_trajectories,
)


# S3DIS label names — we re-expose them through `get_label_id` so the
# (unused) CLIP path in MaskClustering doesn't choke on a missing field.
_S3DIS_LABELS = (
    "unknown", "wall", "floor", "ceiling", "column", "window", "door",
    "chair", "table", "sofa", "bookcase", "stairs", "board", "clutter",
)


@dataclass(frozen=True)
class _HgeFrame:
    frame_id: int
    sensor_id: str
    rgb_path: Path
    depth_path: Path
    T_world_cam: np.ndarray  # 4×4 float64, absolute frame (alignment applied)
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class HgeMaskClusteringDataset:
    """MaskClustering-compatible dataset view over our HGE build cache.

    Expected caller: `MaskClustering/main.py` after we inject `--dataset hge`.
    The class mirrors `dataset/scannet.py` field-by-field so upstream changes
    are easy to merge.

    Args:
        session_dir: the NavVis Capture session
            (`mesh_pipeline/data/navvis_2022-02-06_12.55.11/`).
        mesh_path: the decimated HGE mesh the lift should populate.
            Must match the one used when writing `seg_cache_dir` so vertex
            counts line up (e.g. `data/maps/hge/HGE.semantic.glb` topology).
        seg_cache_dir: our build's seg_cache. Each `frame_{id:06d}.npz`
            holds `(instance_mask: int32 H×W, class_mask: int16 H×W)`.
            `instance_mask` is what MaskClustering consumes.
        out_dir: where to write MaskClustering's intermediate object dicts
            and final cluster tensors. Defaults to `{seg_cache}/maskclustering/`.
        max_frames: optional cap (for smoke tests).
    """

    def __init__(
        self,
        session_dir: Path,
        mesh_path: Path,
        seg_cache_dir: Path,
        *,
        out_dir: Optional[Path] = None,
        max_frames: Optional[int] = None,
        instance_only: bool = True,
    ):
        """
        Args:
            instance_only: when True (default), `get_segmentation` returns
                ONLY the SAM 3 per-frame instance IDs and zeroes out the
                structural-offset entries (`instance_mask >= _STRUCTURAL_ID_OFFSET`,
                i.e. ≥ 1000) that our combiner uses to tag wall/floor/ceiling
                pixels. MaskClustering was published against per-instance
                inputs (CropFormer); feeding it our combined stream causes
                walls to merge across the whole building because every wall
                pixel has the same `_STRUCTURAL_ID_OFFSET + class_id` ID in
                every frame. With `instance_only=True` the only IDs MC sees
                are SAM 3 instances (1..K, frame-local), which matches what
                view-consensus clustering was designed for. Set False to
                feed the combined stream (compatibility / debugging only).
        """
        self.session_dir = Path(session_dir)
        self.mesh_path = Path(mesh_path)
        self.seg_cache_dir = Path(seg_cache_dir)
        self.out_dir = Path(out_dir) if out_dir is not None else self.seg_cache_dir / "maskclustering"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.instance_only = bool(instance_only)

        # Eager frame-table build so `get_frame_list` / `get_*` are O(1).
        self._frames: dict[int, _HgeFrame] = self._load_frames(max_frames=max_frames)
        if not self._frames:
            raise RuntimeError(f"no usable frames found in {session_dir}")

        # Depth resolution (all NavVis center cams share 1280×1920 per
        # sensors.txt, but let's read one to be defensive).
        first = next(iter(self._frames.values()))
        raw = cv2.imread(str(first.depth_path), cv2.IMREAD_ANYDEPTH)
        if raw is None:
            raise RuntimeError(f"cannot read depth {first.depth_path}")
        self.image_size = (raw.shape[1], raw.shape[0])  # (W, H) — o3d convention
        self.depth_scale = 1000.0

        # Required by MaskClustering's code that writes object dicts.
        self.seq_name = "hge"
        self.object_dict_dir = str(self.out_dir / "object")
        Path(self.object_dict_dir).mkdir(exist_ok=True)

        # Path fields that upstream code references (even if we don't use).
        self.mesh_path_str = str(self.mesh_path)
        self.point_cloud_path = str(self.mesh_path)
        self.segmentation_dir = str(self.seg_cache_dir)  # informational
        self.rgb_dir = str(self.session_dir / "raw_data")
        self.depth_dir = str(self.session_dir / "depth_maps")

    # -- MaskClustering contract ---------------------------------------------

    def get_frame_list(self, stride: int = 1) -> list[int]:
        ids = sorted(self._frames.keys())
        return ids[::stride]

    def get_intrinsics(self, frame_id: int) -> o3d.camera.PinholeCameraIntrinsic:
        f = self._frames[frame_id]
        k = o3d.camera.PinholeCameraIntrinsic()
        k.set_intrinsics(f.width, f.height, f.fx, f.fy, f.cx, f.cy)
        return k

    def get_extrinsic(self, frame_id: int) -> np.ndarray:
        """Return the 4×4 cam-to-world (world-from-camera) for the frame.

        MaskClustering's `backproject` (`utils/mask_backprojection.py:22`)
        applies this directly to a camera-frame point cloud via
        `pcld.transform(extrinsics)`. So we must return **T_world_cam**, not
        its inverse. Our NavVis frames already carry `T_world_cam` in the
        absolute frame (alignment applied), identical to the convention our
        own lifter consumes.
        """
        return self._frames[frame_id].T_world_cam

    def get_depth(self, frame_id: int) -> np.ndarray:
        f = self._frames[frame_id]
        raw = cv2.imread(str(f.depth_path), cv2.IMREAD_ANYDEPTH)
        if raw is None:
            raise RuntimeError(f"failed to read depth {f.depth_path}")
        return raw.astype(np.float32) / self.depth_scale

    def get_rgb(self, frame_id: int, change_color: bool = True) -> np.ndarray:
        f = self._frames[frame_id]
        rgb = cv2.imread(str(f.rgb_path))
        if rgb is None:
            raise RuntimeError(f"failed to read rgb {f.rgb_path}")
        if change_color:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb

    def get_segmentation(
        self, frame_id: int, align_with_depth: bool = False
    ) -> np.ndarray:
        """Return per-frame mask image, integer IDs, 0 = background.

        Defaults to **SAM 3 instance IDs only** (`self.instance_only=True`)
        with structural-offset entries (>= `_STRUCTURAL_ID_OFFSET`, i.e.
        ≥ 1000) zeroed out. See `__init__` docstring for why.

        `align_with_depth=True` resizes the mask to depth resolution with
        nearest-neighbor interpolation (matches the path that
        MaskClustering's ScanNet dataset takes when CropFormer output and
        depth disagree on resolution).
        """
        path = self.seg_cache_dir / f"frame_{frame_id:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"seg_cache miss for frame {frame_id}: {path}. Run "
                f"scripts/build_hge_semantics.sh first to populate the cache."
            )
        with np.load(path, allow_pickle=False) as z:
            mask = z["instance_mask"].astype(np.int32)
        if self.instance_only:
            # `_STRUCTURAL_ID_OFFSET = 1000` in cnsg.segmentation.build_hge.
            # Pixels tagged structural (wall/floor/ceiling/etc.) are the
            # ones with ID ≥ 1000; we zero them so MaskClustering's
            # view-consensus only sees per-frame SAM 3 instance IDs.
            mask = np.where(mask >= 1000, 0, mask)
        if align_with_depth and (mask.shape[1], mask.shape[0]) != self.image_size:
            mask = cv2.resize(
                mask, self.image_size, interpolation=cv2.INTER_NEAREST
            )
        return mask

    def get_structural_class_map(
        self, frame_id: int, align_with_depth: bool = False
    ) -> np.ndarray:
        """Return per-pixel S3DIS class IDs (int16), independent of SAM 3.

        Used by the per-cluster majority-class vote that runs AFTER
        MaskClustering (since MC is class-agnostic). Reads the
        `class_mask` array our build pipeline already writes alongside
        `instance_mask`.
        """
        path = self.seg_cache_dir / f"frame_{frame_id:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"seg_cache miss for frame {frame_id}: {path}"
            )
        with np.load(path, allow_pickle=False) as z:
            cls = z["class_mask"].astype(np.int16)
        if align_with_depth and (cls.shape[1], cls.shape[0]) != self.image_size:
            cls = cv2.resize(
                cls, self.image_size, interpolation=cv2.INTER_NEAREST
            )
        return cls

    def get_frame_path(self, frame_id: int) -> tuple[str, str]:
        f = self._frames[frame_id]
        return str(f.rgb_path), str(
            self.seg_cache_dir / f"frame_{frame_id:06d}.npz"
        )

    def get_scene_points(self) -> np.ndarray:
        """Return the vertex positions of the TARGET mesh — the one the
        lift should populate. MaskClustering expects a point-cloud-shaped
        (N, 3) array; we pass mesh vertices directly (triangles unused on
        the ingestion side).
        """
        import trimesh

        m = trimesh.load(self.mesh_path, force="mesh")
        return np.asarray(m.vertices, dtype=np.float64)

    def get_label_features(self) -> dict:
        """MaskClustering uses text-CLIP features here for its open-vocab
        pass. We don't use the CLIP step — we bolt a class-vote onto the
        cluster output instead, which is cheaper and reuses our Mask2Former
        / EoMT structural labels. Return an empty dict; the CLIP step must
        be disabled at call site.
        """
        return {}

    def get_label_id(self) -> tuple[dict[str, int], dict[int, str]]:
        label2id: dict[str, int] = {}
        id2label: dict[int, str] = {}
        for i, name in enumerate(_S3DIS_LABELS):
            label2id[name] = i
            id2label[i] = name
        self.label2id = label2id
        self.id2label = id2label
        return label2id, id2label

    # -- helpers -------------------------------------------------------------

    def _load_frames(
        self, *, max_frames: Optional[int] = None
    ) -> dict[int, _HgeFrame]:
        images = parse_images(self.session_dir / "images.txt")
        sensors = parse_sensors(self.session_dir / "sensors.txt")
        poses_pg = parse_trajectories(self.session_dir / "trajectories.txt")
        T_abs_pg = parse_alignment_global(
            self.session_dir / "proc" / "alignment_global.txt"
        )

        raw_dir = self.session_dir / "raw_data"
        depth_dir = self.session_dir / "depth_maps"

        frames: dict[int, _HgeFrame] = {}
        for i, rec in enumerate(images):
            sensor = sensors.get(rec.sensor_id)
            if sensor is None:
                continue
            pose = poses_pg.get((rec.timestamp, rec.sensor_id))
            if pose is None:
                continue
            rgb_path = raw_dir / rec.relative_path
            depth_path = depth_dir / (Path(rec.relative_path).stem + ".png")
            if not rgb_path.exists() or not depth_path.exists():
                continue

            R = Rotation.from_quat(
                [pose.qx, pose.qy, pose.qz, pose.qw]
            ).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = [pose.tx, pose.ty, pose.tz]
            if T_abs_pg is not None:
                T = T_abs_pg @ T

            frames[i] = _HgeFrame(
                frame_id=i,
                sensor_id=rec.sensor_id,
                rgb_path=rgb_path,
                depth_path=depth_path,
                T_world_cam=T,
                fx=sensor.fx,
                fy=sensor.fy,
                cx=sensor.cx,
                cy=sensor.cy,
                width=sensor.width,
                height=sensor.height,
            )
            if max_frames is not None and len(frames) >= max_frames:
                break
        return frames
