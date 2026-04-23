"""End-to-end HGE semantic-mesh build pipeline.

Chains the four Phase-3 components into a single offline run:

    NavVis frames
        → Mask2FormerBackbone           (structural ADE20K-150 → S3DIS-13)
        → Sam3Segmenter                 (open-vocab foreground instances)
        → per-frame (instance_mask, class_mask) composite
        → lift_masks_to_3d              (project + depth-test + union-find)
        → hierarchy.segment_building    (floor + room IDs)
        → export_habitat                (HM3D-compatible .semantic.glb + .txt)

Runtime env: `cnsg-seg` (py3.12 + torch 2.10+ cu128 + SAM 3 + flash-attn-3).

See `docs/report/01_architecture-lean-migration/phase3-research.md` for the
algorithm spec that justifies the combining strategy: structural classes
(wall, floor, ceiling, window, door, column) come from Mask2Former's
exhaustive per-pixel output; foreground objects (chairs, tables, doors,
stairs, etc.) come from SAM 3's per-instance masks. Where they overlap,
SAM 3 wins (its masks are tighter and instance-aware).

Exit criterion: the run produces `<out_dir>/HGE.semantic.glb` that loads
in Habitat with `semantic_scene.regions` and `semantic_scene.objects`
populating per the HM3D-compatible schema.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import trimesh
from PIL import Image


# --- structural instance IDs (offset so SAM 3 instances can't collide) -----
# SAM 3 per-frame instance IDs are small positive integers (1..K); we reserve
# [1..999] for SAM 3 and use [1000+class_id] for structural background.
# Ensures `(frame_id, mask_id)` keys in the lifter stay unique across sources.
_STRUCTURAL_ID_OFFSET = 1000


# --- configuration records --------------------------------------------------


@dataclass(frozen=True)
class HgeBuildConfig:
    """All knobs for one HGE pipeline run."""

    navvis_session: Path              # path to `navvis_2022-02-06_12.55.11/`
    mesh_path: Path                   # e.g. `HGE_cut.voxelized.ply`
    out_dir: Path                     # where to write HGE.semantic.* files
    stem: str = "HGE"

    # How many NavVis frames to process. Defaults to `None` = all frames.
    max_frames: Optional[int] = None

    # Per-frame budget — skips frames taking > this many seconds.
    frame_timeout_s: float = 30.0

    # Mesh decimation target (face count). Habitat's semantic GLB loader
    # CC-bbox pass crashes on a single CC above ~150k verts. Quadric
    # decimation to 300k faces yields ~150k verts — safely under envelope.
    # `None` = no decimation.
    decimate_target_faces: Optional[int] = 300_000

    # Lift algorithm knobs.
    depth_tolerance_m: float = 0.05

    # SAM 3 config.
    sam3_prompts: tuple[str, ...] = (
        "door", "chair", "table", "desk", "sofa", "bed",
        "stairs", "elevator", "printer", "water fountain",
        "trash can", "cabinet", "shelf", "bookcase", "television",
        "computer", "kitchen counter", "whiteboard",
    )
    sam3_confidence: float = 0.3


# --- per-frame combiner ----------------------------------------------------


def _combine_per_frame(
    sam3_mask: np.ndarray,              # (H, W) int32; 0 = bg
    sam3_class_lut: dict[int, int],     # sam3 instance id → S3DIS class id
    m2f_class: np.ndarray,              # (H, W) int64; S3DIS class ids
) -> tuple[np.ndarray, np.ndarray]:
    """Merge SAM 3 foreground instances with Mask2Former structural labels.

    Output semantics match what `lift_2d_to_3d.Frame` expects:
      - instance_mask: int32, 0 = background, positive = per-frame instance id.
        SAM 3 instance ids stay as-is; pixels only covered by Mask2Former get
        `_STRUCTURAL_ID_OFFSET + class_id` so every structural class becomes
        its own "instance" in the lifter's bookkeeping (all wall pixels across
        all frames will later union-find-merge into one global wall object).
      - class_mask: int16, S3DIS class id. SAM 3 winners use their prompt's
        mapped class; everyone else keeps Mask2Former's per-pixel class.
    """
    H, W = m2f_class.shape
    instance_mask = np.zeros((H, W), dtype=np.int32)
    class_mask = m2f_class.astype(np.int16)

    # 1. Paint structural: instance = offset + class, skip class_id 0 ("unknown").
    #    Build vectorised: where class != 0, instance = class + offset.
    structural_pixels = m2f_class > 0
    instance_mask[structural_pixels] = (
        _STRUCTURAL_ID_OFFSET + m2f_class[structural_pixels]
    ).astype(np.int32)

    # 2. Paint SAM 3 foreground instances on top — overwrites both fields.
    sam3_ids = np.unique(sam3_mask)
    for sid in sam3_ids:
        if sid == 0:
            continue
        mask = sam3_mask == sid
        instance_mask[mask] = int(sid)
        class_mask[mask] = sam3_class_lut.get(int(sid), 13)  # 13 = clutter

    return instance_mask, class_mask


def _sam3_instance_class_lut(
    sam3_output, prompt_to_class: dict[str, int]
) -> dict[int, int]:
    """Map each per-frame SAM 3 instance id (1..K) to its S3DIS class id."""
    lut: dict[int, int] = {}
    for i, prompt in enumerate(sam3_output.class_per_instance, start=1):
        lut[i] = prompt_to_class.get(prompt, 13)  # fallback: clutter
    return lut


# --- NavVis frame iterator -------------------------------------------------


@dataclass(frozen=True)
class _NavVisFrame:
    frame_id: int
    rgb_path: Path
    depth_path: Path
    sensor_id: str
    pose_T_wc: np.ndarray        # (4, 4) world←camera
    fx: float
    fy: float
    cx: float
    cy: float


def _iter_navvis_frames(session: Path) -> Iterable[_NavVisFrame]:
    """Yield one `_NavVisFrame` per (timestamp, sensor_id) in the session.

    Pairs each RGB path with its depth-map (if present) and the sensor's
    intrinsics + trajectory pose.
    """
    from cnsg.localization.capture_io import (
        parse_images, parse_sensors, parse_trajectories,
    )
    from scipy.spatial.transform import Rotation

    images = parse_images(session / "images.txt")
    sensors = parse_sensors(session / "sensors.txt")
    poses = parse_trajectories(session / "trajectories.txt")

    depth_dir = session / "depth_maps"
    raw_dir = session / "raw_data"

    for i, record in enumerate(images):
        sensor = sensors.get(record.sensor_id)
        if sensor is None:
            continue
        pose = poses.get((record.timestamp, record.sensor_id))
        if pose is None:
            continue

        rgb_path = raw_dir / record.relative_path
        # NavVis depth maps are named by image stem with `.png` suffix.
        depth_path = depth_dir / (Path(record.relative_path).stem + ".png")
        if not rgb_path.exists() or not depth_path.exists():
            continue

        # Trajectory stores world ← camera directly as (q_wxyz, t).
        R = Rotation.from_quat(
            [pose.qx, pose.qy, pose.qz, pose.qw]  # scipy expects xyzw
        ).as_matrix()
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, :3] = R
        T_wc[:3, 3] = [pose.tx, pose.ty, pose.tz]

        yield _NavVisFrame(
            frame_id=i,
            rgb_path=rgb_path,
            depth_path=depth_path,
            sensor_id=record.sensor_id,
            pose_T_wc=T_wc,
            fx=sensor.fx,
            fy=sensor.fy,
            cx=sensor.cx,
            cy=sensor.cy,
        )


def _load_navvis_depth(path: Path) -> np.ndarray:
    """Load a NavVis PNG depth map as float32 meters.

    NavVis stores depth as uint16 millimetres; a value of 0 means "no return".
    """
    import cv2

    raw = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if raw is None:
        raise RuntimeError(f"failed to load depth map {path}")
    return (raw.astype(np.float32)) / 1000.0


# --- main pipeline entry point ---------------------------------------------


def build_hge_semantics(cfg: HgeBuildConfig) -> dict:
    """Run the full pipeline. Returns a summary dict.

    Logs + timings go to stdout. Intermediate state stays in memory; the
    final outputs go to `cfg.out_dir` via `cnsg.segmentation.export_habitat`.
    """
    from cnsg.segmentation.export_habitat import export_habitat
    from cnsg.segmentation.hierarchy import segment_building
    from cnsg.segmentation.lift_2d_to_3d import Frame, lift_masks_to_3d
    from cnsg.segmentation.sam3_per_frame import Sam3Segmenter
    from cnsg.segmentation.structural_ade20k import Mask2FormerBackbone
    from cnsg.segmentation.taxonomy import (
        S3DIS_CLASSES, S3DIS_NAME_TO_ID, ade20k_name_to_s3dis,
    )

    t_all = time.time()
    print(f"[build_hge] starting run")
    print(f"[build_hge] session: {cfg.navvis_session}")
    print(f"[build_hge] mesh:    {cfg.mesh_path}")
    print(f"[build_hge] out_dir: {cfg.out_dir}")

    # 1. Load the Poisson mesh once; decimate if above Habitat's envelope.
    t0 = time.time()
    mesh = trimesh.load(cfg.mesh_path, force="mesh")
    print(f"[build_hge] mesh loaded: {len(mesh.vertices):,} verts, "
          f"{len(mesh.faces):,} faces ({time.time()-t0:.1f}s)")

    if cfg.decimate_target_faces is not None and len(mesh.faces) > cfg.decimate_target_faces:
        t0 = time.time()
        import open3d as o3d
        o3m = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
            triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
        )
        dec = o3m.simplify_quadric_decimation(
            target_number_of_triangles=cfg.decimate_target_faces,
        )
        mesh = trimesh.Trimesh(
            vertices=np.asarray(dec.vertices),
            faces=np.asarray(dec.triangles),
            process=False,
        )
        print(
            f"[build_hge] decimated → {len(mesh.vertices):,} verts, "
            f"{len(mesh.faces):,} faces ({time.time()-t0:.1f}s)"
        )

    # 2. Build frame list.
    frames = list(_iter_navvis_frames(cfg.navvis_session))
    if cfg.max_frames is not None:
        frames = frames[: cfg.max_frames]
    print(f"[build_hge] {len(frames)} frames queued")

    # 3. Pre-compute prompt → S3DIS class lookup for SAM 3.
    prompt_to_class = {p: ade20k_name_to_s3dis(p) for p in cfg.sam3_prompts}

    # 4. Load segmentation models once.
    t0 = time.time()
    m2f = Mask2FormerBackbone()
    print(f"[build_hge] Mask2Former loaded ({time.time()-t0:.1f}s)")
    t0 = time.time()
    sam3 = Sam3Segmenter(
        prompts=cfg.sam3_prompts, confidence_threshold=cfg.sam3_confidence,
    )
    print(f"[build_hge] SAM 3 loaded ({time.time()-t0:.1f}s)")

    # 5. Per-frame segment + combine, then hand to the lifter.
    lift_frames: list[Frame] = []
    t_seg = time.time()
    for i, f in enumerate(frames):
        t0 = time.time()
        rgb = Image.open(f.rgb_path).convert("RGB")
        depth = _load_navvis_depth(f.depth_path)

        m2f_out = m2f.segment(rgb)
        sam3_out = sam3.segment(rgb)
        sam3_class_lut = _sam3_instance_class_lut(sam3_out, prompt_to_class)
        instance_mask, class_mask = _combine_per_frame(
            sam3_mask=sam3_out.instance_mask,
            sam3_class_lut=sam3_class_lut,
            m2f_class=m2f_out.s3dis_labels,
        )

        lift_frames.append(
            Frame(
                frame_id=f.frame_id,
                depth=depth,
                instance_mask=instance_mask,
                class_mask=class_mask.astype(np.int16),
                T_world_cam=f.pose_T_wc,
                fx=f.fx, fy=f.fy, cx=f.cx, cy=f.cy,
            )
        )
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_seg
            avg = elapsed / (i + 1)
            eta_min = (len(frames) - (i + 1)) * avg / 60.0
            print(
                f"[build_hge]  seg {i+1:4d}/{len(frames):4d}  "
                f"{time.time()-t0:.2f}s/frame (avg {avg:.2f}s) "
                f"ETA {eta_min:.1f}min"
            )
    print(f"[build_hge] segmentation done ({(time.time()-t_seg)/60:.1f}min)")

    # 6. Lift.
    t0 = time.time()
    verts_np = np.asarray(mesh.vertices, dtype=np.float32)
    lift_result = lift_masks_to_3d(
        verts_np, lift_frames, depth_tolerance=cfg.depth_tolerance_m,
    )
    print(
        f"[build_hge] lift: {lift_result.num_instances} instances "
        f"({time.time()-t0:.1f}s)"
    )

    # 7. Hierarchy (floors + rooms).
    t0 = time.time()
    floor_ids, room_ids = segment_building(verts_np)
    n_floors = int(floor_ids.max())
    n_rooms = int(room_ids.max())
    print(
        f"[build_hge] hierarchy: {n_floors} floors, {n_rooms} rooms "
        f"({time.time()-t0:.1f}s)"
    )

    # 8. Export to Habitat-compatible schema.
    t0 = time.time()
    class_name_lookup = {c.id: c.name for c in S3DIS_CLASSES}
    # `export_habitat` takes class_id + region_id per vertex. We use the
    # lifter's per-vertex class (drops clutter at class==0 via exporter) and
    # the hierarchy's room id as the region.
    manifest = export_habitat(
        mesh=mesh,
        per_vertex_class_id=lift_result.class_ids,
        per_vertex_region_id=room_ids,
        class_id_to_name=class_name_lookup,
        out_dir=cfg.out_dir,
        stem=cfg.stem,
        # Collapse multi-CC same-label regions into single instances. The
        # Phase-2 per-CC export blows past Habitat's CC-bbox envelope at
        # scan-scale (100+ instances × 250k+ verts → segfault). Per-pair
        # grouping keeps instance count ≤ |classes| × |rooms|.
        group_per_class_region=True,
        # Drop instances with fewer than 20 verts — sparse lift coverage
        # produces singletons that also crash Habitat's CC-bbox pass.
        min_verts_per_instance=20,
    )
    print(
        f"[build_hge] export: {manifest.num_instances} instances, "
        f"{manifest.num_regions} regions ({time.time()-t0:.1f}s)"
    )

    summary = {
        "navvis_session": str(cfg.navvis_session),
        "mesh_path": str(cfg.mesh_path),
        "num_frames_processed": len(lift_frames),
        "num_verts": int(len(mesh.vertices)),
        "num_faces": int(len(mesh.faces)),
        "lift_num_instances": int(lift_result.num_instances),
        "hierarchy_num_floors": n_floors,
        "hierarchy_num_rooms": n_rooms,
        "export_num_instances": int(manifest.num_instances),
        "export_num_regions": int(manifest.num_regions),
        "total_time_s": round(time.time() - t_all, 1),
        "out_dir": str(cfg.out_dir),
    }
    (cfg.out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[build_hge] done in {summary['total_time_s']:.1f}s — "
        f"summary → {cfg.out_dir / 'build_summary.json'}"
    )
    return summary


# --- CLI -------------------------------------------------------------------


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--navvis-session", type=Path, required=True)
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--stem", default="HGE")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--depth-tolerance", type=float, default=0.05)
    p.add_argument("--sam3-confidence", type=float, default=0.3)
    p.add_argument(
        "--decimate-target-faces", type=int, default=300_000,
        help="Target face count after decimation. Habitat's CC-bbox pass "
             "crashes on a single CC above ~150k verts; 300k faces gives "
             "~150k verts. Pass 0 to disable.",
    )
    args = p.parse_args()

    cfg = HgeBuildConfig(
        navvis_session=args.navvis_session,
        mesh_path=args.mesh,
        out_dir=args.out_dir,
        stem=args.stem,
        max_frames=args.max_frames,
        depth_tolerance_m=args.depth_tolerance,
        sam3_confidence=args.sam3_confidence,
        decimate_target_faces=(args.decimate_target_faces or None),
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    build_hge_semantics(cfg)


if __name__ == "__main__":
    _main()
