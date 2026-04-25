"""End-to-end HGE semantic-mesh build pipeline.

Chains the four Phase-3 components into a single offline run:

    NavVis frames  +  alignment_global.txt  →  poses in absolute frame
        → Mask2FormerBackbone           (structural ADE20K-150 → S3DIS-13)
        → Sam3Segmenter                 (open-vocab foreground instances,
                                         state-reused across prompts → ~4× faster)
        → per-frame (instance_mask, class_mask) composite
        → (optional) seg_cache npz      (hash-tagged, hit skips M2F + SAM 3)
        → lift_masks_to_3d              (project + depth-test + union-find)
        → coverage sanity gate          ([WARN] when labeled fraction < 15 %)
        → hierarchy.segment_building    (floor + room IDs)
        → export_habitat                (HM3D-compatible .semantic.glb + .txt,
                                         brightened stage GLB + Unknown gray)

Runtime env: `cnsg-seg` (py3.12 + torch 2.10+ cu128 + SAM 3 + flash-attn-3).

See `docs/report/01_architecture-lean-migration/phase3-research.md` for the
algorithm spec that justifies the combining strategy: structural classes
(wall, floor, ceiling, window, door, column) come from Mask2Former's
exhaustive per-pixel output; foreground objects (chairs, tables, doors,
stairs, etc.) come from SAM 3's per-instance masks. Where they overlap,
SAM 3 wins (its masks are tighter and instance-aware).

See `docs/report/02_hge-lift-frame-mismatch/findings.md` for the
alignment-global fix that rescued a 0.44 %-coverage regression — and is
the reason this module now refuses to feed poses to the lifter without
first composing `T_absolute_from_pose_graph` onto them.

Exit criterion: the run produces `<out_dir>/HGE.semantic.glb` that loads
in Habitat with `semantic_scene.regions` and `semantic_scene.objects`
populating per the HM3D-compatible schema, AND `build_summary.json`'s
`coverage.labeled_fraction` is above the sanity threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import trimesh
from PIL import Image


# When this module is run via `python -m ... | tee` or piped through bash,
# stdout is block-buffered by default and progress logs sit invisibly in a
# 4-KB pipe buffer. Flip to line buffering so every `print(...)` hits the
# pipe immediately — the cost is negligible and the user gains live visibility
# into the ~55-minute segmentation loop.
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, OSError):
    pass


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
    # Depth tolerance for the 2D→3D lift. 5 cm is the thorough-literature
    # choice; 15 cm is what we measured as the best-compromise ceiling on our
    # Poisson mesh (see docs/report/04_segmentation-upgrade-plan §measured
    # ceilings). 99.7% of Unknown verts fail depth-test at 5 cm; loosening
    # to 15 cm recovers +4 pp with negligible false-positive increase.
    depth_tolerance_m: float = 0.15

    # SAM 3 open-vocabulary prompts. Curated for ETH HG E floor
    # (landmark-based navigation per docs/papers/MR_Final_Report.pdf §3.3).
    # Categories:
    #   - HG E-specific architecture / decor: fountain, statue, bust, column,
    #     arched doorway, staircase (flight), hall, balustrade, pillar, vault
    #   - Navigation landmarks the LLM refers to: lecture hall door,
    #     display case, notice board, bench, information board, plaque
    #   - Standard furniture (kept from the original generic list where still
    #     relevant): chair, table, desk, door, elevator, bookcase
    # The list is intentionally verbose — SAM 3 runs once per prompt per
    # frame (image encoder cached), so adding prompts costs linear time but
    # recovers landmarks the ADE20K taxonomy lumps into "other".
    sam3_prompts: tuple[str, ...] = (
        # HG E specific landmarks (Bianco et al §3.3)
        "fountain", "statue", "bust", "pillar", "column",
        "arched doorway", "arched ceiling", "archway",
        "staircase", "staircase flight", "balustrade", "handrail",
        "lecture hall door", "entrance door", "information board",
        "display case", "notice board", "plaque", "signboard",
        # Common interior furniture (retained from generic list)
        "door", "chair", "table", "desk", "bench", "sofa",
        "bookcase", "cabinet", "shelf", "whiteboard",
        # Facilities the user may ask about
        "elevator", "printer", "trash can", "water cooler",
    )
    sam3_confidence: float = 0.3

    # Structural backbone. "mask2former" (default, stable) uses
    # `facebook/mask2former-swin-large-ade-semantic`. "eomt" uses
    # `tue-mps/ade20k_semantic_eomt_large_512` — CVPR'25 Highlight, beats
    # Mask2Former accuracy with ~4× speedup per the paper. Both return the
    # same ADE20K-150 label space so the S3DIS remap carries over unchanged.
    # See `docs/report/04_segmentation-upgrade-plan/findings.md`.
    structural_backbone: str = "mask2former"

    # Optional path to a pre-existing photorealistic stage GLB (e.g. NavVis'
    # `HGE.basis.glb` with baked photographic textures). When set, the
    # emitted scene_dataset_config.json points at this file as the stage,
    # so mr_viewer renders a real building instead of flat-shaded palette
    # colours. The palette stage `<stem>.glb` is still written so semantic-
    # visualization tools that expect it keep working. If the file lives
    # outside out_dir it's copied in (so the config's relative path resolves).
    external_stage_glb: Optional[Path] = None

    # Minimum fraction of mesh vertices that must carry a non-zero class_id
    # after the lift. Below this, the build prints a loud banner warning —
    # ships the output anyway (caller decides whether to accept it) but
    # makes silent regressions impossible. 0.0 disables the check.
    min_coverage_fraction: float = 0.15

    # Directory for caching per-frame (instance_mask, class_mask) post-combine.
    # A cache hit skips Mask2Former + SAM 3 inference (dominant ~52 min of a
    # full HGE build). Cache entries are tagged with a hash of
    # `(sam3_prompts, sam3_confidence)` so changing either invalidates
    # individual files loudly rather than serving stale masks. `None` =
    # no caching (always re-run models).
    seg_cache_dir: Optional[Path] = None

    # ----- GPT-5.5 per-frame VLM tagger (Phase-4 upgrade) ------------------
    #
    # When `use_gpt5_tagger=True`, every frame is FIRST sent to OpenAI's
    # GPT-5.5 (or pinned snapshot) for an open-vocab list of landmark
    # phrases — heritage-quality terms like "marble bust on plinth",
    # "ornate stone fountain", "wooden lecture-hall door" that the
    # hand-curated `sam3_prompts` list doesn't cover. The returned phrases
    # become SAM 3's per-frame text prompts (replacing or augmenting the
    # static list, depending on `gpt5_merge_with_curated`).
    #
    # Tagger output is content-addressed and cached on disk
    # (`gpt5_cache_dir/frame_*.gpt5.json`); a full 2 408-frame HGE build
    # is ~$96 once at high fidelity, then $0 on every re-run. The
    # seg_cache hash incorporates the GPT tagger's config-hash so changing
    # the model or prompt template invalidates the downstream
    # (instance_mask, class_mask) entries automatically.
    #
    # See `docs/report/05_vlm-driven-segmentation/findings.md` for the
    # research that picked GPT-5.5 over RAM++ / Qwen3-VL / Florence-2.
    use_gpt5_tagger: bool = False
    gpt5_cache_dir: Optional[Path] = None  # e.g. `data/maps/hge/gpt5_cache`
    gpt5_model: str = "gpt-5.5"
    gpt5_high_fidelity: bool = True
    gpt5_landmarks_only: bool = True       # drop incidental clutter from prompts
    gpt5_max_concurrency: int = 16         # parallel API calls; raise on Tier-5
    gpt5_merge_with_curated: bool = True   # union(VLM_phrases, sam3_prompts)


# --- coverage sanity check --------------------------------------------------


def _coverage_stats(
    class_ids: np.ndarray, class_id_to_name: dict[int, str]
) -> dict:
    """Compute per-class labeled-vertex counts + the overall coverage fraction.

    Returns a dict with:
      - `total_verts`
      - `labeled_verts`
      - `labeled_fraction`
      - `per_class`: {name: count} for every non-zero class
    """
    n_total = int(class_ids.size)
    nonzero = class_ids > 0
    n_labeled = int(nonzero.sum())

    per_class: dict[str, int] = {}
    if n_labeled > 0:
        ids, counts = np.unique(class_ids[nonzero], return_counts=True)
        for cid, c in zip(ids.tolist(), counts.tolist()):
            per_class[class_id_to_name.get(int(cid), f"class_{cid}")] = int(c)

    return {
        "total_verts": n_total,
        "labeled_verts": n_labeled,
        "labeled_fraction": (n_labeled / n_total) if n_total else 0.0,
        "per_class": per_class,
    }


def _log_coverage(stats: dict, min_fraction: float) -> None:
    frac = stats["labeled_fraction"]
    print(
        f"[build_hge] coverage: {stats['labeled_verts']:,} / "
        f"{stats['total_verts']:,} verts labeled ({frac * 100:.2f} %)"
    )
    for name, count in sorted(stats["per_class"].items(), key=lambda kv: -kv[1]):
        print(f"[build_hge]   {name:12s}  {count:>8,} verts")
    if min_fraction > 0 and frac < min_fraction:
        print(
            "[WARN] coverage below sanity threshold: "
            f"expected=>={min_fraction * 100:.0f} %, got={frac * 100:.2f} %, "
            "fallback=shipping anyway. Likely causes: frame/pose mismatch "
            "(see docs/report/02_hge-lift-frame-mismatch/findings.md), "
            "depth-unit mismatch, or a wrong mesh.",
            flush=True,
        )


# --- per-frame segmentation cache ------------------------------------------


def _seg_cache_config_hash(
    prompts: tuple[str, ...],
    confidence: float,
    structural_backbone: str = "mask2former",
    *,
    gpt5_tagger_hash: Optional[str] = None,
) -> str:
    """Stable 16-hex-char fingerprint of the segmentation-affecting knobs.

    Changes to `sam3_prompts`, `sam3_confidence`, or `structural_backbone`
    invalidate every cached frame because all three alter the combined
    (instance_mask, class_mask) output.

    The hash blob OMITS `structural_backbone` when it equals the historical
    default ("mask2former") so cache files written before the backbone knob
    existed still match. Any other backbone (e.g. "eomt") is included and
    therefore gets its own isolated cache partition.

    `gpt5_tagger_hash` is the GPT5Tagger.config_hash when per-frame VLM
    tagging is in use. Including it means swapping the GPT model, schema,
    or prompt-template version invalidates the downstream
    (instance_mask, class_mask) cache automatically — otherwise we'd
    serve stale masks generated against the old prompt set. Omitted from
    the blob when None so non-VLM builds keep their existing cache hash.
    """
    import hashlib

    blob_dict: dict[str, object] = {
        "prompts": sorted(prompts),
        "confidence": round(float(confidence), 6),
    }
    if structural_backbone != "mask2former":
        blob_dict["backbone"] = structural_backbone
    if gpt5_tagger_hash is not None:
        blob_dict["gpt5_tagger"] = gpt5_tagger_hash
    blob = json.dumps(blob_dict, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


_seg_cache_mismatch_warned: set[str] = set()


def _seg_cache_load(
    cache_dir: Path, frame_id: int, expected_hash: str
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return `(instance_mask, class_mask)` if a matching cache entry exists.

    Returns `None` for three distinct reasons (all logged loudly so
    silent-fallback bugs can't hide here):
      1. file missing — normal on first run for a frame
      2. hash mismatch — prompts/confidence changed; caller will re-run
      3. corrupt npz — io error, caller will re-run

    A re-run on miss overwrites the stale file automatically. Hash-mismatch
    warnings dedupe to one message per unique `(got_hash, expected_hash)`
    pair per process so 2408 stale files don't spam 2408 WARN lines.
    """
    path = cache_dir / f"frame_{frame_id:06d}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            got_hash = str(z["config_hash"].item()) if "config_hash" in z.files else ""
            if got_hash != expected_hash:
                key = f"{got_hash}->{expected_hash}"
                if key not in _seg_cache_mismatch_warned:
                    _seg_cache_mismatch_warned.add(key)
                    print(
                        f"[WARN] seg_cache: entries hash={got_hash!r} but "
                        f"expected={expected_hash!r} (config changed). "
                        f"First stale frame: {frame_id}. Re-running "
                        f"segmentation for all mismatched frames — this "
                        f"message prints once per unique mismatch.",
                        flush=True,
                    )
                return None
            return z["instance_mask"].astype(np.int32), z["class_mask"].astype(np.int16)
    except (OSError, ValueError, EOFError) as exc:
        print(
            f"[WARN] seg_cache: expected=readable {path}, got={type(exc).__name__}: "
            f"{exc}, fallback=re-run segmentation",
            flush=True,
        )
        return None


def _seg_cache_save(
    cache_dir: Path,
    frame_id: int,
    config_hash: str,
    instance_mask: np.ndarray,
    class_mask: np.ndarray,
    *,
    instance_phrases: Optional[dict[int, str]] = None,
) -> None:
    """Atomically write a frame's segmentation cache entry.

    `instance_phrases` (optional) is the per-frame map
    `{sam3_instance_id: prompt_phrase}` — i.e. the open-vocab text prompt
    that *generated* each foreground instance. We keep it in a sibling
    `.phrases.json` file rather than stuffing it into the .npz so the
    binary masks stay numpy-only (zero-cost reads) and the downstream
    open-vocab class voter can load just the JSON when it doesn't need
    the masks.

    Older caches without phrases sidecar fall back to S3DIS-13 voting in
    the lifter (loud [WARN] from the voter); new caches enable open-vocab
    cluster labels in the Habitat bundle.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"frame_{frame_id:06d}.npz"
    # np.savez_compressed auto-appends `.npz` if the given filename does not
    # end in `.npz`, so we use an explicit `.tmp.npz` suffix to get a valid
    # sibling that survives the atomic rename below.
    tmp = cache_dir / f"frame_{frame_id:06d}.tmp.npz"
    np.savez_compressed(
        tmp,
        instance_mask=instance_mask.astype(np.int32),
        class_mask=class_mask.astype(np.int16),
        config_hash=np.array(config_hash),
    )
    import os
    os.replace(tmp, path)

    if instance_phrases is not None:
        phrases_path = cache_dir / f"frame_{frame_id:06d}.phrases.json"
        phrases_tmp = cache_dir / f"frame_{frame_id:06d}.phrases.tmp.json"
        # Stringify the int keys so the JSON round-trips cleanly.
        payload = {
            "config_hash": config_hash,
            "phrases": {str(int(k)): v for k, v in instance_phrases.items()},
        }
        phrases_tmp.write_text(json.dumps(payload, ensure_ascii=False))
        os.replace(phrases_tmp, phrases_path)


def _print_seg_progress(
    i: int, n_frames: int, t0_frame: float, t_seg_start: float
) -> None:
    """Tiny shared helper to keep both branches of the seg loop logging
    the same way."""
    elapsed = time.time() - t_seg_start
    avg = elapsed / (i + 1)
    eta_min = (n_frames - (i + 1)) * avg / 60.0
    print(
        f"[build_hge]  seg {i+1:4d}/{n_frames:4d}  "
        f"{time.time() - t0_frame:.2f}s/frame (avg {avg:.2f}s) "
        f"ETA {eta_min:.1f}min"
    )


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


def _sam3_instance_phrase_lut(sam3_output) -> dict[int, str]:
    """Map each per-frame SAM 3 instance id (1..K) to the prompt phrase
    that generated it. Persisted to the cache so the downstream open-vocab
    class voter can label clusters with the actual heritage phrases (e.g.
    "marble bust on plinth") rather than the S3DIS-13 collapse.
    """
    return {i: prompt for i, prompt in enumerate(sam3_output.class_per_instance, start=1)}


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
    intrinsics + trajectory pose. If `alignment_global.txt` is present, every
    emitted pose is pre-composed with `T_absolute_from_pose_graph` so the
    returned `pose_T_wc` is in the absolute frame — matching the mesh that
    downstream consumers (the lifter, Habitat) work in. Without this step,
    trajectories live in the NavVis scanner's internal pose-graph frame while
    the Poisson mesh is in the absolute frame, and projection fails almost
    entirely (see `docs/report/02_hge-lift-frame-mismatch/findings.md`).
    """
    from cnsg.localization.capture_io import (
        parse_alignment_global, parse_images, parse_sensors, parse_trajectories,
    )
    from scipy.spatial.transform import Rotation

    images = parse_images(session / "images.txt")
    sensors = parse_sensors(session / "sensors.txt")
    poses = parse_trajectories(session / "trajectories.txt")
    # Capture format writes `alignment_global.txt` under `{session}/proc/`
    # (confirmed against real NavVis HGE data and scantools source layout).
    T_abs_pg = parse_alignment_global(session / "proc" / "alignment_global.txt")
    if T_abs_pg is None:
        print(
            f"[WARN] _iter_navvis_frames: alignment_global.txt not found in "
            f"{session}; emitting trajectories unaligned. Downstream lifting "
            f"WILL fail silently if the target mesh is in the absolute frame.",
            flush=True,
        )

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

        # Trajectory stores world_pg ← camera (world = NavVis pose-graph frame).
        R = Rotation.from_quat(
            [pose.qx, pose.qy, pose.qz, pose.qw]  # scipy expects xyzw
        ).as_matrix()
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, :3] = R
        T_wc[:3, 3] = [pose.tx, pose.ty, pose.tz]

        # Compose absolute ← camera = absolute ← pose_graph × pose_graph ← camera.
        # Verified against scantools/proc/alignment/scan_align.py which applies
        # `pose = T_session2w * pose` as a LEFT multiply.
        if T_abs_pg is not None:
            T_wc = T_abs_pg @ T_wc

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
    # Structural backbone is pluggable — pick by cfg.structural_backbone.
    # Both expose the same `.segment(PIL.Image) -> FrameSemantics` API.
    if cfg.structural_backbone == "eomt":
        from cnsg.segmentation.structural_eomt import EomtBackbone as _StructuralBackbone
    elif cfg.structural_backbone == "mask2former":
        from cnsg.segmentation.structural_ade20k import Mask2FormerBackbone as _StructuralBackbone
    else:
        raise ValueError(
            f"Unknown structural_backbone {cfg.structural_backbone!r}; "
            f"expected 'mask2former' or 'eomt'"
        )
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
    # When use_gpt5_tagger is on the per-frame prompt set is dynamic, so a
    # fixed lookup is insufficient — we extend it lazily below per-frame.
    prompt_to_class = {p: ade20k_name_to_s3dis(p) for p in cfg.sam3_prompts}

    # 3b. Optional: pre-fetch per-frame VLM tags. One async batch up front so
    # the seg loop can swap SAM 3 prompts per frame without blocking on the
    # API. Cache hits are O(1) reads — re-runs after the first build cost $0.
    frame_tags: dict[int, "FrameTags"] = {}
    gpt5_tagger_hash: Optional[str] = None
    if cfg.use_gpt5_tagger:
        from cnsg.segmentation.gpt5_tagger import GPT5Tagger

        gpt5_cache_dir = cfg.gpt5_cache_dir
        if gpt5_cache_dir is not None:
            gpt5_cache_dir = Path(gpt5_cache_dir)
        tagger = GPT5Tagger(
            cache_dir=gpt5_cache_dir,
            model=cfg.gpt5_model,
            high_fidelity=cfg.gpt5_high_fidelity,
        )
        gpt5_tagger_hash = tagger.config_hash
        items = [(f.frame_id, f.rgb_path) for f in frames]
        print(
            f"[build_hge] gpt5_tagger: {len(items)} frames, model={cfg.gpt5_model}, "
            f"high_fidelity={cfg.gpt5_high_fidelity}, "
            f"max_concurrency={cfg.gpt5_max_concurrency}, "
            f"cache={gpt5_cache_dir}"
        )
        t0 = time.time()
        import asyncio

        frame_tags = asyncio.run(
            tagger.tag_frames_async(items, max_concurrency=cfg.gpt5_max_concurrency)
        )
        # Loud [WARN] when frames got dropped (API failure on every retry).
        # The seg loop falls back to cfg.sam3_prompts for those frames.
        n_dropped = len(items) - len(frame_tags)
        if n_dropped > 0:
            print(
                f"[WARN] gpt5_tagger: expected={len(items)} tags, "
                f"got={len(frame_tags)} (dropped {n_dropped}), "
                f"fallback=use-sam3_prompts-only-for-dropped-frames",
                flush=True,
            )
        print(
            f"[build_hge] gpt5_tagger done: {len(frame_tags)} frames tagged "
            f"({time.time() - t0:.1f}s)"
        )

    # 4. Set up per-frame segmentation cache if requested; defer heavy model
    #    loads until we know we actually need them (pure cache hits skip both).
    cache_dir = cfg.seg_cache_dir
    cache_hash = _seg_cache_config_hash(
        cfg.sam3_prompts, cfg.sam3_confidence, cfg.structural_backbone,
        gpt5_tagger_hash=gpt5_tagger_hash,
    )
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[build_hge] seg_cache: {cache_dir} (hash={cache_hash})")

    m2f = None
    sam3 = None
    n_cache_hits = 0
    n_cache_misses = 0

    def _ensure_models_loaded():
        nonlocal m2f, sam3
        if m2f is not None and sam3 is not None:
            return
        t0 = time.time()
        m2f = _StructuralBackbone()
        print(
            f"[build_hge] {cfg.structural_backbone} backbone loaded "
            f"({time.time()-t0:.1f}s)"
        )
        t0 = time.time()
        sam3 = Sam3Segmenter(
            prompts=cfg.sam3_prompts, confidence_threshold=cfg.sam3_confidence,
        )
        print(f"[build_hge] SAM 3 loaded ({time.time()-t0:.1f}s)")

    # 5a. Per-frame segment-or-cache-hit, WITHOUT accumulating Frames in RAM.
    # Each NavVis frame at 1280×1920 carries ~25 MB of (depth + instance_mask +
    # class_mask). Pre-Phase-3-OOM-fix this loop appended every frame to a
    # `list[Frame]` then passed it to `lift_masks_to_3d` — at 2408 frames that
    # was ~59 GB, which the OOM killer reaped reliably on a 64 GB host. We now
    # split into two phases: Phase 5a ensures the cache is populated (only
    # `(instance_mask, class_mask)` is held briefly per frame, and even those
    # get freed at end of iteration); Phase 5b streams `Frame` objects to the
    # lifter via a generator so peak memory stays at one frame at a time.
    lift_frames_count = 0  # for the lift summary; len() of generator unavailable
    t_seg = time.time()
    for i, f in enumerate(frames):
        t0 = time.time()

        if cache_dir is not None:
            cached = _seg_cache_load(cache_dir, f.frame_id, cache_hash)
            if cached is not None:
                n_cache_hits += 1
                lift_frames_count += 1
                if (i + 1) % 10 == 0 or i == 0:
                    _print_seg_progress(i, len(frames), t0, t_seg)
                continue

        # Cache miss (or no cache configured). Segment, optionally save, drop.
        _ensure_models_loaded()
        rgb = Image.open(f.rgb_path).convert("RGB")
        m2f_out = m2f.segment(rgb)

        # When the GPT-5.5 tagger is on, swap SAM 3's prompt set to
        # (per-frame VLM phrases) ∪ (curated cfg.sam3_prompts). Using a
        # union rather than a replacement guarantees the curated heritage
        # baseline survives even if the VLM call dropped for this frame
        # (tag_frames_async drops failures with a loud [WARN]).
        if cfg.use_gpt5_tagger:
            tags = frame_tags.get(f.frame_id)
            if tags is not None:
                vlm_phrases = tags.sam3_prompts(
                    landmarks_only=cfg.gpt5_landmarks_only
                )
            else:
                vlm_phrases = []
            if cfg.gpt5_merge_with_curated:
                merged = list(dict.fromkeys(vlm_phrases + list(cfg.sam3_prompts)))
            else:
                merged = vlm_phrases or list(cfg.sam3_prompts)
            sam3.set_prompts(merged)
            # Extend the prompt → S3DIS class lookup lazily so new VLM phrases
            # also get a downstream class (falls through to "clutter" via
            # ade20k_name_to_s3dis when they don't match a known term).
            for p in merged:
                if p not in prompt_to_class:
                    prompt_to_class[p] = ade20k_name_to_s3dis(p)

        sam3_out = sam3.segment(rgb)
        sam3_class_lut = _sam3_instance_class_lut(sam3_out, prompt_to_class)
        sam3_phrase_lut = _sam3_instance_phrase_lut(sam3_out)
        instance_mask, class_mask = _combine_per_frame(
            sam3_mask=sam3_out.instance_mask,
            sam3_class_lut=sam3_class_lut,
            m2f_class=m2f_out.s3dis_labels,
        )
        n_cache_misses += 1
        if cache_dir is not None:
            _seg_cache_save(
                cache_dir, f.frame_id, cache_hash, instance_mask, class_mask,
                instance_phrases=sam3_phrase_lut,
            )
        lift_frames_count += 1
        # Free large per-frame tensors before the next iteration so peak RSS
        # stays bounded by one frame's worth even when cache_dir is None.
        del rgb, m2f_out, sam3_out, sam3_class_lut, sam3_phrase_lut, instance_mask, class_mask
        if (i + 1) % 10 == 0 or i == 0:
            _print_seg_progress(i, len(frames), t0, t_seg)
    seg_elapsed_min = (time.time() - t_seg) / 60
    print(
        f"[build_hge] segmentation done ({seg_elapsed_min:.1f}min) — "
        f"cache hits={n_cache_hits} misses={n_cache_misses}"
    )

    if cache_dir is None:
        # No-cache path is structurally incompatible with streaming because
        # we can't re-read what we never wrote. The Phase 3 plan always uses
        # a seg_cache; if a caller drops it on the floor and has 2408+ NavVis
        # frames, fail loud rather than OOM. For small smoke runs (max_frames
        # < ~500) it's still safe to materialize, but the user should opt in
        # explicitly.
        if lift_frames_count > 500:
            raise RuntimeError(
                f"cache_dir is None and {lift_frames_count} frames were "
                f"queued; materialising them all would OOM on a 64 GB host. "
                f"Pass --seg-cache-dir."
            )

    # 5b. Stream Frame objects from cache → lift. One frame in memory at a time.
    def _frame_stream():
        for f in frames:
            depth = _load_navvis_depth(f.depth_path)
            if cache_dir is not None:
                cached = _seg_cache_load(cache_dir, f.frame_id, cache_hash)
                if cached is None:
                    # We just populated the cache for this frame in Phase 5a;
                    # a miss now indicates a hash-stamp race or an io error.
                    raise RuntimeError(
                        f"frame {f.frame_id}: cache populated in Phase 5a "
                        f"but missing in Phase 5b (race or io error)"
                    )
                inst, cls = cached
            else:
                # Re-segment for this frame. Only safe for small smoke runs
                # — Phase 5a's count guard above caps `frames` to ≤500.
                _ensure_models_loaded()
                rgb = Image.open(f.rgb_path).convert("RGB")
                m2f_out = m2f.segment(rgb)
                sam3_out = sam3.segment(rgb)
                sam3_class_lut = _sam3_instance_class_lut(sam3_out, prompt_to_class)
                inst, cls = _combine_per_frame(
                    sam3_mask=sam3_out.instance_mask,
                    sam3_class_lut=sam3_class_lut,
                    m2f_class=m2f_out.s3dis_labels,
                )
            yield Frame(
                frame_id=f.frame_id,
                depth=depth,
                instance_mask=inst,
                class_mask=cls.astype(np.int16),
                T_world_cam=f.pose_T_wc,
                fx=f.fx, fy=f.fy, cx=f.cx, cy=f.cy,
            )

    # 6. Lift.
    t0 = time.time()
    verts_np = np.asarray(mesh.vertices, dtype=np.float32)
    lift_result = lift_masks_to_3d(
        verts_np, _frame_stream(), depth_tolerance=cfg.depth_tolerance_m,
    )
    print(
        f"[build_hge] lift: {lift_result.num_instances} instances "
        f"({time.time()-t0:.1f}s)"
    )

    # 6b. Coverage sanity check. Phase 3's first full HGE build shipped at
    # 0.44 % labeled-vertex coverage because no one noticed the
    # alignment-mismatch bug — Habitat loaded the output fine because the
    # exporter tolerates sparse labels. This gate makes a silent regression
    # loud at build time instead of at evaluation time. See
    # `docs/report/02_hge-lift-frame-mismatch/findings.md`.
    coverage_stats = _coverage_stats(
        lift_result.class_ids, {c.id: c.name for c in S3DIS_CLASSES}
    )
    _log_coverage(coverage_stats, cfg.min_coverage_fraction)

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
        external_stage_glb=cfg.external_stage_glb,
    )
    print(
        f"[build_hge] export: {manifest.num_instances} instances, "
        f"{manifest.num_regions} regions ({time.time()-t0:.1f}s)"
    )

    summary = {
        "navvis_session": str(cfg.navvis_session),
        "mesh_path": str(cfg.mesh_path),
        "num_frames_processed": lift_frames_count,
        "num_verts": int(len(mesh.vertices)),
        "num_faces": int(len(mesh.faces)),
        "lift_num_instances": int(lift_result.num_instances),
        "hierarchy_num_floors": n_floors,
        "hierarchy_num_rooms": n_rooms,
        "export_num_instances": int(manifest.num_instances),
        "export_num_regions": int(manifest.num_regions),
        "total_time_s": round(time.time() - t_all, 1),
        "seg_cache_hits": n_cache_hits,
        "seg_cache_misses": n_cache_misses,
        "coverage": coverage_stats,
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
    p.add_argument("--depth-tolerance", type=float, default=0.15)
    p.add_argument("--sam3-confidence", type=float, default=0.3)
    p.add_argument(
        "--structural-backbone", choices=("mask2former", "eomt"),
        default="mask2former",
        help="Structural-seg backbone. 'eomt' is faster + higher mIoU on "
             "ADE20K (CVPR'25 Highlight, tue-mps/ade20k_semantic_eomt_large_512). "
             "Changing this invalidates the seg_cache.",
    )
    p.add_argument(
        "--decimate-target-faces", type=int, default=300_000,
        help="Target face count after decimation. Habitat's CC-bbox pass "
             "crashes on a single CC above ~150k verts; 300k faces gives "
             "~150k verts. Pass 0 to disable.",
    )
    p.add_argument(
        "--seg-cache-dir", type=Path, default=None,
        help="If set, cache per-frame (instance_mask, class_mask) post-combine "
             "here. Cache hits skip Mask2Former + SAM 3 (~52 min of a full run) "
             "on subsequent invocations. Entries are hash-tagged with prompts "
             "and confidence so stale masks can't silently feed a rerun with "
             "different knobs.",
    )
    p.add_argument(
        "--external-stage-glb", type=Path, default=None,
        help="Pre-existing photorealistic stage GLB (e.g. NavVis' "
             "HGE.basis.glb). If set, the scene_dataset_config.json points "
             "the stage here instead of the palette fallback — mr_viewer "
             "renders a real building.",
    )
    p.add_argument(
        "--use-gpt5-tagger", action="store_true",
        help="Pre-tag every frame with OpenAI GPT-5.5 vision; SAM 3 then "
             "uses the per-frame VLM phrases (∪ curated --sam3_prompts by "
             "default) as text prompts. Requires OPENAI_API_KEY in env or "
             ".env. ~$96 once for 2,408 frames at high fidelity, $0 on "
             "re-runs (content-addressed cache).",
    )
    p.add_argument(
        "--gpt5-cache-dir", type=Path, default=None,
        help="Directory for GPT-5.5 per-frame tag cache. Default suggestion: "
             "data/maps/hge/gpt5_cache.",
    )
    p.add_argument(
        "--gpt5-model", default="gpt-5.5",
        help="OpenAI model id (e.g. gpt-5.5 or pinned snapshot "
             "gpt-5.5-2026-04-23). Pinning reproduces exactly across reruns.",
    )
    p.add_argument(
        "--gpt5-low-fidelity", action="store_true",
        help="Use low-fidelity image tokens for the GPT call (~$0.001/frame "
             "vs ~$0.04 high). Quality drops noticeably; smoke-only.",
    )
    p.add_argument(
        "--gpt5-include-clutter", action="store_true",
        help="Include landmark=False phrases (incidental clutter) in the "
             "SAM 3 prompt set. Default: landmarks-only.",
    )
    p.add_argument(
        "--gpt5-no-merge-curated", action="store_true",
        help="Use ONLY the GPT-5.5 phrases as SAM 3 prompts (not merged "
             "with cfg.sam3_prompts). Default: merge so curated heritage "
             "phrases survive even if the VLM call dropped.",
    )
    p.add_argument(
        "--gpt5-max-concurrency", type=int, default=16,
        help="Parallel GPT-5.5 calls. Tier-5 caps at 15k RPM / 40M TPM; "
             "default 16 is safe for any tier.",
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
        seg_cache_dir=args.seg_cache_dir,
        external_stage_glb=args.external_stage_glb,
        structural_backbone=args.structural_backbone,
        use_gpt5_tagger=args.use_gpt5_tagger,
        gpt5_cache_dir=args.gpt5_cache_dir,
        gpt5_model=args.gpt5_model,
        gpt5_high_fidelity=not args.gpt5_low_fidelity,
        gpt5_landmarks_only=not args.gpt5_include_clutter,
        gpt5_merge_with_curated=not args.gpt5_no_merge_curated,
        gpt5_max_concurrency=args.gpt5_max_concurrency,
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    build_hge_semantics(cfg)


if __name__ == "__main__":
    _main()
