"""Diagnose which filter in `lift_2d_to_3d._project_frame` is collapsing coverage.

Prints a per-stage vertex survival waterfall on a handful of real NavVis frames,
PLUS a direct test of the depth-convention question (planar Z vs Euclidean range)
by comparing the sensor depth value at each projected pixel against BOTH `z_cam`
and `||v_cam||`.

Run in the `cnsg-seg` env (needs numpy, torch, opencv, scipy, trimesh; Mask2Former
call is optional via --with-m2f).

    python -m cnsg.segmentation.debug_lift_coverage \
        --session mesh_pipeline/data/navvis_2022-02-06_12.55.11 \
        --mesh    mesh_pipeline/data/HGE_cut.voxelized.ply \
        --num-frames 3

What it prints per frame:
  - total verts
  - in_front, in_bounds
  - # pixels with sensor_depth > 0 at projected pixel
  - # passing |depth - z_cam|      < {0.05, 0.25, 1.0} m     (planar-Z hypothesis)
  - # passing |depth - ||v_cam|||  < {0.05, 0.25, 1.0} m     (Euclidean hypothesis)
  - histogram of (depth - z_cam) and (depth - ||v_cam||)  for in-bounds verts
  - fraction of image pixels that have non-zero structural labels (if --with-m2f)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation


def _load_frames(session: Path, max_frames: int, *, apply_alignment: bool):
    from cnsg.localization.capture_io import (
        parse_alignment_global, parse_images, parse_sensors, parse_trajectories,
    )

    images = parse_images(session / "images.txt")
    sensors = parse_sensors(session / "sensors.txt")
    poses = parse_trajectories(session / "trajectories.txt")
    T_abs_pg = (
        parse_alignment_global(session / "proc" / "alignment_global.txt")
        if apply_alignment else None
    )
    print(f"[debug] alignment_global.txt -> {'LOADED' if T_abs_pg is not None else 'not applied'}")

    raw_dir = session / "raw_data"
    depth_dir = session / "depth_maps"

    out = []
    for i, rec in enumerate(images):
        sensor = sensors.get(rec.sensor_id)
        if sensor is None:
            continue
        pose = poses.get((rec.timestamp, rec.sensor_id))
        if pose is None:
            continue
        rgb_path = raw_dir / rec.relative_path
        depth_path = depth_dir / (Path(rec.relative_path).stem + ".png")
        if not rgb_path.exists() or not depth_path.exists():
            continue

        R = Rotation.from_quat([pose.qx, pose.qy, pose.qz, pose.qw]).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [pose.tx, pose.ty, pose.tz]
        if T_abs_pg is not None:
            T = T_abs_pg @ T
        out.append({
            "frame_id": i,
            "sensor_id": rec.sensor_id,
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "T_world_cam": T,
            "fx": sensor.fx, "fy": sensor.fy,
            "cx": sensor.cx, "cy": sensor.cy,
        })
        if len(out) >= max_frames:
            break
    return out


def _load_depth_meters(path: Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if raw is None:
        raise RuntimeError(f"failed to load depth map {path}")
    return raw.astype(np.float32) / 1000.0


def _project_and_waterfall(verts_np: np.ndarray, frame: dict, device: str = "cuda") -> dict:
    dev = torch.device(device)
    verts = torch.from_numpy(verts_np).to(dev, dtype=torch.float64)

    T_wc = torch.from_numpy(frame["T_world_cam"]).to(dev, dtype=torch.float64)
    R = T_wc[:3, :3]
    t = T_wc[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t

    v_cam = verts @ R_inv.T + t_inv  # equivalent to R_inv @ v_col (row form)
    z = v_cam[:, 2]
    in_front = z > 1e-3

    fx, fy, cx, cy = frame["fx"], frame["fy"], frame["cx"], frame["cy"]
    u = (fx * v_cam[:, 0] / torch.clamp(z, min=1e-3)) + cx
    v = (fy * v_cam[:, 1] / torch.clamp(z, min=1e-3)) + cy
    ui = torch.round(u).to(torch.long)
    vi = torch.round(v).to(torch.long)

    depth = _load_depth_meters(frame["depth_path"])
    H, W = depth.shape
    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

    valid = in_front & in_bounds

    # Gather depth at projected pixels for valid verts.
    depth_t = torch.from_numpy(depth).to(dev)
    ui_v = ui[valid]
    vi_v = vi[valid]
    d_sensor = depth_t[vi_v, ui_v].to(torch.float64)
    z_v = z[valid]
    euc = torch.linalg.norm(v_cam[valid], dim=1)

    depth_present = d_sensor > 0
    gap_z = d_sensor - z_v
    gap_euc = d_sensor - euc

    def _n(mask: torch.Tensor) -> int:
        return int(mask.sum().item())

    result = {
        "rgb_path": str(frame["rgb_path"]),
        "sensor_id": frame["sensor_id"],
        "depth_shape": list(depth.shape),
        "n_verts": len(verts_np),
        "in_front": _n(in_front),
        "in_bounds_and_front": _n(valid),
        "sensor_depth_present": _n(depth_present),
        "planar_0p05": _n(depth_present & (gap_z.abs() < 0.05)),
        "planar_0p25": _n(depth_present & (gap_z.abs() < 0.25)),
        "planar_1p00": _n(depth_present & (gap_z.abs() < 1.00)),
        "euclid_0p05": _n(depth_present & (gap_euc.abs() < 0.05)),
        "euclid_0p25": _n(depth_present & (gap_euc.abs() < 0.25)),
        "euclid_1p00": _n(depth_present & (gap_euc.abs() < 1.00)),
    }

    # Histograms of gaps for depth-present verts only.
    for label, gap in (("gap_z_planar", gap_z), ("gap_euclidean", gap_euc)):
        g = gap[depth_present].cpu().numpy()
        if len(g) == 0:
            continue
        result[f"{label}_mean"] = float(np.mean(g))
        result[f"{label}_median"] = float(np.median(g))
        result[f"{label}_abs_q50"] = float(np.quantile(np.abs(g), 0.50))
        result[f"{label}_abs_q90"] = float(np.quantile(np.abs(g), 0.90))
        result[f"{label}_abs_q99"] = float(np.quantile(np.abs(g), 0.99))
    return result


def _m2f_coverage(rgb_path: Path) -> dict:
    """Run Mask2Former on one frame → return pixel coverage stats."""
    from cnsg.segmentation.structural_ade20k import Mask2FormerBackbone

    m2f = Mask2FormerBackbone()
    rgb = Image.open(rgb_path).convert("RGB")
    out = m2f.segment(rgb)
    lab = out.s3dis_labels
    H, W = lab.shape
    nonzero = int((lab > 0).sum())
    per_class = {
        int(c): int(((lab == c).sum()))
        for c in np.unique(lab)
    }
    return {
        "rgb_shape": [H, W],
        "total_px": H * W,
        "nonzero_px": nonzero,
        "nonzero_fraction": nonzero / (H * W),
        "per_class_px": per_class,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--num-frames", type=int, default=3)
    p.add_argument("--decimate-target-faces", type=int, default=300_000)
    p.add_argument("--with-m2f", action="store_true",
                   help="also run Mask2Former per frame to measure structural coverage")
    p.add_argument("--no-alignment", action="store_true",
                   help="skip alignment_global.txt (debug mode — reproduce old buggy pipeline)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    print(f"[debug] loading mesh {args.mesh}")
    mesh = trimesh.load(args.mesh, force="mesh")
    print(f"[debug]   raw: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")

    if args.decimate_target_faces and len(mesh.faces) > args.decimate_target_faces:
        import open3d as o3d
        o3m = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
            triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
        )
        dec = o3m.simplify_quadric_decimation(
            target_number_of_triangles=args.decimate_target_faces,
        )
        mesh = trimesh.Trimesh(
            vertices=np.asarray(dec.vertices),
            faces=np.asarray(dec.triangles),
            process=False,
        )
        print(f"[debug]   decimated: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")

    verts_np = np.asarray(mesh.vertices, dtype=np.float64)

    frames = _load_frames(args.session, args.num_frames, apply_alignment=not args.no_alignment)
    print(f"[debug] loaded {len(frames)} frames")

    print("")
    print("=== PROJECTION WATERFALL ===")
    for f in frames:
        r = _project_and_waterfall(verts_np, f, device=args.device)
        print(f"--- frame {f['frame_id']} sensor={r['sensor_id']} depth_shape={r['depth_shape']}")
        print(f"    total_verts              : {r['n_verts']:>8d}")
        print(f"    in_front                 : {r['in_front']:>8d}  ({r['in_front']/r['n_verts']*100:5.1f} %)")
        print(f"    in_bounds (+ in_front)   : {r['in_bounds_and_front']:>8d}  ({r['in_bounds_and_front']/r['n_verts']*100:5.1f} %)")
        print(f"    sensor_depth > 0         : {r['sensor_depth_present']:>8d}")
        print(f"    PLANAR   |d-z| < 5 cm    : {r['planar_0p05']:>8d}")
        print(f"    PLANAR   |d-z| < 25 cm   : {r['planar_0p25']:>8d}")
        print(f"    PLANAR   |d-z| < 1 m     : {r['planar_1p00']:>8d}")
        print(f"    EUCLID   |d-r| < 5 cm    : {r['euclid_0p05']:>8d}")
        print(f"    EUCLID   |d-r| < 25 cm   : {r['euclid_0p25']:>8d}")
        print(f"    EUCLID   |d-r| < 1 m     : {r['euclid_1p00']:>8d}")
        if "gap_z_planar_abs_q50" in r:
            print(f"    gap planar (sensor - z) : q50={r['gap_z_planar_abs_q50']:+.3f}  q90={r['gap_z_planar_abs_q90']:+.3f}  q99={r['gap_z_planar_abs_q99']:+.3f}  mean={r['gap_z_planar_mean']:+.3f}")
            print(f"    gap euclid (sensor - r) : q50={r['gap_euclidean_abs_q50']:+.3f}  q90={r['gap_euclidean_abs_q90']:+.3f}  q99={r['gap_euclidean_abs_q99']:+.3f}  mean={r['gap_euclidean_mean']:+.3f}")

    if args.with_m2f:
        print("")
        print("=== MASK2FORMER STRUCTURAL COVERAGE ===")
        for f in frames:
            r = _m2f_coverage(f["rgb_path"])
            print(f"--- frame {f['frame_id']}  img={r['rgb_shape']}  nonzero={r['nonzero_fraction']*100:5.1f}%   per-class: {r['per_class_px']}")


if __name__ == "__main__":
    main()
