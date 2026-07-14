#!/usr/bin/env python3
"""Batch-generate ``room_id_to_name_map.json`` for every scene in an MP3D dataset.

Loads habitat_sim once and iterates over every ``<dir>/<SCENE>/<SCENE>.glb`` under
the given dataset root, writing a native-region room map next to each scene (same
logic as scripts/generate_room_map_native.py). Existing maps are skipped unless
``--force`` is given.

Usage:
  python scripts/generate_mp3d_room_maps.py \
      --root    data/scene_datasets/mp3d \
      --dataset data/scene_datasets/mp3d/mp3d.scene_dataset_config.json
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import habitat_sim  # noqa: E402
from habitat_sim.utils.settings import default_sim_settings, make_cfg  # noqa: E402


def clean_room_name(name: str) -> str:
    name = (name or "").strip().lower()
    if not name or name in {"void", "unknown", "no label", "none"}:
        return "unknown_room"
    return name


def _center(aabb):
    """Return the aabb center as (x, y, z); ``center`` is a method in this build."""
    c = aabb.center
    c = c() if callable(c) else c
    return float(c[0]), float(c[1]), float(c[2])


def _bottom_y(aabb):
    """Return the floor (min-y) of an aabb, robust to method-vs-property."""
    b = getattr(aabb, "bottom", None)
    if b is not None:
        b = b() if callable(b) else b
        return float(b)
    mn = aabb.min
    mn = mn() if callable(mn) else mn
    return float(mn[1])


def _level_of(region) -> str:
    """MP3D region ids encode the level as a prefix, e.g. ``2_0`` -> level ``2``."""
    rid = region.id or ""
    if "_" in rid:
        return rid.split("_", 1)[0].strip("_")
    if region.level is not None and region.level.id is not None:
        return str(region.level.id)
    return "0"


def generate_for_scene(scene_path: str, dataset: str) -> dict:
    s = dict(default_sim_settings)
    s["scene"] = scene_path
    s["scene_dataset_config_file"] = dataset
    s["color_sensor"] = False
    s["semantic_sensor"] = False
    s["depth_sensor"] = False
    cfg = make_cfg(s)
    cfg.sim_cfg.create_renderer = False
    sim = habitat_sim.Simulator(cfg)
    result: dict[str, dict] = {}
    try:
        sem = sim.semantic_scene
        if sem is None:
            return result
        # Pass 1: gather region geometry + per-level floor heights so that all
        # rooms on the same level share one exact height value (needed for the
        # viewer's floor-number grouping and stair detection).
        regions = []
        level_bottoms: dict[str, list] = {}
        for region in sem.regions:
            rid = (region.id or "").strip("_").lower()
            if not rid:
                continue
            try:
                cx, _cy, cz = _center(region.aabb)
            except Exception:
                cx = cz = 0.0
            try:
                by = _bottom_y(region.aabb)
            except Exception:
                by = 0.0
            try:
                room = clean_room_name(
                    region.category.name() if region.category else ""
                )
            except Exception:
                room = "unknown_room"
            level = _level_of(region)
            regions.append((rid, room, cx, cz, level))
            level_bottoms.setdefault(level, []).append(by)
        level_height = {
            lvl: round(statistics.median(vals), 4)
            for lvl, vals in level_bottoms.items()
        }
        # Pass 2: emit the map, quantising every room's height to its level floor.
        for rid, room, cx, cz, level in regions:
            result[rid] = {
                "name": room,
                "position": [round(cx, 4), level_height.get(level, 0.0), round(cz, 4)],
            }
    finally:
        sim.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root containing <SCENE>/ folders")
    ap.add_argument("--dataset", required=True, help="scene_dataset_config.json")
    ap.add_argument("--force", action="store_true", help="regenerate even if a map exists")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    scenes = sorted(
        name
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
        and os.path.isfile(os.path.join(root, name, f"{name}.glb"))
    )
    print(f"[batch] {len(scenes)} scenes under {root}")

    ok, skipped, failed = 0, 0, 0
    for i, name in enumerate(scenes, 1):
        folder = os.path.join(root, name)
        scene_path = os.path.join(folder, f"{name}.glb")
        out = os.path.join(folder, "room_id_to_name_map.json")
        if os.path.isfile(out) and not args.force:
            print(f"[{i}/{len(scenes)}] {name}: skip (map exists)")
            skipped += 1
            continue
        try:
            result = generate_for_scene(scene_path, args.dataset)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)
            floors = sorted({round(v["position"][1], 1) for v in result.values()})
            print(
                f"[{i}/{len(scenes)}] {name}: {len(result)} regions, "
                f"{len(floors)} floor level(s) -> {os.path.relpath(out)}"
            )
            ok += 1
        except Exception as err:  # keep going on a single bad scene
            print(f"[{i}/{len(scenes)}] {name}: FAILED ({err})")
            failed += 1

    print(f"[batch] done: {ok} written, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
