#!/usr/bin/env python3
"""Generate ``room_id_to_name_map.json`` for a scene using NATIVE semantic regions.

Unlike the HM3D generator (which infers room types heuristically), this uses the
dataset's own region categories (e.g. Matterport3D regions: bedroom, kitchen,
living room, ...). Works for any Habitat dataset whose ``semantic_scene`` exposes
regions with a ``category`` (MP3D, and others).

The map key is ``region.id.strip("_").lower()`` (matching how the viewer looks up
regions), the name is the region's category, and the position is the region AABB
center (its height groups rooms into floors).

Usage:
  python scripts/generate_room_map_native.py \
      --scene  path/to/scene.glb \
      --dataset path/to/scene_dataset_config.json
Writes ``room_id_to_name_map.json`` next to the scene file.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None, help="output json (default: next to scene)")
    args = ap.parse_args()

    s = dict(default_sim_settings)
    s["scene"] = args.scene
    s["scene_dataset_config_file"] = args.dataset
    s["color_sensor"] = False
    s["semantic_sensor"] = False
    s["depth_sensor"] = False
    cfg = make_cfg(s)
    cfg.sim_cfg.create_renderer = False  # no GL context needed
    sim = habitat_sim.Simulator(cfg)

    sem = sim.semantic_scene
    result: dict[str, dict] = {}
    if sem is not None:
        for region in sem.regions:
            rid = (region.id or "").strip("_").lower()
            if not rid:
                continue
            try:
                room = clean_room_name(region.category.name() if region.category else "")
            except Exception:
                room = "unknown_room"
            try:
                c = region.aabb.center
                pos = [round(float(c[0]), 4), round(float(c[1]), 4), round(float(c[2]), 4)]
            except Exception:
                pos = [0, 0, 0]
            result[rid] = {"name": room, "position": pos}
    sim.close()

    out = args.out or os.path.join(os.path.dirname(args.scene), "room_id_to_name_map.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    print(f"[generate_room_map_native] wrote {len(result)} regions -> {out}")
    for k, v in result.items():
        print(f"   {k}: {v['name']}  (y={v['position'][1]})")


if __name__ == "__main__":
    main()
