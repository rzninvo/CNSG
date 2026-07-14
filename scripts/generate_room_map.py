#!/usr/bin/env python3
"""Generate ``room_id_to_name_map.json`` for HM3D scenes.

The mr_viewer expects, for every scene, a ``room_id_to_name_map.json`` next to
the scene ``.glb`` mapping each semantic ``region_id`` to a room ``name`` and a
representative ``position``. HM3D ships the region ids (in ``*.semantic.txt``)
and the per-object categories, but it does NOT provide room-type names or region
positions, so this script derives them:

* ``name``     -> inferred from the object categories contained in the region
                 (heuristic room-type classifier). Falls back to ``room <id>``.
* ``position`` -> centroid of the region computed from the semantic mesh
                 (``*.semantic.glb``). Only ``position[1]`` (height) is actually
                 used by the viewer, to group rooms into floors, but the full
                 centroid is stored for completeness/debugging.

The semantic mesh is loaded with trimesh (no OpenGL / simulator needed), and the
mesh coordinate frame is converted to Habitat's Y-up convention.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import trimesh
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "trimesh is required: install it with `pip install trimesh`"
    ) from exc


# --- Room-type inference -------------------------------------------------------
# Ordered list of (room_name, keywords). The first room whose keyword set has the
# highest weighted score wins. Keywords are matched as substrings of the object
# category names present in the region.
ROOM_RULES: List[Tuple[str, Dict[str, float]]] = [
    ("bathroom", {"toilet": 4, "shower": 3, "bathtub": 3, "sink": 1.5,
                  "towel": 1, "bidet": 3}),
    ("kitchen", {"oven": 4, "stove": 4, "refrigerator": 4, "fridge": 4,
                 "microwave": 3, "dishwasher": 3, "range hood": 3,
                 "kitchen": 3, "countertop": 1.5, "kitchen counter": 2}),
    ("bedroom", {"bed": 4, "headboard": 2, "nightstand": 2, "wardrobe": 1.5,
                 "crib": 3, "dresser": 1.5}),
    ("living room", {"couch": 3, "sofa": 3, "tv": 2, "television": 2,
                     "tv stand": 2, "coffee table": 2, "fireplace": 2,
                     "armchair": 1.5}),
    ("dining room", {"dining table": 4, "dining chair": 2}),
    ("office", {"computer desk": 4, "pc tower": 3, "printer": 3, "monitor": 2,
                "desk": 1.5, "office chair": 3, "keyboard": 1.5}),
    ("garage", {"car": 4, "garage door": 4, "bicycle": 1}),
    ("laundry room", {"washing machine": 4, "dryer": 4, "washer": 3}),
]

# Categories that carry no room-type signal.
GENERIC_CATEGORIES = {
    "wall", "floor", "ceiling", "door", "window", "window frame", "door frame",
    "frame", "handle", "unknown", "misc", "objects", "column", "beam",
    "light", "vent", "ventilation", "switch", "outlet", "baseboard",
    "molding", "curtain", "blinds",
}


def infer_room_name(categories: Dict[str, int]) -> str:
    """Return the most likely room type given a {category: count} mapping."""
    best_name: Optional[str] = None
    best_score = 0.0
    for name, keywords in ROOM_RULES:
        score = 0.0
        for cat, count in categories.items():
            cat_l = cat.lower()
            for kw, weight in keywords.items():
                if kw in cat_l:
                    score += weight * count
        if score > best_score:
            best_score = score
            best_name = name
    return best_name if best_name is not None else ""


# --- Semantic parsing / geometry ----------------------------------------------
def parse_semantic_txt(txt_path: str):
    """Parse an HM3D ``*.semantic.txt`` file.

    Returns ``(color_to_region, region_to_categories)`` where
    ``color_to_region`` maps an (R, G, B) tuple to a region id string and
    ``region_to_categories`` maps a region id to a {category: count} dict.
    """
    color_to_region: Dict[Tuple[int, int, int], str] = {}
    region_to_categories: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    with open(txt_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) != 4:
                continue
            try:
                int(parts[0])
            except ValueError:
                continue
            hex_color = parts[1].strip()
            if len(hex_color) != 6:
                continue
            rgb = (
                int(hex_color[0:2], 16),
                int(hex_color[2:4], 16),
                int(hex_color[4:6], 16),
            )
            category = parts[2].strip().strip('"')
            region = parts[3].strip()
            color_to_region[rgb] = region
            region_to_categories[region][category] += 1
    return color_to_region, {r: dict(c) for r, c in region_to_categories.items()}


def region_centroids_from_mesh(
    glb_path: str, color_to_region: Dict[Tuple[int, int, int], str]
) -> Dict[str, np.ndarray]:
    """Compute a per-region centroid (Habitat frame) from the semantic mesh.

    Each object instance in the HM3D semantic mesh is painted with the colour
    listed in ``*.semantic.txt``. We sample the per-vertex colour from the
    texture, map it to a region and average the world-space vertex positions.
    """
    scene = trimesh.load(glb_path, process=False)
    if isinstance(scene, trimesh.Trimesh):  # single-geometry fallback
        scene = trimesh.Scene(scene)

    # dump(concatenate=True) bakes every node transform into a single mesh, so
    # the vertices are already in world space (avoids manual scene-graph
    # traversal, whose node names differ from the geometry keys).
    mesh = scene.dump(concatenate=True)
    world = np.asarray(mesh.vertices, dtype=float)
    try:
        colors = np.asarray(mesh.visual.to_color().vertex_colors)[:, :3]
    except Exception:
        return {}
    if len(world) != len(colors):
        return {}

    sums: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(3))
    counts: Dict[str, int] = defaultdict(int)
    for rgb, region in color_to_region.items():
        mask = np.all(colors == np.array(rgb), axis=1)
        if not mask.any():
            continue
        sums[region] += world[mask].sum(axis=0)
        counts[region] += int(mask.sum())

    # Convert the trimesh/glTF world frame to Habitat's convention. Validated
    # against the reference map of scene 00800:
    #   habitat_x      = world_x
    #   habitat_height = world_z   (index 1, the only component the viewer uses)
    #   habitat_depth  = -world_y
    centroids: Dict[str, np.ndarray] = {}
    for region, total in sums.items():
        c = total / max(counts[region], 1)
        centroids[region] = np.array([c[0], c[2], -c[1]])
    return centroids


def quantize_floor_heights(
    heights: Dict[str, float], gap: float = 1.5
) -> Dict[str, float]:
    """Group per-region heights into floors and return a per-region floor height.

    The viewer derives a room's floor from the *set* of distinct heights, so all
    rooms on the same floor must share an identical height value. We sort the
    region heights and start a new floor whenever the gap between consecutive
    heights exceeds ``gap`` metres; every region then inherits the mean height
    of its floor cluster.
    """
    if not heights:
        return {}
    ordered = sorted(heights.items(), key=lambda kv: kv[1])
    clusters: List[List[Tuple[str, float]]] = [[ordered[0]]]
    for region, height in ordered[1:]:
        if height - clusters[-1][-1][1] > gap:
            clusters.append([])
        clusters[-1].append((region, height))

    quantized: Dict[str, float] = {}
    for cluster in clusters:
        floor_height = round(float(np.mean([h for _, h in cluster])), 2)
        for region, _ in cluster:
            quantized[region] = floor_height
    return quantized


def build_map(scene_dir: str) -> Dict[str, dict]:
    """Build the room_id_to_name map for a single HM3D scene directory."""
    txt_files = glob.glob(os.path.join(scene_dir, "*.semantic.txt"))
    glb_files = glob.glob(os.path.join(scene_dir, "*.semantic.glb"))
    if not txt_files or not glb_files:
        raise FileNotFoundError(
            f"Scene {scene_dir} is missing *.semantic.txt or *.semantic.glb"
        )

    color_to_region, region_categories = parse_semantic_txt(txt_files[0])
    centroids = region_centroids_from_mesh(glb_files[0], color_to_region)

    # First pass: decide the room name for every region and collect the raw
    # height of the regions we keep, so heights can be quantized into floors.
    resolved: Dict[str, dict] = {}
    raw_heights: Dict[str, float] = {}
    for region in sorted(
        region_categories, key=lambda r: int(r) if r.lstrip("-").isdigit() else 0
    ):
        categories = {
            cat: n
            for cat, n in region_categories[region].items()
            if cat.lower() not in GENERIC_CATEGORIES
        }
        name = infer_room_name(categories)
        centroid = centroids.get(region)

        # Regions without geometry or without any meaningful object are ignored
        # by the viewer, so label them explicitly as unknown_room.
        if centroid is None or not name:
            resolved[region] = None
            continue
        resolved[region] = {"name": name, "centroid": centroid}
        raw_heights[region] = float(centroid[1])

    floor_heights = quantize_floor_heights(raw_heights)

    # Second pass: build the map, disambiguating duplicate room names and
    # snapping each kept region to its floor height.
    used_names: Dict[str, int] = defaultdict(int)
    room_map: Dict[str, dict] = {}
    for region in sorted(
        resolved, key=lambda r: int(r) if r.lstrip("-").isdigit() else 0
    ):
        info = resolved[region]
        if info is None:
            room_map[region] = {"name": "unknown_room", "position": [0, 0, 0]}
            continue

        name = info["name"]
        used_names[name] += 1
        if used_names[name] > 1:
            name = f"{name} {used_names[name]}"

        centroid = info["centroid"]
        room_map[region] = {
            "name": name,
            "position": [round(float(centroid[0]), 2),
                         floor_heights[region],
                         round(float(centroid[2]), 2)],
        }
    return room_map


def find_scene_dirs(root: str) -> List[str]:
    """Return scene directories under ``root`` that contain semantic annots."""
    dirs = []
    for txt in glob.glob(os.path.join(root, "*", "*.semantic.txt")):
        dirs.append(os.path.dirname(txt))
    return sorted(set(dirs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-dir",
        help="A single HM3D scene directory (containing *.semantic.glb/.txt).",
    )
    parser.add_argument(
        "--root",
        default="data/scene_datasets/hm3d/minival",
        help="Root directory to scan for scenes when --scene-dir is omitted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate the map even if room_id_to_name_map.json already exists.",
    )
    args = parser.parse_args()

    scene_dirs = [args.scene_dir] if args.scene_dir else find_scene_dirs(args.root)
    if not scene_dirs:
        raise SystemExit(f"No semantic scenes found under {args.root}")

    for scene_dir in scene_dirs:
        out_path = os.path.join(scene_dir, "room_id_to_name_map.json")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[skip] {out_path} already exists (use --overwrite)")
            continue
        try:
            room_map = build_map(scene_dir)
        except FileNotFoundError as err:
            print(f"[skip] {err}")
            continue
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(room_map, handle, indent=4)
        named = sum(
            1 for v in room_map.values() if v["name"] != "unknown_room"
        )
        print(f"[ok]   {out_path}  ({named}/{len(room_map)} rooms named)")


if __name__ == "__main__":
    main()
