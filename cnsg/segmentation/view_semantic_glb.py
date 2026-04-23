"""Open3D viewer for the exported `<stem>.semantic.glb`.

The stored `COLOR_0` attribute is float32 VEC4 pre-decoded so Habitat's sRGB
encode pass lands on our target hex (see `export_habitat.py:_write_glb_with_
float_colors`). Naively loading into Open3D treats those floats as display
linear values and renders far dimmer than the original palette. This tool
recovers the original uint8 palette bytes (via `_srgb_encode_f32_to_u8`) and
passes them to Open3D so the semantic colors match `<stem>.semantic.txt` hex.

Usage:
    python -m cnsg.segmentation.view_semantic_glb \\
        --semantic-glb data/maps/hge/HGE.semantic.glb \\
        [--point-cloud]  # render as points instead of mesh
        [--stats-only]   # print per-class vert counts, no GUI
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np


def _parse_glb_vertices_and_colors(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (positions [N,3] float32, faces [F,3] int32, colors_u8 [N,3] uint8)."""
    data = path.read_bytes()
    if data[:4] != b"glTF":
        raise ValueError(f"{path} is not a GLB")
    json_len = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20 : 20 + json_len].decode("utf-8"))
    bin_start = 20 + json_len + 8

    prim = gltf["meshes"][0]["primitives"][0]
    accessors = gltf["accessors"]
    bvs = gltf["bufferViews"]

    def _read(acc_idx: int, dtype, shape_cols: int | None) -> np.ndarray:
        acc = accessors[acc_idx]
        bv = bvs[acc["bufferView"]]
        off = bin_start + bv.get("byteOffset", 0)
        count = int(acc["count"])
        comp_size = {5121: 1, 5123: 2, 5125: 4, 5126: 4}[acc["componentType"]]
        cols = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[acc["type"]]
        nbytes = count * cols * comp_size
        arr = np.frombuffer(data[off : off + nbytes], dtype=dtype)
        if shape_cols and shape_cols > 1:
            arr = arr.reshape(count, shape_cols)
        return arr

    positions = _read(prim["attributes"]["POSITION"], np.float32, 3)
    indices = _read(prim["indices"], np.uint32, 1).reshape(-1, 3).astype(np.int32)
    colors_f32 = _read(prim["attributes"]["COLOR_0"], np.float32, 4)

    # Invert _srgb_decode_u8 → recover the target uint8 palette bytes.
    x = np.clip(colors_f32[:, :3], 0.0, 1.0)
    s = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    colors_u8 = np.clip(np.round(s * 255), 0, 255).astype(np.uint8)
    return positions, indices, colors_u8


def _load_semantic_txt(path: Path) -> dict[tuple[int, int, int], str]:
    """Parse `<stem>.semantic.txt` → {(r,g,b): "class_name"}."""
    if not path.exists():
        return {}
    out: dict[tuple[int, int, int], str] = {}
    for line in path.read_text().splitlines()[1:]:  # skip header
        parts = line.split(",", 3)
        if len(parts) < 4:
            continue
        hexstr = parts[1]
        name = parts[2].strip().strip('"')
        r = int(hexstr[0:2], 16)
        g = int(hexstr[2:4], 16)
        b = int(hexstr[4:6], 16)
        out[(r, g, b)] = name
    return out


def _print_coverage(colors_u8: np.ndarray, palette: dict) -> None:
    keys = [tuple(row) for row in colors_u8.tolist()]
    counts = Counter(keys)
    total = len(keys)
    print(f"[view] {total:,} verts total")
    print(f"[view] {counts.get((0,0,0), 0):,} Unknown (hex 000000)")
    labeled = total - counts.get((0, 0, 0), 0)
    print(f"[view] {labeled:,} labeled ({100.0 * labeled / total:.2f} %)")
    print(f"[view] per-class:")
    for (rgb, c) in sorted(counts.items(), key=lambda kv: -kv[1]):
        if rgb == (0, 0, 0):
            continue
        name = palette.get(rgb, "??")
        hexstr = f"{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
        print(f"[view]   {hexstr}  {name:12s}  {c:6d} verts")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--semantic-glb", type=Path, required=True)
    p.add_argument(
        "--semantic-txt",
        type=Path,
        default=None,
        help="defaults to sibling <stem>.semantic.txt",
    )
    p.add_argument("--point-cloud", action="store_true")
    p.add_argument("--stats-only", action="store_true")
    args = p.parse_args()

    positions, indices, colors_u8 = _parse_glb_vertices_and_colors(args.semantic_glb)
    txt = args.semantic_txt or args.semantic_glb.with_suffix(".txt")
    palette = _load_semantic_txt(txt)

    _print_coverage(colors_u8, palette)

    if args.stats_only:
        return

    import open3d as o3d

    colors_f = colors_u8.astype(np.float64) / 255.0
    if args.point_cloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors_f)
        print(f"[view] rendering point cloud ({len(positions):,} verts)")
        o3d.visualization.draw_geometries([pcd])
    else:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(positions.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(indices)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors_f)
        mesh.compute_vertex_normals()
        print(f"[view] rendering mesh ({len(positions):,} verts, {len(indices):,} faces)")
        o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    main()
