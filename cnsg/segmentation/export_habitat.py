"""Export a labeled mesh to the HM3D-compatible Habitat semantic format.

Replaces `mesh_pipeline/src/3D_segmentation/export_hm3d.py`. Produces a set of
files that Habitat's HM3D loader can ingest so `sim.semantic_scene.regions[]`
and `.objects[]` populate natively — no hand-authored `room_id_to_name_map.json`
hack in `mr_viewer.py`, no hardcoded `region_id=16`.

Format authority: `docs/report/01_architecture-lean-migration/habitat-format-spec.md`.

## Algorithm

Habitat's HM3D loader assigns ONE OBB per semantic object, picked as the
**largest connected component** when the same vertex color appears on multiple
disjoint parts (`ccLargestVolToUseForBBox_=1.0` at HM3DSemanticScene.cpp:231).
Smaller CCs sharing the same color lose their bbox silently — which is the
staircase-breaking bug in the legacy exporter.

This exporter sidesteps that by **emitting one instance per CC**:

1. Build a vertex adjacency graph from mesh edges.
2. Keep only edges where both endpoints share the same `(class_id, region_id)`.
3. Run connected-components over that filtered graph.
4. Each non-trivial CC becomes its own semantic object with a unique color and
   its own row in `.semantic.txt`. Every vertex of that CC gets that color in
   the GLB's `COLOR_0`.

Trade: more "objects" in the SemanticScene than there are physical objects.
The LLM prompt side already clusters by label, so this is fine — and the
bboxes are always correct (one CC per object, fully covered).

## Output layout (all in `out_dir`, sharing `stem`)

- `<stem>.glb`             — stage mesh (rendering; vertex-colored)
- `<stem>.semantic.glb`    — semantic mesh (same vertex colors; same geometry)
- `<stem>.semantic.txt`    — HM3D SSD
- `room_id_to_name_map.json` — project-convention, consumed by mr_viewer.py
- `<stem>.scene_dataset_config.json` — minimum config so Habitat finds the others

Not emitted (per spec):
- `.scn` — dead weight under our dataset-config
- `.basis.glb` — no benefit for vertex-color-only semantic meshes
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from cnsg.segmentation.palette import instance_color, instance_hex


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to `path` atomically: write to sibling .tmp then os.replace.

    Protects against partial exports when a write is interrupted.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# --- output bundle -----------------------------------------------------------


@dataclass(frozen=True)
class ExportManifest:
    """Paths of files emitted by `export_habitat`."""

    out_dir: Path
    stem: str
    stage_glb: Path
    semantic_glb: Path
    semantic_txt: Path
    room_id_to_name_map: Path
    scene_dataset_config: Path
    num_instances: int
    num_regions: int


# --- helpers -----------------------------------------------------------------


def _build_vertex_adjacency_same_label(
    mesh: trimesh.Trimesh,
    class_ids: np.ndarray,
    region_ids: np.ndarray,
) -> coo_matrix:
    """Adjacency matrix over mesh vertices, with edges only where labels match.

    Returned in COO form; `connected_components` consumes it directly. An edge
    u–v is included iff both endpoints share `(class_id, region_id)`.
    """
    edges = mesh.edges_unique  # (E, 2) int
    if len(edges) == 0:
        n = len(mesh.vertices)
        return coo_matrix((n, n), dtype=np.uint8)

    u = edges[:, 0]
    v = edges[:, 1]
    same_label = (class_ids[u] == class_ids[v]) & (region_ids[u] == region_ids[v])
    u = u[same_label]
    v = v[same_label]

    # Undirected — add both directions.
    data = np.ones(2 * len(u), dtype=np.uint8)
    row = np.concatenate([u, v])
    col = np.concatenate([v, u])
    n = len(mesh.vertices)
    return coo_matrix((data, (row, col)), shape=(n, n))


def _assign_instance_ids(
    mesh: trimesh.Trimesh,
    class_ids: np.ndarray,
    region_ids: np.ndarray,
    *,
    group_per_class_region: bool = False,
    min_verts_per_instance: int = 1,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Group vertices into instances. Returns `(per_vertex_instance_id, per_instance_label)`.

    Per-vertex instance IDs start at 1; class_id == 0 → instance 0 ("Unknown"
    sentinel, not emitted). Per-instance labels align by index (instance_id i
    has label list[i-1]).

    Stability: instance_ids are assigned deterministically, sorted by
    `(class_id, region_id, min_vertex_index)`. A face permutation of the
    input mesh will produce the same instance_id → (class, region) mapping.

    Filtering: CCs are accepted if they contain any vertex with class_id > 0.
    CCs with no incident triangle faces (isolated vertices from sparse
    projections) are still emitted — Habitat tolerates them, and filtering
    them out destroys valid labels when scan coverage is sparse (e.g. a
    partial NavVis run). Singletons below `min_cc_verts` are dropped.
    """
    n_verts = len(mesh.vertices)
    per_vertex_instance = np.zeros(n_verts, dtype=np.int64)

    if group_per_class_region:
        # One instance per (class, region) pair — no CC splitting. Keeps the
        # instance count small enough for Habitat's CC-bbox pass to not OOM
        # on large meshes with many labels. Trade: disconnected parts sharing
        # a label (staircase runs, split walls) get merged into one object;
        # Habitat's own "largest-CC per color" rule picks one bbox per
        # instance. Use this mode for production scan-scale pipelines
        # where per-CC bbox fidelity costs more than it's worth.
        pairs: dict[tuple[int, int], list[int]] = {}
        for i in range(n_verts):
            cls, reg = int(class_ids[i]), int(region_ids[i])
            if cls == 0:
                continue
            pairs.setdefault((cls, reg), []).append(i)
        # Filter singleton / sub-threshold instances that blow up Habitat's
        # CC-bbox pass when scan coverage is sparse (see build_hge notes).
        candidates_grouped = sorted(
            p for p in pairs.items() if len(p[1]) >= min_verts_per_instance
        )
        instance_labels: list[tuple[int, int]] = []
        for instance_id, ((cls, reg), members) in enumerate(candidates_grouped, start=1):
            per_vertex_instance[np.asarray(members, dtype=np.int64)] = instance_id
            instance_labels.append((cls, reg))
        return per_vertex_instance, instance_labels

    adjacency = _build_vertex_adjacency_same_label(mesh, class_ids, region_ids)
    n_cc, labels = connected_components(adjacency, directed=False)

    # Collect candidate CCs deterministically (class, region, min vertex idx).
    candidates: list[tuple[int, int, int, int]] = []  # (class, region, min_idx, cc)
    for cc in range(n_cc):
        members = np.flatnonzero(labels == cc)
        if len(members) == 0:
            continue
        cls = int(class_ids[members[0]])
        if cls == 0:
            continue  # untagged — stays at instance_id 0
        reg = int(region_ids[members[0]])
        # Sanity: all members share (class, region). The adjacency filter
        # only links same-label vertices, so a violation = filter bug.
        if not np.all(class_ids[members] == cls):
            raise AssertionError(
                f"CC {cc} straddles class boundaries — adjacency filter bug?"
            )
        if not np.all(region_ids[members] == reg):
            raise AssertionError(
                f"CC {cc} straddles region boundaries — adjacency filter bug?"
            )
        candidates.append((cls, reg, int(members[0]), cc))

    candidates.sort()

    instance_labels: list[tuple[int, int]] = []
    for instance_id, (cls, reg, _min_idx, cc) in enumerate(candidates, start=1):
        members = np.flatnonzero(labels == cc)
        per_vertex_instance[members] = instance_id
        instance_labels.append((cls, reg))

    return per_vertex_instance, instance_labels


def _per_vertex_colors_uint8(
    n_verts: int, per_vertex_instance: np.ndarray
) -> np.ndarray:
    """Per-vertex uint8 RGBA from per-vertex instance IDs.

    Vertices of instance 0 get black (0,0,0) — the HM3D Unknown bucket.
    """
    colors = np.zeros((n_verts, 4), dtype=np.uint8)
    colors[:, 3] = 255
    unique_instances = np.unique(per_vertex_instance)
    for iid in unique_instances:
        if iid == 0:
            continue
        r, g, b = instance_color(int(iid))
        mask = per_vertex_instance == iid
        colors[mask, 0] = r
        colors[mask, 1] = g
        colors[mask, 2] = b
    return colors


# Mid-grey assigned to Unknown verts in the stage GLB (visible under flat
# shading without drowning out real labeled instances).
_STAGE_UNKNOWN_GRAY: tuple[int, int, int] = (128, 128, 128)


def _brighten_rgb_u8(rgb_u8: np.ndarray, floor: int) -> np.ndarray:
    """Linearly lift uint8 RGB so the darkest channel value becomes `floor`.

    Purely a *visual* transform for the rendered stage GLB. Bijective inside
    each channel (preserves distinctness of the source palette) because it is
    a strictly-increasing linear map on [0, 255] → [floor, 255]. Semantic
    decoding must not touch this path — it operates on the exact palette in
    `<stem>.semantic.glb` / `.semantic.txt`.

    `floor=0` is the no-op (exact palette). With Habitat's `"shader_type":
    "flat"` stages there is no diffuse/ambient lift, so a raw LCG palette
    drawn uniformly from [0, 255] tends to render dim; `floor≈80` produces
    a readable-but-still-distinguishable stage without affecting the
    semantic sensor, whose decode pass reads the separate `.semantic.glb`.
    """
    if floor <= 0:
        return rgb_u8
    if floor > 255:
        raise ValueError(f"floor must be in [0, 255], got {floor}")
    scale = (255 - floor) / 255.0
    lifted = floor + rgb_u8.astype(np.float64) * scale
    return np.clip(np.round(lifted), 0, 255).astype(np.uint8)


def _srgb_decode_u8(rgb_u8: np.ndarray) -> np.ndarray:
    """Inverse of Habitat's `toSrgb<UnsignedByte>` applied component-wise on [0,1].

    Empirically, Habitat's semantic-mesh loader runs the uint8-VEC4 normalized
    `COLOR_0` attribute through a linear→sRGB encode pass before looking up
    the 24-bit color int in the SSD map. That means the bytes we write to
    `COLOR_0` get sRGB-encoded, and we must write the sRGB-decoded float
    values so the encoding lands back on our target hex.

    This pre-decode is a workaround for what looks like a Magnum/Habitat glTF
    color-space behavior that treats uint8 vertex colors as sRGB-tagged despite
    glTF spec saying vertex colors are linear by default. Documented behavior;
    verified end-to-end in `tests/test_export_habitat_roundtrip.py`.
    """
    v = rgb_u8.astype(np.float64) / 255.0
    linear = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    return linear.astype(np.float32)


def _write_glb_with_float_colors(
    mesh: trimesh.Trimesh, colors_uint8: np.ndarray, out_path: Path
) -> None:
    """Write a GLB whose `COLOR_0` pre-compensates for Habitat's sRGB encode pass.

    Stored as float32 VEC4 so there is no normalized-uint8 ambiguity. The RGB
    bytes are passed through `_srgb_decode_u8` before float conversion so that
    Habitat's `toSrgb<UnsignedByte>()` step lands on the target hex.

    See `docs/report/01_architecture-lean-migration/habitat-format-spec.md`
    §"GLB vertex colors".
    """
    positions = np.asarray(mesh.vertices, dtype=np.float32)
    if not np.all(np.isfinite(positions)):
        bad_count = int((~np.isfinite(positions)).any(axis=1).sum())
        raise ValueError(
            f"mesh has {bad_count} vertices with non-finite coordinates (NaN or Inf); "
            f"glTF's JSON accessor min/max must be finite, and downstream Habitat "
            f"loaders silently corrupt GLBs with non-finite accessor bounds"
        )
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    # Pre-decode RGB (alpha stays linear; Habitat strips alpha).
    rgb_f32 = _srgb_decode_u8(colors_uint8[:, :3])
    alpha_f32 = (colors_uint8[:, 3:4].astype(np.float32) / 255.0)
    colors_f32 = np.concatenate([rgb_f32, alpha_f32], axis=1).astype(np.float32)

    # Serialize per-attribute to bytes, align each sub-block to 4 bytes.
    def _pad4(data: bytes) -> bytes:
        pad = (-len(data)) % 4
        return data + b"\0" * pad

    pos_bytes = _pad4(positions.tobytes())
    color_bytes = _pad4(colors_f32.tobytes())
    idx_bytes = _pad4(faces.reshape(-1).tobytes())
    bin_blob = pos_bytes + color_bytes + idx_bytes

    pos_offset = 0
    color_offset = pos_offset + len(pos_bytes)
    idx_offset = color_offset + len(color_bytes)

    gltf: dict = {
        "asset": {"version": "2.0", "generator": "cnsg.segmentation.export_habitat"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "COLOR_0": 1},
                        "indices": 2,
                        "mode": 4,  # TRIANGLES
                    }
                ]
            }
        ],
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": pos_offset,
                "byteLength": len(positions.tobytes()),
                "target": 34962,  # ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": color_offset,
                "byteLength": len(colors_f32.tobytes()),
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": idx_offset,
                "byteLength": len(faces.reshape(-1).tobytes()),
                "target": 34963,  # ELEMENT_ARRAY_BUFFER
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "type": "VEC3",
                "count": int(positions.shape[0]),
                "min": positions.min(axis=0).tolist(),
                "max": positions.max(axis=0).tolist(),
            },
            {
                "bufferView": 1,
                "componentType": 5126,  # FLOAT — NOT uint8 normalized; see docstring.
                "type": "VEC4",
                "count": int(colors_f32.shape[0]),
            },
            {
                "bufferView": 2,
                "componentType": 5125,  # UNSIGNED_INT
                "type": "SCALAR",
                "count": int(faces.size),
            },
        ],
    }

    # glTF 2.0 §3.4.2: JSON chunk padded to 4-byte alignment with 0x20 (space).
    raw_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_chunk = raw_json + b" " * ((-len(raw_json)) % 4)

    # GLB assembly
    # header: magic (12 46 54 67) + version (2) + total length
    # chunk 0 JSON: length, type=JSON (0x4E4F534A)
    # chunk 1 BIN:  length, type=BIN  (0x004E4942)
    total_len = 12 + 8 + len(json_chunk) + 8 + len(bin_blob)
    out = bytearray()
    out += b"glTF"
    out += struct.pack("<II", 2, total_len)
    out += struct.pack("<II", len(json_chunk), 0x4E4F534A)
    out += json_chunk
    out += struct.pack("<II", len(bin_blob), 0x004E4942)
    out += bin_blob
    _atomic_write_bytes(out_path, bytes(out))


def _validate_category_name(name: str) -> str:
    """Make a category name safe for the .semantic.txt quoted-string format."""
    if '"' in name:
        raise ValueError(
            f'category name may not contain a double-quote: {name!r} (no escape '
            f'mechanism in HM3D SSD format)'
        )
    # LF / CR would terminate the row. Strip to a single line.
    return name.replace("\n", " ").replace("\r", " ")


# --- main entry point --------------------------------------------------------


def export_habitat(
    mesh: trimesh.Trimesh,
    per_vertex_class_id: np.ndarray,
    per_vertex_region_id: np.ndarray,
    class_id_to_name: dict[int, str],
    out_dir: Path,
    stem: str,
    region_id_to_name: dict[int, str] | None = None,
    region_id_to_position: dict[int, tuple[float, float, float]] | None = None,
    *,
    group_per_class_region: bool = False,
    min_verts_per_instance: int = 1,
    stage_rgb_floor: int = 80,
    external_stage_glb: Path | None = None,
) -> ExportManifest:
    """Write an HM3D-compatible semantic scene.

    Args:
        mesh: source mesh. Must have triangle indices (required for Habitat's
            CC analysis; point clouds or wireframes are rejected by Habitat's
            loader downstream).
        per_vertex_class_id: uint array shape (V,). 0 = drop/ignore.
        per_vertex_region_id: uint array shape (V,). 0 = "unknown_room".
        class_id_to_name: semantic class label strings, by id.
        out_dir: destination directory. Created if missing.
        stem: shared basename for all emitted files.
        region_id_to_name: optional map of region_id → human-readable room name.
            Missing keys default to "room_{id}" ("unknown_room" for id=0).
        region_id_to_position: optional map of region_id → [x, y, z] centroid
            used by `mr_viewer.py` as a floor-height heuristic. Missing keys
            default to the region's vertex-mean position.
        stage_rgb_floor: uint8 floor applied to `<stem>.glb` (the visual
            stage) to keep flat-shaded vertex colors readable. The LCG
            palette spans [0, 255] uniformly, so without a lift ~30 % of
            instances render as near-black under Habitat's `shader_type:
            flat` stage. Does NOT affect `<stem>.semantic.glb` —
            SemanticSensor decoding always uses the exact palette.
            Set to 0 for a no-op (identical stage + semantic).
        external_stage_glb: optional path to a pre-existing stage GLB
            (typically a photorealistic basis-compressed mesh from the
            NavVis pipeline, e.g. `mesh_pipeline/data/HGE.basis.glb`).
            If provided, the emitted scene_dataset_config.json references
            this file as the stage instead of the palette-colored fallback.
            The palette stage is still written to disk as `<stem>.glb` so
            semantic-visualization tools that expect it keep working.

    Returns:
        `ExportManifest` with paths and count summary.
    """
    n_verts = len(mesh.vertices)
    if per_vertex_class_id.shape != (n_verts,):
        raise ValueError(
            f"per_vertex_class_id shape {per_vertex_class_id.shape} != ({n_verts},)"
        )
    if per_vertex_region_id.shape != (n_verts,):
        raise ValueError(
            f"per_vertex_region_id shape {per_vertex_region_id.shape} != ({n_verts},)"
        )
    if len(mesh.faces) == 0:
        raise ValueError("mesh must have triangle faces; Habitat's loader needs indices")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_ids = per_vertex_class_id.astype(np.int64, copy=False)
    region_ids = per_vertex_region_id.astype(np.int64, copy=False)

    per_vertex_instance, instance_labels = _assign_instance_ids(
        mesh, class_ids, region_ids,
        group_per_class_region=group_per_class_region,
        min_verts_per_instance=min_verts_per_instance,
    )

    # 1. Colored meshes. `.semantic.glb` uses the exact LCG palette so
    # SemanticSensor's sRGB-encoded lookup lands on the hex in `.semantic.txt`;
    # Unknown verts stay `(0,0,0)` — the HM3D "ignore me" sentinel.
    # `.glb` (rendered stage) gets a brightened copy for the labeled verts and
    # a mid-gray baseline for the Unknown ones — flat-shaded `(0,0,0)` renders
    # as pitch black, which makes the whole scene unreadable whenever labeled
    # coverage is sparse (commonly the case with `group_per_class_region=True`
    # + a min-verts filter).
    colors_uint8 = _per_vertex_colors_uint8(len(mesh.vertices), per_vertex_instance)
    stage_glb = out_dir / f"{stem}.glb"
    semantic_glb = out_dir / f"{stem}.semantic.glb"
    _write_glb_with_float_colors(mesh, colors_uint8, semantic_glb)
    if stage_rgb_floor > 0:
        stage_colors = colors_uint8.copy()
        stage_colors[:, :3] = _brighten_rgb_u8(colors_uint8[:, :3], stage_rgb_floor)
        unknown = per_vertex_instance == 0
        stage_colors[unknown, :3] = _STAGE_UNKNOWN_GRAY
    else:
        stage_colors = colors_uint8
    _write_glb_with_float_colors(mesh, stage_colors, stage_glb)

    # 2. .semantic.txt (the SSD that drives scene.regions / .objects).
    # LF-only line terminator; `write_bytes` avoids Windows LF→CRLF translation
    # (the HM3D parser's `.back().trim(" ,")` does not strip `\r`, silently
    # truncating the last int on CRLF input).
    txt_path = out_dir / f"{stem}.semantic.txt"
    lines = ["HM3D Semantic Annotations"]
    for idx, (cls, reg) in enumerate(instance_labels, start=1):
        name = _validate_category_name(class_id_to_name.get(cls, f"class_{cls}"))
        lines.append(f'{idx},{instance_hex(idx)},"{name}",{reg}')
    _atomic_write_bytes(txt_path, ("\n".join(lines) + "\n").encode("utf-8"))

    # 3. room_id_to_name_map.json (mr_viewer.py convention).
    # Include every region_id that appears with a non-zero class_id, not just
    # those that produced at least one instance — otherwise unknown_room
    # disappears silently when every unknown vertex was also class_id=0.
    kept_region_ids: set[int] = {int(r) for r in np.unique(region_ids[class_ids != 0])}
    kept_region_ids.update(reg for _, reg in instance_labels)
    region_ids_seen = sorted(kept_region_ids)
    region_name_map: dict[str, dict] = {}
    for reg in region_ids_seen:
        name = (
            (region_id_to_name or {}).get(reg)
            or ("unknown_room" if reg == 0 else f"room_{reg}")
        )
        if region_id_to_position and reg in region_id_to_position:
            pos = list(region_id_to_position[reg])
        else:
            # Vertex-mean of everything labeled with this region, in Habitat
            # world-frame (Y-up). Our mesh is authored Z-up per the dataset
            # config's `"up": [0,0,1]` — Habitat rotates so mesh-Z becomes
            # world-Y. `mr_viewer.py` reads `position[1]` as floor height in
            # Habitat's world, so we emit (mesh_x, mesh_z, mesh_y).
            mask = region_ids == reg
            if np.any(mask):
                mean_mesh = mesh.vertices[mask].mean(axis=0).astype(float)
                pos = [float(mean_mesh[0]), float(mean_mesh[2]), float(mean_mesh[1])]
            else:
                pos = [0.0, 0.0, 0.0]
        region_name_map[str(int(reg))] = {"name": name, "position": pos}
    room_json = out_dir / "room_id_to_name_map.json"
    _atomic_write_bytes(
        room_json, (json.dumps(region_name_map, indent=2) + "\n").encode("utf-8")
    )

    # 4. Minimum scene_dataset_config.json so Habitat can load the scene by name.
    # Prefer an external photorealistic stage if the caller provided one
    # (typically `HGE.basis.glb` from mesh_pipeline). The palette stage
    # `<stem>.glb` is always written to disk, but the config points at the
    # prettier mesh when available so mr_viewer renders a real building
    # instead of coloured polygons.
    if external_stage_glb is not None:
        external_stage_glb = Path(external_stage_glb)
        stage_basename = external_stage_glb.name
        if external_stage_glb.parent.resolve() != out_dir.resolve():
            # Copy into out_dir so the config's relative path resolves.
            dest = out_dir / stage_basename
            if not dest.exists() or dest.stat().st_size != external_stage_glb.stat().st_size:
                import shutil
                shutil.copy2(external_stage_glb, dest)
    else:
        stage_basename = f"{stem}.glb"

    ds_config = out_dir / f"{stem}.scene_dataset_config.json"
    ds_config_json = json.dumps(
        {
            "stages": {
                "paths": {".glb": [stage_basename]},
                "default_attributes": {
                    "shader_type": "flat",
                    "up": [0, 0, 1],
                    "front": [0, 1, 0],
                    "origin": [0, 0, 0],
                    "semantic_descriptor_filename": "%%CONFIG_NAME_AS_ASSET_FILENAME%%.semantic.txt",
                    "semantic_asset": "%%CONFIG_NAME_AS_ASSET_FILENAME%%.semantic.glb",
                    "has_semantic_textures": False,
                },
            },
        },
        indent=2,
    ) + "\n"
    _atomic_write_bytes(ds_config, ds_config_json.encode("utf-8"))

    return ExportManifest(
        out_dir=out_dir,
        stem=stem,
        stage_glb=stage_glb,
        semantic_glb=semantic_glb,
        semantic_txt=txt_path,
        room_id_to_name_map=room_json,
        scene_dataset_config=ds_config,
        num_instances=len(instance_labels),
        num_regions=len(region_ids_seen),
    )
