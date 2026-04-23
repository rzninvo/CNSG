"""One-off tool: rewrite `<stem>.glb` with a brightened copy of
`<stem>.semantic.glb`'s vertex colors, leaving the semantic GLB + txt untouched.

Motivation: Habitat renders the stage GLB with `"shader_type": "flat"`, so the
rendered pixel color equals the raw uint8 palette byte — with the 24-bit LCG
palette drawn uniformly over [0, 255], a meaningful fraction of instances land
near black and the scene reads as dim. Post-Phase-3b-4 `export_habitat` now
writes a brightened stage by default (`stage_rgb_floor=80`), but pre-existing
builds have an identical stage + semantic pair. This script re-derives the
brightened stage from the existing semantic GLB without re-running the
55-min segmentation pipeline.

Concretely it:
  1. Parses `<stem>.semantic.glb`'s GLB container.
  2. Reads the float32 VEC4 `COLOR_0` accessor, applies Magnum's
     `toSrgb<UnsignedByte>()` to recover the exact uint8 palette bytes.
  3. Lifts RGB via `_brighten_rgb_u8` (leaves `(0,0,0)` Unknown verts black).
  4. Re-applies `_srgb_decode_u8` so Habitat's encode pass lands on the
     brightened bytes.
  5. Emits a new GLB with the new colors at `<stem>.glb`.

Usage:
    python -m cnsg.segmentation.rebrighten_stage_glb \\
        --semantic-glb data/maps/hge/HGE.semantic.glb \\
        --out-glb      data/maps/hge/HGE.glb \\
        --floor        80
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from cnsg.segmentation.export_habitat import (
    _STAGE_UNKNOWN_GRAY,
    _brighten_rgb_u8,
    _srgb_decode_u8,
)


def _srgb_encode_f32_to_u8(x: np.ndarray) -> np.ndarray:
    """Inverse of `_srgb_decode_u8`; matches Magnum's `toSrgb<UnsignedByte>()`."""
    clipped = np.clip(x, 0.0, 1.0)
    s = np.where(
        clipped <= 0.0031308,
        12.92 * clipped,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )
    return np.clip(np.round(s * 255), 0, 255).astype(np.uint8)


def _parse_glb(path: Path) -> tuple[dict, bytes, int]:
    """Return (gltf_json, bin_blob, bin_start_offset)."""
    data = path.read_bytes()
    if data[:4] != b"glTF":
        raise ValueError(f"{path} is not a GLB (missing magic)")
    json_len = struct.unpack("<I", data[12:16])[0]
    json_start = 20
    json_end = json_start + json_len
    gltf = json.loads(data[json_start:json_end].decode("utf-8"))
    bin_len = struct.unpack("<I", data[json_end : json_end + 4])[0]
    bin_start = json_end + 8
    bin_blob = bytes(data[bin_start : bin_start + bin_len])
    return gltf, bin_blob, bin_start


def rebrighten(semantic_glb: Path, out_glb: Path, *, floor: int = 80) -> dict:
    gltf, bin_blob, _ = _parse_glb(semantic_glb)

    prim = gltf["meshes"][0]["primitives"][0]
    color_acc_idx = prim["attributes"]["COLOR_0"]
    acc = gltf["accessors"][color_acc_idx]
    if acc.get("type") != "VEC4" or acc.get("componentType") != 5126:
        raise ValueError(
            f"COLOR_0 accessor not float32 VEC4 — got {acc.get('type')} / "
            f"componentType={acc.get('componentType')}; this script assumes "
            f"the exporter's float32 VEC4 encoding"
        )
    bv = gltf["bufferViews"][acc["bufferView"]]
    off = bv.get("byteOffset", 0)
    count = int(acc["count"])
    nbytes = count * 16  # float32 VEC4
    colors_f32 = np.frombuffer(
        bin_blob[off : off + nbytes], dtype=np.float32
    ).reshape(count, 4)

    # Decode → uint8 palette bytes (round-trip inverse of _srgb_decode_u8).
    rgb_u8 = _srgb_encode_f32_to_u8(colors_f32[:, :3])
    alpha_u8 = np.clip(np.round(colors_f32[:, 3] * 255), 0, 255).astype(np.uint8)

    # Lift RGB for labeled verts; paint Unknown verts mid-grey so the 99 %
    # of a sparse scan doesn't render as a pit of black. Matches the stage
    # convention in `export_habitat.py`.
    unknown = (rgb_u8 == 0).all(axis=1)
    lifted = _brighten_rgb_u8(rgb_u8, floor=floor)
    lifted[unknown] = _STAGE_UNKNOWN_GRAY

    # Re-encode to the float32 storage that Habitat's sRGB pass compensates for.
    new_rgb_f32 = _srgb_decode_u8(lifted)
    new_alpha_f32 = alpha_u8.astype(np.float32) / 255.0
    new_colors_f32 = np.concatenate(
        [new_rgb_f32, new_alpha_f32.reshape(-1, 1)], axis=1
    ).astype(np.float32)
    new_color_bytes = new_colors_f32.tobytes()
    if len(new_color_bytes) != nbytes:
        raise AssertionError(
            f"color byte-length changed: {nbytes} → {len(new_color_bytes)}"
        )

    new_bin = bytearray(bin_blob)
    new_bin[off : off + nbytes] = new_color_bytes

    raw_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_chunk = raw_json + b" " * ((-len(raw_json)) % 4)
    pad = (-len(new_bin)) % 4
    if pad:
        new_bin.extend(b"\0" * pad)
    total_len = 12 + 8 + len(json_chunk) + 8 + len(new_bin)
    out = bytearray()
    out += b"glTF"
    out += struct.pack("<II", 2, total_len)
    out += struct.pack("<II", len(json_chunk), 0x4E4F534A)
    out += json_chunk
    out += struct.pack("<II", len(new_bin), 0x004E4942)
    out += bytes(new_bin)
    out_glb.write_bytes(bytes(out))

    changed = int((lifted != rgb_u8).any(axis=1).sum())
    return {
        "verts": count,
        "verts_lifted": changed,
        "unknown_verts": int(unknown.sum()),
        "floor": floor,
        "out_glb": str(out_glb),
    }


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--semantic-glb", type=Path, required=True)
    p.add_argument("--out-glb", type=Path, required=True)
    p.add_argument("--floor", type=int, default=80)
    args = p.parse_args()

    summary = rebrighten(args.semantic_glb, args.out_glb, floor=args.floor)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
