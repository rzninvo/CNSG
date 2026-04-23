"""Unit tests for `_brighten_rgb_u8` — the visual lift applied to the stage GLB.

Rationale: Habitat's stage renders with `"shader_type": "flat"` (required so the
SemanticSensor's color→class lookup is byte-exact); the LCG palette in
`cnsg.segmentation.palette` draws uniformly across [0, 255] per channel, so a
substantial fraction of instances land in the near-black quadrant. Brightening
is only applied to `<stem>.glb` (the visual stage); `<stem>.semantic.glb` still
carries the exact palette so `.semantic.txt` decoding is unaffected.

These tests freeze three invariants:
  1. `floor=0` is a no-op (identical output bytes).
  2. The map is strictly monotonic and bijective per-channel on [0, 255] —
     two distinct LCG instance colors stay distinct after brightening, so we
     never merge adjacent instances in the rendered view.
  3. The output's minimum non-zero value meets the requested floor, and the
     maximum stays at 255 (no clipping of already-bright colors).
"""

from __future__ import annotations

import numpy as np

from cnsg.segmentation.export_habitat import _brighten_rgb_u8


def test_floor_zero_is_identity() -> None:
    rgb = np.arange(256, dtype=np.uint8).reshape(256, 1).repeat(3, axis=1)
    out = _brighten_rgb_u8(rgb, floor=0)
    assert np.array_equal(out, rgb)


def test_floor_80_is_monotonic_per_channel() -> None:
    """Monotonic non-decreasing: a palette color can only get brighter, never
    darker, so two LCG colors can't swap brightness ordering after the lift.
    """
    inputs = np.arange(256, dtype=np.uint8).reshape(-1, 1)
    out = _brighten_rgb_u8(inputs, floor=80)[:, 0]
    diffs = np.diff(out.astype(np.int32))
    assert (diffs >= 0).all(), "brightening must be monotonic non-decreasing"


def test_floor_80_preserves_distinctness_of_palette_colors() -> None:
    """The typical real scene has ≤ ~30 instances. The LCG palette sprays
    those over 2^24 colors; after brightening to the 176-value range
    [80, 255] per channel, the 3D RGB cube still has ~176^3 ≈ 5.5 M cells
    so practical collision rate is zero. Verify directly on the first 100
    palette entries — if this test fails, the stage GLB would render two
    different instances with the same colour.
    """
    from cnsg.segmentation.palette import instance_color

    palette = np.array([instance_color(i) for i in range(1, 101)], dtype=np.uint8)
    brightened = _brighten_rgb_u8(palette, floor=80)
    packed = (
        brightened[:, 0].astype(np.int64) * 65536
        + brightened[:, 1].astype(np.int64) * 256
        + brightened[:, 2].astype(np.int64)
    )
    assert len(np.unique(packed)) == 100, "palette colors must stay distinct after lift"


def test_floor_80_lifts_min_and_preserves_max() -> None:
    inputs = np.array([[0, 127, 255]], dtype=np.uint8)
    out = _brighten_rgb_u8(inputs, floor=80)
    assert out[0, 0] == 80, "zero must lift to the requested floor"
    assert out[0, 2] == 255, "255 must stay at 255 (no clipping)"
    assert 80 < out[0, 1] < 255


def test_rejects_out_of_range_floor() -> None:
    import pytest

    with pytest.raises(ValueError):
        _brighten_rgb_u8(np.zeros((1, 3), dtype=np.uint8), floor=256)


def test_semantic_glb_palette_is_not_brightened(tmp_path) -> None:
    """End-to-end: exporter writes an exact palette to `.semantic.glb` even
    when `stage_rgb_floor > 0`. Regression guard against accidentally wiring
    the brighten step into the semantic pass — that would break the whole
    SemanticSensor decode pipeline silently.
    """
    import struct, json
    import trimesh

    from cnsg.segmentation.export_habitat import _srgb_decode_u8, export_habitat
    from cnsg.segmentation.palette import instance_color

    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    class_ids = np.ones(len(mesh.vertices), dtype=np.int64)
    region_ids = np.ones(len(mesh.vertices), dtype=np.int64)

    manifest = export_habitat(
        mesh=mesh,
        per_vertex_class_id=class_ids,
        per_vertex_region_id=region_ids,
        class_id_to_name={1: "box"},
        out_dir=tmp_path,
        stem="toy",
        stage_rgb_floor=80,
    )

    # Parse `.semantic.glb`, extract COLOR_0 floats, convert back to uint8 via
    # Magnum's sRGB encode — this is what Habitat's loader sees. Must match
    # the raw LCG palette color for instance_id=1, not its brightened version.
    data = manifest.semantic_glb.read_bytes()
    assert data[:4] == b"glTF"
    json_len = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20 : 20 + json_len].decode("utf-8"))
    bin_start = 20 + json_len + 8
    prim = gltf["meshes"][0]["primitives"][0]
    color_acc = gltf["accessors"][prim["attributes"]["COLOR_0"]]
    bv = gltf["bufferViews"][color_acc["bufferView"]]
    off = bin_start + bv.get("byteOffset", 0)
    n = color_acc["count"]
    colors_f32 = np.frombuffer(data[off : off + n * 16], dtype=np.float32).reshape(n, 4)

    # Inverse of `_srgb_decode_u8` — matches Magnum's `toSrgb<UnsignedByte>()`.
    x = np.clip(colors_f32[:, :3], 0.0, 1.0)
    s = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    recovered = np.clip(np.round(s * 255), 0, 255).astype(np.uint8)

    expected = np.array(instance_color(1), dtype=np.uint8)
    # Every vertex belongs to instance_id=1, so every row must match.
    assert np.all(recovered[:, 0] == expected[0])
    assert np.all(recovered[:, 1] == expected[1])
    assert np.all(recovered[:, 2] == expected[2])
