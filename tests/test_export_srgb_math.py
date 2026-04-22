"""Unit tests for the sRGB round-trip math in `cnsg.segmentation.export_habitat`.

Habitat's `convertMeshColors(convertToSRGB=true)` path applies
`toSrgb<UnsignedByte>()` component-wise when loading semantic GLBs. Our exporter
pre-applies the inverse (`_srgb_decode_u8`) so the compensating encode lands
back on our target hex byte. This test freezes that round-trip: for every
possible uint8 component value, the decode→encode pipeline must recover the
original byte EXACTLY. A regression in either the decode formula or the encode
assumption would silently break color lookup, bucketing every vertex into
Habitat's "unknown" pile.

Reference: `docs/report/01_architecture-lean-migration/habitat-format-spec.md`
§"GLB vertex colors" and habitat-sim:
  - `src/esp/assets/BaseMesh.cpp:31-48` — `convertMeshColors` / `toSrgb`.
"""

from __future__ import annotations

import math

import numpy as np

from cnsg.segmentation.export_habitat import _srgb_decode_u8
from cnsg.segmentation.palette import _MULTIPLIER


def _habitat_to_srgb_ubyte(linear_float: float) -> int:
    """Replicate Magnum's `Math::Color3::toSrgb<UnsignedByte>()` component-wise.

    Piecewise function from the IEC 61966-2-1 sRGB spec, same one Habitat uses.
    """
    v = linear_float
    if v <= 0.0031308:
        s = 12.92 * v
    else:
        s = 1.055 * (v ** (1.0 / 2.4)) - 0.055
    s = max(0.0, min(1.0, s))
    return int(round(s * 255))


def test_srgb_roundtrip_is_byte_exact_for_all_uint8_inputs() -> None:
    """Every byte [0..255] must survive decode → encode → uint8 unchanged.

    Concretely: given target uint8 T, write `_srgb_decode_u8([T])` as float
    `COLOR_0`. Habitat reads the float, packs back via `toSrgb<UnsignedByte>`.
    Assert the recovered byte equals T for all 256 values.
    """
    inputs = np.arange(256, dtype=np.uint8).reshape(256, 1)
    decoded = _srgb_decode_u8(inputs)  # shape (256, 1), float32
    recovered = np.array(
        [_habitat_to_srgb_ubyte(float(decoded[i, 0])) for i in range(256)]
    )
    mismatches = np.where(recovered != np.arange(256))[0]
    assert len(mismatches) == 0, (
        f"sRGB round-trip broke for bytes: {mismatches.tolist()} "
        f"(recovered: {recovered[mismatches].tolist()})"
    )


def test_palette_multiplier_is_coprime_with_color_space() -> None:
    """The LCG bijectivity hinges on `gcd(_MULTIPLIER, 2^24) == 1`.

    Asserted mathematically rather than sampled so a future edit of `_MULTIPLIER`
    to an even number gets caught immediately.
    """
    assert math.gcd(_MULTIPLIER, 1 << 24) == 1, (
        f"_MULTIPLIER={_MULTIPLIER} must be coprime with 2^24 for the palette "
        f"permutation to be bijective; got gcd={math.gcd(_MULTIPLIER, 1 << 24)}"
    )
