"""Deterministic, collision-free RGB palette for per-object vertex coloring.

Habitat's HM3D semantic loader reads per-vertex uint8 RGB, packs it into a
24-bit integer `(R<<16)|(G<<8)|B`, and looks up `(semantic_id, region_id)` in
the `.semantic.txt`. The mapping must be byte-exact: every vertex of a given
instance uses the same RGB, and the hex in `.semantic.txt` must match those
bytes.

Design choice: a 24-bit linear congruential permutation. A multiplier coprime
with 2^24 (any odd integer works) makes
    f(i) = (i * M) mod 2^24
a bijection over [0, 2^24). We set M ≈ 2^24 / φ (odd) for visually-distinct
neighboring colors. Bijectivity guarantees **zero collisions** for any set of
distinct instance IDs up to 2^24 − 1 ≈ 16 million, far more than any real
building.

Reserved: `instance_id = 0` maps to `(0, 0, 0)` and is the HM3D "Unknown"
sentinel. Callers must start real object IDs at 1. See
`docs/report/01_architecture-lean-migration/habitat-format-spec.md`.
"""

from __future__ import annotations

# 2^24 / φ ≈ 10_371_398. Dropping one to the next odd integer keeps it coprime
# with 2^24. Any odd multiplier works; this one just makes consecutive IDs
# land in different hue neighborhoods.
_MULTIPLIER = 10_371_397

_COLOR_SPACE = 1 << 24
_MAX_INSTANCE_ID = _COLOR_SPACE - 1  # 16_777_215


def instance_color(instance_id: int) -> tuple[int, int, int]:
    """Deterministic uint8 RGB for instance_id ∈ [1, 2^24 − 1].

    Guaranteed unique across all valid instance_ids (bijective on [0, 2^24)).
    """
    if instance_id <= 0:
        raise ValueError(
            f"instance_id must be ≥ 1 (0 is reserved for the HM3D Unknown "
            f"sentinel); got {instance_id}"
        )
    if instance_id > _MAX_INSTANCE_ID:
        raise ValueError(
            f"instance_id must fit in 24 bits (≤ {_MAX_INSTANCE_ID}); got {instance_id}"
        )

    packed = (instance_id * _MULTIPLIER) & 0xFFFFFF
    r = (packed >> 16) & 0xFF
    g = (packed >> 8) & 0xFF
    b = packed & 0xFF
    return r, g, b


def instance_hex(instance_id: int) -> str:
    """Uppercase 6-digit hex for the `.semantic.txt` second column."""
    r, g, b = instance_color(instance_id)
    return f"{r:02X}{g:02X}{b:02X}"


def pack_rgb(r: int, g: int, b: int) -> int:
    """Pack uint8 RGB → the 24-bit integer Habitat's loader computes."""
    return (int(r) << 16) | (int(g) << 8) | int(b)


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """Inverse of instance_hex; accepts any-case 6-digit hex (no prefix)."""
    if len(hex_str) != 6:
        raise ValueError(f"expected 6-digit hex, got {hex_str!r}")
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return r, g, b
