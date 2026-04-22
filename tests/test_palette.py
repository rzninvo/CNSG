"""Unit tests for the deterministic RGB palette."""

from __future__ import annotations

import pytest

from cnsg.segmentation.palette import (
    hex_to_rgb,
    instance_color,
    instance_hex,
    pack_rgb,
)


def test_id_zero_is_rejected() -> None:
    """Instance 0 is the HM3D 'Unknown' sentinel — never assign it a color."""
    with pytest.raises(ValueError):
        instance_color(0)


def test_color_is_deterministic_across_calls() -> None:
    for i in (1, 2, 100, 65_537, 16_000_000):
        assert instance_color(i) == instance_color(i)
        assert instance_hex(i) == instance_hex(i)


def test_hex_matches_rgb_byte_for_byte() -> None:
    """The crucial Habitat round-trip: hex in .semantic.txt == vertex RGB bytes."""
    for i in range(1, 1000):
        r, g, b = instance_color(i)
        hx = instance_hex(i)
        assert hex_to_rgb(hx) == (r, g, b)
        # Habitat's 24-bit pack of the vertex colors must equal the int form of the hex.
        assert pack_rgb(r, g, b) == int(hx, 16)


def test_no_collisions_within_realistic_object_count() -> None:
    """Far more object IDs than any real building is going to have."""
    n = 50_000
    colors = {instance_color(i) for i in range(1, n + 1)}
    assert len(colors) == n


def test_no_id_produces_black() -> None:
    """(0,0,0) is reserved for HM3D's synthetic Unknown row. No real instance may be black."""
    for i in range(1, 10_000):
        assert instance_color(i) != (0, 0, 0)


def test_hex_uppercase_lowercase_both_parse() -> None:
    assert hex_to_rgb("ABCDEF") == hex_to_rgb("abcdef")


def test_hex_wrong_length_raises() -> None:
    with pytest.raises(ValueError):
        hex_to_rgb("ABC")
