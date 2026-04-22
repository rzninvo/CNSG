"""Unit tests for `cnsg.segmentation.taxonomy`.

Validates the ADE20K→S3DIS remap table, the prompt list, and the
LUT-building helper used at inference time.
"""

from __future__ import annotations

import pytest

from cnsg.segmentation.taxonomy import (
    OBJECT_PROMPTS,
    S3DIS_CLASSES,
    S3DIS_ID_TO_NAME,
    S3DIS_NAME_TO_ID,
    ade20k_name_to_s3dis,
    build_ade20k_remap_lut,
)


def test_s3dis_ids_are_contiguous_from_zero() -> None:
    ids = [c.id for c in S3DIS_CLASSES]
    assert ids == list(range(len(S3DIS_CLASSES))), ids


def test_s3dis_id_zero_is_unknown_reserved() -> None:
    """ID 0 is reserved for 'drop/ignore' per the exporter contract."""
    assert S3DIS_ID_TO_NAME[0] == "unknown"


def test_s3dis_name_id_inverses() -> None:
    for cls in S3DIS_CLASSES:
        assert S3DIS_NAME_TO_ID[cls.name] == cls.id
        assert S3DIS_ID_TO_NAME[cls.id] == cls.name


def test_all_six_structural_classes_present() -> None:
    """Wall/floor/ceiling/door/window/column — the 6/6 that drove the backbone choice."""
    required = {"wall", "floor", "ceiling", "door", "window", "column"}
    assert required <= set(S3DIS_NAME_TO_ID.keys())


def test_stairs_is_distinct_from_clutter() -> None:
    """Stairs is an intentional extension past the S3DIS-13 canon (for navigation)."""
    assert "stairs" in S3DIS_NAME_TO_ID
    assert S3DIS_NAME_TO_ID["stairs"] != S3DIS_NAME_TO_ID["clutter"]


@pytest.mark.parametrize(
    "ade_name, expected_s3dis",
    [
        ("wall", "wall"),
        ("floor", "floor"),
        ("flooring", "floor"),
        ("ceiling", "ceiling"),
        ("door", "door"),
        ("window", "window"),
        ("windowpane", "window"),
        ("column", "column"),
        ("pillar", "column"),
        ("chair", "chair"),
        ("armchair", "chair"),
        ("table", "table"),
        ("coffee table", "table"),
        ("desk", "table"),
        ("sofa", "sofa"),
        ("couch", "sofa"),
        ("bookcase", "bookcase"),
        ("stairs", "stairs"),
        ("staircase", "stairs"),
        ("bed", "clutter"),
        ("refrigerator", "clutter"),
        # Unknown / absent classes fall through to clutter.
        ("this is not a real ade class", "clutter"),
        # Multi-term ADE20K label: first recognised term wins.
        ("doorway, door, opening", "door"),
        # Empty string handling.
        ("", "clutter"),
    ],
)
def test_ade20k_to_s3dis_mapping(ade_name: str, expected_s3dis: str) -> None:
    got = ade20k_name_to_s3dis(ade_name)
    assert got == S3DIS_NAME_TO_ID[expected_s3dis], (
        f"ade20k {ade_name!r}: got {S3DIS_ID_TO_NAME.get(got, got)!r}, "
        f"expected {expected_s3dis!r}"
    )


def test_build_lut_preserves_ade20k_indices() -> None:
    """The LUT produced here must be indexable by Mask2Former's raw class id."""
    id2label = {0: "wall", 1: "building", 2: "sky", 3: "floor", 14: "door"}
    lut = build_ade20k_remap_lut(id2label)
    # Highest id is 14 → lut has 15 entries.
    assert len(lut) == 15
    assert lut[0] == S3DIS_NAME_TO_ID["wall"]
    assert lut[3] == S3DIS_NAME_TO_ID["floor"]
    assert lut[14] == S3DIS_NAME_TO_ID["door"]
    # Gaps (ids 4..13 not in id2label) default to clutter.
    assert lut[5] == S3DIS_NAME_TO_ID["clutter"]
    # ADE 'building' and 'sky' don't map to any S3DIS class → clutter.
    assert lut[1] == S3DIS_NAME_TO_ID["clutter"]
    assert lut[2] == S3DIS_NAME_TO_ID["clutter"]


def test_build_lut_empty() -> None:
    assert build_ade20k_remap_lut({}) == []


def test_sam3_prompt_list_covers_navigation_essentials() -> None:
    required = {"door", "stairs", "elevator", "chair", "table"}
    assert required <= set(OBJECT_PROMPTS)


def test_sam3_prompts_have_no_duplicates() -> None:
    assert len(set(OBJECT_PROMPTS)) == len(OBJECT_PROMPTS)
