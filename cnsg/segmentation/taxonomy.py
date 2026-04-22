"""Class taxonomy for the Phase 3 segmentation pipeline.

Two vocabularies live here:

1. **S3DIS-13** closed-vocab: the target label set for the structural backbone
   (walls, floors, ceilings, doors, windows, etc.) Each vertex in the final
   `.semantic.txt` gets one of these IDs.
2. **ADE20K-150**: the source vocabulary of the Mask2Former backbone we're
   using (`facebook/mask2former-swin-large-ade-semantic`). We translate ADE20K
   logits to S3DIS IDs via `ADE20K_TO_S3DIS`.

Also registered: the SAM 3 open-vocab prompt set — objects whose 2D masks we
want SAM 3 to return for the 2D→3D lift. Designed for landmark-style
navigation (doors, stairs, furniture, appliances, bathroom fixtures, utilities
that people use as referents when describing routes).

Authority for the ADE20K→S3DIS mapping: the research validation in
`docs/report/01_architecture-lean-migration/phase3-research.md`, grounded in
ADE20K's `objectInfo150.csv`.
"""

from __future__ import annotations

from dataclasses import dataclass


# -- S3DIS-13 ---------------------------------------------------------------
# Stanford 2D-3D-S's 13 categories. ID 0 is reserved for "unknown / drop"
# (matches `cnsg.segmentation.export_habitat`'s class_id==0 convention).


@dataclass(frozen=True)
class S3DISClass:
    id: int
    name: str


S3DIS_CLASSES: tuple[S3DISClass, ...] = (
    S3DISClass(id=0, name="unknown"),
    S3DISClass(id=1, name="wall"),
    S3DISClass(id=2, name="floor"),
    S3DISClass(id=3, name="ceiling"),
    S3DISClass(id=4, name="beam"),
    S3DISClass(id=5, name="column"),
    S3DISClass(id=6, name="window"),
    S3DISClass(id=7, name="door"),
    S3DISClass(id=8, name="table"),
    S3DISClass(id=9, name="chair"),
    S3DISClass(id=10, name="sofa"),
    S3DISClass(id=11, name="bookcase"),
    S3DISClass(id=12, name="board"),
    S3DISClass(id=13, name="clutter"),
    # Extension beyond the S3DIS-13 canon: "stairs" is structurally distinct
    # from "wall"/"floor"/"ceiling" and critical for multi-floor navigation.
    # Not in the original S3DIS taxonomy, but ADE20K has a direct class for it
    # (index 53) so we preserve it rather than collapse into clutter.
    S3DISClass(id=14, name="stairs"),
)

S3DIS_NAME_TO_ID: dict[str, int] = {c.name: c.id for c in S3DIS_CLASSES}
S3DIS_ID_TO_NAME: dict[int, str] = {c.id: c.name for c in S3DIS_CLASSES}


# -- ADE20K → S3DIS remap ---------------------------------------------------
# Key: ADE20K category name (lower-cased primary term; ADE20K often carries
# multi-name labels like "door, double, door" — we match on the FIRST term).
# Value: S3DIS class ID above. ADE20K categories not listed fall through to
# `clutter` (13) by default — see `ade20k_name_to_s3dis` for the rule.
#
# Source: ADE20K's objectInfo150.csv (rows 1..150, 1-indexed in the file,
# 0-indexed in Mask2Former's output). We key on NAME not INDEX because HF
# Transformers' Mask2Former returns class names via id2label and the raw
# index convention differs between the 1-indexed CSV and 0-indexed model.

_ADE20K_TO_S3DIS_BY_NAME: dict[str, int] = {
    # Structural (6/6 direct coverage — the whole reason to keep a closed-vocab head).
    "wall": 1,
    "floor": 2,
    "flooring": 2,
    "ceiling": 3,
    "windowpane": 6,
    "window": 6,
    "door": 7,
    "column": 5,
    "pillar": 5,
    # Furniture.
    "chair": 9,
    "armchair": 9,
    "swivel chair": 9,
    "seat": 9,
    "table": 8,
    "coffee table": 8,
    "desk": 8,
    "counter": 8,
    "sofa": 10,
    "couch": 10,
    "bookcase": 11,
    "shelf": 11,
    "cabinet": 11,
    # Architectural bonus (not in S3DIS-13 canon; explicit extension).
    "stairs": 14,
    "staircase": 14,
    "stairway": 14,
    "step": 14,
    # Named furniture that S3DIS-13 puts under "clutter" — we use 13 to avoid
    # loss but the downstream consumer knows these are distinct via the
    # OBJECT_PROMPTS list below.
    "bed": 13,
    "bathtub": 13,
    "toilet": 13,
    "sink": 13,
    "refrigerator": 13,
    "microwave": 13,
    "oven": 13,
    "dishwasher": 13,
    "stove": 13,
    "television": 13,
    "tv": 13,
    "computer": 13,
    "lamp": 13,
    "light": 13,
    "fireplace": 13,
    "trash can": 13,
    "plant": 13,
    "potted plant": 13,
    "rug": 13,
    "curtain": 13,
    "blind": 13,
    "radiator": 13,
}


def ade20k_name_to_s3dis(ade_name: str) -> int:
    """Map an ADE20K category name (any of its multi-terms) to an S3DIS ID.

    Unknown ADE20K categories fall back to `clutter` (13) per S3DIS convention.
    ADE20K labels occasionally bundle multiple names per class ("door, double,
    door"); we split on commas and try the first-known term.
    """
    if not ade_name:
        return S3DIS_NAME_TO_ID["clutter"]
    # ADE20K's multi-term labels: try each term in order; first hit wins.
    for term in (t.strip().lower() for t in ade_name.split(",")):
        if term in _ADE20K_TO_S3DIS_BY_NAME:
            return _ADE20K_TO_S3DIS_BY_NAME[term]
    return S3DIS_NAME_TO_ID["clutter"]


def build_ade20k_remap_lut(id2label: dict[int, str]) -> list[int]:
    """Given a Mask2Former `id2label` (ADE20K-150), produce a LUT indexed by
    ADE20K class ID → S3DIS class ID.

    The returned list has `len(id2label)` entries so a numpy remap is just
    `s3dis_labels = np.asarray(lut)[ade20k_labels]`.
    """
    max_id = max(id2label) if id2label else -1
    lut = [S3DIS_NAME_TO_ID["clutter"]] * (max_id + 1)
    for ade_id, ade_name in id2label.items():
        lut[ade_id] = ade20k_name_to_s3dis(ade_name)
    return lut


# -- SAM 3 open-vocab prompt list ------------------------------------------
# Objects worth asking SAM 3 for in an indoor navigation context. SAM 3 is
# exhaustive-concept (not referring-expression), so each prompt should name
# a *category* the user might navigate to, not a specific instance.
#
# Ordered by navigation salience: objects a person is most likely to use as
# a landmark come first.

OBJECT_PROMPTS: tuple[str, ...] = (
    # High-salience navigation landmarks.
    "door",
    "stairs",
    "elevator",
    "chair",
    "table",
    "desk",
    "sofa",
    "bed",
    # Structural (also covered by closed-vocab head, but include for fallback).
    "window",
    "column",
    # Utilities people often reference.
    "printer",
    "water fountain",
    "trash can",
    "toilet",
    "sink",
    "refrigerator",
    # Furniture secondary.
    "bookcase",
    "cabinet",
    "shelf",
    "television",
    "computer",
    "plant",
    # Often-relevant fixtures.
    "fireplace",
    "kitchen counter",
    "whiteboard",
    "projector screen",
)
