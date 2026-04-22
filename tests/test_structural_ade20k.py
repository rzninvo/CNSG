"""Tests for `cnsg.segmentation.structural_ade20k`.

Split into (a) fast pure-function tests and (b) an integration test that
actually loads `facebook/mask2former-swin-large-ade-semantic` and segments
a real NavVis indoor frame. The integration test is marked `slow` and
skipped by default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cnsg.segmentation.structural_ade20k import remap_ade_labels_to_s3dis
from cnsg.segmentation.taxonomy import S3DIS_NAME_TO_ID


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# --- pure-function unit tests (no model load) -------------------------------


def test_remap_returns_correct_s3dis_ids_for_structural_classes() -> None:
    id2label = {0: "wall", 1: "floor", 2: "ceiling", 3: "chair"}
    ade = np.array([[0, 1], [2, 3]])  # (2, 2)
    out = remap_ade_labels_to_s3dis(ade, id2label)
    assert out.shape == ade.shape
    assert out.dtype == np.int64
    assert out[0, 0] == S3DIS_NAME_TO_ID["wall"]
    assert out[0, 1] == S3DIS_NAME_TO_ID["floor"]
    assert out[1, 0] == S3DIS_NAME_TO_ID["ceiling"]
    assert out[1, 1] == S3DIS_NAME_TO_ID["chair"]


def test_remap_clutters_out_of_range_ids() -> None:
    id2label = {0: "wall", 3: "door"}
    ade = np.array([0, 3, 7, 99])  # 7 and 99 absent from id2label
    out = remap_ade_labels_to_s3dis(ade, id2label)
    assert out[0] == S3DIS_NAME_TO_ID["wall"]
    assert out[1] == S3DIS_NAME_TO_ID["door"]
    # Out-of-range → clutter, not crash.
    assert out[2] == S3DIS_NAME_TO_ID["clutter"]
    assert out[3] == S3DIS_NAME_TO_ID["clutter"]


def test_remap_preserves_shape_for_larger_arrays() -> None:
    id2label = {0: "wall", 1: "floor"}
    ade = np.zeros((64, 128), dtype=np.int64)
    ade[:32, :] = 1  # floor half
    out = remap_ade_labels_to_s3dis(ade, id2label)
    assert out.shape == (64, 128)
    assert np.all(out[:32, :] == S3DIS_NAME_TO_ID["floor"])
    assert np.all(out[32:, :] == S3DIS_NAME_TO_ID["wall"])


def test_remap_multi_term_label() -> None:
    """ADE20K labels sometimes bundle multiple names ('door, double, door')."""
    id2label = {0: "door, double, door"}
    ade = np.array([0])
    out = remap_ade_labels_to_s3dis(ade, id2label)
    assert out[0] == S3DIS_NAME_TO_ID["door"]


def test_remap_unmapped_ade_class_is_clutter() -> None:
    id2label = {0: "wall", 1: "sky"}  # 'sky' is not in our remap table
    ade = np.array([0, 1])
    out = remap_ade_labels_to_s3dis(ade, id2label)
    assert out[0] == S3DIS_NAME_TO_ID["wall"]
    assert out[1] == S3DIS_NAME_TO_ID["clutter"]


def test_fp16_on_cpu_raises() -> None:
    """fp16 on CPU would run ~100× slower and be numerically unstable.
    The backbone should reject this combination loudly. (Finding #6.)"""
    import torch

    from cnsg.segmentation.structural_ade20k import Mask2FormerBackbone

    with pytest.raises(ValueError, match="float16 requires CUDA"):
        Mask2FormerBackbone(device="cpu", dtype=torch.float16)


# --- integration (real model, real NavVis frame, marked slow) ---------------


REPO_ROOT = Path(__file__).resolve().parent.parent
NAVVIS_IMAGE = (
    REPO_ROOT
    / "mesh_pipeline" / "data" / "navvis_2022-02-06_12.55.11"
    / "raw_data" / "images_undistr_center" / "00000-cam0__center.jpg"
)


@pytest.mark.slow
@pytest.mark.skipif(
    not NAVVIS_IMAGE.exists(),
    reason="NavVis image not on disk; run scripts/download_data.sh first.",
)
def test_integration_segment_real_indoor_frame() -> None:
    """Real Mask2Former + real NavVis indoor frame.

    Acceptance: at least one structural class (wall / floor / ceiling) covers
    >5% of pixels. A model that recognises no structural class on an indoor
    hallway is useless. This is a minimum-viability gate — quantitative mIoU
    belongs in a separate benchmark harness.
    """
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        pytest.skip("integration test needs CUDA")

    from cnsg.segmentation.structural_ade20k import Mask2FormerBackbone

    backbone = Mask2FormerBackbone()
    result = backbone.segment(Image.open(NAVVIS_IMAGE))

    total = result.s3dis_labels.size
    wall_frac = float((result.s3dis_labels == S3DIS_NAME_TO_ID["wall"]).sum()) / total
    floor_frac = float((result.s3dis_labels == S3DIS_NAME_TO_ID["floor"]).sum()) / total
    ceil_frac = float((result.s3dis_labels == S3DIS_NAME_TO_ID["ceiling"]).sum()) / total
    door_frac = float((result.s3dis_labels == S3DIS_NAME_TO_ID["door"]).sum()) / total

    print(
        f"\n[integration] NavVis frame: "
        f"wall={wall_frac:.2%} floor={floor_frac:.2%} "
        f"ceiling={ceil_frac:.2%} door={door_frac:.2%}"
    )
    structural_total = wall_frac + floor_frac + ceil_frac
    assert structural_total > 0.10, (
        f"structural classes (wall+floor+ceiling) cover only {structural_total:.2%} — "
        f"backbone may be unsuited for indoor scenes"
    )
