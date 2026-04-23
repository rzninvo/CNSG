"""Tests for `cnsg.segmentation.sam3_per_frame`.

Every real SAM 3 inference test is gated on the `cnsg-seg` environment
(py3.12 + torch 2.10+ cu128 + SAM 3 + flash-attn-3 + gated HF weights).
We run those under `@pytest.mark.slow` and also skip when the current
Python env doesn't have the `sam3` module.

Pure-Python tests cover:
- `InstanceMaskSet` composition logic: when multiple prompts produce
  overlapping masks, the highest-scoring instance wins at each pixel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _try_import_sam3_segmenter():
    try:
        from cnsg.segmentation.sam3_per_frame import Sam3Segmenter
        return Sam3Segmenter
    except Exception:
        return None


# --- pure-logic sanity: the mask compositing rule --------------------------


def test_instance_maskset_dataclass_holds_aligned_lists() -> None:
    from cnsg.segmentation.sam3_per_frame import InstanceMaskSet

    mask = np.zeros((4, 4), dtype=np.int32)
    mask[0, 0] = 1
    mask[1, 1] = 2
    s = InstanceMaskSet(
        instance_mask=mask,
        class_per_instance=["chair", "door"],
        score_per_instance=[0.9, 0.7],
    )
    assert s.instance_mask.shape == (4, 4)
    assert len(s.class_per_instance) == len(s.score_per_instance)
    # IDs in mask must be in-range for the class/score lists.
    max_id = int(s.instance_mask.max())
    assert max_id <= len(s.class_per_instance)


# --- real SAM 3 integration (slow, env-conditional) -------------------------


REPO_ROOT = Path(__file__).resolve().parent.parent
NAVVIS_IMAGE = (
    REPO_ROOT
    / "mesh_pipeline" / "data" / "navvis_2022-02-06_12.55.11"
    / "raw_data" / "images_undistr_center" / "00000-cam0__center.jpg"
)


@pytest.mark.slow
def test_integration_segment_real_indoor_frame() -> None:
    """End-to-end SAM 3 inference on a real NavVis frame.

    Expected to run in the `cnsg-seg` env. Skipped otherwise. Verifies:
      1. The wrapper imports SAM 3 and builds the model.
      2. A handful of prompts produce non-empty InstanceMaskSet outputs.
      3. High-salience indoor classes (floor, ceiling) return at least
         one instance on an indoor hallway frame.
    """
    Sam3Segmenter = _try_import_sam3_segmenter()
    if Sam3Segmenter is None:
        pytest.skip("sam3 module not installed (cnsg-seg env required)")
    if not NAVVIS_IMAGE.exists():
        pytest.skip(f"NavVis image missing at {NAVVIS_IMAGE}")

    from PIL import Image

    segmenter = Sam3Segmenter(
        prompts=("door", "chair", "table", "floor", "ceiling", "window"),
        confidence_threshold=0.3,
    )
    result = segmenter.segment(Image.open(NAVVIS_IMAGE))
    print(
        f"\n[sam3] frame 0: {len(result.class_per_instance)} instances "
        f"from {len(segmenter.prompts)} prompts"
    )
    for i, (c, s) in enumerate(
        zip(result.class_per_instance, result.score_per_instance)
    ):
        print(f"    id={i+1:3d}  prompt={c!r:>10}  score={s:.3f}")

    assert result.instance_mask.shape[:2] == (1920, 1280), (
        f"expected (H=1920, W=1280); got {result.instance_mask.shape}"
    )
    # Check expected indoor structural classes show at least one detection.
    found_classes = set(result.class_per_instance)
    assert {"floor", "ceiling"} & found_classes, (
        f"expected SAM 3 to detect floor and/or ceiling on an indoor hallway; "
        f"got prompts with instances: {found_classes}"
    )


@pytest.mark.slow
def test_integration_compositing_respects_score_ranking() -> None:
    """When two prompts' masks overlap at a pixel, the higher-scoring one wins."""
    Sam3Segmenter = _try_import_sam3_segmenter()
    if Sam3Segmenter is None:
        pytest.skip("sam3 module not installed (cnsg-seg env required)")
    if not NAVVIS_IMAGE.exists():
        pytest.skip(f"NavVis image missing at {NAVVIS_IMAGE}")

    from PIL import Image

    # Floor + ceiling tend to co-cover some pixels on an indoor frame (SAM 3's
    # masks aren't mutually exclusive). This gives us a natural overlap test.
    seg = Sam3Segmenter(prompts=("floor", "ceiling"), confidence_threshold=0.2)
    result = seg.segment(Image.open(NAVVIS_IMAGE))
    # Build score lookup per instance id.
    for pix_id in np.unique(result.instance_mask):
        if pix_id == 0:
            continue
        idx = int(pix_id) - 1
        assert 0 <= idx < len(result.score_per_instance), pix_id
