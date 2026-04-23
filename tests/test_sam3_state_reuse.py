"""Unit test: `Sam3Segmenter.segment` calls `processor.set_image` ONCE per
image and reuses the state across all prompts.

Rationale: the image encoder (ViT) dominates SAM 3 per-frame cost. Naively
calling `set_image` inside every prompt iteration incurs ~N× the encode cost
for N prompts. Our HGE pipeline has 18 prompts per frame — re-using the
state drops per-frame SAM 3 time from ~800 ms to ~100 ms on RTX 5090.

This test substitutes the SAM 3 model + processor with mocks so we don't
depend on flash-attn-3 / CUDA / network weights, then asserts the call
count invariant on the real `Sam3Segmenter.segment` path.

Regression target: if someone refactors `segment()` and accidentally moves
`set_image` back inside the prompt loop, this test fails immediately.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image


class _FakeTensor:
    """Lightweight stand-in for torch tensors that SAM 3's processor would
    return — supports the `.to(dtype).cpu().numpy()` + `.numel()` chain
    without pulling in a GPU tensor."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def numel(self) -> int:
        return int(np.prod(self._arr.shape))

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._arr

    def detach(self):
        return self


def _make_fake_processor(masks_per_prompt: int = 1):
    """Build a MagicMock that records every call to `set_image` /
    `set_text_prompt`. Returns a tuple `(processor, fake_model)`.

    `set_text_prompt` always returns the same shape regardless of prompt.
    """
    processor = MagicMock()
    # `set_image` returns the usual state dict; record call count via spec.
    processor.set_image.return_value = {"original_height": 4, "original_width": 4}
    masks = _FakeTensor(np.ones((masks_per_prompt, 1, 4, 4), dtype=bool))
    scores = _FakeTensor(np.full((masks_per_prompt,), 0.8, dtype=np.float32))
    processor.set_text_prompt.return_value = {"masks": masks, "scores": scores}
    fake_model = MagicMock()
    return processor, fake_model


def test_set_image_called_once_per_frame() -> None:
    """Three prompts on one frame → `set_image` fires exactly once."""
    from cnsg.segmentation import sam3_per_frame

    fake_proc, fake_model = _make_fake_processor(masks_per_prompt=2)

    with patch.object(sam3_per_frame, "build_sam3_image_model", create=True,
                      return_value=fake_model, spec=True, spec_set=False) \
            if False else patch.dict("sys.modules", {}):
        # We bypass the real import by constructing Sam3Segmenter via __new__.
        seg = sam3_per_frame.Sam3Segmenter.__new__(sam3_per_frame.Sam3Segmenter)
        seg._device = "cpu"
        seg._autocast_dtype = torch.float32
        seg._model = fake_model
        seg._processor = fake_proc
        seg._prompts = ("chair", "door", "table")

        img = Image.new("RGB", (4, 4))
        out = seg.segment(img)

    assert fake_proc.set_image.call_count == 1, (
        f"expected set_image called exactly once, got {fake_proc.set_image.call_count}"
    )
    assert fake_proc.set_text_prompt.call_count == 3, (
        f"expected set_text_prompt called once per prompt, got "
        f"{fake_proc.set_text_prompt.call_count}"
    )
    # Each prompt returned 2 masks → 6 instances total, all three class labels present.
    assert len(out.class_per_instance) == 6
    assert set(out.class_per_instance) == {"chair", "door", "table"}


def test_no_prompts_still_calls_set_image_for_consistency() -> None:
    """Empty prompt list → set_image fires once, set_text_prompt never.

    (The image encode isn't wasted — callers typically set prompts right after
    constructing the segmenter; keeping the call in the main `segment` path
    makes the function's behaviour uniform.)
    """
    from cnsg.segmentation import sam3_per_frame

    fake_proc, fake_model = _make_fake_processor()

    seg = sam3_per_frame.Sam3Segmenter.__new__(sam3_per_frame.Sam3Segmenter)
    seg._device = "cpu"
    seg._autocast_dtype = torch.float32
    seg._model = fake_model
    seg._processor = fake_proc
    seg._prompts = ()

    img = Image.new("RGB", (4, 4))
    out = seg.segment(img)

    assert fake_proc.set_image.call_count == 1
    assert fake_proc.set_text_prompt.call_count == 0
    assert out.instance_mask.shape == (4, 4)
    assert out.instance_mask.sum() == 0
