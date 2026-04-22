"""ADE20K Mask2Former backbone + per-pixel remap to S3DIS-13.

Serves as the closed-vocabulary structural head of the Phase 3 pipeline.
Model: `facebook/mask2former-swin-large-ade-semantic` (HuggingFace, not gated,
pure PyTorch — verified Blackwell-compatible with torch 2.7+ cu128). Produces
per-pixel semantic labels for 150 ADE20K classes; we remap to S3DIS-13 via
`cnsg.segmentation.taxonomy.build_ade20k_remap_lut`.

Used for: every structural class we care about (wall, floor, ceiling, window,
door, column — 6/6 of the navigation-critical set). Non-structural ADE20K
classes either map to S3DIS `clutter` or to one of our furniture IDs (chair,
table, sofa, bookcase, stairs).

Authority: `docs/report/01_architecture-lean-migration/phase3-research.md`
§"Structural backbone — actionable spec".

Runtime env: `cnsg` (py3.9) — NOT the `cnsg-seg` py3.12 env, because
`transformers` Mask2Former doesn't need SAM 3's py3.12 stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image

from cnsg.segmentation.taxonomy import build_ade20k_remap_lut


DEFAULT_MODEL_ID = "facebook/mask2former-swin-large-ade-semantic"


@dataclass(frozen=True)
class FrameSemantics:
    """Per-pixel S3DIS labels for one RGB frame, plus provenance."""

    s3dis_labels: np.ndarray          # shape (H, W), int64, S3DIS class IDs
    ade_labels: np.ndarray            # shape (H, W), int64, raw ADE20K IDs
    source_size: tuple[int, int]      # (H, W) of the original input


def remap_ade_labels_to_s3dis(
    ade_labels: np.ndarray, id2label: dict[int, str]
) -> np.ndarray:
    """Per-pixel remap of ADE20K IDs → S3DIS IDs.

    Pure function: unit-testable without loading Mask2Former.

    Args:
        ade_labels: int array of shape (...,), values in [0, max(id2label)].
            Unknown IDs above the LUT's max fall back to S3DIS `clutter`.
        id2label: the Mask2Former checkpoint's id2label dict.

    Returns:
        Same-shape array of S3DIS class IDs.
    """
    lut = np.asarray(build_ade20k_remap_lut(id2label), dtype=np.int64)
    ade = np.asarray(ade_labels, dtype=np.int64)
    # Guard against out-of-bounds IDs — clip to `clutter`.
    from cnsg.segmentation.taxonomy import S3DIS_NAME_TO_ID

    clutter = S3DIS_NAME_TO_ID["clutter"]
    out = np.where(
        (ade >= 0) & (ade < len(lut)), lut[np.clip(ade, 0, len(lut) - 1)], clutter
    )
    return out.astype(np.int64)


class Mask2FormerBackbone:
    """Thin wrapper over HF Transformers' Mask2Former for semantic segmentation.

    Heavy resources (model weights + processor) are loaded once at
    construction and reused across frames. Not thread-safe (underlying torch
    model holds one CUDA stream).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        from transformers import (
            Mask2FormerForUniversalSegmentation,
            Mask2FormerImageProcessor,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            # fp16 on GPU is the inference sweet-spot; stick to fp32 on CPU.
            dtype = torch.float16 if device.startswith("cuda") else torch.float32
        # Guard against pathological combos. Running fp16 on CPU is ~100× slower
        # and numerically unstable; easier to fail loudly than debug silently.
        if dtype == torch.float16 and not device.startswith("cuda"):
            raise ValueError(
                f"dtype=torch.float16 requires CUDA; got device={device}"
            )
        self._device = device
        self._dtype = dtype

        self._processor = Mask2FormerImageProcessor.from_pretrained(model_id)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
        self._model = model.to(device=device, dtype=dtype).eval()

        # ADE20K's id2label ships with the checkpoint config.
        self._id2label: dict[int, str] = dict(self._model.config.id2label)

    @property
    def id2label(self) -> dict[int, str]:
        return dict(self._id2label)

    @torch.inference_mode()
    def segment(self, image: Image.Image) -> FrameSemantics:
        """Predict S3DIS labels per pixel for one PIL image."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self._processor(images=image, return_tensors="pt")
        # Move each tensor individually; non-tensor inputs are passed through.
        inputs = {k: v.to(device=self._device, dtype=self._dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else (v.to(self._device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        outputs = self._model(**inputs)

        # HF processor handles up-sampling back to the original image size.
        # Returns a list[Tensor(H, W)] with ADE20K class indices per pixel.
        W, H = image.size  # PIL order is (W, H)
        seg = self._processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(H, W)]
        )[0].cpu().numpy().astype(np.int64)

        s3dis = remap_ade_labels_to_s3dis(seg, self._id2label)
        return FrameSemantics(s3dis_labels=s3dis, ade_labels=seg, source_size=(H, W))

    def segment_many(
        self, images: Iterable[Image.Image]
    ) -> list[FrameSemantics]:
        """Sequential convenience. Not a true batched forward — Mask2Former's
        processor handles variable-size inputs, but each PIL image pushes
        through separately for simplicity. If HGE throughput demands it,
        Phase 3b can add a true padded-batch path.
        """
        return [self.segment(img) for img in images]
