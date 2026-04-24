"""EoMT backbone — drop-in replacement for the Mask2Former structural head.

EoMT ("Your ViT is Secretly a Segmentation Model", CVPR'25 Highlight —
arXiv:2503.19108) is an encoder-only transformer that matches Mask2Former's
accuracy on ADE20K semantic segmentation while running ~4× faster on ViT-L.
Checkpoint: `tue-mps/ade20k_semantic_eomt_large_512` (HF hub, open license,
ADE20K-150 classes).

Why swap:
  - Same ADE20K-150 label space as our existing Mask2Former model → our
    ADE→S3DIS remap LUT (`cnsg.segmentation.taxonomy`) carries over unchanged.
  - Higher mIoU + substantially faster inference (the paper claims 4× on L).
  - Pure PyTorch + `transformers` — no custom CUDA ops; known to run on
    Blackwell / cu128 (we verified `AutoImageProcessor.from_pretrained` works
    in the `cnsg-seg` env before writing this module).

API matches `Mask2FormerBackbone` exactly (same `FrameSemantics` return type)
so `build_hge._ensure_models_loaded` can pick either without code changes.

Runtime env: any env with torch 2.1+ and transformers 4.45+.
Reference: https://github.com/tue-mps/eomt
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image

from cnsg.segmentation.taxonomy import build_ade20k_remap_lut


DEFAULT_MODEL_ID = "tue-mps/ade20k_semantic_eomt_large_512"


@dataclass(frozen=True)
class FrameSemantics:
    """Per-pixel S3DIS labels for one RGB frame, plus provenance.

    Intentionally identical shape to `structural_ade20k.FrameSemantics` — the
    lifter treats them interchangeably.
    """

    s3dis_labels: np.ndarray          # (H, W) int64, S3DIS class IDs
    ade_labels: np.ndarray            # (H, W) int64, raw ADE20K IDs
    source_size: tuple[int, int]      # (H, W) of the input image


def remap_ade_labels_to_s3dis(
    ade_labels: np.ndarray, id2label: dict[int, str]
) -> np.ndarray:
    """Per-pixel remap of ADE20K IDs → S3DIS IDs.

    Same pure-function pattern as `structural_ade20k.remap_ade_labels_to_s3dis`.
    Factored out so unit tests don't need to load EoMT weights.
    """
    lut = np.asarray(build_ade20k_remap_lut(id2label), dtype=np.int64)
    ade = np.asarray(ade_labels, dtype=np.int64)
    from cnsg.segmentation.taxonomy import S3DIS_NAME_TO_ID

    clutter = S3DIS_NAME_TO_ID["clutter"]
    out = np.where(
        (ade >= 0) & (ade < len(lut)),
        lut[np.clip(ade, 0, len(lut) - 1)],
        clutter,
    )
    return out.astype(np.int64)


class EomtBackbone:
    """Thin wrapper over HF `transformers` EoMT for ADE20K-semantic seg.

    Loads the model once at construction. Not thread-safe. Use `segment()` per
    frame; `segment_many()` for a trivial sequential loop.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        from transformers import (
            AutoImageProcessor,
            AutoModelForUniversalSegmentation,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.float16 if device.startswith("cuda") else torch.float32
        if dtype == torch.float16 and not device.startswith("cuda"):
            raise ValueError(
                f"dtype=torch.float16 requires CUDA; got device={device}"
            )
        self._device = device
        self._dtype = dtype
        self._model_id = model_id

        self._processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForUniversalSegmentation.from_pretrained(model_id)
        self._model = model.to(device=device, dtype=dtype).eval()

        # ADE20K's id2label ships with the checkpoint config.
        self._id2label: dict[int, str] = dict(self._model.config.id2label)

    @property
    def id2label(self) -> dict[int, str]:
        return dict(self._id2label)

    @property
    def model_id(self) -> str:
        return self._model_id

    @torch.inference_mode()
    def segment(self, image: Image.Image) -> FrameSemantics:
        """Predict S3DIS labels per pixel for one PIL image."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self._processor(images=image, return_tensors="pt")
        # Move each tensor individually; non-tensor inputs pass through.
        inputs = {
            k: (
                v.to(device=self._device, dtype=self._dtype)
                if isinstance(v, torch.Tensor) and v.is_floating_point()
                else (v.to(self._device) if isinstance(v, torch.Tensor) else v)
            )
            for k, v in inputs.items()
        }

        outputs = self._model(**inputs)

        W, H = image.size
        seg = (
            self._processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(H, W)]
            )[0]
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        s3dis = remap_ade_labels_to_s3dis(seg, self._id2label)
        return FrameSemantics(s3dis_labels=s3dis, ade_labels=seg, source_size=(H, W))

    def segment_many(
        self, images: Iterable[Image.Image]
    ) -> list[FrameSemantics]:
        """Sequential convenience; batched throughput optimization is future work."""
        return [self.segment(img) for img in images]
