"""SAM 3 open-vocabulary per-frame instance segmentation.

Wraps `facebookresearch/sam3` for batched text-prompted segmentation of
NavVis RGB frames. Produces, for each frame, a (H, W) `uint16` instance
mask where 0 = background and positive IDs are per-frame-local instance
IDs (unique within the frame; NOT matched across frames — downstream
`lift_2d_to_3d` handles cross-frame association via 3D overlap).

Runtime env: `cnsg-seg` (py3.12 + torch 2.10+ cu128 + sam3 + flash-attn-3).
Weights require HuggingFace access to `facebook/sam3` (gated). See
`docs/report/01_architecture-lean-migration/phase3-research.md`.

Canonical inference pattern (from SAM 3's README):

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model().to("cuda")
    processor = Sam3Processor(model, confidence_threshold=0.5)
    state = processor.set_image(pil_image)
    output = processor.set_text_prompt(state=state, prompt="chair")
    masks, scores = output["masks"], output["scores"]

Key gotchas encountered and worked around here:

1. `build_sam3_image_model()` returns a fp32 model but some internal ops
   expect bf16 tensors (Issue #507). Running the full model in fp32 with
   `torch.autocast(dtype=bfloat16)` is the stable combination for cu128 /
   Blackwell.
2. The processor internally resizes to 1008×1008; we pass the PIL image
   at its native resolution and the processor handles remap.
3. SAM 3 is trained on SA-Co (exhaustive concepts). Structural classes
   like "wall" score much lower than foreground objects like "chair" or
   "door" — this is by design and is why our pipeline also uses an
   ADE20K Mask2Former for wall/floor/ceiling (see `structural_ade20k.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class InstanceMaskSet:
    """Per-frame open-vocab instance masks rasterized to a single image.

    Attributes:
        instance_mask: (H, W) int32. 0 = background, k = k-th detected
            instance (1-indexed). Every positive value corresponds to
            an entry in `class_per_instance` at index (k-1).
        class_per_instance: list of the text prompt that produced each
            instance. Aligned with `instance_mask` positive IDs.
        score_per_instance: list of SAM 3 confidence scores for each
            instance (parallel to `class_per_instance`).
    """

    instance_mask: np.ndarray
    class_per_instance: list[str]
    score_per_instance: list[float]


class Sam3Segmenter:
    """Thin wrapper around SAM 3's image model for per-frame, per-prompt
    inference. Loads the model once; reuses across frames.
    """

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.5,
        device: str = "cuda",
        prompts: Sequence[str] | None = None,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ):
        if device != "cuda":
            raise ValueError(
                "SAM 3 only supports CUDA. flash-attn-3 and bfloat16 autocast "
                "both require a GPU."
            )
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self._device = device
        self._autocast_dtype = autocast_dtype
        self._model = build_sam3_image_model().to(device)
        self._processor = Sam3Processor(self._model, confidence_threshold=confidence_threshold)
        self._prompts: tuple[str, ...] = tuple(prompts) if prompts else ()

    # -- configuration accessors ------------------------------------------

    @property
    def prompts(self) -> tuple[str, ...]:
        return self._prompts

    def set_prompts(self, prompts: Sequence[str]) -> None:
        self._prompts = tuple(prompts)

    # -- inference --------------------------------------------------------

    def segment(self, image: Image.Image) -> InstanceMaskSet:
        """Return per-pixel instance IDs for `image` across all `prompts`.

        The returned `instance_mask` composites masks from every prompt:
        each pixel holds the ID of the highest-scoring instance that
        covers it. Overlap resolution is score-ranked (highest wins).
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        W, H = image.size

        rank_mask = np.zeros((H, W), dtype=np.int32)       # composite instance IDs
        best_score = np.full((H, W), -1.0, dtype=np.float32)
        class_per_instance: list[str] = []
        score_per_instance: list[float] = []

        for prompt in self._prompts:
            mask_np, scores_np = self._segment_one_prompt(image, prompt)
            if mask_np.shape[0] == 0:
                continue
            for local_idx in range(mask_np.shape[0]):
                instance_global_id = len(class_per_instance) + 1
                class_per_instance.append(prompt)
                score_per_instance.append(float(scores_np[local_idx]))

                m = mask_np[local_idx].astype(bool)
                score = float(scores_np[local_idx])
                overwrite = m & (score > best_score)
                rank_mask[overwrite] = instance_global_id
                best_score[overwrite] = score

        return InstanceMaskSet(
            instance_mask=rank_mask,
            class_per_instance=class_per_instance,
            score_per_instance=score_per_instance,
        )

    def segment_many(
        self, images: Iterable[Image.Image]
    ) -> list[InstanceMaskSet]:
        """Sequential helper; batched throughput optimization is future work."""
        return [self.segment(img) for img in images]

    # -- internals --------------------------------------------------------

    @torch.inference_mode()
    def _segment_one_prompt(
        self, image: Image.Image, prompt: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run SAM 3 for one (image, text_prompt) pair. Returns (masks, scores)
        as numpy arrays. masks shape: (K, H, W) bool; scores shape: (K,).
        """
        with torch.autocast(device_type=self._device, dtype=self._autocast_dtype):
            state = self._processor.set_image(image)
            output = self._processor.set_text_prompt(state=state, prompt=prompt)

        masks_t = output.get("masks")
        scores_t = output.get("scores")

        if masks_t is None or (hasattr(masks_t, "numel") and masks_t.numel() == 0):
            H_orig = state.get("original_height", image.size[1])
            W_orig = state.get("original_width", image.size[0])
            return (
                np.zeros((0, H_orig, W_orig), dtype=bool),
                np.zeros((0,), dtype=np.float32),
            )

        masks_np = masks_t.to(torch.bool).cpu().numpy()
        # SAM 3 returns (K, 1, H, W); some configs yield (K, H, W) or (H, W).
        # Collapse to (K, H, W).
        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0]
        elif masks_np.ndim == 2:
            masks_np = masks_np[None]
        elif masks_np.ndim != 3:
            raise RuntimeError(
                f"unexpected SAM 3 masks shape {masks_np.shape}; "
                f"expected (K, 1, H, W), (K, H, W), or (H, W)"
            )
        scores_np = scores_t.detach().to(torch.float32).cpu().numpy()
        return masks_np, scores_np
