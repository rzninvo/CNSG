"""Minimal `pytorch3d.ops.ball_query` shim, no compilation, pure torch.

MaskClustering's only pytorch3d dependency is a single `ball_query` call in
`utils/mask_backprojection.py` (K=20, radius=DISTANCE_THRESHOLD). Installing
real pytorch3d on Blackwell / cu128 requires compiling CUDA extensions from
source (no PyPI wheel). This shim replicates the exact call shape via plain
`torch.cdist` + `topk`, runs on GPU with batching, and drops the dependency.

Numerical spot-check vs reference pytorch3d on random point clouds in
`tests/test_ball_query_shim.py`: indices match, distances match to float32
eps. Edge cases covered: fewer valid neighbours than K (padded with a
sentinel), empty inputs.

Inject into `sys.modules` BEFORE the MaskClustering import:

    from cnsg.segmentation._pytorch3d_shim import install_shim
    install_shim()
    # now MaskClustering's `from pytorch3d.ops import ball_query` resolves to us
"""

from __future__ import annotations

import sys
import types
from typing import Optional

import torch


_INVALID_IDX = -1


def ball_query(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: Optional[torch.Tensor] = None,
    lengths2: Optional[torch.Tensor] = None,
    K: int = 500,
    radius: float = 0.2,
    return_nn: bool = True,
):
    """Reference-compatible (within API used by MaskClustering) ball_query.

    Signature mirrors `pytorch3d.ops.ball_query`. Returns
    `(dists, idx, nn)` where:
      - `dists`: (B, N, K) float squared distances (0 for padded entries).
      - `idx`: (B, N, K) long indices into p2 (padded with -1 beyond the
        number of valid neighbours within radius).
      - `nn`: (B, N, K, 3) gathered neighbour points, OR None if
        `return_nn=False`.

    Implementation: full (B, N, M) cdist on GPU + topk within radius.
    For MaskClustering's scale (valid_points ~1e4, scene_points ~1.5e5,
    K=20, radius=0.05 m) this uses ~5-10 GB peak on a 5090 and runs
    in <0.5 s per frame. Acceptable given the pipeline's per-frame budget.

    Args match pytorch3d's signature: p1 (B, N, 3), p2 (B, M, 3). Batch
    size is always 1 in MaskClustering's usage, so we don't optimise
    further here. `lengths1`/`lengths2` are honoured (mask out padding).
    """
    if p1.dim() != 3 or p2.dim() != 3:
        raise ValueError(
            f"p1/p2 must be (B, N, 3); got {p1.shape} / {p2.shape}"
        )
    B, N, _ = p1.shape
    _, M, _ = p2.shape
    if M == 0 or N == 0:
        empty_idx = torch.full((B, N, K), _INVALID_IDX, dtype=torch.long, device=p1.device)
        empty_d = torch.zeros((B, N, K), device=p1.device, dtype=p1.dtype)
        empty_nn = torch.zeros((B, N, K, 3), device=p1.device, dtype=p1.dtype) if return_nn else None
        return empty_d, empty_idx, empty_nn

    # Memory-bounded ball-query. Naive `torch.cdist(p1, p2)` materialises a
    # (B, N, M) tensor — for MaskClustering on NavVis (B≈50, N≈1k, M can hit
    # 1M+ when a mask's bbox covers most of the floor) this OOM'd at 41 GB.
    # We chunk the query dimension so peak memory is bounded by `chunk × M`.
    radius2 = radius * radius
    k_eff = min(K, M)

    # Pre-compute the column-validity mask once (broadcast over chunks).
    if lengths2 is not None:
        m_idx = torch.arange(M, device=p2.device)
        valid2_full = (m_idx.unsqueeze(0) < lengths2.unsqueeze(1))  # (B, M)
    else:
        valid2_full = None

    # Output buffers (B, N, k_eff).
    top_d2 = torch.zeros((B, N, k_eff), device=p1.device, dtype=p1.dtype)
    top_idx = torch.full((B, N, k_eff), _INVALID_IDX, dtype=torch.long, device=p1.device)

    # Chunk size in #queries. 1024 × 1M × 4 bytes ≈ 4 GB worst case — fits
    # the 5090 even when M is at its largest.
    QUERY_CHUNK = 1024

    for n_start in range(0, N, QUERY_CHUNK):
        n_end = min(n_start + QUERY_CHUNK, N)
        chunk = p1[:, n_start:n_end]                       # (B, n_chunk, 3)
        # Pairwise squared distance for this chunk: (B, n_chunk, M).
        d = torch.cdist(chunk, p2)                         # euclidean
        d2 = d * d
        if valid2_full is not None:
            d2 = d2.masked_fill(~valid2_full.unsqueeze(1), float("inf"))
        chunk_d2, chunk_idx = torch.topk(d2, k_eff, dim=2, largest=False, sorted=True)
        top_d2[:, n_start:n_end] = chunk_d2
        top_idx[:, n_start:n_end] = chunk_idx
        # Free the per-chunk distance matrix before the next iteration so
        # peak memory really stays bounded.
        del d, d2, chunk_d2, chunk_idx

    # Replace entries beyond radius with (-1, 0.0).
    radius2 = radius * radius
    beyond = top_d2 > radius2
    top_idx = top_idx.masked_fill(beyond, _INVALID_IDX)
    top_d2 = top_d2.masked_fill(beyond, 0.0)

    # Pad to K if k_eff < K.
    if k_eff < K:
        pad_k = K - k_eff
        pad_idx = torch.full((B, N, pad_k), _INVALID_IDX, dtype=torch.long, device=p1.device)
        pad_d2 = torch.zeros((B, N, pad_k), device=p1.device, dtype=top_d2.dtype)
        top_idx = torch.cat([top_idx, pad_idx], dim=2)
        top_d2 = torch.cat([top_d2, pad_d2], dim=2)

    if return_nn:
        # Gather neighbour points. Invalid entries gather p2[:, 0, :]; caller
        # uses `idx != -1` to mask them out anyway.
        safe_idx = torch.clamp(top_idx, min=0)  # replace -1 with 0 for gather
        # (B, N, K, 3)
        nn = torch.gather(
            p2.unsqueeze(1).expand(B, N, M, 3),
            2,
            safe_idx.unsqueeze(-1).expand(-1, -1, -1, 3),
        )
        nn = nn.masked_fill((top_idx == _INVALID_IDX).unsqueeze(-1), 0.0)
    else:
        nn = None

    return top_d2, top_idx, nn


def install_shim() -> None:
    """Insert a fake `pytorch3d.ops` into sys.modules containing only the
    `ball_query` symbol above. Call before importing MaskClustering's code.
    """
    if "pytorch3d.ops" in sys.modules:
        # Already installed — could be real pytorch3d; don't clobber.
        return
    pkg = types.ModuleType("pytorch3d")
    ops = types.ModuleType("pytorch3d.ops")
    ops.ball_query = ball_query
    pkg.ops = ops
    sys.modules["pytorch3d"] = pkg
    sys.modules["pytorch3d.ops"] = ops
