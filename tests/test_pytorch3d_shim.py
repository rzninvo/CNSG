"""Unit tests for the pytorch3d.ops.ball_query shim.

MaskClustering is the only downstream user of `pytorch3d.ops.ball_query` in
our stack. We avoid installing pytorch3d (no PyPI wheel for Blackwell cu128)
by injecting a torch-native replacement via `install_shim()`. These tests
pin the behaviour against a reference-style ball-query so a regression in
the shim doesn't silently break the MaskClustering lifter.

Reference: pytorch3d's `ball_query` returns `(dists, idx, nn)` where
`dists` are SQUARED euclidean distances and `idx` has `-1` for padded
entries (beyond radius OR padded column). We mirror that contract exactly.
"""

from __future__ import annotations

import numpy as np
import torch

from cnsg.segmentation._pytorch3d_shim import ball_query


def _brute_ball_query(p1, p2, K, radius):
    """Pure-numpy reference: per query point find K nearest in p2 within radius.

    Returns (dists2, idx, nn) with the same padding convention.
    """
    B, N, _ = p1.shape
    M = p2.shape[1]
    dists2 = np.zeros((B, N, K), dtype=np.float32)
    idx = np.full((B, N, K), -1, dtype=np.int64)
    nn = np.zeros((B, N, K, 3), dtype=np.float32)
    for b in range(B):
        for i in range(N):
            d2 = ((p2[b] - p1[b, i]) ** 2).sum(axis=-1)
            order = np.argsort(d2)[:K]
            k_eff = min(K, M)
            for k in range(k_eff):
                j = order[k]
                if d2[j] <= radius * radius:
                    dists2[b, i, k] = d2[j]
                    idx[b, i, k] = int(j)
                    nn[b, i, k] = p2[b, j]
    return dists2, idx, nn


def test_matches_brute_force_on_random_points() -> None:
    rng = np.random.default_rng(0)
    p1 = rng.uniform(-1, 1, size=(1, 30, 3)).astype(np.float32)
    p2 = rng.uniform(-1, 1, size=(1, 200, 3)).astype(np.float32)

    d2_ref, idx_ref, nn_ref = _brute_ball_query(p1, p2, K=8, radius=0.5)

    p1_t = torch.from_numpy(p1)
    p2_t = torch.from_numpy(p2)
    d2, idx, nn = ball_query(p1_t, p2_t, K=8, radius=0.5, return_nn=True)

    # topk is stable by distance but ties may reorder indices — compare
    # value-wise on the distances (stable), and membership-wise on idx.
    np.testing.assert_allclose(d2.numpy(), d2_ref, atol=1e-5)
    for b in range(1):
        for i in range(p1.shape[1]):
            assert set(idx[b, i].tolist()) == set(idx_ref[b, i].tolist()), (
                f"idx mismatch at batch={b} query={i}: "
                f"shim={idx[b, i].tolist()} ref={idx_ref[b, i].tolist()}"
            )
    np.testing.assert_allclose(nn.numpy(), nn_ref, atol=1e-5)


def test_pads_with_minus_one_when_fewer_neighbours_than_k() -> None:
    """If only 3 points fall inside radius but K=10, entries 3..9 must be -1."""
    p1 = torch.tensor([[[0.0, 0.0, 0.0]]])
    # 3 close points + 10 far points
    close = torch.tensor([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])
    far = torch.tensor([[100.0, 0, 0]] * 10)
    p2 = torch.cat([close, far]).unsqueeze(0)

    d2, idx, nn = ball_query(p1, p2, K=10, radius=0.1, return_nn=True)
    assert (idx[0, 0, :3] != -1).all()
    assert (idx[0, 0, 3:] == -1).all()
    assert (d2[0, 0, 3:] == 0).all()


def test_empty_inputs_return_padded() -> None:
    """Zero query points or zero reference points → everything padded."""
    p1 = torch.zeros((1, 0, 3))
    p2 = torch.rand(1, 100, 3)
    d2, idx, _ = ball_query(p1, p2, K=5, radius=0.1, return_nn=False)
    assert d2.shape == (1, 0, 5)
    assert idx.shape == (1, 0, 5)

    p1 = torch.rand(1, 10, 3)
    p2 = torch.zeros((1, 0, 3))
    d2, idx, _ = ball_query(p1, p2, K=5, radius=0.1, return_nn=False)
    assert (idx == -1).all()


def test_return_nn_false_returns_none() -> None:
    p1 = torch.rand(1, 5, 3)
    p2 = torch.rand(1, 50, 3)
    _, _, nn = ball_query(p1, p2, K=3, radius=0.5, return_nn=False)
    assert nn is None


def test_dists_are_squared_not_euclidean() -> None:
    """pytorch3d contract: returned dists are SQUARED, not euclidean."""
    p1 = torch.tensor([[[0.0, 0, 0]]])
    p2 = torch.tensor([[[3.0, 4, 0]]])  # dist = 5, dist^2 = 25
    d2, idx, _ = ball_query(p1, p2, K=1, radius=10.0, return_nn=False)
    assert idx[0, 0, 0] == 0
    assert abs(d2[0, 0, 0].item() - 25.0) < 1e-5


def test_radius_threshold_trims_far_points() -> None:
    p1 = torch.tensor([[[0.0, 0, 0]]])
    p2 = torch.tensor([[[0.5, 0, 0], [0.9, 0, 0], [2.0, 0, 0]]])
    # radius = 1.0 should keep the first two, drop the third
    d2, idx, _ = ball_query(p1, p2, K=5, radius=1.0, return_nn=False)
    assert sorted(idx[0, 0].tolist()) == [-1, -1, -1, 0, 1]
