"""Unit tests for the per-frame segmentation cache in build_hge.

The cache sits between Mask2Former + SAM 3 inference (dominant ~52 min of a
2408-frame HGE build) and the downstream lift. A cache hit must:

  1. round-trip the exact (instance_mask, class_mask) arrays byte-for-byte
  2. reject entries written under a different `(prompts, confidence)` config
     so stale masks can't silently contaminate a rerun with different knobs
  3. behave gracefully when the cache file is corrupt or missing (warn
     loudly per CLAUDE.md §5 — no silent fallbacks)

Regression target: if someone drops the config-hash check or changes the
array dtypes, these tests fail immediately rather than wait ~50 min for a
full rebuild to surface the problem.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cnsg.segmentation.build_hge import (
    _seg_cache_config_hash,
    _seg_cache_load,
    _seg_cache_save,
)


def test_config_hash_is_stable_across_prompt_reordering() -> None:
    """Reordering prompts must NOT invalidate the cache — `sorted` normalises."""
    h1 = _seg_cache_config_hash(("door", "chair", "table"), 0.3)
    h2 = _seg_cache_config_hash(("table", "door", "chair"), 0.3)
    assert h1 == h2


def test_config_hash_changes_with_prompt_set() -> None:
    h1 = _seg_cache_config_hash(("door", "chair"), 0.3)
    h2 = _seg_cache_config_hash(("door", "chair", "table"), 0.3)
    assert h1 != h2


def test_config_hash_changes_with_confidence() -> None:
    h1 = _seg_cache_config_hash(("door",), 0.3)
    h2 = _seg_cache_config_hash(("door",), 0.5)
    assert h1 != h2


def test_roundtrip_byte_exact(tmp_path: Path) -> None:
    """Save then load → same int32 + int16 arrays, no silent dtype drift."""
    h = _seg_cache_config_hash(("door",), 0.3)
    inst = np.array([[0, 1, 2], [0, 1001, 3]], dtype=np.int32)
    cls = np.array([[0, 13, 1], [0, 1, 13]], dtype=np.int16)
    _seg_cache_save(tmp_path, frame_id=42, config_hash=h,
                    instance_mask=inst, class_mask=cls)
    out = _seg_cache_load(tmp_path, frame_id=42, expected_hash=h)
    assert out is not None
    got_inst, got_cls = out
    np.testing.assert_array_equal(got_inst, inst)
    np.testing.assert_array_equal(got_cls, cls)
    assert got_inst.dtype == np.int32
    assert got_cls.dtype == np.int16


def test_miss_returns_none(tmp_path: Path) -> None:
    assert _seg_cache_load(tmp_path, frame_id=0, expected_hash="abc") is None


def test_hash_mismatch_rejects_cache(tmp_path: Path, capsys) -> None:
    """Different (prompts, confidence) → cache miss + [WARN] log (no silent stale return)."""
    h1 = _seg_cache_config_hash(("door",), 0.3)
    h2 = _seg_cache_config_hash(("window",), 0.3)
    assert h1 != h2
    _seg_cache_save(
        tmp_path, frame_id=1, config_hash=h1,
        instance_mask=np.zeros((4, 4), dtype=np.int32),
        class_mask=np.zeros((4, 4), dtype=np.int16),
    )
    out = _seg_cache_load(tmp_path, frame_id=1, expected_hash=h2)
    assert out is None
    # Must have warned — no silent fallback.
    captured = capsys.readouterr().out
    assert "[WARN]" in captured


def test_corrupt_file_is_warned_and_returns_none(tmp_path: Path, capsys) -> None:
    """Corrupt .npz → cache miss + [WARN] log; caller falls back to fresh inference."""
    (tmp_path / "frame_000000.npz").write_bytes(b"this is not an npz file")
    out = _seg_cache_load(tmp_path, frame_id=0, expected_hash="anything")
    assert out is None
    assert "[WARN]" in capsys.readouterr().out


def test_save_is_atomic(tmp_path: Path) -> None:
    """Intermediate .tmp file must not linger after a successful save."""
    h = _seg_cache_config_hash(("door",), 0.3)
    _seg_cache_save(
        tmp_path, frame_id=7, config_hash=h,
        instance_mask=np.ones((2, 2), dtype=np.int32),
        class_mask=np.ones((2, 2), dtype=np.int16),
    )
    assert (tmp_path / "frame_000007.npz").exists()
    assert not (tmp_path / "frame_000007.npz.tmp").exists()
