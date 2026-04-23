"""Unit test for `_coverage_stats` + `_log_coverage` in build_hge.

Rationale: the first full HGE build shipped at 0.44 % coverage because no
sanity check existed. This pins the stat calculation and the loud-warning
behaviour for any future regression — a wrong mesh or frame-convention bug
will now fail LOUDLY at build time (coverage fraction) rather than at
evaluation time (navigation looks broken).
"""

from __future__ import annotations

import numpy as np

from cnsg.segmentation.build_hge import _coverage_stats, _log_coverage


def test_stats_counts_unique_classes() -> None:
    class_ids = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3], dtype=np.int64)
    names = {1: "wall", 2: "floor", 3: "ceiling"}
    s = _coverage_stats(class_ids, names)
    assert s["total_verts"] == 9
    assert s["labeled_verts"] == 6
    assert s["labeled_fraction"] == 6 / 9
    assert s["per_class"] == {"wall": 2, "floor": 1, "ceiling": 3}


def test_stats_handles_zero_labels() -> None:
    class_ids = np.zeros(100, dtype=np.int64)
    s = _coverage_stats(class_ids, {1: "wall"})
    assert s["labeled_verts"] == 0
    assert s["labeled_fraction"] == 0.0
    assert s["per_class"] == {}


def test_stats_labels_unknown_class_ids() -> None:
    """An ID not in `class_id_to_name` gets a fallback label."""
    class_ids = np.array([0, 99], dtype=np.int64)
    s = _coverage_stats(class_ids, {1: "wall"})
    assert s["per_class"] == {"class_99": 1}


def test_log_emits_warn_below_threshold(capsys) -> None:
    class_ids = np.array([0] * 95 + [1] * 5, dtype=np.int64)  # 5 % coverage
    s = _coverage_stats(class_ids, {1: "wall"})
    _log_coverage(s, min_fraction=0.20)
    captured = capsys.readouterr().out
    assert "[WARN] coverage below sanity threshold" in captured
    assert "5.00 %" in captured  # we log to 2 decimals
    assert "02_hge-lift-frame-mismatch" in captured


def test_log_silent_above_threshold(capsys) -> None:
    class_ids = np.concatenate(
        [np.zeros(80, dtype=np.int64), np.ones(20, dtype=np.int64)]
    )  # 20 % coverage
    s = _coverage_stats(class_ids, {1: "wall"})
    _log_coverage(s, min_fraction=0.15)
    out = capsys.readouterr().out
    assert "coverage" in out         # stats print
    assert "[WARN]" not in out       # but no regression banner


def test_log_disabled_when_min_is_zero(capsys) -> None:
    class_ids = np.zeros(100, dtype=np.int64)  # 0 % coverage
    s = _coverage_stats(class_ids, {1: "wall"})
    _log_coverage(s, min_fraction=0.0)
    assert "[WARN]" not in capsys.readouterr().out
