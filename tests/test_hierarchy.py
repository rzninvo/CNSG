"""Unit tests for `cnsg.segmentation.hierarchy`.

Synthetic scenes:

- `test_segment_floors_detects_two_floors`: two stacked planar levels at z=0
  and z=3 → should yield 2 floors.
- `test_segment_floors_single_level`: one level → 1 floor.
- `test_segment_rooms_splits_two_rectangular_rooms`: two L-shaped rooms
  separated by a single doorway on one floor → 2 rooms from watershed.
- `test_segment_building_end_to_end`: combine a two-floor stack where each
  floor has two rooms.
"""

from __future__ import annotations

import numpy as np
import pytest

from cnsg.segmentation.hierarchy import (
    FloorBand,
    assign_floor_ids,
    segment_building,
    segment_floors,
    segment_rooms,
)


# --- floor tests ------------------------------------------------------------


def _make_flat_floor(
    n_per_plane: int,
    z_floor: float,
    z_ceiling: float,
    x_range=(-5.0, 5.0),
    y_range=(-5.0, 5.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Dense points on a horizontal floor and ceiling at the given heights."""
    if rng is None:
        rng = np.random.default_rng(0)
    xs = rng.uniform(x_range[0], x_range[1], size=n_per_plane * 2)
    ys = rng.uniform(y_range[0], y_range[1], size=n_per_plane * 2)
    zs_floor = np.full(n_per_plane, z_floor) + rng.normal(0, 0.005, size=n_per_plane)
    zs_ceiling = np.full(n_per_plane, z_ceiling) + rng.normal(0, 0.005, size=n_per_plane)
    zs = np.concatenate([zs_floor, zs_ceiling])
    return np.stack([xs, ys, zs], axis=1)


def test_segment_floors_detects_two_floors() -> None:
    rng = np.random.default_rng(42)
    floor1 = _make_flat_floor(5000, z_floor=0.0, z_ceiling=2.8, rng=rng)
    floor2 = _make_flat_floor(5000, z_floor=3.0, z_ceiling=5.8, rng=rng)
    verts = np.concatenate([floor1, floor2], axis=0)

    floors = segment_floors(verts)
    assert len(floors) == 2, [(f.z_min, f.z_max) for f in floors]

    # Bands should partition the input range with the split near z≈2.9 (the
    # midpoint between the floor-1 ceiling at 2.8 and the floor-2 floor at 3.0).
    assert floors[0].z_min < 0.1
    assert 2.7 < floors[0].z_max < 3.0
    assert 2.7 < floors[1].z_min < 3.0
    assert floors[1].z_max > 5.6
    # Bands abut (no gap, no overlap).
    assert floors[0].z_max == floors[1].z_min


def test_segment_floors_single_level() -> None:
    verts = _make_flat_floor(5000, z_floor=0.0, z_ceiling=3.0)
    floors = segment_floors(verts)
    assert len(floors) == 1


def test_segment_floors_empty_input() -> None:
    verts = np.empty((0, 3))
    assert segment_floors(verts) == []


def test_assign_floor_ids_partitions_vertices() -> None:
    rng = np.random.default_rng(1)
    floor1 = _make_flat_floor(2000, z_floor=0.0, z_ceiling=2.5, rng=rng)
    floor2 = _make_flat_floor(2000, z_floor=3.2, z_ceiling=5.7, rng=rng)
    verts = np.concatenate([floor1, floor2], axis=0)

    floors = segment_floors(verts)
    ids = assign_floor_ids(verts, floors)
    # All vertices should be assigned to a non-zero floor.
    assert np.all(ids > 0), f"unassigned vertex count: {int((ids == 0).sum())}"
    # First half should end up on floor 1; second half on floor 2 (allowing
    # DBSCAN's merge to have happened correctly).
    unique = set(ids.tolist())
    assert unique == {1, 2}


# --- room tests -------------------------------------------------------------


def _make_two_room_floor(
    floor_z: float = 0.0,
    wall_height: float = 2.5,
    wall_sample_density: int = 10000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build a floor with two rectangular rooms separated by a wall + doorway.

    Layout (top-down):
      room A: x ∈ [-5, -0.5], y ∈ [-3, 3]
      room B: x ∈ [0.5, 5],   y ∈ [-3, 3]
      wall at x = 0 with a doorway at y ∈ [-0.5, 0.5]
      outer walls at x=±5 and y=±3.
    """
    if rng is None:
        rng = np.random.default_rng(2)

    verts = []

    # Floor plane of room A and room B (for the floor histogram).
    for (x0, x1) in [(-5, -0.5), (0.5, 5)]:
        n = 4000
        xs = rng.uniform(x0, x1, size=n)
        ys = rng.uniform(-3, 3, size=n)
        zs = np.full(n, floor_z) + rng.normal(0, 0.005, size=n)
        verts.append(np.stack([xs, ys, zs], axis=1))

    # Ceiling planes matching floor extents so segment_floors can bracket.
    for (x0, x1) in [(-5, -0.5), (0.5, 5)]:
        n = 4000
        xs = rng.uniform(x0, x1, size=n)
        ys = rng.uniform(-3, 3, size=n)
        zs = np.full(n, floor_z + wall_height) + rng.normal(0, 0.005, size=n)
        verts.append(np.stack([xs, ys, zs], axis=1))

    # Walls: generate points at head-height (floor_z + 1.65 ± 0.1) along the
    # wall lines. That puts enough density in the head-height slice for the
    # BEV histogram to detect walls.
    def _wall_line(x0, y0, x1, y1, n):
        t = rng.uniform(0, 1, size=n)
        xs = x0 + t * (x1 - x0)
        ys = y0 + t * (y1 - y0)
        zs = rng.uniform(floor_z + 1.5, floor_z + 1.8, size=n)
        return np.stack([xs, ys, zs], axis=1)

    # Dividing wall at x=0 minus doorway at y ∈ [-0.5, 0.5].
    verts.append(_wall_line(0, -3, 0, -0.5, 1500))
    verts.append(_wall_line(0, 0.5, 0, 3, 1500))
    # Outer walls.
    verts.append(_wall_line(-5, -3, -5, 3, 2000))
    verts.append(_wall_line(5, -3, 5, 3, 2000))
    verts.append(_wall_line(-5, -3, 5, -3, 2000))
    verts.append(_wall_line(-5, 3, 5, 3, 2000))

    return np.concatenate(verts, axis=0)


def test_segment_rooms_splits_two_rooms() -> None:
    """A floor partitioned by a wall+doorway should yield two rooms."""
    rng = np.random.default_rng(3)
    verts = _make_two_room_floor(rng=rng)

    # Establish the floor band first (full floor+ceiling extent).
    band = FloorBand(floor_id=1, z_min=0.0, z_max=2.5)
    result = segment_rooms(verts, band, bev_bin_size=0.15, closing_radius_cells=2)
    assert result.n_rooms == 2, (
        f"expected 2 rooms from the two-rectangle fixture; got {result.n_rooms}"
    )

    # Each room must own a non-trivial number of vertices.
    counts = np.bincount(result.room_ids)
    # counts[0] = unassigned; counts[1:] are the rooms.
    assert len(counts) >= 3
    assert counts[1] > 1000 and counts[2] > 1000


def test_segment_building_end_to_end() -> None:
    """Two floors × two rooms = 4 global rooms."""
    rng = np.random.default_rng(4)
    verts_f1 = _make_two_room_floor(floor_z=0.0, rng=rng)
    verts_f2 = _make_two_room_floor(floor_z=3.5, rng=rng)
    verts = np.concatenate([verts_f1, verts_f2], axis=0)

    floor_ids, room_ids = segment_building(
        verts, bev_bin_size=0.15, closing_radius_cells=2
    )

    assert set(np.unique(floor_ids).tolist()) <= {0, 1, 2}
    # Both floors must get at least one room assigned to > 1000 verts.
    for fid in (1, 2):
        mask = floor_ids == fid
        rooms_on_floor = set(np.unique(room_ids[mask]).tolist()) - {0}
        assert len(rooms_on_floor) >= 2, (
            f"floor {fid} should have at least 2 rooms, got {rooms_on_floor}"
        )


def test_segment_rooms_handles_empty_head_height_slice() -> None:
    """If no vertices exist at head-height, the whole floor becomes one room."""
    # Only floor plane — no walls, no ceiling at the head-height slice.
    rng = np.random.default_rng(5)
    xs = rng.uniform(-5, 5, size=1000)
    ys = rng.uniform(-5, 5, size=1000)
    zs = np.zeros_like(xs) + rng.normal(0, 0.01, size=1000)
    verts = np.stack([xs, ys, zs], axis=1)

    band = FloorBand(floor_id=1, z_min=-0.1, z_max=0.1)
    result = segment_rooms(verts, band)
    assert result.n_rooms == 1, (
        f"floor with no head-height verts should collapse to one room; got {result.n_rooms}"
    )


def test_shape_guard_on_positions() -> None:
    with pytest.raises(ValueError):
        segment_floors(np.zeros((10, 2)))
