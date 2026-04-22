"""Floor + room segmentation of a 3D point cloud / mesh.

Produces per-vertex `(floor_id, room_id)` by:

1. **Floors**: 1D height-histogram peak detection. Two consecutive peaks per
   DBSCAN cluster bracket a floor's ceiling+floor planes.
2. **Rooms**: for each floor, slice vertices at head-height, build a 2D
   top-down occupancy histogram, apply morphological closing to connect walls,
   then watershed on the distance transform to separate room cells.

Algorithm ported from HOV-SG (MIT license):
    https://github.com/hovsg/HOV-SG/blob/main/hovsg/graph/graph.py
    (segment_floors / segment_rooms)

Why re-port rather than pip-import: HOV-SG vendors habitat-sim, Open-CLIP, and
other heavy deps we don't need for just the geometric hierarchy step. The
algorithms themselves are ~200 LOC of numpy/scipy/skimage/open3d.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks


# -- Floors -----------------------------------------------------------------


@dataclass(frozen=True)
class FloorBand:
    """A detected floor: everything in `z_min ≤ z ≤ z_max`."""

    floor_id: int
    z_min: float
    z_max: float

    def contains(self, z: np.ndarray) -> np.ndarray:
        return (z >= self.z_min) & (z <= self.z_max)


def segment_floors(
    vertex_positions: np.ndarray,
    *,
    bin_height: float = 0.01,
    smoothing_sigma: float = 2.0,
    peak_min_distance: float = 0.2,
    min_floor_thickness: float = 1.0,
) -> list[FloorBand]:
    """Detect horizontal floor bands in a vertical stack of building levels.

    Strategy: a 1D histogram of Z shows sharp peaks at each floor's flat
    surfaces (floor plane + ceiling plane). We smooth and peak-pick, then
    pair consecutive peaks — peaks[0]/peaks[1] bracket floor 1, peaks[2]/
    peaks[3] bracket floor 2, etc. Each pair must be at least
    `min_floor_thickness` apart to rule out two spurious peaks in the same
    plane. This diverges from HOV-SG's DBSCAN-of-peaks because DBSCAN merges
    adjacent-floor peaks when ceiling-to-next-floor gaps are tight (common
    in real buildings).

    Args:
        vertex_positions: (N, 3) array; Z is the vertical axis (ENU / Z-up).
        bin_height: histogram bin size in meters.
        smoothing_sigma: Gaussian smoothing of the 1D histogram.
        peak_min_distance: minimum separation between peaks, in meters.
        min_floor_thickness: a (floor_plane, ceiling_plane) pair must be at
            least this far apart to be accepted as a floor. Defaults to 1 m —
            below a typical residential ceiling height so it's a lax filter.

    Returns:
        Ordered list of `FloorBand` with `floor_id` starting at 1 (0 reserved
        for "unknown floor" in the export schema).
    """
    if vertex_positions.ndim != 2 or vertex_positions.shape[1] != 3:
        raise ValueError(
            f"vertex_positions must be (N, 3); got shape {vertex_positions.shape}"
        )
    z = vertex_positions[:, 2].astype(np.float64)
    if len(z) == 0:
        return []

    # 1. Height histogram.
    z_min, z_max = float(z.min()), float(z.max())
    if z_max <= z_min:
        return [FloorBand(floor_id=1, z_min=z_min, z_max=z_max + bin_height)]
    n_bins = max(2, int(np.ceil((z_max - z_min) / bin_height)))
    counts, edges = np.histogram(z, bins=n_bins, range=(z_min, z_max))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # 2. Gaussian smooth the 1D count profile.
    smoothed = ndimage.gaussian_filter1d(counts.astype(np.float64), sigma=smoothing_sigma)

    # 3. Peak detection. `distance` is in bin units. Pad zeros on both ends
    # so the floor/ceiling peaks at the extremes of z are surfaced as local
    # maxima by `find_peaks` (which requires both neighbours).
    padded = np.concatenate(([0.0], smoothed, [0.0]))
    bin_distance = max(1, int(round(peak_min_distance / bin_height)))
    peaks_padded, _ = find_peaks(padded, distance=bin_distance)
    peaks = peaks_padded - 1  # undo padding offset
    peaks = peaks[(peaks >= 0) & (peaks < len(smoothed))]
    if len(peaks) < 2:
        return [FloorBand(floor_id=1, z_min=z_min, z_max=z_max)]

    peak_z = np.sort(centers[peaks])

    # 4. Greedy pair consecutive peaks into (floor, ceiling) brackets. Accept
    # only pairs whose vertical separation exceeds `min_floor_thickness`.
    bands: list[tuple[float, float]] = []
    i = 0
    while i < len(peak_z) - 1:
        lo, hi = float(peak_z[i]), float(peak_z[i + 1])
        if (hi - lo) >= min_floor_thickness:
            bands.append((lo, hi))
            i += 2
        else:
            # The pair is too thin to be a real floor (probably two peaks on
            # the same plane). Drop the first and retry with the next pair.
            i += 1

    if not bands:
        # No valid pairs — fall back to one band covering the full extent.
        return [FloorBand(floor_id=1, z_min=z_min, z_max=z_max)]

    # 5. Extend bottom-most down to z_min and top-most up to z_max so every
    # vertex falls into some band.
    bands.sort(key=lambda b: b[0])
    bands[0] = (z_min, bands[0][1])
    bands[-1] = (bands[-1][0], z_max)

    # 6. Adjust consecutive bands to share the midpoint between them (no
    # gaps, no overlaps) so `assign_floor_ids` yields a clean partition.
    out: list[tuple[float, float]] = []
    for j, (lo, hi) in enumerate(bands):
        if j > 0:
            prev_hi = out[j - 1][1]
            mid = 0.5 * (prev_hi + lo)
            out[j - 1] = (out[j - 1][0], mid)
            lo = mid
        out.append((lo, hi))

    return [
        FloorBand(floor_id=i + 1, z_min=lo, z_max=hi) for i, (lo, hi) in enumerate(out)
    ]


def assign_floor_ids(
    vertex_positions: np.ndarray, floors: list[FloorBand]
) -> np.ndarray:
    """Per-vertex floor_id (0 for "no floor matched")."""
    n = len(vertex_positions)
    out = np.zeros(n, dtype=np.int64)
    if not floors:
        return out
    z = vertex_positions[:, 2]
    for band in floors:
        out[band.contains(z)] = band.floor_id
    return out


# -- Rooms ------------------------------------------------------------------


@dataclass
class RoomLabels:
    """Per-vertex room_id assignment for a single floor."""

    floor_id: int
    room_ids: np.ndarray          # shape (N,), 0 = "no room matched"
    n_rooms: int


def segment_rooms(
    vertex_positions: np.ndarray,
    floor_band: FloorBand,
    *,
    slice_height_min: float = 1.5,
    slice_height_max: float = 1.8,
    bev_bin_size: float = 0.1,
    closing_radius_cells: int = 3,
    min_room_area_cells: int = 50,
) -> RoomLabels:
    """Partition vertices on one floor into rooms via 2D watershed on a BEV
    histogram of head-height vertices.

    Strategy (HOV-SG):

    1. Slice vertices at head-height (1.5–1.8 m above floor). Walls show up
       as dense strips there while room interiors are empty.
    2. Rasterize the (x, y) of those vertices into a 2D histogram. High cells
       = walls; low cells = open floor space.
    3. Threshold + morphological closing → a binary wall map.
    4. Invert to get open-space mask. Connected components of that mask are
       candidate rooms.
    5. Watershed on the distance transform to separate rooms that touch
       through a doorway.
    6. Back-project each vertex's (x, y) to its watershed cell.

    Returned `RoomLabels.room_ids` is 0 for vertices that fall outside any
    reliable room cell (e.g. on a wall, or inside too-small a cell).
    """
    # Defer skimage import to keep the cnsg package lightweight for consumers
    # that don't run segmentation.
    try:
        from skimage.segmentation import watershed
        from skimage.morphology import disk, binary_closing
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "segment_rooms requires scikit-image. Install with: pip install scikit-image"
        ) from e

    n_verts = len(vertex_positions)
    room_ids = np.zeros(n_verts, dtype=np.int64)
    if n_verts == 0:
        return RoomLabels(floor_id=floor_band.floor_id, room_ids=room_ids, n_rooms=0)

    z = vertex_positions[:, 2]
    in_floor = floor_band.contains(z)
    if not np.any(in_floor):
        return RoomLabels(floor_id=floor_band.floor_id, room_ids=room_ids, n_rooms=0)

    # The actual floor plane may be above `floor_band.z_min` (which is often a
    # midpoint between two adjacent floors' peaks, not the real floor plane).
    # Use the lower quartile of z within the band — robust to noise while
    # tracking the real floor surface.
    floor_z_real = float(np.quantile(z[in_floor], 0.05))
    slice_lo = floor_z_real + slice_height_min
    slice_hi = floor_z_real + slice_height_max
    in_slice = in_floor & (z >= slice_lo) & (z <= slice_hi)

    if not np.any(in_slice):
        # No head-height vertices means no wall information: treat the whole
        # floor as a single room.
        room_ids[in_floor] = 1
        return RoomLabels(floor_id=floor_band.floor_id, room_ids=room_ids, n_rooms=1)

    xy_slice = vertex_positions[in_slice, :2]
    xy_all = vertex_positions[in_floor, :2]

    # Common grid from the union so we can back-project every floor vertex.
    x_min, y_min = np.minimum(xy_slice.min(axis=0), xy_all.min(axis=0))
    x_max, y_max = np.maximum(xy_slice.max(axis=0), xy_all.max(axis=0))
    nx = max(2, int(np.ceil((x_max - x_min) / bev_bin_size)))
    ny = max(2, int(np.ceil((y_max - y_min) / bev_bin_size)))

    # 1. Rasterize head-height slice to a 2D wall-density histogram.
    wall_hist, _, _ = np.histogram2d(
        xy_slice[:, 0],
        xy_slice[:, 1],
        bins=[nx, ny],
        range=[[x_min, x_max], [y_min, y_max]],
    )

    # 2. Binary wall map: cells with any wall presence, then close small gaps.
    wall_mask = wall_hist > 0
    if closing_radius_cells > 0:
        wall_mask = binary_closing(wall_mask, footprint=disk(closing_radius_cells))

    # 3. Open-space seed: invert + label connected components.
    open_mask = ~wall_mask
    # But also restrict to cells with *some* floor coverage to avoid labeling
    # exterior free space as rooms.
    floor_hist, _, _ = np.histogram2d(
        xy_all[:, 0],
        xy_all[:, 1],
        bins=[nx, ny],
        range=[[x_min, x_max], [y_min, y_max]],
    )
    open_mask &= floor_hist > 0

    # Distance transform of the open mask: peaks are room centers.
    dist_raw = ndimage.distance_transform_edt(open_mask).astype(np.float64)
    # Smooth the distance field so per-cell noise doesn't generate spurious
    # local maxima inside a single room. `peak_local_max` on an unsmoothed
    # distance transform routinely finds 5–20+ maxima per real room.
    dist = ndimage.gaussian_filter(dist_raw, sigma=2.0)

    # Markers for watershed: local maxima of the smoothed distance transform.
    # `min_distance` must be large enough that two distinct rooms' centers
    # can't both appear as peaks through a narrow connection (e.g. a doorway).
    # We set it to half the expected minimum room width, in cell units.
    from skimage.feature import peak_local_max

    min_room_width_cells = int(round(2.0 / bev_bin_size))  # assume >= 2 m room
    peak_distance = max(closing_radius_cells * 2, min_room_width_cells)
    peak_coords = peak_local_max(
        dist,
        min_distance=peak_distance,
        labels=open_mask.astype(np.int32),
    )
    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (r, c) in enumerate(peak_coords, start=1):
        markers[r, c] = i

    if markers.max() == 0:
        # Single component → one room.
        room_ids[in_floor] = 1
        return RoomLabels(floor_id=floor_band.floor_id, room_ids=room_ids, n_rooms=1)

    # 4. Watershed.
    labels_grid = watershed(-dist, markers=markers, mask=open_mask)

    # Drop tiny rooms (noise) by relabeling them to 0.
    for rid in range(1, labels_grid.max() + 1):
        if (labels_grid == rid).sum() < min_room_area_cells:
            labels_grid[labels_grid == rid] = 0
    # Compact labels (so IDs are 1..K with no gaps).
    unique_rooms = sorted(set(int(v) for v in np.unique(labels_grid) if v > 0))
    compact_map = {old: new for new, old in enumerate(unique_rooms, start=1)}
    relabel_fn = np.vectorize(lambda v: compact_map.get(int(v), 0))
    if unique_rooms:
        labels_grid = relabel_fn(labels_grid).astype(np.int64)
    else:
        labels_grid = np.zeros_like(labels_grid, dtype=np.int64)

    # 5. Back-project every floor vertex to its watershed cell.
    floor_idx = np.flatnonzero(in_floor)
    xs = vertex_positions[floor_idx, 0]
    ys = vertex_positions[floor_idx, 1]
    col = np.clip(((xs - x_min) / bev_bin_size).astype(np.int64), 0, nx - 1)
    row = np.clip(((ys - y_min) / bev_bin_size).astype(np.int64), 0, ny - 1)
    # Note: histogram2d uses [x_bins, y_bins] layout; labels_grid[ix, iy]
    # matches the histogram, so we use col=ix, row=iy here.
    room_ids[floor_idx] = labels_grid[col, row]

    return RoomLabels(
        floor_id=floor_band.floor_id,
        room_ids=room_ids,
        n_rooms=len(unique_rooms),
    )


def segment_building(
    vertex_positions: np.ndarray,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Full pipeline: return (per_vertex_floor_id, per_vertex_room_id).

    `room_id` is a **global** label across the whole building: floor 1 gets
    rooms 1..R1; floor 2 gets R1+1..R1+R2; etc. Zero means "no room matched".
    """
    floors = segment_floors(vertex_positions)
    floor_ids = assign_floor_ids(vertex_positions, floors)
    room_ids = np.zeros(len(vertex_positions), dtype=np.int64)

    # Extract per-floor kwargs if caller wanted them (else use defaults).
    room_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        in {
            "slice_height_min",
            "slice_height_max",
            "bev_bin_size",
            "closing_radius_cells",
            "min_room_area_cells",
        }
    }

    offset = 0
    for band in floors:
        result = segment_rooms(vertex_positions, band, **room_kwargs)
        mask = result.room_ids > 0
        room_ids[mask] = result.room_ids[mask] + offset
        offset += result.n_rooms

    return floor_ids, room_ids
