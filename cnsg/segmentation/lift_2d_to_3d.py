"""2D → 3D mask lifting for the Phase 3 segmentation pipeline.

Given posed RGB-D frames and per-frame instance + class masks (from SAM 3 +
ADE20K Mask2Former respectively), assigns a global `instance_id` and a
`class_id` to every vertex of the target mesh.

Algorithm (validated in `docs/report/01_architecture-lean-migration/phase3-research.md`):

  Stage 1 — Per-frame project + depth-test + mask lookup.
    For each vertex v and frame k, project v into k's image plane using the
    frame's pose and intrinsics. A vertex is "visible" in k iff:
      (a) v is in front of the camera (z > 0),
      (b) its pixel lands inside the image,
      (c) the recorded depth at that pixel is within `depth_tolerance` of
          the vertex's camera-space z — i.e. another surface isn't occluding.
    For visible vertices, read the instance_mask pixel → local (k, mask_id).

  Stage 2 — Cross-frame union-find merge.
    Build disjoint-set forest over (frame_id, mask_id) tuples. Two local
    instances are unioned when their 3D vertex sets overlap by more than
    `overlap_threshold` (relative to the smaller set).

  Stage 3 — Per-vertex majority vote.
    For each vertex, count how many (k, m) observations fall into each
    union-find root. Root with the most votes wins → the global instance_id.
    Class ID is the mode of Mask2Former's per-pixel S3DIS class over all
    votes for the winning instance.

Why this works for NavVis: 2408 frames × 4 camera rigs over 600 timestamps
produces ~dozens-to-hundreds of valid projections per visible vertex,
making the vote statistically stable. The metric depth maps on disk make
the occlusion check trivially reliable.

No MinkowskiEngine, no spconv, no PyTorch3D — just torch + scipy + numpy.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import torch


# --- input record ------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """One posed RGB-D capture with its SAM 3 + Mask2Former outputs.

    All arrays are `numpy` so this dataclass stays device-agnostic; the
    lifter moves tensors to GPU internally.
    """

    frame_id: int
    depth: np.ndarray                  # (H, W) float32 metric depth (meters)
    instance_mask: np.ndarray          # (H, W) int16/int32; 0 = "background"
    class_mask: np.ndarray             # (H, W) int8/int16; S3DIS class IDs
    T_world_cam: np.ndarray            # (4, 4) float64 pose (world ← camera)
    fx: float
    fy: float
    cx: float
    cy: float


# --- output record -----------------------------------------------------------


@dataclass
class LiftResult:
    """Per-vertex assignments + provenance."""

    instance_ids: np.ndarray           # (N,) int64; 0 = "unassigned"
    class_ids: np.ndarray              # (N,) int64; S3DIS class IDs
    num_instances: int
    # Per-instance label: (instance_id -> class_id). Useful for emitters.
    instance_to_class: dict[int, int] = field(default_factory=dict)
    # Per-instance supporting (frame_id, mask_id) pairs — the SAM 3 instances
    # whose 2D masks union-find-merged into this 3D cluster. The open-vocab
    # phrase voter consumes this to look up each supporting mask's
    # generating prompt phrase (from the seg_cache `frame_*.phrases.json`
    # sidecars) and majority-vote a heritage label per cluster.
    instance_to_mask_list: dict[int, list[tuple[int, int]]] = field(default_factory=dict)


# --- union-find --------------------------------------------------------------


class _DSU:
    """Plain int-keyed Disjoint Set Union with path compression + union by rank."""

    __slots__ = ("parent", "rank")

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}

    def make_set(self, x: int) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression.
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# --- stage 1: per-frame projection + vote accumulation ----------------------


def _project_frame(
    vertices: torch.Tensor,
    frame: Frame,
    depth_tol: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (vertex_idx_valid, instance_id_valid, class_id_valid) for one frame.

    Args:
        vertices: (N, 3) float32 tensor on `device`.
        frame: source capture.
        depth_tol: meters; |sensor_depth − vertex_z| must be below this.

    Returns:
        Three 1D tensors of matching length, holding the global vertex index,
        the per-frame local instance ID (from `instance_mask`), and the
        per-pixel S3DIS class ID (from `class_mask`).
    """
    # T_cam_world = inverse(T_world_cam). Rigid transform inverse (not LU).
    T_wc = torch.from_numpy(frame.T_world_cam).to(device=device, dtype=torch.float64)
    R = T_wc[:3, :3]
    t = T_wc[:3, 3]
    R_inv = R.transpose(0, 1)
    t_inv = -R_inv @ t

    verts_f64 = vertices.to(torch.float64)
    v_cam = verts_f64 @ R_inv.transpose(0, 1) + t_inv  # equivalent to R_inv @ v + t_inv

    z = v_cam[:, 2]
    in_front = z > 1e-3

    u = (frame.fx * v_cam[:, 0] / torch.clamp(z, min=1e-3)) + frame.cx
    v = (frame.fy * v_cam[:, 1] / torch.clamp(z, min=1e-3)) + frame.cy

    H, W = frame.depth.shape
    # Round to nearest pixel (NOT truncation — `.to(torch.long)` rounds toward
    # zero, which lets through sub-pixel slivers at the image edge that
    # shouldn't project to pixel 0).
    ui = torch.round(u).to(torch.long)
    vi = torch.round(v).to(torch.long)
    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

    valid = in_front & in_bounds

    # Early-exit if nothing is visible.
    if not torch.any(valid):
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty

    # Gather sensor depth + masks at projected pixels. Only gather valid verts.
    depth_t = torch.from_numpy(frame.depth).to(device)
    instance_t = torch.from_numpy(frame.instance_mask.astype(np.int64)).to(device)
    class_t = torch.from_numpy(frame.class_mask.astype(np.int64)).to(device)

    ui_v = ui[valid]
    vi_v = vi[valid]
    d_sensor = depth_t[vi_v, ui_v].to(torch.float64)
    z_v = z[valid]
    # Only count a vertex as observed if its projected depth matches the
    # sensor depth AND the sensor depth is > 0 (missing-depth pixels are 0).
    depth_ok = (d_sensor > 0) & (torch.abs(d_sensor - z_v) < depth_tol)

    inst = instance_t[vi_v, ui_v]
    cls = class_t[vi_v, ui_v]
    # Exclude background pixels (instance_mask == 0).
    mask_ok = inst > 0

    keep = depth_ok & mask_ok
    vertex_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)[keep]
    return vertex_indices, inst[keep], cls[keep]


# --- main entry point --------------------------------------------------------


def lift_masks_to_3d(
    vertices: np.ndarray,
    frames: Iterable[Frame],
    *,
    depth_tolerance: float = 0.05,
    overlap_threshold: float = 0.5,
    device: Optional[str] = None,
) -> LiftResult:
    """Full 3-stage lift.

    Args:
        vertices: (N, 3) float array of mesh vertex positions in WORLD frame.
        frames: iterable of `Frame` records. Streamed — not held in memory
            simultaneously.
        depth_tolerance: meters. Absolute tolerance between sensor depth and
            projected-vertex depth. 5 cm works for NavVis-grade captures.
        overlap_threshold: fraction of the smaller local-instance vertex set
            that must be shared for a union-find merge.
        device: "cuda" / "cpu" / None (auto).
    """
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be (N, 3); got {vertices.shape}")
    n_verts = len(vertices)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    verts_t = torch.from_numpy(vertices).to(dev, dtype=torch.float32)

    # (vertex_idx, local_instance_key, class_id) triples. Use lists of
    # numpy arrays rather than one giant contiguous buffer so we can free
    # per-frame GPU allocations as we go.
    all_vertex_idx: list[np.ndarray] = []
    all_local_keys: list[np.ndarray] = []
    all_class_ids: list[np.ndarray] = []

    # (frame_id, mask_id) → unique integer `local_key` for DSU efficiency.
    local_key_of: dict[tuple[int, int], int] = {}
    seen_frame_ids: set[int] = set()

    def _key(fid: int, mid: int) -> int:
        k = local_key_of.get((fid, mid))
        if k is None:
            k = len(local_key_of)
            local_key_of[(fid, mid)] = k
        return k

    for frame in frames:
        # Guard #4: duplicate frame_ids would silently alias (frame_id, mask_id)
        # keys from different physical frames together, merging unrelated objects.
        if frame.frame_id in seen_frame_ids:
            raise ValueError(
                f"duplicate frame_id {frame.frame_id}; every Frame must have a unique id"
            )
        seen_frame_ids.add(frame.frame_id)
        # Guard #8: validate mask dtype — booleans and floats would pass
        # `inst > 0` semantics by accident but corrupt DSU keys.
        if not np.issubdtype(frame.instance_mask.dtype, np.integer):
            raise TypeError(
                f"frame {frame.frame_id}: instance_mask dtype must be integer; "
                f"got {frame.instance_mask.dtype}"
            )
        vi, inst, cls = _project_frame(verts_t, frame, depth_tolerance, dev)
        if vi.numel() == 0:
            continue
        vi_np = vi.cpu().numpy()
        inst_np = inst.cpu().numpy()
        cls_np = cls.cpu().numpy()
        # Map (frame_id, instance_mask_value) → dense local_key integer.
        keys = np.asarray(
            [_key(frame.frame_id, int(m)) for m in inst_np], dtype=np.int64
        )
        all_vertex_idx.append(vi_np)
        all_local_keys.append(keys)
        all_class_ids.append(cls_np.astype(np.int64))

    if not all_vertex_idx:
        return LiftResult(
            instance_ids=np.zeros(n_verts, dtype=np.int64),
            class_ids=np.zeros(n_verts, dtype=np.int64),
            num_instances=0,
        )

    vertex_idx = np.concatenate(all_vertex_idx)
    local_keys = np.concatenate(all_local_keys)
    class_ids = np.concatenate(all_class_ids)
    del all_vertex_idx, all_local_keys, all_class_ids

    # ---- Stage 2: union-find merge across frames.
    # Build key → set-of-vertex-indices.
    key_to_verts: dict[int, set[int]] = defaultdict(set)
    for v, k in zip(vertex_idx, local_keys):
        key_to_verts[int(k)].add(int(v))

    dsu = _DSU()
    for k in key_to_verts:
        dsu.make_set(k)

    # Sparse co-occurrence: for each (key_a, key_b) pair, count how many
    # vertices are claimed by both. This replaces the previous O(K²)-per-vertex
    # candidate-pair enumeration (which blew up at ~500 keys/vertex in the
    # NavVis case). Algorithm: for each vertex, increment pair_count[(a, b)]
    # for every unique pair of keys observed at that vertex. Total work is
    # O(Σ_v K_v²) summed across vertices — exactly matches actual
    # co-observations and skips keys that never share a vertex.
    vertex_to_keys: dict[int, list[int]] = defaultdict(list)
    for v, k in zip(vertex_idx, local_keys):
        vertex_to_keys[int(v)].append(int(k))

    pair_intersection: dict[tuple[int, int], int] = defaultdict(int)
    for keys_at_v in vertex_to_keys.values():
        unique = sorted(set(keys_at_v))
        if len(unique) <= 1:
            continue
        for i in range(len(unique)):
            a = unique[i]
            for j in range(i + 1, len(unique)):
                pair_intersection[(a, unique[j])] += 1

    # Union only pairs whose intersection ÷ min(|a|, |b|) exceeds threshold.
    for (a, b), inter in pair_intersection.items():
        min_size = min(len(key_to_verts[a]), len(key_to_verts[b]))
        if inter / min_size > overlap_threshold:
            dsu.union(a, b)

    # ---- Stage 3: per-vertex majority vote.
    # Compact root → dense instance_id (starting at 1; 0 = unassigned).
    root_to_instance: dict[int, int] = {}

    def _instance_for_root(r: int) -> int:
        iid = root_to_instance.get(r)
        if iid is None:
            iid = len(root_to_instance) + 1
            root_to_instance[r] = iid
        return iid

    per_vertex_instance = np.zeros(n_verts, dtype=np.int64)
    per_vertex_class = np.zeros(n_verts, dtype=np.int64)

    # For each vertex: accumulate (root, class) votes. First decide the
    # winning root (instance), then within that root pick the class by
    # per-vertex majority — NOT by global instance-wide majority. A
    # door-labelled vertex inside a wall-majority instance still gets
    # "door" on that specific vertex. (Code review finding #2.)
    per_vertex_root_votes: dict[int, Counter] = defaultdict(Counter)
    per_vertex_root_class_votes: dict[int, dict[int, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    instance_class_votes: dict[int, Counter] = defaultdict(Counter)

    for v, k, c in zip(vertex_idx, local_keys, class_ids):
        vi, ci = int(v), int(c)
        root = dsu.find(int(k))
        per_vertex_root_votes[vi][root] += 1
        per_vertex_root_class_votes[vi][root][ci] += 1
        instance_class_votes[root][ci] += 1

    for vi, root_votes in per_vertex_root_votes.items():
        winning_root, _count = root_votes.most_common(1)[0]
        iid = _instance_for_root(winning_root)
        per_vertex_instance[vi] = iid
        # Per-vertex class = mode of class votes for the winning root only.
        class_counter = per_vertex_root_class_votes[vi][winning_root]
        per_vertex_class[vi] = class_counter.most_common(1)[0][0]

    instance_to_class = {
        root_to_instance[r]: cls_counter.most_common(1)[0][0]
        for r, cls_counter in instance_class_votes.items()
        if r in root_to_instance
    }

    # Build instance → supporting (frame_id, mask_id) pairs by inverting
    # local_key_of and routing each key through the DSU root → instance map.
    # Cheap (O(K) where K = number of unique (frame_id, mask_id) keys).
    instance_to_mask_list: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (fid, mid), key in local_key_of.items():
        if key not in dsu.parent:  # key never observed any vertex; skip
            continue
        root = dsu.find(key)
        iid = root_to_instance.get(root)
        if iid is None:
            continue
        instance_to_mask_list[iid].append((fid, mid))

    return LiftResult(
        instance_ids=per_vertex_instance,
        class_ids=per_vertex_class,
        num_instances=len(root_to_instance),
        instance_to_class=instance_to_class,
        instance_to_mask_list=dict(instance_to_mask_list),
    )
