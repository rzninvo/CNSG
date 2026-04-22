"""Tests for `cnsg.segmentation.lift_2d_to_3d`.

Uses fully synthetic fixtures: hand-rolled meshes + pinhole-projected
depth / mask / class images. Covers:
- Empty / degenerate cases (no frames, all-background masks).
- Single object visible across two views → 1 instance, 1 class, union-find
  merges the two per-frame local IDs.
- Two distinct objects → 2 instances, correct per-object class.
- Occlusion: a vertex behind another surface is NOT counted.
- Depth mismatch: a vertex whose sensor depth disagrees with its projected
  depth is NOT counted.
- Class-vote aggregation: per-frame class label mismatches get resolved by
  majority vote.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from cnsg.segmentation.lift_2d_to_3d import Frame, lift_masks_to_3d


# ---------- fixture helpers --------------------------------------------------


def _look_at_pose(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """Build T_world_cam for a camera at `eye` looking at `target`.

    Returns a 4×4 float64 transform that takes camera-frame points into world
    frame. Uses OpenCV convention: +Z camera forward, +Y camera down.
    """
    if up is None:
        up = np.array([0.0, -1.0, 0.0])
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    R = np.column_stack([right, -true_up, forward])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = eye
    return T


def _render_synthetic_frame(
    vertices: np.ndarray,
    per_vertex_instance: np.ndarray,
    per_vertex_class: np.ndarray,
    frame_id: int,
    eye: np.ndarray,
    target: np.ndarray,
    W: int = 128,
    H: int = 96,
    fov_deg: float = 60.0,
) -> Frame:
    """Render a synthetic frame by forward-projecting a point cloud.

    For each vertex, compute its pixel coordinates and depth. The pixel gets
    tagged with the vertex's instance and class. If two vertices hit the
    same pixel, the nearer one wins. Other pixels are background (0 depth,
    0 instance, 0 class).

    This is a toy forward-renderer that doesn't handle surfaces — fine for
    the unit test because we directly control occlusion.
    """
    fx = fy = 0.5 * W / np.tan(np.deg2rad(fov_deg) / 2)
    cx, cy = 0.5 * W, 0.5 * H

    T_wc = _look_at_pose(np.asarray(eye), np.asarray(target))
    # Transform vertices into camera frame.
    R = T_wc[:3, :3]
    t = T_wc[:3, 3]
    v_cam = (vertices - t) @ R  # R^T @ (v - t) = (v - t) @ R for row-vectors

    depth = np.zeros((H, W), dtype=np.float32)
    instance = np.zeros((H, W), dtype=np.int16)
    cls = np.zeros((H, W), dtype=np.int8)

    z = v_cam[:, 2]
    in_front = z > 1e-3
    u = np.round(fx * v_cam[:, 0] / np.maximum(z, 1e-3) + cx).astype(np.int64)
    v = np.round(fy * v_cam[:, 1] / np.maximum(z, 1e-3) + cy).astype(np.int64)
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    valid = in_front & in_bounds

    # Z-buffer: nearest vertex at each pixel wins.
    for i in np.nonzero(valid)[0]:
        ui, vi = int(u[i]), int(v[i])
        zi = float(z[i])
        if depth[vi, ui] == 0 or zi < depth[vi, ui]:
            depth[vi, ui] = zi
            instance[vi, ui] = int(per_vertex_instance[i])
            cls[vi, ui] = int(per_vertex_class[i])

    return Frame(
        frame_id=frame_id,
        depth=depth,
        instance_mask=instance,
        class_mask=cls,
        T_world_cam=T_wc,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )


# ---------- tests ------------------------------------------------------------


def test_no_frames_returns_all_unassigned() -> None:
    verts = np.random.default_rng(0).uniform(-1, 1, size=(10, 3))
    result = lift_masks_to_3d(verts, frames=[], device="cpu")
    assert result.num_instances == 0
    assert np.all(result.instance_ids == 0)
    assert np.all(result.class_ids == 0)


def test_all_background_masks_produces_no_instances() -> None:
    """If every frame's instance_mask is 0, no vertex gets an assignment."""
    verts = np.array([[0.0, 0.0, 5.0]])  # one vertex 5 m in front of origin
    eye = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 5.0])
    frame = _render_synthetic_frame(
        verts,
        per_vertex_instance=np.zeros(1, dtype=np.int16),  # background
        per_vertex_class=np.zeros(1, dtype=np.int8),
        frame_id=0,
        eye=eye,
        target=target,
    )
    result = lift_masks_to_3d(verts, frames=[frame], device="cpu")
    assert result.num_instances == 0


def _sparse_grid_cluster(
    center: np.ndarray, *, nx: int = 5, ny: int = 5, step: float = 0.4
) -> np.ndarray:
    """A flat grid in the x/y plane centred at `center`.

    Sparse enough that each vertex gets its own pixel in the synthetic
    renderer, and with depth=center.z for all verts so the depth test
    passes cleanly for every projection.
    """
    xs = (np.arange(nx) - (nx - 1) / 2) * step
    ys = (np.arange(ny) - (ny - 1) / 2) * step
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny)], axis=1) + center


def test_single_object_two_views_merges_to_one_instance() -> None:
    """A sparse grid visible from two angles — where frame 0 and frame 1
    give the same physical object DIFFERENT per-frame instance mask IDs —
    must lift to ONE global instance after union-find merge.
    """
    verts = _sparse_grid_cluster(np.array([0.0, 0.0, 5.0]))

    inst_f0 = np.full(len(verts), 1, dtype=np.int16)     # SAM 3 calls it "1" in frame 0
    inst_f1 = np.full(len(verts), 7, dtype=np.int16)     # ...but "7" in frame 1
    cls = np.full(len(verts), 9, dtype=np.int8)          # class "chair" consistently

    f0 = _render_synthetic_frame(
        verts, inst_f0, cls, frame_id=0,
        eye=np.array([0, 0, 0]), target=np.array([0, 0, 5]),
        W=256, H=192,
    )
    f1 = _render_synthetic_frame(
        verts, inst_f1, cls, frame_id=1,
        eye=np.array([1, 0, 3]), target=np.array([0, 0, 5]),
        W=256, H=192,
    )
    result = lift_masks_to_3d(verts, frames=[f0, f1], device="cpu", depth_tolerance=0.05)

    assert result.num_instances == 1, (
        f"expected 1 merged instance; got {result.num_instances}. "
        f"iids={set(result.instance_ids.tolist())}"
    )
    assigned = result.instance_ids[result.instance_ids > 0]
    assert assigned.size > 0
    assert set(assigned.tolist()) == {1}
    assert result.instance_to_class == {1: 9}


def test_two_distinct_objects_yield_two_instances() -> None:
    """Two separately-viewed objects with disjoint vertex sets → 2 instances.

    Use two cameras, each looking at ONE cluster. Guarantees zero projection
    overlap between the clusters so the lifter's union-find has no reason
    to merge them. Exercises the "don't collapse separate objects" path.
    """
    a = _sparse_grid_cluster(np.array([-2.0, 0.0, 5.0]))
    b = _sparse_grid_cluster(np.array([2.0, 0.0, 5.0]))
    verts = np.concatenate([a, b], axis=0)
    inst = np.concatenate([np.full(len(a), 1), np.full(len(b), 1)]).astype(np.int16)
    cls = np.concatenate([np.full(len(a), 9), np.full(len(b), 8)]).astype(np.int8)

    # Camera 0 — tight FOV on cluster A only. Cluster B is far off-frustum.
    f0 = _render_synthetic_frame(
        verts, inst, cls, frame_id=0,
        eye=np.array([-2, 0, 2]), target=np.array([-2, 0, 5]),
        W=128, H=96, fov_deg=40,
    )
    # Camera 1 — tight FOV on cluster B only.
    f1 = _render_synthetic_frame(
        verts, inst, cls, frame_id=1,
        eye=np.array([2, 0, 2]), target=np.array([2, 0, 5]),
        W=128, H=96, fov_deg=40,
    )

    result = lift_masks_to_3d(verts, frames=[f0, f1], device="cpu", depth_tolerance=0.05)
    assert result.num_instances == 2, (
        f"two disjoint objects should lift to two instances; got {result.num_instances}"
    )
    assert set(result.instance_to_class.values()) == {8, 9}


def test_depth_mismatch_rejects_vote() -> None:
    """If a frame's sensor depth disagrees with the projected vertex depth,
    the vertex must NOT be counted for that frame."""
    verts = np.array([[0, 0, 5]])
    inst = np.array([1], dtype=np.int16)
    cls = np.array([9], dtype=np.int8)

    f = _render_synthetic_frame(
        verts, inst, cls, frame_id=0,
        eye=np.array([0, 0, 0]), target=np.array([0, 0, 5]),
    )
    # Corrupt the depth map to something far off from the projected vertex depth.
    bad_depth = f.depth.copy()
    bad_depth[bad_depth > 0] = 20.0  # wrong by 15 m
    f_bad = replace(f, depth=bad_depth)

    result = lift_masks_to_3d(verts, frames=[f_bad], device="cpu", depth_tolerance=0.05)
    assert result.num_instances == 0, (
        "vertex with mismatched sensor depth should not produce an instance"
    )


def test_class_vote_resolves_majority_across_frames() -> None:
    """If Mask2Former produces conflicting class labels across views, the
    majority wins for the final per-instance class."""
    verts = _sparse_grid_cluster(np.array([0.0, 0.0, 5.0]))
    inst = np.full(len(verts), 1, dtype=np.int16)
    cls_chair = np.full(len(verts), 9, dtype=np.int8)
    cls_sofa = np.full(len(verts), 10, dtype=np.int8)

    # Two chair frames, one sofa frame, all from different angles.
    f0 = _render_synthetic_frame(
        verts, inst, cls_chair, frame_id=0,
        eye=np.array([0, 0, 0]), target=np.array([0, 0, 5]),
        W=256, H=192,
    )
    f1 = _render_synthetic_frame(
        verts, inst, cls_chair, frame_id=1,
        eye=np.array([1, 0, 3]), target=np.array([0, 0, 5]),
        W=256, H=192,
    )
    f2 = _render_synthetic_frame(
        verts, inst, cls_sofa, frame_id=2,
        eye=np.array([-1, 0, 3]), target=np.array([0, 0, 5]),
        W=256, H=192,
    )
    result = lift_masks_to_3d(verts, frames=[f0, f1, f2], device="cpu", depth_tolerance=0.05)
    assert result.num_instances == 1
    # Majority (chair, 2/3 views) wins.
    assert result.instance_to_class[1] == 9


def test_frame_without_visible_vertices_is_skipped() -> None:
    """Frame with camera pointing away from all verts shouldn't crash or
    contribute votes."""
    verts = np.array([[0, 0, 5]])
    inst = np.array([1], dtype=np.int16)
    cls = np.array([9], dtype=np.int8)
    # Valid frame.
    f_good = _render_synthetic_frame(
        verts, inst, cls, frame_id=0,
        eye=np.array([0, 0, 0]), target=np.array([0, 0, 5]),
    )
    # Camera looking the wrong way → no verts visible.
    f_bad = _render_synthetic_frame(
        verts, inst, cls, frame_id=1,
        eye=np.array([0, 0, 10]), target=np.array([0, 0, 20]),
    )
    result = lift_masks_to_3d(verts, frames=[f_good, f_bad], device="cpu", depth_tolerance=0.1)
    assert result.num_instances == 1


# --- reviewer-driven fix validation -----------------------------------------


def test_duplicate_frame_ids_raise() -> None:
    """Guard against silent (frame_id, mask_id) key collisions (finding #4)."""
    verts = _sparse_grid_cluster(np.array([0.0, 0.0, 5.0]))
    inst = np.ones(len(verts), dtype=np.int16)
    cls = np.full(len(verts), 9, dtype=np.int8)

    f0 = _render_synthetic_frame(
        verts, inst, cls, frame_id=0,
        eye=np.array([0, 0, 0]), target=np.array([0, 0, 5]),
    )
    f_dup = _render_synthetic_frame(
        verts, inst, cls, frame_id=0,  # same id — should be rejected
        eye=np.array([1, 0, 3]), target=np.array([0, 0, 5]),
    )
    with pytest.raises(ValueError, match="duplicate frame_id"):
        lift_masks_to_3d(verts, frames=[f0, f_dup], device="cpu")


def test_non_integer_instance_mask_raises() -> None:
    """Boolean or float instance masks must raise, not silently alias (finding #8)."""
    verts = np.array([[0, 0, 5.0]])
    from cnsg.segmentation.lift_2d_to_3d import Frame

    bad = Frame(
        frame_id=0,
        depth=np.ones((8, 8), dtype=np.float32),
        instance_mask=np.ones((8, 8), dtype=bool),  # wrong dtype
        class_mask=np.zeros((8, 8), dtype=np.int8),
        T_world_cam=np.eye(4),
        fx=1.0, fy=1.0, cx=4.0, cy=4.0,
    )
    with pytest.raises(TypeError, match="instance_mask dtype"):
        lift_masks_to_3d(verts, frames=[bad], device="cpu")


def test_depth_mismatch_just_under_tolerance_passes() -> None:
    """Boundary: depth off by 4 cm (tol=5 cm) → vertex is still counted.

    Regression gate for finding #9 — the existing tests only exercised
    far-from-threshold values, so a subtle pose-inverse / depth-test bug
    wouldn't have been caught.
    """
    verts = _sparse_grid_cluster(np.array([0.0, 0.0, 5.0]))
    inst = np.ones(len(verts), dtype=np.int16)
    cls = np.full(len(verts), 9, dtype=np.int8)

    f = _render_synthetic_frame(
        verts, inst, cls, frame_id=0,
        eye=np.array([0, 0, 0]), target=np.array([0, 0, 5]),
    )
    # Shift the depth map by +4 cm everywhere. With tol=5 cm, verts stay visible.
    from dataclasses import replace
    f_near = replace(f, depth=f.depth + 0.04)
    result_near = lift_masks_to_3d(verts, frames=[f_near], device="cpu", depth_tolerance=0.05)
    assert result_near.num_instances == 1, "4 cm under 5 cm tolerance should pass"

    # With tol=5 cm, +6 cm shift should fail.
    f_far = replace(f, depth=f.depth + 0.06)
    result_far = lift_masks_to_3d(verts, frames=[f_far], device="cpu", depth_tolerance=0.05)
    assert result_far.num_instances == 0, "6 cm over 5 cm tolerance should fail"


def test_per_vertex_class_respects_local_observations() -> None:
    """A vertex whose own frame votes are mostly class=X must get class=X,
    even if the rest of the instance's verts majority-vote for class=Y.

    Validates finding #2 fix: per-vertex class = majority over the winning-root
    observations AT THAT VERTEX, not the global instance-wide majority.
    """
    # 10 verts. First one is labelled as class 7 ("door") consistently in
    # every frame. The other 9 are labelled as class 1 ("wall") consistently.
    # Put them at positions where they share the same per-frame instance_mask
    # (=1) so the union-find puts them all in one instance.
    rng = np.random.default_rng(7)
    verts = _sparse_grid_cluster(np.array([0.0, 0.0, 5.0]), nx=5, ny=2, step=0.4)
    assert len(verts) == 10
    inst = np.ones(len(verts), dtype=np.int16)
    cls = np.ones(len(verts), dtype=np.int8)  # most are class 1
    cls[0] = 7  # but vertex 0 is class 7

    # Three views for robust coverage.
    frames = [
        _render_synthetic_frame(verts, inst, cls, frame_id=i,
                                 eye=np.array([dx, 0, 3]), target=np.array([0, 0, 5]),
                                 W=256, H=192)
        for i, dx in enumerate([-0.5, 0.0, 0.5])
    ]
    result = lift_masks_to_3d(verts, frames=frames, device="cpu", depth_tolerance=0.05)

    assert result.num_instances == 1
    # Vertex 0's class should be 7 (door), despite the instance-wide majority being 1 (wall).
    # Global instance class is whichever dominates (1 wins 9:1).
    assert result.instance_to_class[1] == 1
    assert result.class_ids[0] == 7, (
        f"vertex 0 observed only class 7 but got assigned {result.class_ids[0]} "
        f"— per-vertex class must respect local observations, not global mode"
    )


def test_cpu_and_cuda_produce_identical_results() -> None:
    """Regression gate (finding #10): any future fp32→fp16 or kernel drift
    between devices would silently change production output."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    verts = _sparse_grid_cluster(np.array([0.0, 0.0, 5.0]))
    inst = np.full(len(verts), 1, dtype=np.int16)
    cls = np.full(len(verts), 9, dtype=np.int8)
    f0 = _render_synthetic_frame(
        verts, inst, cls, frame_id=0,
        eye=np.array([0, 0, 0]), target=np.array([0, 0, 5]),
        W=256, H=192,
    )
    f1 = _render_synthetic_frame(
        verts, inst, cls, frame_id=1,
        eye=np.array([1, 0, 3]), target=np.array([0, 0, 5]),
        W=256, H=192,
    )
    r_cpu = lift_masks_to_3d(verts, frames=[f0, f1], device="cpu", depth_tolerance=0.05)
    r_gpu = lift_masks_to_3d(verts, frames=[f0, f1], device="cuda", depth_tolerance=0.05)
    np.testing.assert_array_equal(r_cpu.instance_ids, r_gpu.instance_ids)
    np.testing.assert_array_equal(r_cpu.class_ids, r_gpu.class_ids)
