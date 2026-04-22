"""Phase 2 roundtrip test: synthetic mesh → export → habitat_sim load → assert.

Exit criterion (plan §Phase 2):
  > For any input (vertex_labels, instance_ids, region_ids), running the
  > exporter and re-loading through `habitat_sim.Simulator` recovers the
  > same triple (modulo Habitat's CC-largest-vol selection on disconnected
  > regions).

Fixture: two disjoint unit cubes, each in its own room with its own class.
We expect the loader to report 2 SemanticRegions and 2 real SemanticObjects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

try:
    import habitat_sim
except Exception:  # pragma: no cover
    habitat_sim = None


pytestmark = pytest.mark.skipif(
    habitat_sim is None, reason="habitat_sim not available in this environment"
)


def _make_toy_two_box_mesh() -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    """Two axis-aligned unit cubes, disjoint; per-vertex (class_id, region_id).

    Cube A (class=1, region=1) at origin; cube B (class=2, region=2) offset on +x.
    """
    box_a = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    box_b = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    box_b.apply_translation([3.0, 0.0, 0.0])
    mesh = trimesh.util.concatenate([box_a, box_b])

    assert len(box_a.vertices) == 8, "expected a unit cube to have 8 vertices"

    class_ids = np.concatenate(
        [np.ones(8, dtype=np.int64) * 1, np.ones(8, dtype=np.int64) * 2]
    )
    region_ids = np.concatenate(
        [np.ones(8, dtype=np.int64) * 1, np.ones(8, dtype=np.int64) * 2]
    )
    return mesh, class_ids, region_ids


def _make_sim(stem: str, out_dir: Path) -> "habitat_sim.Simulator":
    """Configure habitat_sim to load our exported scene."""
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = stem  # dataset config resolves this via glob
    backend_cfg.scene_dataset_config_file = str(out_dir / f"{stem}.scene_dataset_config.json")
    backend_cfg.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)


def test_two_boxes_two_regions_roundtrip(tmp_path: Path) -> None:
    """Export a 2-box mesh and verify Habitat reports 2 regions + 2 objects."""
    from cnsg.segmentation.export_habitat import export_habitat

    mesh, class_ids, region_ids = _make_toy_two_box_mesh()

    manifest = export_habitat(
        mesh=mesh,
        per_vertex_class_id=class_ids,
        per_vertex_region_id=region_ids,
        class_id_to_name={1: "box_a", 2: "box_b"},
        region_id_to_name={1: "room_a", 2: "room_b"},
        out_dir=tmp_path,
        stem="toy",
    )
    assert manifest.num_instances == 2, "two disjoint boxes should map to two CCs"
    assert manifest.num_regions == 2

    sim = _make_sim("toy", tmp_path)
    try:
        scene = sim.semantic_scene
        # Habitat's HM3D loader auto-inserts a synthetic Unknown region with
        # index = -1 (HM3DSemanticScene.cpp:99-100). We emit two real regions;
        # the loader reports three. Match on IDs, not raw count.
        region_ids = [r.id for r in scene.regions]
        # `"_-1"` is Habitat's synthetic Unknown region (HM3DSemanticScene.cpp:99-100).
        # Exclude it with an exact match, not endswith (which would also drop
        # legitimate IDs like `"_-10"` or `"_31"` that happen to end in "-1").
        real_region_ids = sorted(rid for rid in region_ids if rid != "_-1")
        assert real_region_ids == ["_1", "_2"], (
            f"expected real regions [_1, _2], got {real_region_ids} "
            f"(all: {region_ids})"
        )

        real_objects = [o for o in scene.objects if o is not None and o.semantic_id > 0]
        assert len(real_objects) == 2, (
            f"expected 2 real objects, got {len(real_objects)}"
        )

        category_names = sorted(o.category.name() for o in real_objects)
        assert category_names == ["box_a", "box_b"], category_names

        # Each real region must contain its expected object. This is the
        # assertion that actually proves the region-id field in .semantic.txt
        # wired through correctly (the hardcoded `16` bug was that ALL
        # objects ended up in one fake region).
        region_by_id = {r.id: r for r in scene.regions}
        r1_objs = [o.category.name() for o in region_by_id["_1"].objects if o]
        r2_objs = [o.category.name() for o in region_by_id["_2"].objects if o]
        assert r1_objs == ["box_a"], r1_objs
        assert r2_objs == ["box_b"], r2_objs

        # NB: scene.objects[*].obb.sizes is [0,0,0] even on real HM3D scenes —
        # the OBB data isn't exposed via the Python binding in this habitat-sim
        # version. We skip that assertion. Per-vertex bbox correctness is
        # verified indirectly: if vertex colors did NOT match the SSD map,
        # Habitat would log `Inserted Unknown semantic Color` warnings per
        # distinct color and bucket those vertices into synthetic IDs > N_SSD.
        # No such warnings appearing (manually verified 2026-04-22) means the
        # sRGB round-trip works.
    finally:
        sim.close()


def test_room_id_to_name_map_has_expected_keys(tmp_path: Path) -> None:
    """The mr_viewer.py-style room map must be keyed by stringified region ids."""
    import json

    from cnsg.segmentation.export_habitat import export_habitat

    mesh, class_ids, region_ids = _make_toy_two_box_mesh()

    manifest = export_habitat(
        mesh=mesh,
        per_vertex_class_id=class_ids,
        per_vertex_region_id=region_ids,
        class_id_to_name={1: "box_a", 2: "box_b"},
        region_id_to_name={1: "room_a", 2: "room_b"},
        out_dir=tmp_path,
        stem="toy",
    )
    data = json.loads(manifest.room_id_to_name_map.read_text())
    assert set(data.keys()) == {"1", "2"}
    assert data["1"]["name"] == "room_a"
    assert data["2"]["name"] == "room_b"
    assert len(data["1"]["position"]) == 3


# ---------------------------------------------------------------------------
# Phase 2b: edge cases + real-scan-scale smoke.
# ---------------------------------------------------------------------------


def test_multi_cc_same_region_emits_one_instance_per_component(tmp_path: Path) -> None:
    """Two disjoint geometric pieces with IDENTICAL (class, region) must emit two instances.

    This is the staircase-bug fix. The legacy exporter would have emitted one
    row and relied on Habitat's largest-CC selection, silently dropping the
    smaller component's bbox. Our exporter emits one row per CC so each piece
    has its own bbox.
    """
    from cnsg.segmentation.export_habitat import export_habitat

    # Two cubes: disjoint geometry, same class, same region.
    box_a = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    box_b = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    box_b.apply_translation([3.0, 0.0, 0.0])
    mesh = trimesh.util.concatenate([box_a, box_b])
    n = len(mesh.vertices)
    class_ids = np.ones(n, dtype=np.int64) * 7        # same class
    region_ids = np.ones(n, dtype=np.int64) * 1       # same region

    manifest = export_habitat(
        mesh=mesh,
        per_vertex_class_id=class_ids,
        per_vertex_region_id=region_ids,
        class_id_to_name={7: "stair_run"},
        region_id_to_name={1: "stairwell"},
        out_dir=tmp_path,
        stem="multi_cc",
    )
    assert manifest.num_instances == 2, (
        f"two disjoint pieces with the same (class, region) must split into "
        f"two instances; got {manifest.num_instances}"
    )
    assert manifest.num_regions == 1

    txt_lines = manifest.semantic_txt.read_text().splitlines()
    # Header + 2 instance rows
    assert len(txt_lines) == 3
    # Both rows share category + region_id but have distinct ids and distinct colors.
    row_1 = txt_lines[1].split(",")
    row_2 = txt_lines[2].split(",")
    assert row_1[0] == "1" and row_2[0] == "2"
    assert row_1[1] != row_2[1], "two instances must have distinct colors"
    assert row_1[3] == "1" and row_2[3] == "1", "both must live in region 1"
    assert '"stair_run"' in txt_lines[1] and '"stair_run"' in txt_lines[2]

    sim = _make_sim("multi_cc", tmp_path)
    try:
        scene = sim.semantic_scene
        real_objects = [o for o in scene.objects if o is not None and o.semantic_id > 0]
        assert len(real_objects) == 2
        # Both in the same region.
        region_by_id = {r.id: r for r in scene.regions}
        r1_objs = [o.category.name() for o in region_by_id["_1"].objects if o]
        # `region.objects` iteration order is not guaranteed by the binding;
        # compare as a multiset.
        from collections import Counter
        assert Counter(r1_objs) == Counter(["stair_run", "stair_run"]), r1_objs
    finally:
        sim.close()


def test_single_connected_component_produces_single_instance(tmp_path: Path) -> None:
    """Degenerate case: one mesh, all verts one label → exactly one instance."""
    from cnsg.segmentation.export_habitat import export_habitat

    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    n = len(mesh.vertices)
    class_ids = np.ones(n, dtype=np.int64) * 3
    region_ids = np.ones(n, dtype=np.int64) * 1

    manifest = export_habitat(
        mesh=mesh,
        per_vertex_class_id=class_ids,
        per_vertex_region_id=region_ids,
        class_id_to_name={3: "table"},
        out_dir=tmp_path,
        stem="one_cc",
    )
    assert manifest.num_instances == 1
    assert manifest.num_regions == 1


def test_class_id_zero_vertices_are_dropped(tmp_path: Path) -> None:
    """class_id == 0 is the 'ignore' bucket; those vertices must not become instances."""
    from cnsg.segmentation.export_habitat import export_habitat

    box_a = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    box_b = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    box_b.apply_translation([3.0, 0.0, 0.0])
    mesh = trimesh.util.concatenate([box_a, box_b])
    # box_a is class=1, box_b is class=0 (dropped).
    class_ids = np.concatenate(
        [np.ones(8, dtype=np.int64), np.zeros(8, dtype=np.int64)]
    )
    region_ids = np.ones(len(mesh.vertices), dtype=np.int64)

    manifest = export_habitat(
        mesh=mesh,
        per_vertex_class_id=class_ids,
        per_vertex_region_id=region_ids,
        class_id_to_name={1: "keeper"},
        out_dir=tmp_path,
        stem="with_drop",
    )
    # Only box_a should produce an instance.
    assert manifest.num_instances == 1
    txt = manifest.semantic_txt.read_text()
    assert '"keeper"' in txt


def test_hge_scale_real_mesh_subset_exports_and_loads(tmp_path: Path) -> None:
    """Real scan-scale smoke test: load a connected subset of the actual HGE
    voxelized mesh, synthesise plausible labels, export, and verify Habitat
    loads it cleanly.

    Why a subset, not the full 4M-vert mesh: at ~4M verts + ~2000 distinct
    instance colors, Habitat's per-color connected-component bbox pass OOMs
    or segfaults (reproduced 2026-04-22, out of scope for Phase 2 — tracked
    for Phase 3/4 when the real segmentation pipeline lands and we can
    decimate upstream). The subset here is ~200k verts — real-scan-scale
    enough to flush out any mesh-format bugs that toy cubes would miss,
    while staying well under Habitat's load envelope.

    Labels are synthetic (proper semantic labels come from Phase 3). This
    test is about ensuring the exporter survives real-world vertex counts,
    mesh extents (~130 m across), non-manifold topology, and produces a
    GLB that Habitat accepts. It is NOT a correctness test for segmentation
    content.
    """
    from cnsg.segmentation.export_habitat import export_habitat

    hge_ply = Path(__file__).resolve().parent.parent / "mesh_pipeline" / "data" / "HGE_cut.voxelized.ply"
    if not hge_ply.exists():
        pytest.skip(f"HGE voxelized mesh not on disk at {hge_ply}")

    full_mesh = trimesh.load(hge_ply, force="mesh")

    # Take the first N faces as a connected submesh: preserves topology,
    # fast, and yields a spatially-coherent chunk of the building.
    n_faces_keep = 200_000
    face_subset = np.asarray(full_mesh.faces[:n_faces_keep])
    used_verts = np.unique(face_subset.reshape(-1))
    vtx_remap = -np.ones(len(full_mesh.vertices), dtype=np.int64)
    vtx_remap[used_verts] = np.arange(len(used_verts))
    mesh = trimesh.Trimesh(
        vertices=full_mesh.vertices[used_verts],
        faces=vtx_remap[face_subset],
        process=False,
    )
    assert len(mesh.vertices) > 50_000, f"expected meaningful scale; got {len(mesh.vertices)} verts"

    # Synthetic labels: split z into 3 bands (floor / mid / ceiling), split
    # x into 2 regions. Yields up to 6 (class, region) pairs + many CCs.
    z = mesh.vertices[:, 2]
    x = mesh.vertices[:, 0]
    z_lo, z_hi = np.percentile(z, [33, 66])
    x_mid = np.median(x)
    class_ids = np.where(z < z_lo, 1, np.where(z < z_hi, 2, 3)).astype(np.int64)
    region_ids = np.where(x < x_mid, 1, 2).astype(np.int64)

    manifest = export_habitat(
        mesh=mesh,
        per_vertex_class_id=class_ids,
        per_vertex_region_id=region_ids,
        class_id_to_name={1: "floor", 2: "wall", 3: "ceiling"},
        region_id_to_name={1: "west_wing", 2: "east_wing"},
        out_dir=tmp_path,
        stem="hge_smoke",
    )
    assert manifest.num_instances >= 6, (
        f"expected at least 6 instances from 3 classes × 2 regions; "
        f"got {manifest.num_instances}"
    )
    assert manifest.num_regions == 2

    sim = _make_sim("hge_smoke", tmp_path)
    try:
        scene = sim.semantic_scene
        real_objects = [o for o in scene.objects if o is not None and o.semantic_id > 0]
        # Should match what the exporter emitted.
        assert len(real_objects) == manifest.num_instances
        # Both regions populated.
        region_ids_loaded = {r.id for r in scene.regions if not r.id.endswith("-1")}
        assert region_ids_loaded == {"_1", "_2"}
    finally:
        sim.close()
