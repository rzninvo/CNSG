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
        real_region_ids = sorted(rid for rid in region_ids if not rid.endswith("-1"))
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
