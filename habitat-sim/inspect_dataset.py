#!/usr/bin/env python3
"""Quick dataset inspector: regions (room types), objects (OBBs), navmesh."""
import ctypes
import sys

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import numpy as np
import habitat_sim
from habitat_sim.utils.settings import default_sim_settings, make_cfg


def inspect(scene, dataset):
    s = dict(default_sim_settings)
    s["scene"] = scene
    s["scene_dataset_config_file"] = dataset
    s["color_sensor"] = False
    s["semantic_sensor"] = False
    s["depth_sensor"] = False
    cfg = make_cfg(s)
    # No GL context needed for semantic-scene + navmesh inspection.
    cfg.sim_cfg.create_renderer = False
    sim = habitat_sim.Simulator(cfg)
    sem = sim.semantic_scene

    print("=" * 70)
    print("SCENE:", scene)
    print("=" * 70)
    if sem is None:
        print("NO semantic scene!")
        sim.close()
        return

    regions = list(sem.regions)
    objects = list(sem.objects)
    print(f"regions: {len(regions)}   objects: {len(objects)}")

    # Region room-type names + sample ids
    print("\n-- regions (id | category/room-type | #objects) --")
    for r in regions[:20]:
        cat = None
        try:
            cat = r.category.name() if r.category is not None else None
        except Exception:
            cat = "?"
        print(f"   id={getattr(r,'id','?')!r:>10}  room={cat!r:<18} objs={len(list(r.objects))}")

    # Object OBBs non-zero?
    nonzero = 0
    sample = []
    for o in objects:
        try:
            he = o.obb.half_extents
            if he[0] > 0 or he[1] > 0 or he[2] > 0:
                nonzero += 1
                if len(sample) < 8:
                    sample.append((o.id, o.category.name() if o.category else "?",
                                   [round(float(x), 2) for x in he]))
        except Exception:
            pass
    print(f"\nobjects with non-zero OBB: {nonzero}/{len(objects)}")
    print("sample objects (id | category | half_extents):")
    for oid, cat, he in sample:
        print(f"   {oid!r:>10}  {cat!r:<18} {he}")

    # Navmesh
    pf = sim.pathfinder
    print(f"\nnavmesh loaded: {pf.is_loaded}")
    if pf.is_loaded:
        try:
            print(f"navigable area: {pf.navigable_area:.2f} m^2")
        except Exception:
            pass
        # sample heights to gauge floors
        ys = []
        for _ in range(400):
            p = pf.get_random_navigable_point()
            if p is not None and np.isfinite(p).all():
                ys.append(round(float(p[1]), 1))
        if ys:
            uniq = sorted(set(ys))
            print(f"navigable Y levels (sampled): {uniq[:15]}")
    sim.close()


if __name__ == "__main__":
    inspect(sys.argv[1], sys.argv[2])
