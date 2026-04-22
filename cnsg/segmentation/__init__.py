"""Offline 3D segmentation + Habitat export pipeline (Phase 2/3).

- `palette.py`: deterministic color-per-object RGB palette.
- `export_habitat.py`: writes HM3D-compatible .semantic.glb + .semantic.txt +
  room_id_to_name_map.json + minimal scene_dataset_config.json.

Format authority: `docs/report/01_architecture-lean-migration/habitat-format-spec.md`.
"""
