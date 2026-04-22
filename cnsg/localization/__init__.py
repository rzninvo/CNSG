"""In-process visual localization — replaces the LaMAR Docker pipeline.

At query time: SuperPoint features + MegaLoc retrieval + LightGlue matching
+ pycolmap PnP/RANSAC. Pure Python, pure pip install. No Docker, no
subprocess, no scantools dependency.

Map artifacts (COLMAP reconstruction + per-image feature DB + retrieval index)
are built once offline — see `cnsg.localization.map_builder` — and loaded
read-only from `Settings.localization.map_dir`.
"""
