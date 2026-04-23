"""Diff two `build_summary.json` files side-by-side.

Use to compare HGE builds across fix commits — e.g. pre-alignment-fix vs
post-alignment-fix — so regressions and improvements are easy to eyeball
and paste into findings reports.

    python -m cnsg.segmentation.compare_build_summaries \\
        --before data/maps/hge/pre_alignment_fix_backup/build_summary.json \\
        --after  data/maps/hge/build_summary.json

Prints a markdown table of the scalar metrics and a separate per-class
coverage table when both summaries carry a `coverage` block (produced by
builds at or after commit 10a2266).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


_SCALAR_FIELDS = [
    ("num_frames_processed",   "frames"),
    ("num_verts",              "verts (post-decimation)"),
    ("lift_num_instances",     "raw lift instances"),
    ("export_num_instances",   "exported instances (post-filter)"),
    ("export_num_regions",     "exported regions"),
    ("hierarchy_num_floors",   "floors"),
    ("hierarchy_num_rooms",    "rooms"),
    ("total_time_s",           "total wall time (s)"),
]


def _fmt(val) -> str:
    if isinstance(val, (int, float)):
        return f"{val:,}" if isinstance(val, int) else f"{val:.3f}"
    if val is None:
        return "-"
    return str(val)


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _cov_section(before: dict, after: dict) -> str:
    """Render the per-class coverage table if both summaries have it."""
    cov_a = before.get("coverage")
    cov_b = after.get("coverage")
    out = []

    if cov_a is None and cov_b is None:
        return "(no `coverage` key in either summary — both pre-10a2266)"

    out.append("## Overall coverage\n")
    out.append("| Metric | Before | After |")
    out.append("|---|---|---|")
    for k in ("total_verts", "labeled_verts", "labeled_fraction"):
        va = (cov_a or {}).get(k)
        vb = (cov_b or {}).get(k)
        if k == "labeled_fraction":
            va = None if va is None else f"{va * 100:.2f} %"
            vb = None if vb is None else f"{vb * 100:.2f} %"
        out.append(f"| {k} | {_fmt(va)} | {_fmt(vb)} |")

    out.append("\n## Per-class verts (non-zero)\n")
    out.append("| Class | Before | After | Δ |")
    out.append("|---|---|---|---|")
    classes = sorted(
        set((cov_a or {}).get("per_class", {})) | set((cov_b or {}).get("per_class", {}))
    )
    for cls in classes:
        va = int((cov_a or {}).get("per_class", {}).get(cls, 0))
        vb = int((cov_b or {}).get("per_class", {}).get(cls, 0))
        delta = vb - va
        sign = "+" if delta > 0 else ""
        out.append(f"| {cls} | {va:,} | {vb:,} | {sign}{delta:,} |")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    args = p.parse_args()

    before = _load(args.before)
    after = _load(args.after)

    print(f"# Build diff\n")
    print(f"- **before**: `{args.before}`")
    print(f"- **after** : `{args.after}`")
    print()
    print("## Scalars\n")
    print("| Metric | Before | After |")
    print("|---|---|---|")
    for key, label in _SCALAR_FIELDS:
        print(f"| {label} | {_fmt(before.get(key))} | {_fmt(after.get(key))} |")
    print()
    print(_cov_section(before, after))


if __name__ == "__main__":
    main()
