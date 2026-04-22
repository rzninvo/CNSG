"""Minimal parser for the NavVis (LaMAR-style) Capture format.

Replaces the `scantools.capture` dependency. The Capture format is
CSV-with-header text files; we only need the three that describe
sensors, per-image paths, and ground-truth poses.

Used at:
- map build time (enumerate images to extract features from, look up
  intrinsics per sensor), and
- smoke-test / evaluation time (ground-truth pose lookup per image).

It is NOT used on the hot localization path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CameraIntrinsics:
    """Camera model + pinhole parameters."""

    sensor_id: str
    model: str           # e.g. 'PINHOLE'
    width: int
    height: int
    params: tuple[float, ...]  # (fx, fy, cx, cy) for PINHOLE

    @property
    def fx(self) -> float:
        return self.params[0]

    @property
    def fy(self) -> float:
        return self.params[1]

    @property
    def cx(self) -> float:
        return self.params[2]

    @property
    def cy(self) -> float:
        return self.params[3]


@dataclass(frozen=True)
class Pose:
    """6-DoF pose. Convention recorded explicitly per construction site."""

    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float

    @property
    def quat_wxyz(self) -> tuple[float, float, float, float]:
        return (self.qw, self.qx, self.qy, self.qz)

    @property
    def translation(self) -> tuple[float, float, float]:
        return (self.tx, self.ty, self.tz)


@dataclass(frozen=True)
class ImageRecord:
    timestamp: int
    sensor_id: str
    relative_path: str  # e.g. 'images_undistr_center/00000-cam0__center.jpg'


def _iter_data_lines(path: Path) -> Iterable[list[str]]:
    """Yield comma-separated fields from a Capture text file, skipping header comments.

    NavVis Capture files start with a `# ...` header line describing columns;
    data lines are comma-separated with arbitrary whitespace padding.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield [cell.strip() for cell in line.split(",")]


def parse_sensors(path: Path) -> dict[str, CameraIntrinsics]:
    """Parse `sensors.txt`. Returns only camera sensors (skips wifi/bt/lidar).

    Format: `sensor_id, name, sensor_type, [sensor_params]+`
    For cameras: `sensor_id, name, camera, MODEL, W, H, fx, fy, cx, cy, ...`
    """
    sensors: dict[str, CameraIntrinsics] = {}
    for cells in _iter_data_lines(path):
        if len(cells) < 3 or cells[2] != "camera":
            continue
        sensor_id = cells[0]
        model = cells[3]
        width = int(cells[4])
        height = int(cells[5])
        params = tuple(float(x) for x in cells[6:])
        sensors[sensor_id] = CameraIntrinsics(
            sensor_id=sensor_id, model=model, width=width, height=height, params=params
        )
    if not sensors:
        raise ValueError(f"No camera sensors found in {path}")
    return sensors


def parse_images(path: Path) -> list[ImageRecord]:
    """Parse `images.txt`. Format: `timestamp, sensor_id, image_path`."""
    records: list[ImageRecord] = []
    for cells in _iter_data_lines(path):
        if len(cells) < 3:
            continue
        records.append(
            ImageRecord(
                timestamp=int(cells[0]),
                sensor_id=cells[1],
                relative_path=cells[2],
            )
        )
    if not records:
        raise ValueError(f"No image records found in {path}")
    return records


def parse_trajectories(path: Path) -> dict[tuple[int, str], Pose]:
    """Parse `trajectories.txt`. Format:

        `timestamp, device_id, qw, qx, qy, qz, tx, ty, tz, *covariance`

    The Capture-format convention for this pose is **world-from-camera**
    (`T_wc`) — the pose that, when applied to a point in the camera frame,
    gives its coordinates in the world frame. This matches what downstream
    consumers (Habitat agent placement) expect.
    """
    poses: dict[tuple[int, str], Pose] = {}
    for cells in _iter_data_lines(path):
        if len(cells) < 9:
            continue
        key = (int(cells[0]), cells[1])
        poses[key] = Pose(
            qw=float(cells[2]),
            qx=float(cells[3]),
            qy=float(cells[4]),
            qz=float(cells[5]),
            tx=float(cells[6]),
            ty=float(cells[7]),
            tz=float(cells[8]),
        )
    if not poses:
        raise ValueError(f"No poses found in {path}")
    return poses
