"""Project config.

Typed, env-loadable settings. Replaces the scattered hardcodes in the legacy
`mr_viewer.py` (agent eye height, densify step, max bbox count, etc.).

Phase 1 scope: localization settings only. Further sections (scene, LLM,
navigation) are added as their modules land.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


class LocalizationSettings(BaseSettings):
    """Visual localization (hloc + LightGlue + MegaLoc + pycolmap PnP)."""

    model_config = SettingsConfigDict(env_prefix="CNSG_LOCALIZATION_", env_file=".env", extra="ignore")

    # Where the precomputed map artifacts live (COLMAP recon + feature DB + retrieval index).
    # Built once offline, read-only at serve time.
    map_dir: Path = Field(default_factory=lambda: _project_root() / "data" / "maps" / "hge")

    # NavVis Capture-format session the map was built from. Consumed at
    # localizer-load time to pre-compose `alignment_global.txt` onto returned
    # poses so they live in the absolute frame (= what Habitat uses).
    session_dir: Path = Field(
        default_factory=lambda: (
            _project_root() / "mesh_pipeline" / "data" / "navvis_2022-02-06_12.55.11"
        )
    )

    # Feature extractor (Phase 1: SuperPoint).
    feature_name: str = "superpoint_max"  # hloc config name

    # Matcher (Phase 1: LightGlue).
    matcher_name: str = "superpoint+lightglue"  # hloc config name

    # Retrieval (Phase 1: MegaLoc — SOTA on LaMAR; falls back to NetVLAD if unavailable).
    retrieval_name: str = "megaloc"
    retrieval_num_pairs: int = 10

    # RANSAC / PnP.
    ransac_max_error_px: float = 12.0
    ransac_min_inliers: int = 20

    # Default camera intrinsics for phone/web queries when EXIF is missing.
    # f = 0.7 * max(W, H), principal point at image center (matches LaMAR's heuristic).
    default_focal_ratio: float = 0.7


class Settings(BaseSettings):
    """Top-level settings. Compose sub-section settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    project_root: Path = Field(default_factory=_project_root)
    localization: LocalizationSettings = Field(default_factory=LocalizationSettings)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Singleton settings accessor. Safe to call from hot paths (cached)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
