"""Unit tests for cnsg.segmentation.gpt5_tagger.

Pin the contract without hitting the OpenAI API. The tagger has three
behaviours we lean on downstream:

  - schema correctness: FrameTags + TaggedObject parse / round-trip.
  - cache: hits when config matches, misses (and re-saves) when config
    hash differs, atomic-write via .tmp.
  - consumer helpers: sam3_prompts dedup + landmarks_only filter.

Network code is mocked at the OpenAI client boundary. We construct a
fake client that returns a pre-built `parsed` payload, so every test
runs offline / in CI without an API key.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cnsg.segmentation.gpt5_tagger import (
    FrameTags,
    GPT5Tagger,
    SCHEMA_VERSION,
    PROMPT_TEMPLATE_VERSION,
    TaggedObject,
    _CacheEntry,
    _config_hash,
)


# ----- fixtures -------------------------------------------------------------


def _make_frame_tags() -> FrameTags:
    return FrameTags(
        objects=[
            TaggedObject(phrase="marble bust on plinth", count=1, landmark=True),
            TaggedObject(phrase="stone column", count=4, landmark=True),
            TaggedObject(phrase="stone column", count=4, landmark=True),  # dup → coalesces
            TaggedObject(phrase="paper on table", count=2, landmark=False),
        ],
        scene_description="An arched corridor with stone columns and a marble bust.",
    )


def _make_jpg(tmp_path: Path, name: str = "frame.jpg") -> Path:
    """Tiny valid JPEG so PIL/openai don't choke if anything reads its bytes."""
    p = tmp_path / name
    # Minimum viable JPEG: SOI + EOI markers. The tagger only base64-encodes
    # the bytes; nothing in the test code path actually decodes the image.
    p.write_bytes(b"\xff\xd8\xff\xd9")
    return p


# ----- schema --------------------------------------------------------------


def test_frame_tags_round_trip() -> None:
    tags = _make_frame_tags()
    serialized = tags.model_dump_json()
    restored = FrameTags.model_validate_json(serialized)
    assert restored.scene_description == tags.scene_description
    assert len(restored.objects) == 4
    assert restored.objects[0].phrase == "marble bust on plinth"


def test_tagged_object_count_must_be_positive() -> None:
    with pytest.raises(Exception):
        TaggedObject(phrase="x", count=0, landmark=True)


# ----- consumer helpers ----------------------------------------------------


def test_sam3_prompts_dedup_case_insensitive() -> None:
    tags = FrameTags(
        objects=[
            TaggedObject(phrase="Stone Column", count=2, landmark=True),
            TaggedObject(phrase="stone column", count=4, landmark=True),
            TaggedObject(phrase="marble bust", count=1, landmark=True),
        ],
        scene_description="x",
    )
    out = tags.sam3_prompts()
    assert out == ["Stone Column", "marble bust"]  # first-seen wins, dup case-insens


def test_sam3_prompts_landmarks_only_filter() -> None:
    tags = _make_frame_tags()
    landmarks = tags.landmark_phrases()
    # Drops the "paper on table" non-landmark; keeps both "stone column"
    # mentions but dedups.
    assert "paper on table" not in landmarks
    assert "marble bust on plinth" in landmarks
    assert "stone column" in landmarks


def test_sam3_prompts_dedup_off_keeps_duplicates() -> None:
    tags = _make_frame_tags()
    raw = tags.sam3_prompts(dedup=False)
    assert raw.count("stone column") == 2


# ----- config hash ---------------------------------------------------------


def test_config_hash_changes_with_any_input() -> None:
    base = _config_hash(
        model="gpt-5.5",
        prompt_template_version=1,
        schema_version=1,
        high_fidelity=True,
    )
    assert base != _config_hash(
        model="gpt-5.5", prompt_template_version=1,
        schema_version=1, high_fidelity=False,
    )
    assert base != _config_hash(
        model="gpt-5.6", prompt_template_version=1,
        schema_version=1, high_fidelity=True,
    )
    assert base != _config_hash(
        model="gpt-5.5", prompt_template_version=2,
        schema_version=1, high_fidelity=True,
    )
    assert len(base) == 16


# ----- cache ---------------------------------------------------------------


def test_cache_hit_returns_stored_tags(tmp_path: Path) -> None:
    tagger = GPT5Tagger(cache_dir=tmp_path, model="gpt-5.5")
    entry = _CacheEntry(
        config_hash=tagger.config_hash,
        frame_tags=_make_frame_tags(),
        model="gpt-5.5",
    )
    tagger._cache_save(7, entry)
    # Don't initialize a client at all — pure cache hit.
    tags = tagger.tag_frame(image_path=Path("/nonexistent.jpg"), frame_id=7)
    assert tags.scene_description == entry.frame_tags.scene_description


def test_cache_miss_when_config_hash_changes(tmp_path: Path) -> None:
    tagger = GPT5Tagger(cache_dir=tmp_path, model="gpt-5.5")
    # Plant an entry with a different config hash.
    entry = _CacheEntry(
        config_hash="deadbeefdeadbeef",
        frame_tags=_make_frame_tags(),
        model="gpt-5.5",
    )
    tagger._cache_save(3, entry)
    # Should miss → would call the API, but we patch _call_api_sync.
    fake_response = _CacheEntry(
        config_hash=tagger.config_hash,
        frame_tags=FrameTags(objects=[], scene_description="empty"),
        model="gpt-5.5",
    )
    with patch.object(tagger, "_call_api_sync", return_value=fake_response) as mock_api:
        tags = tagger.tag_frame(image_path=_make_jpg(tmp_path), frame_id=3)
    mock_api.assert_called_once()
    assert tags.scene_description == "empty"


def test_cache_round_trip_via_disk(tmp_path: Path) -> None:
    tagger_a = GPT5Tagger(cache_dir=tmp_path, model="gpt-5.5")
    fake_response = _CacheEntry(
        config_hash=tagger_a.config_hash,
        frame_tags=_make_frame_tags(),
        model="gpt-5.5",
    )
    with patch.object(tagger_a, "_call_api_sync", return_value=fake_response):
        tagger_a.tag_frame(image_path=_make_jpg(tmp_path), frame_id=42)
    # New tagger instance — should hit disk cache and skip the API.
    tagger_b = GPT5Tagger(cache_dir=tmp_path, model="gpt-5.5")
    with patch.object(tagger_b, "_call_api_sync") as mock_api:
        tags = tagger_b.tag_frame(image_path=Path("/x.jpg"), frame_id=42)
    mock_api.assert_not_called()
    assert tags.scene_description == fake_response.frame_tags.scene_description


def test_cache_atomic_write_uses_tmp_then_rename(tmp_path: Path) -> None:
    """The cache save must go via a sibling .tmp file then os.replace, so a
    crash mid-write doesn't poison the cache file."""
    tagger = GPT5Tagger(cache_dir=tmp_path)
    seen: list[Path] = []

    real_replace = __import__("os").replace

    def _spy(src, dst):
        seen.append(Path(src))
        return real_replace(src, dst)

    fake = _CacheEntry(
        config_hash=tagger.config_hash,
        frame_tags=_make_frame_tags(),
        model="gpt-5.5",
    )
    with patch("cnsg.segmentation.gpt5_tagger.os.replace", side_effect=_spy):
        tagger._cache_save(99, fake)
    assert seen, "os.replace was never called — write wasn't atomic"
    assert seen[0].suffix == ".tmp"


def test_no_cache_dir_skips_disk(tmp_path: Path) -> None:
    tagger = GPT5Tagger(cache_dir=None)
    fake = _CacheEntry(
        config_hash=tagger.config_hash,
        frame_tags=_make_frame_tags(),
        model="gpt-5.5",
    )
    with patch.object(tagger, "_call_api_sync", return_value=fake):
        tags = tagger.tag_frame(image_path=_make_jpg(tmp_path), frame_id=0)
    # Subsequent calls should also hit the API since there's no cache.
    with patch.object(tagger, "_call_api_sync", return_value=fake) as mock_api:
        tagger.tag_frame(image_path=_make_jpg(tmp_path), frame_id=0)
    mock_api.assert_called_once()
    assert tags.scene_description == fake.frame_tags.scene_description


def test_corrupted_cache_falls_back_to_api(tmp_path: Path) -> None:
    """A garbled cache file must not crash the tagger; we re-call the API
    and overwrite. Loud [WARN] fires so it's noticed."""
    tagger = GPT5Tagger(cache_dir=tmp_path)
    cache_path = tagger._cache_path(11)
    cache_path.write_text("{ not valid json")  # type: ignore[union-attr]
    fake = _CacheEntry(
        config_hash=tagger.config_hash,
        frame_tags=_make_frame_tags(),
        model="gpt-5.5",
    )
    with patch.object(tagger, "_call_api_sync", return_value=fake) as mock_api:
        tagger.tag_frame(image_path=_make_jpg(tmp_path), frame_id=11)
    mock_api.assert_called_once()


# ----- input payload assembly ----------------------------------------------


def test_build_input_includes_base64_image_and_system_prompt(tmp_path: Path) -> None:
    tagger = GPT5Tagger(cache_dir=tmp_path, high_fidelity=True)
    img = _make_jpg(tmp_path)
    payload = tagger._build_input(img)
    assert payload[0]["role"] == "system"
    assert "ETH" in payload[0]["content"] or "navigable" in payload[0]["content"]
    user_content = payload[1]["content"]
    img_part = next(p for p in user_content if p["type"] == "input_image")
    assert img_part["image_url"].startswith("data:image/jpeg;base64,")
    assert img_part["detail"] == "high"


def test_build_input_low_fidelity_flag(tmp_path: Path) -> None:
    tagger = GPT5Tagger(cache_dir=tmp_path, high_fidelity=False)
    payload = tagger._build_input(_make_jpg(tmp_path))
    img_part = next(p for p in payload[1]["content"] if p["type"] == "input_image")
    assert img_part["detail"] == "low"


# ----- async batch ---------------------------------------------------------


def test_tag_frames_async_skips_failed_frames(tmp_path: Path) -> None:
    tagger = GPT5Tagger(cache_dir=tmp_path)
    fake_ok = _CacheEntry(
        config_hash=tagger.config_hash,
        frame_tags=_make_frame_tags(),
        model="gpt-5.5",
    )

    async def _fake_call(image_path):
        if "boom" in str(image_path):
            raise RuntimeError("boom")
        return fake_ok

    with patch.object(tagger, "_call_api_async", side_effect=_fake_call):
        items = [
            (1, tmp_path / "ok.jpg"),
            (2, tmp_path / "boom.jpg"),
            (3, tmp_path / "ok2.jpg"),
        ]
        for fid, p in items:
            p.write_bytes(b"\xff\xd8\xff\xd9")
        out = asyncio.run(tagger.tag_frames_async(items, max_concurrency=2))
    assert set(out) == {1, 3}  # frame 2 dropped, others survived
    assert out[1].scene_description == fake_ok.frame_tags.scene_description


# ----- versioning sentinel -------------------------------------------------


def test_versions_pin_for_cache_invalidation() -> None:
    """If you bump PROMPT_TEMPLATE_VERSION or SCHEMA_VERSION you also need
    to update this assertion — the dance forces you to remember that
    every existing on-disk cache invalidates."""
    assert PROMPT_TEMPLATE_VERSION == 2
    assert SCHEMA_VERSION == 1
