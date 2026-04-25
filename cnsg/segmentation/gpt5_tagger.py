"""Per-frame open-vocab object tagger using OpenAI GPT-5.5 vision.

Replaces the hand-curated 33-prompt SAM 3 list (`taxonomy.OBJECT_PROMPTS`)
with a per-frame VLM call that returns the actual landmark phrases the
frame contains. The output is cached by frame content + tagger config so
re-runs of the build cost $0 after the first pass.

## Why this exists

Two bottlenecks in the existing pipeline that this targets:

1. The hand-curated prompt list misses heritage classes. A row of fluted
   columns in HG E might be detected as "column" but a marble bust on a
   plinth, an arched cartouche, an ornate stair railing or a lecture-hall
   door go unprompted — SAM 3 never sees them so the lifter never lifts
   them. GPT-5.5 names what it actually sees, so SAM 3 gets per-frame
   prompts tuned to the frame's contents.

2. The downstream class voter resolves clusters through S3DIS-13
   (`taxonomy.S3DIS_CLASSES`), which has no bucket for "marble bust",
   "stone fountain", "plinth", "balustrade", or "lecture-hall door" — so
   real landmarks fall into `clutter`. The cluster voter (separate
   module) consumes this tagger's output instead, and emits open-vocab
   labels that survive into the Habitat bundle.

## Cache design

`{cache_dir}/frame_{frame_id:06d}.gpt5.json` holds the structured output
plus a config hash. The hash keys on (model, prompt_template_version,
schema_version, fidelity) — change any of those and the cache invalidates
automatically. Costs ≈ $0.04/frame for a 1920×1280 high-fidelity call;
2,408-frame full HGE build is ~$96 once, then $0 on every re-run.

## Runtime env

`cnsg-seg`, with `openai>=1` + `python-dotenv` + `pydantic>=2` installed
(see scripts/install.sh / requirements.txt for the canonical pin).

## Usage

    from cnsg.segmentation.gpt5_tagger import GPT5Tagger

    tagger = GPT5Tagger(cache_dir=Path("data/maps/hge/gpt5_cache"))
    tags = tagger.tag_frame(image_path=Path("frame_000000.jpg"), frame_id=0)
    sam3_prompts = tags.sam3_prompts()  # list[str], dedup'd
    landmarks = tags.landmark_phrases()  # subset where landmark=True

CLI for one-off testing on a single frame:

    python -m cnsg.segmentation.gpt5_tagger \\
        --frame mesh_pipeline/data/navvis_.../00000-cam0__center.jpg \\
        --frame-id 0 \\
        --cache-dir data/maps/hge/gpt5_cache
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Schema (pydantic) — mirrored on the OpenAI structured-output side
# ---------------------------------------------------------------------------

# Bumping these invalidates the on-disk cache automatically (factored into
# the config hash). Keep them monotonic — never reuse a number.
#
# v1 (2026-04-25): initial. Smoke test on HGE frame 0 (vaulted corridor with
#                  arches + columns + staircase) returned only
#                  `wide stone staircase` — the columns/arches got captured
#                  in scene_description but were dropped from objects[]
#                  because "skip structural mass" was read too aggressively.
# v2 (2026-04-25): explicitly include columns / pillars / arches / archways /
#                  busts / fountains / staircases / handrails / decorative
#                  architectural elements as INSTANCES; only skip the
#                  boundary surfaces (wall paint, floor tile, ceiling).
PROMPT_TEMPLATE_VERSION = 2
SCHEMA_VERSION = 1


class TaggedObject(BaseModel):
    """One distinct foreground object the VLM saw in the frame."""

    phrase: str = Field(
        ...,
        description=(
            "Specific noun phrase suitable as a SAM 3 text prompt — "
            "e.g. 'marble bust on plinth', 'stone fountain', "
            "'wooden lecture-hall door'. NOT generic terms like 'object'."
        ),
    )
    count: int = Field(
        ...,
        ge=1,
        description="Number of distinct instances of this object visible in the frame.",
    )
    landmark: bool = Field(
        ...,
        description=(
            "True if a person giving navigation directions inside the "
            "building would naturally reference this object "
            "('walk past the marble bust'). False for incidental clutter."
        ),
    )


class FrameTags(BaseModel):
    """Structured output schema for one VLM call on one NavVis frame."""

    objects: list[TaggedObject] = Field(
        default_factory=list,
        description="List of distinct foreground objects visible.",
    )
    scene_description: str = Field(
        ...,
        description="One-sentence summary of what the frame shows.",
    )

    # ---- consumer-side helpers -------------------------------------------

    def sam3_prompts(
        self,
        *,
        landmarks_only: bool = False,
        dedup: bool = True,
    ) -> list[str]:
        """Phrases ready to feed straight into SAM 3 as text prompts.

        Args:
            landmarks_only: when True, drop incidental objects.
            dedup: case-insensitive de-duplication, preserving first-seen order.
        """
        phrases = [
            o.phrase.strip() for o in self.objects
            if (not landmarks_only) or o.landmark
        ]
        if dedup:
            seen: set[str] = set()
            out: list[str] = []
            for p in phrases:
                key = p.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(p)
            return out
        return phrases

    def landmark_phrases(self) -> list[str]:
        """Shortcut: just the landmark-flagged phrases."""
        return self.sam3_prompts(landmarks_only=True)


# ---------------------------------------------------------------------------
# Prompt + schema for the OpenAI Responses API
# ---------------------------------------------------------------------------

# Calibrated against HG E specifically — a NavVis VLX scan of ETH Zurich's
# Hauptgebäude HG E floor (Beaux-Arts heritage). Generic "list objects"
# prompts return clutter-heavy garbage; this prompt forces the VLM toward
# navigation-relevant landmarks at heritage-vocabulary precision.
SYSTEM_PROMPT = """\
You identify navigable landmarks in interior architectural photographs for an indoor navigation system.

The image is a frame from a NavVis VLX 360 scan of ETH Zurich's Hauptgebäude HG E floor — a Beaux-Arts heritage building with arched corridors, stone columns and pillars, marble busts, ornate fountains, lecture halls, ceremonial staircases, balustrades, archways, and information boards.

Your output feeds an open-vocabulary segmentation model (SAM 3) and a downstream LLM that gives a person walking through the building step-by-step directions.

WHAT TO INCLUDE (each as a separate instance, with a count when there are several):
- Architectural landmarks: COLUMNS, PILLARS, ARCHES, ARCHWAYS, ARCHED DOORWAYS, BALUSTRADES, BANISTERS, HANDRAILS, COLUMN CAPITALS, CORNICES, MEDALLIONS, friezes, plinths.
- Heritage decoration: marble or stone BUSTS, STATUES, FOUNTAINS, plaques, cartouches, decorative wall medallions, display cases.
- Navigation landmarks: DOORS (lecture-hall doors, glass doors, arched doors), STAIRCASES, STAIRCASE FLIGHTS, ELEVATORS / lifts, INFORMATION BOARDS, NOTICE BOARDS, signs, exit signs.
- Furniture and fixtures: chairs, tables, benches, bookcases, planters, fire extinguishers, vending machines, water fountains.
- For repeated similar instances (e.g. a row of 5 columns or 3 busts), give the count of distinct visible instances. DO count repeated columns or busts even when they form a row.

WHAT TO SKIP:
- Boundary SURFACES of the building: wall paint, floor tile, ceiling plaster, vaulted ceiling shell. ("Vaulted ceiling" is a surface; skip. The "stone columns supporting the vault" are individual landmarks; INCLUDE them.)
- Tiny objects (<30 cm) or barely visible / heavily blurred objects.

PHRASE STYLE:
- Specific, descriptive noun phrases ("marble bust on plinth", "fluted stone column", "wooden lecture-hall door", "ornate stone fountain", "wrought-iron handrail") — NOT generic terms ("object", "thing", "feature", "structure").
- The phrase must be promptable to an open-vocab segmentation model.

LANDMARK FLAG:
- `landmark=true` for objects a person would naturally reference when navigating ("walk past the marble bust", "turn at the columns", "the lecture-hall door is on the left").
- `landmark=false` for incidental clutter (paper, cables, small books on a table, utility pipes, signage that's purely informational and not a wayfinding cue).

EDGE CASES:
- If a frame is clearly a featureless wall or floor close-up with no landmarks, return an empty `objects` list and a brief `scene_description`.
- Outdoor frames (the scan briefly exits the building): tag visible large outdoor landmarks (trees, lampposts, building facades) the same way.

Return only the JSON object that conforms to the provided schema.
"""


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------


def _config_hash(
    *,
    model: str,
    prompt_template_version: int,
    schema_version: int,
    high_fidelity: bool,
) -> str:
    """Stable 16-char hash for cache invalidation when tagger config changes."""
    raw = json.dumps(
        {
            "model": model,
            "prompt_template_version": prompt_template_version,
            "schema_version": schema_version,
            "high_fidelity": bool(high_fidelity),
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class _CacheEntry:
    """Wire format for the on-disk cache."""

    config_hash: str
    frame_tags: FrameTags
    raw_response_id: Optional[str] = None
    elapsed_s: Optional[float] = None
    model: Optional[str] = None
    timestamp: Optional[float] = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "config_hash": self.config_hash,
                "frame_tags": self.frame_tags.model_dump(),
                "raw_response_id": self.raw_response_id,
                "elapsed_s": self.elapsed_s,
                "model": self.model,
                "timestamp": self.timestamp,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "_CacheEntry":
        d = json.loads(text)
        return cls(
            config_hash=d["config_hash"],
            frame_tags=FrameTags.model_validate(d["frame_tags"]),
            raw_response_id=d.get("raw_response_id"),
            elapsed_s=d.get("elapsed_s"),
            model=d.get("model"),
            timestamp=d.get("timestamp"),
        )


# ---------------------------------------------------------------------------
# Tagger
# ---------------------------------------------------------------------------


class GPT5Tagger:
    """Per-frame open-vocab tagger backed by OpenAI's Responses API.

    One instance per pipeline run; reuses the OpenAI client and the
    structured-output schema across calls. Cache is content-addressed
    by `(model, prompt_template_version, schema_version, fidelity)` so
    iterating the prompt invalidates automatically.

    Args:
        cache_dir: where to write `frame_{id:06d}.gpt5.json`. Created
            if missing. Set to `None` to bypass caching (useful for
            one-off testing).
        model: OpenAI model id. Default `gpt-5.5` (current flagship,
            April 2026). Pin a snapshot like `gpt-5.5-2026-04-23` for
            reproducibility on long-running builds.
        high_fidelity: when True, sends the image at high fidelity
            (~6,240 image tokens for 1920×1280, $0.04/frame at GPT-5.5
            pricing). When False, low fidelity (~65 + 129 per tile,
            ~$0.001/frame) — use for quick smoke tests only.
        api_key: OpenAI API key. Falls through to `OPENAI_API_KEY`
            env var (loaded from `.env` via python-dotenv if present)
            when None. Raises at first call site if neither is set.
        max_retries / retry_backoff_s: applied per-frame on transient
            failures (network, rate-limit). Frame-level retries; build-
            level concurrency is the caller's job.
    """

    DEFAULT_MODEL = "gpt-5.5"

    def __init__(
        self,
        *,
        cache_dir: Optional[Path] = None,
        model: str = DEFAULT_MODEL,
        high_fidelity: bool = True,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_backoff_s: float = 5.0,
        prompt_template_version: int = PROMPT_TEMPLATE_VERSION,
        schema_version: int = SCHEMA_VERSION,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.high_fidelity = bool(high_fidelity)
        self.max_retries = int(max_retries)
        self.retry_backoff_s = float(retry_backoff_s)
        self.prompt_template_version = int(prompt_template_version)
        self.schema_version = int(schema_version)
        self._api_key = api_key
        self._client = None  # lazy
        self._async_client = None  # lazy

        self.config_hash = _config_hash(
            model=self.model,
            prompt_template_version=self.prompt_template_version,
            schema_version=self.schema_version,
            high_fidelity=self.high_fidelity,
        )

    # ------ public sync API -----------------------------------------------

    def tag_frame(self, *, image_path: Path, frame_id: int) -> FrameTags:
        """Return the cached or freshly-computed tags for one frame.

        Cache hit: round-trip is a single JSON read, ~1 ms.
        Cache miss: one Responses API call, 2-6 s wall.
        """
        cached = self._cache_load(frame_id)
        if cached is not None:
            return cached

        result = self._call_api_sync(image_path)
        self._cache_save(frame_id, result)
        return result.frame_tags

    # ------ public async API ---------------------------------------------

    async def tag_frame_async(
        self, *, image_path: Path, frame_id: int
    ) -> FrameTags:
        cached = self._cache_load(frame_id)
        if cached is not None:
            return cached
        result = await self._call_api_async(image_path)
        self._cache_save(frame_id, result)
        return result.frame_tags

    async def tag_frames_async(
        self,
        items: list[tuple[int, Path]],
        *,
        max_concurrency: int = 16,
    ) -> dict[int, FrameTags]:
        """Tag many frames concurrently with a bounded semaphore.

        Returns a `{frame_id: FrameTags}` dict. Per-frame failures are
        wrapped in a [WARN] log and excluded from the result rather than
        crashing the batch — caller checks for missing keys.
        """
        sem = asyncio.Semaphore(max_concurrency)

        async def _one(fid: int, p: Path):
            async with sem:
                try:
                    tags = await self.tag_frame_async(image_path=p, frame_id=fid)
                    return fid, tags
                except Exception as e:  # noqa: BLE001 — broad on purpose
                    print(
                        f"[WARN] gpt5_tagger frame {fid}: expected=tags, "
                        f"got={type(e).__name__}: {e}, fallback=skip-frame",
                        flush=True,
                    )
                    return fid, None

        results = await asyncio.gather(*(_one(fid, p) for fid, p in items))
        return {fid: tags for fid, tags in results if tags is not None}

    # ------ cache helpers --------------------------------------------------

    def _cache_path(self, frame_id: int) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"frame_{frame_id:06d}.gpt5.json"

    def _cache_load(self, frame_id: int) -> Optional[FrameTags]:
        path = self._cache_path(frame_id)
        if path is None or not path.exists():
            return None
        try:
            entry = _CacheEntry.from_json(path.read_text())
        except Exception as e:  # noqa: BLE001
            print(
                f"[WARN] gpt5_tagger cache read frame {frame_id}: "
                f"expected=valid_json, got={type(e).__name__}: {e}, "
                f"fallback=re-call-api",
                flush=True,
            )
            return None
        if entry.config_hash != self.config_hash:
            # Stale cache from a different config — treat as miss; the
            # next save will overwrite. We don't delete here since the
            # caller might want to inspect the old tags.
            return None
        return entry.frame_tags

    def _cache_save(self, frame_id: int, entry: _CacheEntry) -> None:
        path = self._cache_path(frame_id)
        if path is None:
            return
        # Atomic write so a Ctrl-C mid-write doesn't poison the cache.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(entry.to_json())
        os.replace(tmp, path)

    # ------ API call (sync wraps async) -----------------------------------

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        # Best-effort .env load so a project-root .env populates OPENAI_API_KEY
        # without the caller having to do it. If python-dotenv isn't installed
        # we just rely on the existing process env.
        try:
            from dotenv import load_dotenv  # noqa: WPS433 (intentional local import)

            load_dotenv()
        except ImportError:
            pass
        from openai import OpenAI  # noqa: WPS433

        api_key = self._api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "[FATAL] gpt5_tagger: no OPENAI_API_KEY in env. "
                "Set it in a .env file at the project root or export it "
                "before running."
            )
        self._client = OpenAI(api_key=api_key)
        return self._client

    def _ensure_async_client(self):
        if self._async_client is not None:
            return self._async_client
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        from openai import AsyncOpenAI

        api_key = self._api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "[FATAL] gpt5_tagger: no OPENAI_API_KEY in env."
            )
        self._async_client = AsyncOpenAI(api_key=api_key)
        return self._async_client

    def _build_input(self, image_path: Path) -> list[dict]:
        """Construct the Responses-API `input` payload for one frame.

        Image is sent as a base64 data URL — works for any local file,
        no need for the OpenAI Files API (which would add a round-trip).
        """
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        # Pick MIME from extension; default to JPEG (NavVis frames are .jpg).
        suffix = Path(image_path).suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(
            suffix, "jpeg"
        )
        detail = "high" if self.high_fidelity else "low"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/{mime};base64,{b64}",
                        "detail": detail,
                    },
                ],
            },
        ]

    def _call_api_sync(self, image_path: Path) -> _CacheEntry:
        client = self._ensure_client()
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            try:
                t0 = time.time()
                response = client.responses.parse(
                    model=self.model,
                    input=self._build_input(image_path),
                    text_format=FrameTags,
                )
                elapsed = time.time() - t0
                parsed = response.output_parsed
                if parsed is None:
                    raise RuntimeError(
                        "[WARN] gpt5_tagger: response.output_parsed is None — "
                        "model returned a non-conformant payload"
                    )
                return _CacheEntry(
                    config_hash=self.config_hash,
                    frame_tags=parsed,
                    raw_response_id=getattr(response, "id", None),
                    elapsed_s=round(elapsed, 3),
                    model=self.model,
                    timestamp=time.time(),
                )
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if attempt < self.max_retries:
                    delay = self.retry_backoff_s * (2 ** attempt)
                    print(
                        f"[WARN] gpt5_tagger attempt {attempt+1}/{self.max_retries+1}: "
                        f"expected=parsed-FrameTags, got={type(e).__name__}: {e}, "
                        f"fallback=retry-after-{delay:.1f}s",
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
                break
        raise RuntimeError(
            f"[FATAL] gpt5_tagger: {self.max_retries+1} attempts failed for "
            f"{image_path}; last error: {last_exc}"
        )

    async def _call_api_async(self, image_path: Path) -> _CacheEntry:
        client = self._ensure_async_client()
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            try:
                t0 = time.time()
                response = await client.responses.parse(
                    model=self.model,
                    input=self._build_input(image_path),
                    text_format=FrameTags,
                )
                elapsed = time.time() - t0
                parsed = response.output_parsed
                if parsed is None:
                    raise RuntimeError(
                        "[WARN] gpt5_tagger: response.output_parsed is None"
                    )
                return _CacheEntry(
                    config_hash=self.config_hash,
                    frame_tags=parsed,
                    raw_response_id=getattr(response, "id", None),
                    elapsed_s=round(elapsed, 3),
                    model=self.model,
                    timestamp=time.time(),
                )
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if attempt < self.max_retries:
                    delay = self.retry_backoff_s * (2 ** attempt)
                    print(
                        f"[WARN] gpt5_tagger async attempt {attempt+1}: "
                        f"expected=parsed-FrameTags, got={type(e).__name__}: {e}, "
                        f"fallback=retry-after-{delay:.1f}s",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
                    continue
                break
        raise RuntimeError(
            f"[FATAL] gpt5_tagger async: {self.max_retries+1} attempts failed "
            f"for {image_path}; last error: {last_exc}"
        )


# ---------------------------------------------------------------------------
# CLI for one-off testing
# ---------------------------------------------------------------------------


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--frame", type=Path, required=True, help="path to a JPG/PNG frame")
    p.add_argument("--frame-id", type=int, default=0, help="frame id for cache key")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="cache directory; omit for no caching",
    )
    p.add_argument("--model", type=str, default=GPT5Tagger.DEFAULT_MODEL)
    p.add_argument(
        "--low-fidelity",
        action="store_true",
        help="use low-fidelity image tokens (cheap smoke test)",
    )
    p.add_argument(
        "--landmarks-only",
        action="store_true",
        help="print only the landmark-flagged phrases",
    )
    args = p.parse_args()

    tagger = GPT5Tagger(
        cache_dir=args.cache_dir,
        model=args.model,
        high_fidelity=not args.low_fidelity,
    )
    t0 = time.time()
    tags = tagger.tag_frame(image_path=args.frame, frame_id=args.frame_id)
    elapsed = time.time() - t0

    print(f"\n# Scene: {tags.scene_description}")
    print(f"# {len(tags.objects)} object(s) in frame, {elapsed:.1f}s wall\n")
    print(f"{'phrase':<45} {'count':>5}  {'landmark':>9}")
    print("-" * 65)
    for o in tags.objects:
        print(f"{o.phrase:<45} {o.count:>5}  {str(o.landmark):>9}")
    print("\nSAM 3 prompts (deduped):")
    for p in tags.sam3_prompts(landmarks_only=args.landmarks_only):
        print(f"  - {p}")


if __name__ == "__main__":
    _main()
