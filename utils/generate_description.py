#!/usr/bin/env python3
"""Generate a grounded path description from sequential JSON observations."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from dotenv import load_dotenv

load_dotenv()

# Structural labels do not help with navigation cues.
IGNORED_LABELS = {
    "ceiling",
    "floor",
    "wall",
    "walls",
    "ceiling trim",
    "wall trim",
    "railing",
}


@dataclass
class FrameSummary:
    name: str
    objects: Sequence[str]
    relationships: Sequence[str]

    def to_prompt_line(self) -> str:
        segments: List[str] = []
        if self.objects:
            segments.append("Key objects: " + ", ".join(self.objects))
        if self.relationships:
            segments.append("Relations: " + "; ".join(self.relationships))
        if not segments:
            segments.append("Limited landmarks visible.")
        return f"{self.name}: " + " | ".join(segments)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a grounded path description from observation JSON files."
    )
    default_input = Path(__file__).with_name("output")
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=default_input,
        help=f"Directory containing path JSON files (default: {default_input})",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        help="Destination text file for the generated description "
        "(default: input_dir/path_description.txt)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="ChatGPT model identifier to use for generation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=40,
        help="Limit the number of frames included in the prompt (default: 40)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and print the prompt without calling the ChatGPT API.",
    )
    return parser.parse_args()


def load_frames(input_dir: Path, max_frames: int) -> List[Dict[str, Any]]:
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    json_paths = sorted(path for path in input_dir.glob("*.json") if path.is_file())
    if not json_paths:
        raise SystemExit(f"No JSON files found in: {input_dir}")

    selected_paths = json_paths[:max_frames] if max_frames else json_paths

    frames: List[Dict[str, Any]] = []
    for path in selected_paths:
        try:
            frames.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Could not parse JSON file {path}: {exc}") from exc
    return frames


def distance_preference(distance: float | None) -> float:
    if distance is None:
        return 0.6
    if distance <= 0:
        return 0.3
    diff = abs(distance - 4.0)
    if diff <= 1.0:
        return 1.0
    if diff <= 2.0:
        return 0.75
    if diff <= 3.0:
        return 0.5
    return 0.3


def distance_bucket(distance: float | None) -> str | None:
    if distance is None or distance <= 0:
        return None
    if distance < 1:
        return "very close"
    if distance < 3:
        return "near"
    if distance <= 5:
        return "mid-range"
    if distance <= 6:
        return "slightly far"
    return "far"


def format_object_entry(obj: Dict[str, Any]) -> str | None:
    label = str(obj.get("label", "")).strip()
    if not label:
        return None
    if label.lower() in IGNORED_LABELS:
        return None

    percent = obj.get("pixel_percent")
    distance = obj.get("distance_from_camera")
    details: List[str] = []

    if isinstance(percent, (float, int)) and percent > 0:
        details.append(f"{percent:.2f}% area")
    if isinstance(distance, (float, int)) and distance > 0:
        distance_val = float(distance)
        details.append(f"{distance_val:.1f}m away")
        bucket = distance_bucket(distance_val)
        if bucket:
            details.append(bucket)

    return f"{label} ({', '.join(details)})" if details else label


def object_priority(obj: Dict[str, Any]) -> tuple[float, float, float, float]:
    percent = obj.get("pixel_percent")
    distance = obj.get("distance_from_camera")
    percent_val = float(percent) if isinstance(percent, (float, int)) else 0.0
    distance_val = float(distance) if isinstance(distance, (float, int)) else 0.0
    preference = distance_preference(distance_val if distance_val > 0 else None)
    scored_percent = percent_val * preference
    return (
        scored_percent,
        preference,
        percent_val,
        -distance_val if distance_val > 0 else 0.0,
    )


def extract_objects(visible_objects: Dict[str, Any], limit: int = 6) -> List[str]:
    candidates: List[Dict[str, Any]] = []
    for obj in visible_objects.values():
        label = str(obj.get("label", "")).lower()
        if label in IGNORED_LABELS:
            continue
        candidates.append(obj)

    candidates.sort(key=object_priority, reverse=True)

    results: List[str] = []
    for obj in candidates[:limit]:
        formatted = format_object_entry(obj)
        if formatted:
            results.append(formatted)
    return results


def extract_relationships(
    relationships: Iterable[Dict[str, Any]], limit: int = 6
) -> List[str]:
    filtered: List[str] = []
    for item in relationships:
        subject = str(item.get("subject", "")).strip()
        relation = str(item.get("relation", "")).strip()
        obj = str(item.get("object", "")).strip()
        if not subject or not relation or not obj:
            continue
        if subject.lower() in IGNORED_LABELS or obj.lower() in IGNORED_LABELS:
            continue
        distance = item.get("distance_m")
        suffix = ""
        if isinstance(distance, (float, int)) and distance > 0:
            suffix = f" ({distance:.1f}m)"
        filtered.append(f"{subject} {relation} {obj}{suffix}")
        if len(filtered) >= limit:
            break
    return filtered


def summarise_frames(frames: Sequence[Dict[str, Any]]) -> List[FrameSummary]:
    summaries: List[FrameSummary] = []
    for frame in frames:
        name = str(frame.get("image_index") or frame.get("frame", "frame")).strip()
        object_data = frame.get("objects") or frame.get("visible_objects") or {}
        relation_data = (
            frame.get("spatial_relations") or frame.get("relationships") or []
        )
        objects = extract_objects(object_data)
        relationships = extract_relationships(relation_data)
        summaries.append(
            FrameSummary(
                name=name,
                objects=objects,
                relationships=relationships,
            )
        )
    return summaries


def build_prompt(scene_index: str | None, summaries: Sequence[FrameSummary]) -> str:
    intro_lines = [
        "You are a navigation assistant helping someone retrace a short walk through a home.",
        "You will see a sequence of snapshots with visible objects and spatial relationships.",
        "Write a single natural, human-sounding description of the path — clear, fluent, and easy to imagine.",
        "Describe what a person would perceive or notice while moving along the route.",
        "Mention only objects and landmarks that appear in the observations, but describe them naturally (e.g. 'you pass a couch on your right' instead of listing data).",
        "Focus on spatial flow and transitions between views — use simple phrases that sound like real directions.",
        "Do not include numbers, measurements, or technical details — just a smooth narrative under 180 words.",
    ]

    header = "\n".join(intro_lines)
    scene_line = (
        f"Scene index: {scene_index}" if scene_index else "Scene index: unknown"
    )
    observation_lines = "\n".join(
        f"- {summary.to_prompt_line()}" for summary in summaries
    )

    return textwrap.dedent(
        f"""
        {header}

        {scene_line}
        Observations:
        {observation_lines}
        """
    ).strip()


def generate_description(prompt: str, model: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY environment variable is not set.")

    try:
        import openai  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "The openai package is required. Install it via 'pip install openai'."
        ) from exc

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise navigation assistant. Only reference landmarks that appear "
                "in the observations. Avoid embellishments or invented objects."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    kwargs = {"model": model, "temperature": 0.2, "max_tokens": 400}

    try:
        if hasattr(openai, "OpenAI"):
            client = openai.OpenAI(api_key=api_key)
            print("[INFO] Calling ChatGPT with the following messages: ", messages)
            response = client.chat.completions.create(messages=messages, **kwargs)
            return response.choices[0].message.content.strip()

        openai.api_key = api_key
        response = openai.ChatCompletion.create(messages=messages, **kwargs)
        print("[INFO] Calling ChatGPT with the following messages: ", messages)
        return response.choices[0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"ChatGPT generation failed: {exc}") from exc


def write_output(output_path: Path, description: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(description + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    frames = load_frames(args.input_dir, args.max_frames)
    scene_index = frames[0].get("scene_index")
    summaries = summarise_frames(frames)
    prompt = build_prompt(scene_index, summaries)

    if args.dry_run:
        print(prompt)
        return

    description = generate_description(prompt, args.model)
    output_path = args.output_path or (args.input_dir / "path_description.txt")
    write_output(output_path, description)
    print(f"Wrote path description to {output_path}")


# * Used as API
def generate_path_description(
    input_dir: Path,
    model: str = "gpt-4o-mini",
    max_frames: int = 40,
) -> str:
    """
    Full pipeline: loads frames, builds prompt, queries the model, and returns description.
    Does NOT write anything to disk.
    """
    frames = load_frames(input_dir, max_frames)
    scene_index = frames[0].get("scene_index")
    summaries = summarise_frames(frames)
    prompt = build_prompt(scene_index, summaries)
    return generate_description(prompt, model)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user.")
