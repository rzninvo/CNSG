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

RELEVANCE_SCORES_OBJECTS = {
    "door": 0.8,
    "chandelier": 0.8,
    "wardrobe": 0.7,
    "tv": 0.8,
    "cabinet": 0.7,
    "blanket": 0.2,
    "pad": 0.2,
    "bed": 0.6,
    "pillow": 0.5,
    "nightstand": 0.8,
    "book": 0.7,
    "table lamp": 0.6,
    "plush toy": 0.8,
    "window": 0.7,
    "armchair": 0.8,
    "mat": 0.2,
    "towel": 0.5,
    "bucket": 0.2,
    "tap": 0.2,
    "hand soap": 0.2,
    "toilet": 0.8,
    "toilet brush": 0.4,
    "lamp": 0.3,
    "shower curtain": 0.2,
    "curtain": 0.5,
    "photo": 0.7,
    "toy": 0.8,
    "board": 0.6,
    "ventilation": 0.2,
    "attic door": 0.2,
    "light": 0.3,
    "vent": 0.3,
    "bicycle": 0.9,
    "box": 0.6,
    "couch": 0.7,
    "basket": 0.6,
    "magazine": 0.8,
    "stack of papers": 0.4,
    "picture": 0.7,
    "folder": 0.7,
    "table": 0.9,
    "chair": 0.9,
    "handbag": 0.5,
    "pc tower": 0.7,
    "trashcan": 0.7,
    "computer desk": 1.0,
    "printer": 0.8,
    "telephone": 0.8,
    "desk lamp": 0.5,
    "plant": 0.7,
    "shirt": 0.3,
    "bag": 0.4,
    "newspaper": 0.6,
    "balustrade": 0.7,
    "stairs": 1.0,
    "window curtain": 0.6,
    "curtain rod": 0.2,
    "speaker": 0.5,
    "led tv": 1.0,
    "fireplace": 1.0,
    "flower": 0.7,
    "decorative plate": 0.8,
    "floor mat": 0.3,
    "pillar": 0.3,
    "fire alarm": 0.2,
    "alarm control": 0.3,
    "ceiling vent": 0.2,
    "wall clock": 0.8,
    "flag": 0.8,
    "kitchen appliance": 0.7,
    "coffee mug": 0.6,
    "worktop": 0.5,
    "sink": 0.7,
    "knife holder": 0.5,
    "microwave": 0.7,
    "kitchen countertop item": 0.3,
    "oven and stove": 0.8,
    "fruit bowl": 0.4,
    "dishwasher": 0.8,
    "bath sink": 0.7,
    "toilet paper": 0.5,
    "toilet seat": 0.6,
    "door handle": 0.4,
    "bathroom shelf": 0.3,
    "doormat": 0.2,
    "ventilation hood": 0.2,
    "dresser": 0.6,
    "casket": 0.1,
    "wall hanging decoration": 0.3,
    "laundry basket": 0.6,
    "electric box": 0.2,
    "electrical controller": 0.2,
    "tissue box": 0.2,
    "shower dial": 0.3,
    "bath": 0.8,
    "bathroom cabinet": 0.4,
    "bathroom accessory": 0.2,
    "mirror": 0.9,
    "soap bottle": 0.3,
    "mirror frame": 0.3,
    "wall lamp": 0.3,
    "shoe": 0.5,
    "iron board": 0.7,
    "iron": 0.8,
    "clothes": 0.5,
    "clothes hanger rod": 0.2,
    "case": 0.6,
    "storage box": 0.3,
    "briefcase": 0.6,
    "backpack": 0.6,
    "boxes": 0.4,
    "kitchen shelf": 0.7,
    "bottle of soap": 0.2,
}


@dataclass
class FrameSummary:
    name: str
    objects: Sequence[str]
    relationships: Sequence[str]

    def to_prompt_line(self) -> str:
        if not self.objects:
            return f"{self.name}: Limited visibility in this frame."

        object_part = ", ".join(self.objects[:2])
        rel_part = "; ".join(self.relationships[:2]) if self.relationships else ""

        description = f"In {self.name}, you see {object_part}"
        if rel_part:
            description += f". You also notice that {rel_part}"
        description += "."

        return description


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


def describe_relative_position(cam_coords):
    x, y, z = cam_coords
    horiz = "right" if x > 0.5 else "left" if x < -0.5 else "ahead"
    return horiz


def format_object_entry(obj: Dict[str, Any]) -> str | None:
    label = str(obj.get("label", "")).strip()
    if not label or label.lower() in IGNORED_LABELS:
        return None

    # View-based positioning
    direction = None
    # View-based positioning from centroid_cam (x, y)
    centroid = obj.get("centroid_cam")
    if isinstance(centroid, list) and len(centroid) >= 2:
        x, y = centroid[0], centroid[1]
        if y > 0.3:
            vert = "upper"
        elif y < -0.3:
            vert = "lower"
        else:
            vert = ""

        if x < -0.3:
            horiz = "left"
        elif x > 0.3:
            horiz = "right"
        else:
            horiz = "center"

        if vert and horiz != "center":
            direction = f"{vert}-{horiz}"
        elif vert:
            direction = vert
        elif horiz:
            direction = horiz
        else:
            direction = "center"

        return f"{label} ({direction})"


def object_priority(obj: Dict[str, Any]) -> tuple[float, float, float, float]:
    label = str(obj.get("label", "")).strip().lower()
    percent = obj.get("pixel_percent")
    distance = obj.get("distance_from_camera")

    percent_val = float(percent) if isinstance(percent, (float, int)) else 0.0
    distance_val = float(distance) if isinstance(distance, (float, int)) else 0.0

    preference = distance_preference(distance_val if distance_val > 0 else None)
    relevance = RELEVANCE_SCORES_OBJECTS.get(label, 0.2)  # default relevance if missing

    scored_percent = percent_val * preference * relevance
    return (
        scored_percent,
        relevance,
        percent_val,
        -distance_val if distance_val > 0 else 0.0,
    )


def extract_objects(visible_objects: Dict[str, Any], limit: int = 3) -> List[str]:
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
    relationships: Iterable[Dict[str, Any]], limit: int = 2
) -> List[str]:
    def natural_direction(relation: str) -> str:
        # Converts relation labels to natural expressions
        table = {
            "left_of": "to the left of",
            "right_of": "to the right of",
            "in_front_of": "in front of",
            "behind": "behind",
            "beneath_of": "below",
            "on_top_of": "on top of",
            "above": "above",
            "next_to": "next to",
        }
        return table.get(relation, relation.replace("_", " "))

    filtered: List[str] = []
    for item in relationships:
        subj = str(item.get("subject", "")).strip()
        rel = str(item.get("relation", "")).strip()
        obj = str(item.get("object", "")).strip()
        if not subj or not rel or not obj:
            continue
        if subj.lower() in IGNORED_LABELS or obj.lower() in IGNORED_LABELS:
            continue
        natural_rel = natural_direction(rel)
        filtered.append(f"{subj} is {natural_rel} {obj}")
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


def build_prompt(
    scene_index: str | None, summaries: Sequence[FrameSummary], user_input: str
) -> str:
    intro_lines = [
        "You are a navigation assistant helping someone retrace a short walk through a home.",
        "You will see a sequence of snapshots with visible objects and spatial relationships.",
        "Use the spatial layout of the objects to describe the path clearly and naturally.",
        "Write a single human-sounding description of the walk — fluent and easy to follow.",
        "You can mention positions like left/right, in front of, next to, behind.",
        "Prefer common, easily recognized landmarks like large furniture, doors, appliances, and windows. Avoid small, decorative, or rarely used objects like mats, lamps, or soap bottles.",
        "Avoid numeric measurements or technical descriptions. Focus on intuitive guidance under 120 words.",
        f"User question: {user_input}",
    ]

    header = "\n".join(intro_lines)
    scene_line = (
        f"Scene index: {scene_index}" if scene_index else "Scene index: unknown"
    )
    observation_lines = "\n".join(summary.to_prompt_line() for summary in summaries)

    return textwrap.dedent(
        f"""
        {header}

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

    kwargs = {"model": model, "temperature": 0.6, "max_tokens": 400}

    try:
        if hasattr(openai, "OpenAI"):
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(messages=messages, **kwargs)
            return response.choices[0].message.content.strip()

        openai.api_key = api_key
        response = openai.ChatCompletion.create(messages=messages, **kwargs)
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
    prompt = build_prompt(scene_index, summaries, user_input="Where is the kitchen?")

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
    user_input: str,
    model: str = "gpt-4o-mini",
    max_frames: int = 40,
    dry_run: bool = False,
) -> str:
    """
    Full pipeline: loads frames, builds prompt, optionally queries the model, and returns description or prompt.
    Does NOT write anything to disk.
    """
    frames = load_frames(input_dir, max_frames)
    scene_index = frames[0].get("scene_index")
    summaries = summarise_frames(frames)
    prompt = build_prompt(scene_index, summaries, user_input)

    if dry_run:
        return prompt

    return generate_description(prompt, model)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user.")
