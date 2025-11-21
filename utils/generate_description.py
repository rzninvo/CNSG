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
import torch

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
    "chandelier": 0,
    "wardrobe": 0.7,
    "tv": 0.8,
    "cabinet": 0.6,
    "blanket": 0,
    "pad": 0,
    "bed": 0.6,
    "pillow": 0,
    "nightstand": 0.6,
    "book": 0,
    "table lamp": 0.4,
    "plush toy": 0,
    "window": 0.3,
    "armchair": 0.8,
    "mat": 0,
    "towel": 0,
    "bucket": 0,
    "tap": 0,
    "hand soap": 0,
    "toilet": 0.7,
    "toilet brush": 0,
    "lamp": 0.3,
    "shower curtain": 0,
    "curtain": 0,
    "photo": 0,
    "toy": 0,
    "board": 0,
    "ventilation": 0,
    "attic door": 0,
    "light": 0,
    "vent": 0,
    "bicycle": 0.6,
    "box": 0,
    "couch": 0.7,
    "basket": 0,
    "magazine": 0,
    "stack of papers": 0,
    "picture": 0.2,
    "folder": 0,
    "table": 0.9,
    "chair": 0.8,
    "handbag": 0,
    "pc tower": 0,
    "trashcan": 0,
    "computer desk": 0.7,
    "printer": 0.5,
    "telephone": 0.5,
    "desk lamp": 0.5,
    "plant": 0,
    "shirt": 0,
    "bag": 0,
    "newspaper": 0,
    "balustrade": 0,
    "stairs": 0.8,
    "window curtain": 0,
    "curtain rod": 0,
    "speaker": 0,
    "led tv": 0.8,
    "fireplace": 0.8,
    "flower": 0,
    "decorative plate": 0,
    "floor mat": 0,
    "pillar": 0,
    "fire alarm": 0,
    "alarm control": 0,
    "ceiling vent": 0,
    "wall clock": 0.5,
    "flag": 0.4,
    "kitchen appliance": 0,
    "coffee mug": 0,
    "worktop": 0.5,
    "sink": 0.7,
    "knife holder": 0,
    "microwave": 0.7,
    "kitchen countertop item": 0,
    "oven and stove": 0.8,
    "fruit bowl": 0,
    "dishwasher": 0.8,
    "bath sink": 0.7,
    "toilet paper": 0,
    "toilet seat": 0,
    "door handle": 0,
    "bathroom shelf": 0.3,
    "doormat": 0,
    "ventilation hood": 0,
    "dresser": 0,
    "casket": 0,
    "wall hanging decoration": 0,
    "laundry basket": 0,
    "electric box": 0,
    "electrical controller": 0,
    "tissue box": 0,
    "shower dial": 0,
    "bath": 0.8,
    "bathroom cabinet": 0.5,
    "bathroom accessory": 0,
    "mirror": 0.8,
    "soap bottle": 0,
    "mirror frame": 0.2,
    "wall lamp": 0.5,
    "shoe": 0,
    "iron board": 0,
    "iron": 0,
    "clothes": 0,
    "clothes hanger rod": 0,
    "case": 0,
    "storage box": 0,
    "briefcase": 0,
    "backpack": 0,
    "boxes": 0,
    "kitchen shelf": 0.7,
    "bottle of soap": 0,
}

@dataclass
class FrameSummary:
    name: str
    clusters: Sequence[str]
    relations: Sequence[str]

    def to_prompt_line(self, num_clusters_per_frame = 2) -> str:
        if not self.clusters:
            return f"{self.name}: Limited visibility in this frame."


        object_part = ", ".join(self.clusters[:num_clusters_per_frame])
        rel_part = "; ".join(self.relations[:num_clusters_per_frame]) if self.relations else ""

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


def get_distance_score(distance: float, target_distance: float = 3.0) -> float:
    if distance == 0:
        return 0.2
    
    diff = abs(distance - target_distance)
    if diff <= 1.0:
        return 1.0
    if diff <= 2.0:
        return 0.75
    if diff <= 3.0:
        return 0.5
    return 0.2


def distance_bucket(distance: float | None) -> str | None:
    if distance is None or distance <= 0:
        return None
    if distance < 1:
        return "very close"
    if distance < 3:
        return "close"
    if distance <= 5:
        return "mid-distance"
    if distance <= 6:
        return "slightly far"
    return "far"

def format_object_entry(cluster: Dict[str, Any]) -> str | None:
    label = str(cluster.get("label", "")).strip()
    cluster_str_id = str(cluster['cluster_str_id']).strip()

    # View-based positioning
    direction = None
    # View-based positioning from centroid_cam (x, y)
    centroid_cam = cluster.get("centroid_cam") # Position of the object in the camera pose
    if isinstance(centroid_cam, list) and len(centroid_cam) >= 2:
        x, y = centroid_cam[0], centroid_cam[1]
        if y > 0.3:
            vert = "upper"
        elif y < -0.3:
            vert = "lower"
        else:
            vert = "center"

        if x < -0.3:
            horiz = "left"
        elif x > 0.3:
            horiz = "right"
        else:
            horiz = "center"

        if vert == "center" and horiz == "center":
            direction = "center"
        else:
            direction = f"{vert}-{horiz}"
        
        position = f"(relative position: {direction}), (distance: {distance_bucket(cluster.get('distance_from_camera'))})"

        # Add information about the room name and the floor number
        room = cluster.get("room", "").strip()
        floor_number = cluster.get("floor_number")
        if room and floor_number is not None:
            position += f", (room: {room}), (floor: {floor_number})"

        return f"{cluster_str_id} [{position}]"


def object_priority(obj: Dict[str, Any]) -> tuple[float, float, float, float]:
    label = str(obj.get("label", "")).strip().lower()
    size = obj.get("linear_size", 0.0)
    pixel_percent = obj.get("pixel_percent")
    distance = obj.get("distance_from_camera")

    # percent_val = float(percent) if isinstance(percent, (float, int)) else 0.0
    distance_val = float(distance) if isinstance(distance, (float, int)) else 0.0
    distance_val = max(distance_val, 0.0)

    distance_score = get_distance_score(distance_val, target_distance=4.0)
    relevance = RELEVANCE_SCORES_OBJECTS.get(label, 0.5)

    scored_percent = pixel_percent * distance_score * relevance

    return (
        scored_percent,
        size,
        )


def select_n_clusters(clusters: Dict[str, Any], limit: int = 3, target_object: str = "") -> List[str]:
    candidates: List[Dict[str, Any]] = []
    print(f"[PIPPO] Selecting top {limit} clusters. Target object: '{target_object}'")
    for cluster in clusters.values():
        label = str(cluster.get("label", "")).lower()
        cluster_str_id = str(cluster['cluster_str_id']).lower()
        # print("Evaluating cluster:", cluster_str_id, "label:", label, "target_object:", target_object)
        cluster["priority_score"] = object_priority(cluster)[0]
        print("Cluster:", cluster_str_id, "Label:", label, "Priority Score:", cluster["priority_score"])
        
        if target_object and label.lower().strip() == target_object.lower().strip() and target_object != "":
            print(f"[PLUTO] Found target object '{target_object}' in cluster '{cluster_str_id}'. Setting highest priority.")
            cluster["priority_score"] = 9999.0 # * Set the highest priority for the target object
        if label in IGNORED_LABELS or not cluster_str_id or cluster_str_id == "":
            continue
        candidates.append(cluster)

    # Sort by priority score (higher is better), then by size (higher is better)
    candidates.sort(key=lambda c: (c["priority_score"], c.get("linear_size", 0.0)), reverse=True)
    
    results: List[str] = []
    for cluster in candidates[:limit]:
        formatted = format_object_entry(cluster)
        if formatted:
            results.append(formatted)
    return results, candidates[:limit]


def extract_relations(
    relationships: Iterable[Dict[str, Any]], limit: int = 2
) -> List[str]:
    # TODO -> need to have better relations
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
    clusters = {}
    filtered: List[str] = []
    for item in relationships:
        subj = item.get("subject", "")
        rel = item.get("relation", "")
        obj = item.get("object", "")
        subj_str_id = subj.get("cluster_str_id", "")
        obj_str_id = obj.get("cluster_str_id", "")

        if not subj or not rel or not obj:
            continue
        if subj_str_id.lower() in IGNORED_LABELS or obj_str_id.lower() in IGNORED_LABELS:
            continue
        natural_rel = natural_direction(rel)
        filtered.append(f"{subj_str_id} is {natural_rel} {obj_str_id}")

        if subj_str_id not in clusters:
            clusters[subj_str_id] = []
        if obj_str_id not in clusters:
            clusters[obj_str_id] = []
            
        clusters[subj_str_id] += subj.get("obj_str_ids", [])
        clusters[obj_str_id] += obj.get("obj_str_ids", [])
        if len(filtered) >= limit:
            break
    return filtered, clusters


def summarise_frames(frames: Sequence[Dict[str, Any]], num_clusters_per_frame = 2, target_name: str = "") -> List[FrameSummary]:
    summaries: List[FrameSummary] = []
    clusters_to_draw = {}

    # for each room get the top num_clusters_per_frame clusters based on relevance scores 
    # (no more than num_clusters_per_room clusters can be selected from the same room)
    # dictionary room_name -> [cluster1, cluster2, ...]

    # objs_in_room: Dict[str, List] = {}
    rooms_visited: List[str] = []
    for i, frame in enumerate(frames):

        name = str(frame.get("image_index"))
        clusters = frame.get("objects", {})
        spatial_relations = frame.get("spatial_relations", [])

        if i==0:
            turn_direction = frame.get("turn_direction")

        phrases, selected_clusters  = select_n_clusters(clusters, num_clusters_per_frame, target_name)
        
        relations, clusters_in_relations = extract_relations(spatial_relations)
            
        # clusters_to_draw = {"cluster_str_id": ["obj_str_id1", "obj_str_id2", ...], ...}

        # print("Frame", name, "objects:", selected_objects)
        for cluster in selected_clusters:
            cluster_str_id = cluster.get("cluster_str_id", "")
            obj_str_ids = cluster.get("obj_str_ids", [])

            # # Get room name (where this cluster belongs) and update the objs_in_room dictionary
            # room_name = cluster["room"].strip()
            # if room_name not in objs_in_room:
            #     objs_in_room[room_name] = []
            # if cluster not in objs_in_room[room_name]:
            #     objs_in_room[room_name].append(cluster)
            # else:
            #     # Need to update the existing cluster info
            #     for idx, existing_cluster in enumerate(objs_in_room[room_name]):
            #         if existing_cluster["cluster_str_id"] == cluster_str_id:
            #             if cluster["pixel_count"] > existing_cluster["pixel_count"]:
            #                 objs_in_room[room_name][idx] = cluster
            #                 break

            # print("Selected Cluster", cluster_str_id, "with obj IDs:", obj_str_ids)
            if cluster_str_id in clusters_to_draw:
                clusters_to_draw[cluster_str_id] = list(set(clusters_to_draw[cluster_str_id] + obj_str_ids))
            else:
                clusters_to_draw[cluster_str_id] = obj_str_ids
        
        # for cluster_str_id, obj_str_ids in clusters_in_relations.items():
        #     # print("Relation Cluster", cluster_str_id, "with obj IDs:", obj_str_ids)
        #     if cluster_str_id in clusters_to_draw:
        #         clusters_to_draw[cluster_str_id] = list(set(clusters_to_draw[cluster_str_id] + obj_str_ids))
        #     else:
        #         clusters_to_draw[cluster_str_id] = obj_str_ids

        print("IDs in frame", name, ":", clusters_to_draw)

        summaries.append(
            FrameSummary(
                name=name,
                clusters=phrases,
                relations=[],
            )
        )

        current_room = frame.get("current_room", "unknown_room")
        print("[PLUTO] Current room in frame", name, "is:", current_room)
        if current_room not in rooms_visited and "unknown" not in current_room.lower():
            rooms_visited.append(current_room)

    # print("All collected IDs:", raw_ids)
    return summaries, clusters_to_draw, rooms_visited, turn_direction

import re
def clean_text_from_ids(text: str) -> str:
    """
    Removes '_<ID>' (underscore followed by one or more digits) from a string.
    """
    # Regex pattern: _ (underscore) followed by \d+ (one or more digits)
    # The re.sub() function replaces all matches with an empty string ('').
    cleaned_text = re.sub(r"_\d+", "", text)
    
    # Optional: Remove any double spaces that might result from the removal 
    # (e.g., if "obj_123 text" becomes "obj  text")
    cleaned_text = re.sub(r"  +", " ", cleaned_text).strip()
    
    return cleaned_text

def build_prompt(
    scene_index: str | None, summaries: Sequence[FrameSummary], user_input: str, rooms_visited, turn_direction, num_clusters_per_frame: int = 2, target: str = ""
) -> str:
    
    

    # intro_lines = [
    #     "You are a navigation assistant helping me reach a target goal inside a building.",
    #     "You will see a sequence of frames data with visible objects, including floor, relative position to the viewer, the distance to the viewer and the name of the room they belong to.",
    #     "The frames are taken in chronological order along the path from the start to the target location.",
    #     "Write a human-sounding description of the walk, fluent and easy to follow for a real person.",
    #     "Avoid numeric measurements or technical descriptions. Focus on intuitive guidance under 120 words. You can use less than 120 if appropriate.",
    #     # "You must imagine to guide me from start to end of the path.",
    #     "You must mention at most two objects per room, by picking the most informative and useful ones for the scope of navigation and movement in the building.",
    #     "If the path goes through stairs, just mention 'go up / down to stairs to reach the $room_name' without mentioning objects in the stairs area.",
    #     "If you see the target location or object, you have to directly mention it and stop referencing other objects.",
    #     "Only reference objects that appear in the observations. Avoid embellishments or invented objects.",
    #     "When you mention an object, always its ID (e.g., 'chair_5') to uniquely identify it.",
    #     f"User question: {user_input}",
    #     # f"Target: {target}\n"
    #     "Here are the observations from the path:"
    # ]

    # header = "\n".join(intro_lines)
    # scene_line = (
    #     f"Scene index: {scene_index}" if scene_index else "Scene index: unknown"
    # )
    observation_lines = "\n".join(summary.to_prompt_line(num_clusters_per_frame=num_clusters_per_frame) for summary in summaries)
    print("Visited rooms:", rooms_visited)

    # Rooms visited in order: \n{', '.join(rooms_visited)}

    turn_istruction = ""
    if turn_direction != "forward":
        turn_istruction = f" Initially, turn {turn_direction}."

    return textwrap.dedent(
        f"""
        User question: {user_input}

        Observations:
        {turn_istruction}
        {observation_lines}

        Rooms visited in order: \n{', '.join(rooms_visited)}

        """
    ).strip()

def few_shot_examples() -> str:
    return textwrap.dedent(
        """
        ### Example 1
        User question: Where is the wall clock in the kitchen?
        Observations:
        In frame-000000, you see door_0 [(relative position: lower-right), (distance: close), (room: upper bedroom), (floor: 1)], couch_102 [(relative position: lower-left), (distance: mid-distance), (room: office), (floor: 1)].
        In frame-000001, you see door_0 [(relative position: center-right), (distance: close), (room: upper bedroom), (floor: 1)], stairs_142 [(relative position: lower-center), (distance: close), (room: living room), (floor: 0)].
        In frame-000002, you see stairs_142 [(relative position: lower-center), (distance: close), (room: living room), (floor: 0)], door_2 [(relative position: lower-left), (distance: mid-distance), (room: living room), (floor: 0)].
        In frame-000003, you see door_2 [(relative position: center-left), (distance: close), (room: living room), (floor: 0)], armchair_59 [(relative position: lower-left), (distance: mid-distance), (room: living room), (floor: 0)].
        In frame-000004, you see armchair_59 [(relative position: lower-right), (distance: mid-distance), (room: living room), (floor: 0)], couch_103 [(relative position: lower-left), (distance: close), (room: living room), (floor: 0)].
        In frame-000005, you see wall clock_175 [(relative position: upper-left), (distance: slightly far), (room: kitchen), (floor: 0)], couch_103 [(relative position: lower-left), (distance: close), (room: living room), (floor: 0)].
        In frame-000006, you see wall clock_175 [(relative position: upper-left), (distance: slightly far), (room: kitchen), (floor: 0)], couch_103 [(relative position: lower-left), (distance: close), (room: living room), (floor: 0)].
        In frame-000007, you see wall clock_175 [(relative position: upper-center), (distance: mid-distance), (room: kitchen), (floor: 0)], chair_126 [(relative position: lower-right), (distance: mid-distance), (room: kitchen), (floor: 0)]. 
        Rooms visited in order: upper bedroom, office, living room, kitchen
        Response:
        Go down the stairs_142 in front of you, and reach the living room. Here you'll find an armchair_59 to your left and a couch_103 on your right. Continue straight ahead into the kitchen where the wall clock_175 is visible on the upper-left wall.

        ### Example 2
        User question: How do I get to the fireplace?
        Observations:
        In frame-000000, you see door_7 [(relative position: lower-left), (distance: close), (room: entryway), (floor: 0)], picture_114 [(relative position: center-left), (distance: close), (room: entryway), (floor: 0)].
        In frame-000001, you see picture_114 [(relative position: center-left), (distance: very close), (room: entryway), (floor: 0)], flag_176 [(relative position: lower-right), (distance: close), (room: kitchen), (floor: 0)].
        In frame-000002, you see chair_126 [(relative position: lower-left), (distance: close), (room: kitchen), (floor: 0)], door_6 [(relative position: center-right), (distance: close), (room: kitchen), (floor: 0)].
        In frame-000003, you see fireplace_153 [(relative position: lower-left), (distance: slightly far), (room: living room), (floor: 0)], armchair_60 [(relative position: lower-left), (distance: mid-distance), (room: living room), (floor: 0)].
        In frame-000004, you see fireplace_153 [(relative position: lower-center), (distance: slightly far), (room: living room), (floor: 0)], kitchen cabinet_178 [(relative position: lower-left), (distance: mid-distance), (room: kitchen), (floor: 0)].
        In frame-000005, you see fireplace_153 [(relative position: lower-center), (distance: close), (room: living room), (floor: 0)], armchair_59 [(relative position: lower-right), (distance: mid-distance), (room: living room), (floor: 0)].
        Rooms visited in order: entryway, kitchen, living room
        Response:
        From the entryway, reach the corridor where you see a flag_176 to enter the kitchen. Continue straight ahead into the living room, where the fireplace_153 is located to the lower-left.

        ### Example 3
        User question: I want to go to the fireplace.
        Observations:
        Observations:
        In frame-000000, you see fireplace_153 [(relative position: lower-left), (distance: slightly far), (room: living room), (floor: 0)], armchair_59 [(relative position: lower-right), (distance: mid-distance), (room: living room), (floor: 0)].
        In frame-000001, you see fireplace_153 [(relative position: lower-center), (distance: slightly far), (room: living room), (floor: 0)], armchair_59 [(relative position: lower-right), (distance: mid-distance), (room: living room), (floor: 0)].
        In frame-000002, you see fireplace_153 [(relative position: lower-center), (distance: close), (room: living room), (floor: 0)], led tv_151 [(relative position: center), (distance: close), (room: living room), (floor: 0)].
        Rooms visited in order: living room
        Response:
        You can already see the fireplace_153, it's located right in front of you in the living room, just below the led tv_151.

        ### Example 4
        User question: I am hungry, can you guide me to the refrigerator?
        Observations:
        In frame-000000, you see refrigerator_177 [(relative position: lower-left), (distance: mid-distance), (room: kitchen), (floor: 0)], kitchen cabinet_178 [(relative position: center-right), (distance: close), (room: kitchen), (floor: 0)].
        In frame-000001, you see refrigerator_177 [(relative position: lower-right), (distance: close), (room: kitchen), (floor: 0)], kitchen cabinet_178 [(relative position: center-right), (distance: very close), (room: kitchen), (floor: 0)].
        Rooms visited in order: kitchen
        Response:
        The refrigerator_177 is located in front of you to your left.

        ### Example 5
        User question: I want to go to the kitchen sink.
        Observations:
        In frame-000000, you see sink_184 [(relative position: lower-right), (distance: far), (room: kitchen), (floor: 0)], fireplace_153 [(relative position: lower-right), (distance: mid-distance), (room: living room), (floor: 0)].
        In frame-000001, you see sink_184 [(relative position: lower-right), (distance: mid-distance), (room: kitchen), (floor: 0)], kitchen cabinet_178 [(relative position: center-left), (distance: mid-distance), (room: kitchen), (floor: 0)].
        Rooms visited in order: living room, kitchen
        Response:
        From the living room, head towards the kitchen, and you will find the sink_184 located to your lower-right.

        ### Example 6
        User question: How do I reach the tv in the upper bedroom?
        Observations:
        In frame-000000, you see door_16 [(relative position: lower-center), (distance: close), (room: lower bedroom), (floor: 0)], door_14 [(relative position: lower-center), (distance: mid-distance), (room: entryway), (floor: 0)].
        In frame-000001, you see door_14 [(relative position: lower-right), (distance: close), (room: entryway), (floor: 0)], door_15 [(relative position: center-left), (distance: close), (room: dining room), (floor: 0)].
        In frame-000002, you see chair_152 [(relative position: lower-left), (distance: mid-distance), (room: kitchen), (floor: 0)], door_12 [(relative position: center-right), (distance: close), (room: kitchen), (floor: 0)].
        In frame-000003, you see armchair_74 [(relative position: lower-left), (distance: mid-distance), (room: living room), (floor: 0)], couch_126 [(relative position: lower-right), (distance: mid-distance), (room: living room), (floor: 0)].
        In frame-000004, you see armchair_73 [(relative position: lower-left), (distance: mid-distance), (room: living room), (floor: 0)], door_8 [(relative position: lower-left), (distance: mid-distance), (room: living room), (floor: 0)].
        In frame-000005, you see door_8 [(relative position: lower-left), (distance: close), (room: living room), (floor: 0)], stairs_170 [(relative position: lower-right), (distance: close), (room: living room), (floor: 0)].
        In frame-000006, you see picture_136 [(relative position: upper-right), (distance: close), (room: living room), (floor: 0)], picture_137 [(relative position: center-left), (distance: close), (room: living room), (floor: 0)].
        In frame-000007, you see stairs_170 [(relative position: lower-center), (distance: mid-distance), (room: living room), (floor: 0)], couch_126 [(relative position: lower-right), (distance: mid-distance), (room: living room), (floor: 0)].
        In frame-000008, you see stairs_170 [(relative position: center), (distance: very close), (room: living room), (floor: 0)], door_7 [(relative position: upper-left), (distance: slightly far), (room: office), (floor: 1)].
        In frame-000009, you see door_3 [(relative position: upper-left), (distance: mid-distance), (room: upper bathroom), (floor: 1)], bicycle_119 [(relative position: center-right), (distance: close), (room: office), (floor: 1)].
        In frame-000010, you see tv_30 [(relative position: lower-right), (distance: close), (room: upper bedroom), (floor: 1)], door_1 [(relative position: center-right), (distance: very close), (room: upper bedroom), (floor: 1)].
        Rooms visited in order: lower bedroom, entryway, kitchen, living room, office, upper bedroom
        Response:
        Exit the lower bedroom through the door_16. Take the corridor on your right to the kitchen. Continue straight into the livingroom where you will find couch_126 on you right. From there, go up the stairs_170 to the first floor. Once upstairs, pass the bicycle and go through the door_1 to enter the upper bedroom, where the tv_30 is located to your lower-right.

        ### Example 7
        User question: where is the sink in the kitchen?
        Observations:
        In frame-000000, you see sink_218 [(relative position: lower-left), (distance: mid-distance), (room: kitchen), (floor: 0)], armchair_74 [(relative position: lower-right), (distance: mid-distance), (room: living room), (floor: 0)].
        In frame-000001, you see sink_218 [(relative position: lower-right), (distance: close), (room: kitchen), (floor: 0)], kitchen cabinet_208 [(relative position: lower-left), (distance: close), (room: kitchen), (floor: 0)].
        Rooms visited in order: kitchen
        Response:
        The sink_218 is at your lower-left in the kitchen, next to the kitchen cabinet_208.

        ### End of examples.
        """
    ).strip()

def generate_description(prompt: str, model = None, tokenizer = None) -> str:
    if model == None or tokenizer == None:
        print("[INFO] Generating description using OpenAI ChatGPT API...")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY environment variable is not set.")

        try:
            import openai  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "The openai package is required. Install it via 'pip install openai'."
            ) from exc
        
        system_prompt = """
            You are a navigation assistant helping the user locate a target object inside a building.

            You will receive a sequence of frames describing visible objects.  
            Each object includes:  
            - the floor,  
            - the relative position to the viewer,  
            - the distance from the viewer,  
            - and the room it belongs to.

            The frames appear in chronological order along the user's path from the starting point toward the target.

            Before starting the walk description, consider an initial turn direction if provided.
            Your task is to write a human-sounding description of the walk, fluent and easy to follow.  
            Avoid technical language or numeric measurements. Use intuitive guidance and stay under 120 words (using fewer words when possible).

            Mention at least one and at most two objects per room, choosing only the most informative for navigation.  
            If the path includes stairs, simply write: “go up/down the stairs to reach the <room_name>”, without describing objects on the stairs.

            If you see the target location or object, mention it immediately and stop referencing any further objects.

            Only refer to objects that appear in the observations. Never invent or embellish details.  
            When referencing an object, always include its ID (e.g., “chair_5”).

            You will then receive a user question and the list of observations from the path, as well as the rooms visited in order. Imagine you are moving from the starting room to the target location, and provide clear path instructions.
        """

        system_prompt += few_shot_examples()

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ]

        kwargs = {"model": "gpt-4o", "temperature": 0.6}

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
    else:
    #     messages = [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": "Who are you?"},
    #     ]
        
    # # --- Create inputs ---
    # input_ids = tokenizer.apply_chat_template(
    #     messages,
    #     add_generation_prompt=True,
    #     return_tensors="pt",
    # ).to(model.device)

    # attention_mask = torch.ones_like(input_ids)
    # print("Pre generation")
    # # --- Generate ---
    # with torch.no_grad():
    #     outputs = model.generate(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         max_new_tokens=50
    #     )
    # print("Post generation")

    # generated = outputs[0][input_ids.shape[-1]:]
    # print(tokenizer.decode(generated, skip_special_tokens=True))
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            dtype=torch.float16,
            device_map="auto"
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who are you?"},
        ]

        # --- Create inputs ---
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        attention_mask = torch.ones_like(input_ids)

        # --- Generate ---
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50
            )

        generated = outputs[0][input_ids.shape[-1]:]
        print(tokenizer.decode(generated, skip_special_tokens=True))
    return None



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
    model,
    tokenizer,
    max_frames: int = 40,
    dry_run: bool = False,
    target_name: str = "",
    room_name: str = "",
) -> str:
    """
    Full pipeline: loads frames, builds prompt, optionally queries the model, and returns description or prompt.
    Does NOT write anything to disk.
    """
    frames = load_frames(input_dir, max_frames)
    scene_index = frames[0].get("scene_index")
    num_clusters_per_frame = 2
    summaries, clusters_to_draw, rooms_visited, turn_direction = summarise_frames(frames, num_clusters_per_frame=num_clusters_per_frame, target_name=target_name)
    if room_name != "" and room_name not in rooms_visited:
        rooms_visited.append(room_name)
    print("[PLUTO] Rooms visited:", rooms_visited)
    prompt = build_prompt(scene_index, summaries, user_input, rooms_visited, turn_direction, num_clusters_per_frame=num_clusters_per_frame, target=target_name if target_name else room_name)

    print(prompt)
    print("\n\n[generate_path_description] - Cluster to draw:", clusters_to_draw)
    if dry_run: 
        return None, clusters_to_draw

    if model == None or tokenizer == None:
        print("[generate_path_description] - Using OpenAI backend for description generation.")
        description = generate_description(prompt)
    else:
        print("[generate_path_description] - Using Local LLM backend for description generation.")
        description = generate_description(prompt, model, tokenizer)

    draw_all_clusters = True #! TODO set false then
    if draw_all_clusters:
        clusters_to_draw_final = clusters_to_draw
    else:
        clusters_to_draw_final = {}
        for cluster_str_id in clusters_to_draw:
            if cluster_str_id in description:
                clusters_to_draw_final[cluster_str_id] = clusters_to_draw[cluster_str_id]
    print("\n Descrition before cleaning:", description)
    description = clean_text_from_ids(description)

    return description, clusters_to_draw_final

def generate_relevance_scores(clusters) -> List[tuple[str, float]]:
    """
    Generate a score for each of the clusters (clusters use the format inside the frames JSON). They are a dictionary of cluster_str_id -> cluster info.
    """
    pass



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user.")
