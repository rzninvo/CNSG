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
    turn_direction: str | None = None

    def to_prompt_line(self, num_clusters_per_frame = 2) -> str:
        if not self.clusters:
            return f"{self.name}: Limited visibility in this frame."


        object_part = ", ".join(self.clusters[:num_clusters_per_frame])

        direction_part = ""
        frame_number = int(self.name.split("-")[-1])
        if frame_number == 0:
            connector = "Initially"
        else:
            connector = "From previous frame"

        if self.turn_direction == "forward":
            direction_part = f"{connector}, continue forward.\n"
        elif self.turn_direction == "left" or self.turn_direction == "right":
            direction_part = f"{connector}, turn {self.turn_direction}.\n"
        elif self.turn_direction == "behind" and frame_number == 0:
            direction_part = f"{connector}, turn around.\n"


        description = f"{direction_part} In {self.name}, you see {object_part}."
    
        return description
    
    def get_turn_direction(self) -> str:
        return self.turn_direction if self.turn_direction else "forward"


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
    # View-based positioning using NDC (Normalized Device Coordinates)
    # NDC coordinates: x in [-1, 1] (left to right), y in [-1, 1] (bottom to top)
    ndc_x = cluster.get("ndc_x")
    ndc_y = cluster.get("ndc_y")
    # print("NDC coordinates for", cluster_str_id, ":", ndc_x, ndc_y)
    
    if ndc_x is not None and ndc_y is not None:
        # Position detection with NDC coordinates (screen space)
        # Vertical axis: ndc_y (-1 bottom to +1 top)
        if ndc_y > 0.3:
            vert = "upper"
        elif ndc_y < -0.3:
            vert = "lower"
        else:
            vert = "center"

        # Horizontal axis: ndc_x (-1 left to +1 right)
        if ndc_x < -0.2:
            horiz = "left"
        elif ndc_x > 0.2:
            horiz = "right"
        else:
            horiz = "center"

        if vert == "center" and horiz == "center":
            direction = "center"
        else:
            direction = f"{vert}-{horiz}"
        
        # position = f"(relative position: {direction}), (distance: {distance_bucket(cluster.get('distance_from_camera'))})"
        position = f"(relative position: {direction})" #! TODO removed distance
    else:
        # Fallback if NDC not available
        direction = "unknown"
        position = "(relative position: unknown)"
    
    # Add information about the room name and the floor number (always, regardless of NDC availability)
    room = cluster.get("room", "").strip()
    floor_number = cluster.get("floor_number")
    if room and floor_number is not None:
        # position += f", (room: {room}), (floor: {floor_number})"
        position += f", (room: {room})" #! NOTE removed floor

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
    for cluster in clusters.values():
        label = str(cluster.get("label", "")).lower()
        cluster_str_id = str(cluster['cluster_str_id']).lower()
        cluster["priority_score"] = object_priority(cluster)[0]
        
        if target_object and label.lower().strip() == target_object.lower().strip() and target_object != "":
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
    target_found = False
    for i, frame in enumerate(frames):

        name = str(frame.get("image_index"))
        clusters = frame.get("objects", {})

        turn_direction = frame.get("turn_direction")

        phrases, selected_clusters  = select_n_clusters(clusters, num_clusters_per_frame, target_name)
        
            
        # clusters_to_draw = {"cluster_str_id": ["obj_str_id1", "obj_str_id2", ...], ...}

        if not target_found:
            summaries.append(
                FrameSummary(
                    name=name,
                    clusters=phrases,
                    turn_direction=turn_direction,
                    relations=[],
                )
            ) 

        for cluster in selected_clusters:
            cluster_str_id = cluster.get("cluster_str_id", "")
            obj_str_ids = cluster.get("obj_str_ids", [])

            # print("Selected Cluster", cluster_str_id, "with obj IDs:", obj_str_ids)
            if cluster_str_id in clusters_to_draw:
                clusters_to_draw[cluster_str_id] = list(set(clusters_to_draw[cluster_str_id] + obj_str_ids))
            else:
                clusters_to_draw[cluster_str_id] = obj_str_ids
            
            if target_name and target_name in cluster_str_id.lower():
                target_found = True #! NOTE added, check if this works better than before
        

        
        current_room = frame.get("current_room", {})
        if current_room is None:
            continue
        current_room_name = current_room.get("name", "unknown_room")
        rooms_visited_names = [room.get("name") for room in rooms_visited if isinstance(room, dict)]
        if current_room_name not in rooms_visited_names and "unknown" not in current_room_name.lower():
            rooms_visited.append(current_room)

    # print("All collected IDs:", raw_ids)
    return summaries, clusters_to_draw, rooms_visited

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

def build_prompt(summaries: Sequence[FrameSummary], user_input: str, rooms_visited, num_clusters_per_frame: int = 2) -> str:
    
    observation_lines = "\n".join(summary.to_prompt_line(num_clusters_per_frame=num_clusters_per_frame) for summary in summaries)

    # Rooms visited in order: \n{', '.join(rooms_visited)}


    visited_room_strings = []
    for room in rooms_visited:
        if not isinstance(room, dict):
            continue
        visited_room_strings.append(f"{room.get('name')} (floor: {room.get('floor_number')})")
    
    return textwrap.dedent(
        f"""
        User question: {user_input}

        Observations:
        {observation_lines}

        Rooms visited in order: \n{', '.join(visited_room_strings)}
        The user is in {rooms_visited[0].get("name")} (floor: {rooms_visited[0].get("floor_number")}) and the target is in {rooms_visited[-1].get("name")} (floor: {rooms_visited[-1].get("floor_number")}).
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
        Rooms visited in order: entryway, kitchen, living room
        Response:
        From the entryway, reach the corridor where you see a flag_176 to enter the kitchen. Continue straight ahead into the living room, where the fireplace_153 is located to the lower-left.

        ### Example 3
        User question: I want to go to the fireplace.
        Observations:
        Observations:
        In frame-000000, you see fireplace_153 [(relative position: lower-left), (distance: slightly far), (room: living room), (floor: 0)], armchair_59 [(relative position: lower-right), (distance: mid-distance), (room: living room), (floor: 0)].
        Rooms visited in order: living room
        Response:
        You can already see the fireplace_153, it's located right in front of you in the living room, just below the led tv_151.

        ### Example 4
        User question: I am hungry, can you guide me to the refrigerator?
        Observations:
        In frame-000000, you see refrigerator_177 [(relative position: lower-left), (distance: mid-distance), (room: kitchen), (floor: 0)], kitchen cabinet_178 [(relative position: center-right), (distance: close), (room: kitchen), (floor: 0)].
        Rooms visited in order: kitchen
        Response:
        The refrigerator_177 is located in front of you to your left.

        ### Example 5
        User question: I want to go to the kitchen sink.
        Observations:
        In frame-000000, you see sink_184 [(relative position: lower-right), (distance: far), (room: kitchen), (floor: 0)], fireplace_153 [(relative position: lower-right), (distance: mid-distance), (room: living room), (floor: 0)].
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
        Rooms visited in order: kitchen
        Response:
        The sink_218 is at your lower-left in the kitchen, next to the kitchen cabinet_208.

        ### End of examples.
        """
    ).strip()

def generate_description(user_prompt: str, model = None, tokenizer = None) -> str:

    
    
    if model is None and tokenizer is None:
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
        system_prompt += few_shot_examples() #! NOTE added few shot examples only for OpenAI API (out of memory issues with local model)
    
    else:
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
            Your task is to write a human-sounding description of the path.  
            Avoid technical language or numeric measurements. Use intuitive guidance and stay under 120 words (using fewer words when possible).

            Mention at least one and at most two objects per room, choosing only the most informative for navigation.  
            If the path includes stairs, simply write: “go up/down the stairs to reach the <room_name>”, without describing objects on the stairs.

            If you see the target location or object, mention it immediately and stop referencing any further objects.

            Only refer to objects that appear in the observations. Never invent or embellish details.  
            Use the object IDs when referencing them (e.g., “chair_5”).

            You will then receive a user question and the list of observations from the path, as well as the rooms visited in order. 
            Imagine you are moving from the starting room to the target location, and provide clear path instructions.
        """

    # print("System prompt:\n", system_prompt)
    print("User prompt:\n", user_prompt)

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user", 
            "content": user_prompt
        },
    ]
    
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
                max_new_tokens=500
            )

        generated = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()
        print("[LOCAL MODEL] Generated response:", response)
        return response
    


# * Used as API
def generate_path_description(
    frames: List[Dict[str, Any]],
    user_input: str,
    model,
    tokenizer,
    max_frames: int = 40,
    dry_run: bool = False,
    target_name: str = "",
    room_name: str = "",
    floor_number: int | None = None,
) -> str:
    """
    Full pipeline: loads frames, builds prompt, optionally queries the model, and returns description or prompt.
    Does NOT write anything to disk.
    """
    frames = frames[:max_frames] if max_frames else frames
    num_clusters_per_frame = 2
    summaries, clusters_to_draw, rooms_visited = summarise_frames(frames, num_clusters_per_frame=num_clusters_per_frame, target_name=target_name)
    current_room_names = [room.get("name") for room in rooms_visited if isinstance(room, dict)]
    if room_name != "" and room_name not in current_room_names:
        rooms_visited.append({"name": room_name, "floor_number": floor_number})
    prompt = build_prompt(summaries, user_input, rooms_visited, num_clusters_per_frame=num_clusters_per_frame)

    print("\n\n[generate_path_description] - Cluster to draw:", clusters_to_draw)
    if dry_run: 
        return None, clusters_to_draw

    if model == None or tokenizer == None:
        print("[generate_path_description] - Using OpenAI backend for description generation.")
        description = generate_description(prompt)
    else:
        print("[generate_path_description] - Using Local LLM backend for description generation.")
        description = generate_description(prompt, model, tokenizer)

    draw_all_clusters = True #! TODO set to false to visualize only clusters mentioned by the LLM
    if draw_all_clusters:
        clusters_to_draw_final = clusters_to_draw
    else:
        clusters_to_draw_final = {}
        for cluster_str_id in clusters_to_draw:
            if cluster_str_id in description:
                clusters_to_draw_final[cluster_str_id] = clusters_to_draw[cluster_str_id]
    print("\nDescription before cleaning:", description)
    description = clean_text_from_ids(description)

    return description, clusters_to_draw_final



