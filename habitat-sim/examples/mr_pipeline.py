#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
import ctypes
import math
import os
import string
import sys
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import json
import datetime
import re
import threading
import queue


import json
import os
from pathlib import Path
from typing import Any, Dict, List, Iterable, Sequence
from dataclasses import dataclass
import textwrap
from openai import OpenAI


flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import magnum as mn
import numpy as np
from magnum import shaders, text
from magnum.platform.glfw import Application

import habitat_sim
from habitat_sim.utils import common as utils

from habitat_sim import ReplayRenderer, ReplayRendererConfiguration, physics
from habitat_sim.logging import LoggingContext, logger
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_sim.utils.settings import default_sim_settings, make_cfg


# Try importing the base viewer.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./")))
from viewer import (
    HabitatSimInteractiveViewer as BaseViewer,
    MouseMode,
    MouseGrabber,
    Timer,
)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.generate_description import generate_path_description
from utils.conversation_gui import *

from habitat.utils.visualizations import maps
from habitat_sim.utils.common import d3_40_colors_rgb


# Initialize OpenAI client
try:
    from dotenv import load_dotenv

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    print(
        "[WARNING] OpenAI API key not found or error loading .env file. ChatGPT features will be disabled."
    )
    client = None


class NewViewer(BaseViewer):
    MOVE, LOOK = 0.04, 1.5  # New definition for these two attributes

    def __init__(self, sim_settings: Dict[str, Any], q_app: QApplication) -> None:
        scene_path = sim_settings["scene"]
        super().__init__(sim_settings)
        self.q_app = q_app

        self.cnt = 0
        self.action_queue = queue.Queue()

        # draw object bounding boxes when enabled
        self.show_object_bboxes = False
        self._object_bbox_colors: Dict[int, mn.Color4] = {}
        self._bbox_label_screen_positions: List[Tuple[str, mn.Vector2]] = []

        self.map_room_id_to_name = {}
        self.room_objects_occurences = {}

        if True:
            base_path = os.path.dirname(scene_path)
            scene_name = os.path.splitext(os.path.basename(scene_path))[0]
            semantic_path = os.path.join(
                base_path, f"{scene_name.split('.')[0]}.semantic.txt"
            )
            map_file_path = os.path.join(base_path, "room_id_to_name_map.json")
            # room_object_file_path = os.path.join(base_path, "scene_room_object_occurences.json")

            print(f"Base path: {base_path}")
            print(f"Semantic path: {semantic_path}")
            print(f"Map file path: {map_file_path}")
            # print(f"Room-object occurences file path: {room_object_file_path}")

            if os.path.exists(map_file_path):
                with open(map_file_path, "r", encoding="utf-8") as f:
                    self.map_room_id_to_name = json.load(f)
            else:
                raise FileNotFoundError(f"Map file not found: {map_file_path}")

            ignore_categories = [
                "ceiling",
                "floor",
                "wall",
                "handle",
                "window frame",
                "door frame",
                "frame",
                "unknown",
                "stairs",
                "staircase",
                "stair",
                "stairway",
            ]
            semantic_info = self.get_semantic_info(
                semantic_path,
                map_room_id_to_name=self.map_room_id_to_name,
                ignore_categories=ignore_categories,
            )

            self.room_objects_occurences = semantic_info
            # print("\nSemantic information of the scene:")
            # print(semantic_info)

            # self.print_scene_semantic_info()

            # Demonstrate shortest path functionality
            # dummy_goal = mn.Vector3(-1.6096749, 3.163378, -7.154511)
            # self.shortest_path(self.sim, dummy_goal)

        ###########################################

    def _process_queued_actions(self):
        """Execute actions enqueued from other threads."""
        try:
            while True:
                action, args, kwargs = self.action_queue.get_nowait()
                try:
                    action(*args, **kwargs)
                except Exception as e:
                    print(f"Error executing queued action {action}: {e}")

                self.action_queue.task_done()

        except queue.Empty:
            pass

    def enqueue_shortest_path(self, goal_pos):
        self.action_queue.put((self.shortest_path, (self.sim, goal_pos), {}))

    # display a topdown map with matplotlib
    def display_map(self, topdown_map, key_points=None):
        plt.figure(figsize=(36, 24))
        ax = plt.subplot(1, 1, 1)
        ax.axis("off")
        plt.imshow(topdown_map)
        # plot points on map
        if key_points is not None:
            for point in key_points:
                plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
        # plt.show(block=False)

        self.topdown_map_counter = getattr(self, "topdown_map_counter", 0)
        plt.savefig(
            f"output/topdown_map{self.topdown_map_counter}.png", bbox_inches="tight"
        )
        self.topdown_map_counter += 1
        plt.close()
        # logger.info(f"Saved: output/topdown_map.png")

    def display_sample(
        self, rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])
    ):
        rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

        arr = [rgb_img]
        titles = ["rgb"]
        if semantic_obs.size != 0:
            semantic_img = Image.new(
                "P", (semantic_obs.shape[1], semantic_obs.shape[0])
            )
            semantic_img.putpalette(d3_40_colors_rgb.flatten())
            semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
            semantic_img = semantic_img.convert("RGBA")
            arr.append(semantic_img)
            titles.append("semantic")

        if depth_obs.size != 0:
            depth_img = Image.fromarray(
                (depth_obs / 10 * 255).astype(np.uint8), mode="L"
            )
            arr.append(depth_img)
            titles.append("depth")

        plt.figure(figsize=(36, 24))
        for i, data in enumerate(arr):
            ax = plt.subplot(1, 3, i + 1)
            ax.axis("off")
            ax.set_title(titles[i])
            plt.imshow(data)

        # Inizializza contatore e cartella output solo la prima volta
        if not hasattr(self, "output_counter"):
            self.output_counter = 0

        # incrementa contatore

        filename = f"output/sample_output_{self.output_counter}.png"
        self.output_counter += 1

        plt.savefig(filename, bbox_inches="tight")
        plt.close()
        # logger.info(f"Saved: {filename}")

    def densify_path(self, path_points, step_size=1.0, min_step_size=0.7):
        points = np.array(path_points)

        new_points = [points[0]]
        for i in range(1, len(points)):
            p0, p1 = points[i - 1], points[i]
            segment = p1 - p0
            dist = np.linalg.norm(segment)
            if dist == 0:
                continue
            if (
                dist <= step_size
                and np.linalg.norm(new_points[-1] - p1) > min_step_size
            ):
                new_points.append(p1)
                continue
            direction = segment / dist
            n_steps = int(dist / step_size)
            for s in range(1, n_steps + 1):
                new_point = p0 + direction * step_size * s
                new_points.append(new_point)
        if np.linalg.norm(new_points[-1] - points[-1]) > 0.3:  # NOTE removed
            new_points.append(points[-1])
        return np.array(new_points)

    def get_semantic_info(self, file_path, map_room_id_to_name, ignore_categories=[]):
        semantic_info = {}
        with open(file_path, "r") as f:
            for line in f:
                line_parts = line.strip().split(",")
                if len(line_parts) != 4:
                    continue

                room_info = map_room_id_to_name.get(line_parts[3])
                if room_info and "name" in room_info:
                    room_id = room_info["name"]
                else:
                    room_id = "unknown_room"

                category_id = line_parts[2].strip('"')

                if room_id not in semantic_info:
                    semantic_info[room_id] = {}

                if category_id not in ignore_categories:
                    if category_id not in semantic_info[room_id]:
                        semantic_info[room_id][category_id] = 1
                    else:
                        semantic_info[room_id][category_id] += 1

        return semantic_info

    def extract_visible_objects(self, sim, observations) -> Optional[Dict[str, Any]]:
        """
        Given current sim and observations (must include semantic sensor),
        returns a dict of visible objects and spatial relations between them.
        """
        if "semantic_sensor" not in observations:
            logger.warning(
                "No semantic sensor found; skipping visible object extraction."
            )
            return None

        semantic = observations["semantic_sensor"]
        total_pixels = semantic.size
        ids, counts = np.unique(semantic, return_counts=True)

        visible_objects = {}
        for obj_id, pixel_count in zip(ids, counts):
            if obj_id == 0 or pixel_count < 50:
                continue  # skip background/noise/small fragments

            try:
                obj = sim.semantic_scene.objects[obj_id]
            except IndexError:
                continue

            label = (
                obj.category.name() if obj.category is not None else f"object_{obj_id}"
            )
            # Prefer oriented bounding box if available
            obb = getattr(obj, "obb", None)
            if obb is not None:
                center = mn.Vector3(obb.center)
                half_extents = mn.Vector3(obb.half_extents)
                rot_vec = mn.Vector4(obb.rotation)
                rotation = mn.Quaternion(
                    mn.Vector3(rot_vec[0], rot_vec[1], rot_vec[2]), rot_vec[3]
                )

                # Compute oriented corners
                corner_offsets = [
                    mn.Vector3(-half_extents.x, -half_extents.y, -half_extents.z),
                    mn.Vector3(half_extents.x, -half_extents.y, -half_extents.z),
                    mn.Vector3(-half_extents.x, half_extents.y, -half_extents.z),
                    mn.Vector3(half_extents.x, half_extents.y, -half_extents.z),
                    mn.Vector3(-half_extents.x, -half_extents.y, half_extents.z),
                    mn.Vector3(half_extents.x, -half_extents.y, half_extents.z),
                    mn.Vector3(-half_extents.x, half_extents.y, half_extents.z),
                    mn.Vector3(half_extents.x, half_extents.y, half_extents.z),
                ]
                corners_world = [
                    rotation.transform_vector(offset) + center
                    for offset in corner_offsets
                ]

                bbox_world = [
                    [
                        float(min(v.x for v in corners_world)),
                        float(min(v.y for v in corners_world)),
                        float(min(v.z for v in corners_world)),
                    ],
                    [
                        float(max(v.x for v in corners_world)),
                        float(max(v.y for v in corners_world)),
                        float(max(v.z for v in corners_world)),
                    ],
                ]
                centroid_world = [float(center.x), float(center.y), float(center.z)]
            else:
                # fallback to aabb if obb missing
                aabb = obj.aabb
                vmin = aabb.min() if callable(getattr(aabb, "min", None)) else aabb.min
                vmax = aabb.max() if callable(getattr(aabb, "max", None)) else aabb.max
                bbox_world = [
                    [float(vmin[0]), float(vmin[1]), float(vmin[2])],
                    [float(vmax[0]), float(vmax[1]), float(vmax[2])],
                ]
                centroid_world = [float(x) for x in aabb.center()]

            # Convert centroid to camera coordinates
            sensor_state = (
                sim.get_agent(self.agent_id).get_state().sensor_states["color_sensor"]
            )
            rot_mn = utils.quat_to_magnum(sensor_state.rotation)
            T_world_cam = mn.Matrix4.from_(
                rot_mn.inverted().to_matrix(),
                -rot_mn.inverted().transform_vector(sensor_state.position),
            )
            centroid_cam = T_world_cam.transform_point(mn.Vector3(*centroid_world))
            centroid_cam = np.array([centroid_cam.x, centroid_cam.y, centroid_cam.z])
            dist = float(np.linalg.norm(centroid_cam))

            visible_objects[str(obj_id)] = {
                "label": label,
                "pixel_count": int(pixel_count),
                "pixel_percent": float(100 * pixel_count / total_pixels),
                "centroid_world": centroid_world,
                "bbox_world": bbox_world,
                "centroid_cam": centroid_cam.tolist(),
                "distance_from_camera": dist,
                "linear_size": self.compute_object_size({"bbox_world": bbox_world}),
            }

        print(f"[DEBUG] Visible Objects (size): ")
        for obj_id, obj_data in visible_objects.items():
            print(
                f"  ID {obj_id}: {obj_data['label']}, linear_size={obj_data['linear_size']:.2f} m"
            )

        # Compute spatial relations
        relations = self.compute_spatial_relations(visible_objects)

        return {
            "visible_objects": visible_objects,
            "spatial_relations": relations,
        }

    def compute_spatial_relations(
        self,
        visible_objects,
        max_distance=1.5,
        vertical_thresh=0.25,
        horizontal_bias=1.2,
        size_ratio_thresh=3.0,
    ):
        """
        Compute spatial relations between nearby objects, filtering out irrelevant ones.
        - Ignora relazioni tra oggetti con scala troppo diversa
        - Favorisce relazioni tra oggetti visibili di scala comparabile
        - Usa la direzione più dominante per definire la relazione
        - Applica regole semantiche per oggetti specifici (es. tavoli, porte, ecc.)
        """

        # Definisci quali relazioni sono ammissibili per certi oggetti
        # Formato: "object_label": {"allowed": [...]}
        SEMANTIC_RULES = {
            "table": {
                "allowed": ["on_top_of", "beneath_of", "left_of", "right_of"],
            },
            "desk": {
                "allowed": ["on_top_of", "beneath_of", "left_of", "right_of"],
            },
            "door": {
                "allowed": [
                    "on_top_of",
                    "left_of",
                    "right_of",
                    "in_front_of",
                ],
            },
            "window": {
                "allowed": [
                    "on_top_of",
                    "left_of",
                    "right_of",
                ],
            },
            "ceiling": {
                "allowed": ["beneath_of"],
            },
            "floor": {
                "allowed": ["on_top_of"],
            },
        }

        def is_relation_valid(obj_label, relation):
            """
            Verifica se una relazione è semanticamente valida per un dato oggetto.
            """
            if obj_label not in SEMANTIC_RULES:
                return True  # Se non ci sono regole, accetta tutto

            rules = SEMANTIC_RULES[obj_label]

            if "allowed" in rules and relation not in rules["allowed"]:
                return False

            return True

        relations = []
        keys = list(visible_objects.keys())

        if len(keys) <= 1:
            return relations

        centroids = {
            k: np.array(visible_objects[k]["centroid_cam"], dtype=float) for k in keys
        }

        sizes = {k: float(visible_objects[k].get("linear_size", 0.0)) for k in keys}

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                obj_a, obj_b = visible_objects[keys[i]], visible_objects[keys[j]]
                size_a, size_b = sizes[keys[i]], sizes[keys[j]]

                if size_a <= 0 or size_b <= 0:
                    continue

                ratio = max(size_a, size_b) / min(size_a, size_b)
                if ratio > size_ratio_thresh:
                    continue

                ca, cb = centroids[keys[i]], centroids[keys[j]]
                diff = cb - ca
                dist = np.linalg.norm(diff)

                if dist > max_distance:
                    continue

                dx, dy, dz = diff
                abs_dx, abs_dy, abs_dz = abs(dx), abs(dy), abs(dz)

                # Calcola la relazione dominante
                rel_ab = None
                if (
                    abs_dy > vertical_thresh
                    and abs_dy > (abs_dx + abs_dz) / horizontal_bias
                ):
                    rel_ab = "on_top_of" if dy > 0 else "beneath_of"
                elif abs_dx > abs_dz:
                    rel_ab = "left_of" if dx > 0 else "right_of"
                else:
                    rel_ab = "in_front_of" if dz < 0 else "behind"

                # Verifica la validità semantica per entrambi gli oggetti
                # (obj_b è il "subject", obj_a è l'"object" nella relazione)
                if not is_relation_valid(obj_a["label"], rel_ab):
                    continue

                relations.append(
                    {
                        "subject": obj_a["label"],
                        "object": obj_b["label"],
                        "relation": rel_ab,
                        "distance_m": round(float(dist), 3),
                    }
                )

        return relations

    def compute_object_size(self, obj_entry: dict[str, any]) -> float:
        """
        Return approximate linear size (in meters) of an object based on its bounding box.
        Works with bbox_world from extract_visible_objects.
        """
        bbox = obj_entry.get("bbox_world")
        if not bbox or len(bbox) != 2:
            return 0.0
        vmin, vmax = np.array(bbox[0]), np.array(bbox[1])
        dims = np.abs(vmax - vmin)  # [dx, dy, dz] in meters
        volume = float(np.prod(dims))
        linear_size = float(np.cbrt(volume))  # cubic root gives a single scalar size
        return linear_size

    def _normalize_label(self, label: str) -> str:
        """Normalize object labels to handle synonyms and groupings."""
        l = label.strip().lower()
        if l in {
            "wall",
            "ceiling",
            "floor",
            "window frame",
            "door frame",
            "frame",
            "unknown",
        }:
            return l

        if l in {"door", "doorway", "door frame", "attic door"}:
            return "doorway"

        if l in {"stairs", "stair", "step", "stairway"}:
            return "staircase"

        return l

    def _is_informative(
        self,
        label_norm: str,
        *,
        mode: str = "blacklist",
        blacklist=None,
        whitelist=None,
    ) -> bool:
        """
        mode="blacklist": keep everything except the blacklist
        mode="whitelist": keep only the whitelist
        """
        if mode not in {"blacklist", "whitelist"}:
            mode = "blacklist"
        if mode == "blacklist":
            blacklist = set(
                blacklist
                or [
                    "wall",
                    "floor",
                    "ceiling",
                    "frame",
                    "window frame",
                    "unknown",
                    "ceiling_light",
                    "light",
                    "lamp",
                ]
            )
            return label_norm not in blacklist
        else:
            whitelist = set(
                whitelist
                or [
                    "doorway",
                    "staircase",
                    "elevator",
                    "escalator",
                    "corridor",
                    "intersection",
                    "railing",
                    "exit_sign",
                    "sign",
                    "sofa",
                    "table",
                    "wardrobe",
                    "balcony",
                    "bridge",
                ]
            )
            return label_norm in whitelist

    def _xy_dist(self, a, b):
        # ground-plane distance using world x,z (index 0 and 2)
        return math.hypot(a[0] - b[0], a[2] - b[2])

    def _merge_aabbs(self, aabb_a, aabb_b):
        # each aabb: [[xmin,ymin,zmin],[xmax,ymax,zmax]]
        return [
            [
                min(aabb_a[0][0], aabb_b[0][0]),
                min(aabb_a[0][1], aabb_b[0][1]),
                min(aabb_a[0][2], aabb_b[0][2]),
            ],
            [
                max(aabb_a[1][0], aabb_b[1][0]),
                max(aabb_a[1][1], aabb_b[1][1]),
                max(aabb_a[1][2], aabb_b[1][2]),
            ],
        ]

    def _cluster_same_label(self, instances, distance_thresh=1.0):
        """
        Greedy clustering per label on ground plane.
        - instances: list of dicts with keys:
            label_norm, centroid_world, bbox_world, pixel_count, pixel_percent, distance_from_camera, raw_ids (set)
        - distance_thresh: meters (tune per scene scale; 0.8~1.5 works well)

        Returns list of merged clusters for that label.
        """
        clusters = []  # each: dict like instances, aggregated

        # Sort big to small so large areas seed clusters
        instances_sorted = sorted(
            instances, key=lambda x: x["pixel_count"], reverse=True
        )

        for inst in instances_sorted:
            assigned = False
            for cl in clusters:
                if (
                    self._xy_dist(inst["centroid_world"], cl["centroid_world"])
                    <= distance_thresh
                ):
                    # merge into cluster (weighted by pixel_count)
                    w_old = cl["pixel_count"]
                    w_new = inst["pixel_count"]
                    w_sum = w_old + w_new

                    # weighted centroid (world)
                    cx = (
                        cl["centroid_world"][0] * w_old
                        + inst["centroid_world"][0] * w_new
                    ) / w_sum
                    cy = (
                        cl["centroid_world"][1] * w_old
                        + inst["centroid_world"][1] * w_new
                    ) / w_sum
                    cz = (
                        cl["centroid_world"][2] * w_old
                        + inst["centroid_world"][2] * w_new
                    ) / w_sum
                    cl["centroid_world"] = [cx, cy, cz]

                    # choose min distance to camera (useful for narration)
                    cl["distance_from_camera"] = min(
                        cl["distance_from_camera"], inst["distance_from_camera"]
                    )

                    # union bbox
                    cl["bbox_world"] = self._merge_aabbs(
                        cl["bbox_world"], inst["bbox_world"]
                    )

                    # accumulate pixels
                    cl["pixel_count"] = w_sum
                    cl["pixel_percent"] += inst["pixel_percent"]

                    # track raw semantic ids merged
                    cl["raw_ids"].update(inst["raw_ids"])
                    assigned = True
                    break

            if not assigned:
                clusters.append(
                    {
                        **inst,
                        "raw_ids": set(inst["raw_ids"]),  # ensure it’s a fresh set
                    }
                )

        return clusters

    def postprocess_visible_objects(
        self,
        visible_objects: dict,
        *,
        pixel_percent_min=0.02,
        mode="blacklist",
        blacklist=None,
        whitelist=None,
        per_label_cluster_thresh_m=1.0,
        top_k_per_label=None,
        recompute_relations=True,
    ):
        """
        - visible_objects: the dict produced by extract_visible_objects()["visible_objects"]
        Returns:
            {
            "objects": { "<uid>": {...} },
            "spatial_relations": [ ... ]   # recomputed on deduped set (if requested)
            }
        """
        # 1) Normalizza + filtro base
        buckets = {}  # label_norm -> list[instance]
        for raw_id, inst in visible_objects.items():
            label_norm = self._normalize_label(inst["label"])
            if not self._is_informative(
                label_norm, mode=mode, blacklist=blacklist, whitelist=whitelist
            ):
                continue
            if inst.get("pixel_percent", 0.0) < pixel_percent_min:
                continue

            buckets.setdefault(label_norm, []).append(
                {
                    "label": inst["label"],
                    "label_norm": label_norm,
                    "pixel_count": inst["pixel_count"],
                    "pixel_percent": inst.get("pixel_percent", 0.0),
                    "centroid_world": inst["centroid_world"],
                    "bbox_world": inst.get("bbox_world"),  # opzionale
                    "centroid_cam": inst["centroid_cam"],
                    "distance_from_camera": inst["distance_from_camera"],
                    "linear_size": float(inst.get("linear_size", 0.0)),  # <<— NEW
                    "raw_ids": {raw_id},
                }
            )

        # 2) Clustering per label
        dedup_list = []
        for label_norm, insts in buckets.items():
            clusters = self._cluster_same_label(
                insts, distance_thresh=per_label_cluster_thresh_m
            )

            # (opzionale) top-K cluster per label
            if top_k_per_label is not None and len(clusters) > top_k_per_label:
                clusters = sorted(
                    clusters, key=lambda x: x["pixel_count"], reverse=True
                )[:top_k_per_label]

            dedup_list.extend(clusters)

        # 3) Build oggetti unici + stima linear_size di cluster
        objects = {}
        per_label_counts = {}
        for cl in dedup_list:
            cnt = per_label_counts.get(cl["label_norm"], 0) + 1
            per_label_counts[cl["label_norm"]] = cnt
            uid = f"{cl['label_norm']}_{cnt:02d}"

            # linear_size del cluster: media pesata per pixel_count sulle raw_ids
            # (robusta anche se _cluster_same_label non propaga linear_size)
            raw_ids = list(cl.get("raw_ids", []))
            if raw_ids:
                sizes, weights = [], []
                for rid in raw_ids:
                    v = visible_objects.get(rid, {})
                    sz = float(v.get("linear_size", 0.0))
                    if sz > 0:
                        sizes.append(sz)
                        weights.append(float(v.get("pixel_count", 1)))
                cluster_linear_size = (
                    float(np.average(sizes, weights=weights))
                    if sizes
                    else float(cl.get("linear_size", 0.0))
                )
            else:
                cluster_linear_size = float(cl.get("linear_size", 0.0))

            objects[uid] = {
                "label": cl["label_norm"],
                "pixel_count": int(cl["pixel_count"]),
                "pixel_percent": float(cl["pixel_percent"]),
                "centroid_world": [float(v) for v in cl["centroid_world"]],
                "bbox_world": cl.get("bbox_world"),
                "centroid_cam": cl["centroid_cam"],
                "distance_from_camera": float(cl["distance_from_camera"]),
                "linear_size": cluster_linear_size,  # <<— NEW
                "merged_raw_ids": sorted(list(cl["raw_ids"])),
            }

        # 4) Relazioni ricalcolate sul set deduplicato
        relations = (
            self.compute_spatial_relations(objects) if recompute_relations else []
        )

        # 5) Ordinamento: salienza primaria pixel_count, secondaria grandezza
        objects = dict(
            sorted(
                objects.items(),
                key=lambda item: (
                    item[1]["pixel_count"],
                    item[1].get("linear_size", 0.0),
                ),
                reverse=True,
            )
        )

        return {"objects": objects, "spatial_relations": relations}

    def shortest_path(self, sim, goal: mn.Vector3):  # TODO REMOVE ALL THE PRINTS
        if not sim.pathfinder.is_loaded:
            print("Pathfinder not initialized, aborting.")
        else:
            seed = 4  # 4  # @param {type:"integer"}
            sim.pathfinder.seed(seed)

            agent_state = sim.get_agent(self.agent_id).get_state()
            initial_agent_state_position = agent_state.position
            initial_agent_state_rotation = agent_state.rotation

            path = habitat_sim.ShortestPath()
            path.requested_start = mn.Vector3(initial_agent_state_position)
            path.requested_end = goal
            found_path = sim.pathfinder.find_path(path)
            path_points = path.points

            print("Path found : " + str(found_path))
            print("Start : " + str(path.requested_start))
            print("Goal : " + str(path.requested_end))
            print("Path points : " + str(path_points))

            path_points = self.densify_path(path_points, step_size=3.0)

            save_images = False

            output_dir = "output"
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            if found_path:
                if save_images:
                    meters_per_pixel = 0.025
                    height = sim.scene_aabb.y().min

                    top_down_map = maps.get_topdown_map(
                        sim.pathfinder, height, meters_per_pixel=meters_per_pixel
                    )
                    recolor_map = np.array(
                        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
                    )
                    top_down_map = recolor_map[top_down_map]
                    grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
                    # convert world trajectory points to maps module grid points
                    trajectory = [
                        maps.to_grid(
                            path_point[2],
                            path_point[0],
                            grid_dimensions,
                            pathfinder=sim.pathfinder,
                        )
                        for path_point in path_points
                    ]
                    grid_tangent = mn.Vector2(
                        trajectory[1][1] - trajectory[0][1],
                        trajectory[1][0] - trajectory[0][0],
                    )
                    path_initial_tangent = grid_tangent / grid_tangent.length()
                    initial_angle = math.atan2(
                        path_initial_tangent[0], path_initial_tangent[1]
                    )
                    # draw the agent and trajectory on the map
                    maps.draw_path(top_down_map, trajectory)
                    maps.draw_agent(
                        top_down_map, trajectory[0], initial_angle, agent_radius_px=8
                    )
                    # print("\nDisplay the map with agent and path overlay:")
                    self.display_map(top_down_map)

                # @markdown 4. (optional) Place agent and render images at trajectory points (if found).
                display_path_agent_renders = True  # @param{type:"boolean"}
                if display_path_agent_renders:
                    # print("Rendering observations at path points:")
                    tangent = path_points[1] - path_points[0]
                    agent_state = habitat_sim.AgentState()
                    for ix, point in enumerate(path_points):
                        if ix < len(path_points) - 1:
                            tangent = path_points[ix + 1] - point
                            agent_state.position = point
                            tangent_orientation_matrix = mn.Matrix4.look_at(
                                point, point + tangent, np.array([0, 1.0, 0])
                            )
                            tangent_orientation_q = mn.Quaternion.from_matrix(
                                tangent_orientation_matrix.rotation()
                            )
                            agent_state.rotation = utils.quat_from_magnum(
                                tangent_orientation_q
                            )

                            agent = sim.get_agent(self.agent_id)
                            agent.set_state(agent_state)

                            observations = sim.get_sensor_observations()

                            # use get with default None to safely handle missing sensors
                            rgb = observations.get("color_sensor", None)
                            semantic = observations.get("semantic_sensor", None)
                            depth = observations.get("depth_sensor", None)

                            if rgb is not None:
                                if save_images:
                                    # Save RGB/semantic preview as before
                                    if semantic is not None:
                                        self.display_sample(
                                            rgb_obs=rgb, semantic_obs=semantic
                                        )
                                    else:
                                        self.display_sample(rgb_obs=rgb)

                                # Extract visible objects + relations
                                frame_meta = self.extract_visible_objects(
                                    sim, observations
                                )
                                if frame_meta is not None:
                                    dedup = self.postprocess_visible_objects(
                                        frame_meta["visible_objects"],
                                        pixel_percent_min=0.02,
                                        mode="blacklist",
                                        per_label_cluster_thresh_m=1.2,
                                        top_k_per_label=3,
                                        recompute_relations=True,
                                    )

                                    sensor_state = (
                                        sim.get_agent(0)
                                        .get_state()
                                        .sensor_states["color_sensor"]
                                    )
                                    rot_mn = utils.quat_to_magnum(sensor_state.rotation)
                                    T_world_sensor = mn.Matrix4.from_(
                                        rot_mn.to_matrix(), sensor_state.position
                                    )

                                    frame_data = {
                                        "scene_index": sim.curr_scene_name,
                                        "image_index": f"frame-{ix:06d}",
                                        "scene_pose": np.array(T_world_sensor).tolist(),
                                        "objects": dedup["objects"],  # deduped version
                                        "spatial_relations": dedup["spatial_relations"],
                                        "timestamp": datetime.datetime.now().isoformat(),
                                    }

                                    with open(f"output/frame_{ix:06d}.json", "w") as f:
                                        json.dump(frame_data, f, indent=2)
                                        print(
                                            f"✅ Saved metadata: output/frame_{ix:06d}.json"
                                        )
                            else:
                                print("No color sensor found in observations.")

            agent_state.position = initial_agent_state_position
            agent_state.rotation = initial_agent_state_rotation
            agent = sim.get_agent(self.agent_id)
            agent.set_state(agent_state)

    def print_scene_semantic_info(self) -> None:
        scene = self.sim.semantic_scene
        if scene is not None:
            print(
                f"Scene has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
            )

            for region in scene.regions:
                print(f"\nRegion id:{region.id}" f" center:{region.aabb.center}")
                for obj in region.objects:
                    print(
                        f"\tObject id:{obj.id}, category:{obj.category.name()},"
                        f" center:{self.compute_xyz_center(obj.aabb)}"
                    )

    def compute_xyz_center(self, obj_aabb):
        # ottieni i vertici min e max dell'AABB
        vmin = (
            obj_aabb.min() if callable(getattr(obj_aabb, "min", None)) else obj_aabb.min
        )
        vmax = (
            obj_aabb.max() if callable(getattr(obj_aabb, "max", None)) else obj_aabb.max
        )

        # accedi per indice (funziona sia per Vector3 di Magnum che per array-like)
        cx = 0.5 * (float(vmin[0]) + float(vmax[0]))
        cy = 0.5 * (float(vmin[1]) + float(vmax[1]))
        cz = 0.5 * (float(vmin[2]) + float(vmax[2]))
        return cx, cy, cz

    def _get_bbox_color(self, obj_id: int) -> mn.Color4:
        """
        Returns a consistent color for the given semantic object id.
        """
        if obj_id in self._object_bbox_colors:
            return self._object_bbox_colors[obj_id]

        palette = d3_40_colors_rgb
        palette_len = len(palette)
        if palette_len == 0:
            color = mn.Color4(1.0, 0.0, 0.0, 1.0)
            self._object_bbox_colors[obj_id] = color
            return color
        try:
            idx = int(obj_id) % palette_len
        except (ValueError, TypeError):
            idx = hash(str(obj_id)) % palette_len
        rgb = palette[idx] / 255.0
        color = mn.Color4(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
        self._object_bbox_colors[obj_id] = color
        return color

    def _project_to_screen(self, world_point: mn.Vector3) -> Optional[mn.Vector2]:
        """
        Project a world-space point to screen pixel coordinates.
        """
        if self.render_camera is None:
            return None

        cam = self.render_camera.render_camera
        proj = cam.projection_matrix
        cam_mat = cam.camera_matrix

        # Transform point from world -> camera -> clip space
        clip = proj.transform_point(cam_mat.transform_point(world_point))

        # clip is already divided by w; no need to access clip.w
        ndc = clip  # Normalized device coordinates (-1..1)

        if not (-1.0 <= ndc.x <= 1.0 and -1.0 <= ndc.y <= 1.0 and -1.0 <= ndc.z <= 1.0):
            return None

        framebuffer = mn.Vector2(self.framebuffer_size)
        x = (ndc.x * 0.5 + 0.5) * framebuffer[0]
        y = (1.0 - (ndc.y * 0.5 + 0.5)) * framebuffer[1]
        return mn.Vector2(x, y)

    def _draw_bbox_label_overlay(self) -> None:
        if not self._bbox_label_screen_positions:
            return

        mn.gl.Renderer.enable(mn.gl.Renderer.Feature.BLENDING)
        self.shader.bind_vector_texture(self.glyph_cache.texture)

        framebuffer = mn.Vector2(self.framebuffer_size)
        for label, screen_pos in self._bbox_label_screen_positions:
            if not label:
                continue

            label_renderer = text.Renderer2D(
                self.display_font,
                self.glyph_cache,
                BaseViewer.DISPLAY_FONT_SIZE,
                text.Alignment.TOP_CENTER,
            )
            label_renderer.reserve(len(label))
            label_renderer.render(label)

            transform = mn.Matrix3.projection(framebuffer) @ mn.Matrix3.translation(
                screen_pos
            )
            self.shader.transformation_projection_matrix = transform
            self.shader.color = mn.Color4(1.0, 1.0, 1.0, 1.0)
            self.shader.draw(label_renderer.mesh)

        mn.gl.Renderer.disable(mn.gl.Renderer.Feature.BLENDING)

    def _draw_object_bboxes(self, debug_line_render: Any) -> None:
        """
        Draw axis-aligned bounding boxes for every semantic object.
        """
        scene = self.sim.semantic_scene
        if scene is None:
            return

        self._bbox_label_screen_positions.clear()
        debug_line_render.set_line_width(2.5)
        max_boxes = 20
        target_labels = {"wall clock", "sofa", "armchair", "couch"}
        candidates = []

        for obj in scene.objects:
            label = ""
            if obj.category is not None and hasattr(obj.category, "name"):
                label = obj.category.name()
            label_norm = label.strip().lower()
            if label_norm not in target_labels:
                continue

            if not label:
                label = f"object_{obj.id}"

            obb = getattr(obj, "obb", None)
            if obb is None:
                continue

            center = mn.Vector3(obb.center)
            half_extents = mn.Vector3(obb.half_extents)
            rot_vec = mn.Vector4(obb.rotation)
            rotation = mn.Quaternion(
                mn.Vector3(rot_vec[0], rot_vec[1], rot_vec[2]), rot_vec[3]
            )

            # Precompute the eight OBB corners in world space.
            corner_offsets = [
                mn.Vector3(-half_extents[0], -half_extents[1], -half_extents[2]),
                mn.Vector3(half_extents[0], -half_extents[1], -half_extents[2]),
                mn.Vector3(-half_extents[0], half_extents[1], -half_extents[2]),
                mn.Vector3(half_extents[0], half_extents[1], -half_extents[2]),
                mn.Vector3(-half_extents[0], -half_extents[1], half_extents[2]),
                mn.Vector3(half_extents[0], -half_extents[1], half_extents[2]),
                mn.Vector3(-half_extents[0], half_extents[1], half_extents[2]),
                mn.Vector3(half_extents[0], half_extents[1], half_extents[2]),
            ]
            corners = [
                rotation.transform_vector(offset) + center for offset in corner_offsets
            ]

            volume = max(8.0 * half_extents[0] * half_extents[1] * half_extents[2], 0.0)
            candidates.append(
                (volume, obj.id, label, corners, center, rotation, half_extents)
            )

        candidates.sort(key=lambda item: item[0], reverse=True)

        edges = [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ]

        for (
            volume,
            obj_id,
            label,
            corners,
            center,
            rotation,
            half_extents,
        ) in candidates[:max_boxes]:
            color = self._get_bbox_color(obj_id)

            for edge in edges:
                start = corners[edge[0]]
                end = corners[edge[1]]
                debug_line_render.draw_transformed_line(
                    start,
                    end,
                    color,
                )

            top_center = (
                center
                + rotation.transform_vector(mn.Vector3(0.0, half_extents[1], 0.0))
                + mn.Vector3(0.0, 0.05, 0.0)
            )
            screen_pos = self._project_to_screen(top_center)
            if screen_pos is not None:
                self._bbox_label_screen_positions.append((label, screen_pos))

    def debug_draw(self):
        """
        Additional draw commands to be called during draw_event.
        """
        super().debug_draw()
        if self.show_object_bboxes:
            self._draw_object_bboxes(self.debug_line_render)
        else:
            self._bbox_label_screen_positions.clear()

    def draw_event(
        self,
        simulation_call: Optional[Callable] = None,
        global_call: Optional[Callable] = None,
        active_agent_id_and_sensor_name: Tuple[int, str] = (0, "color_sensor"),
    ) -> None:
        """
        Calls continuously to re-render frames and swap the two frame buffers
        at a fixed rate.
        """
        if self.q_app:
            self.q_app.processEvents()

        agent_acts_per_sec = self.fps

        mn.gl.default_framebuffer.clear(
            mn.gl.FramebufferClear.COLOR | mn.gl.FramebufferClear.DEPTH
        )

        # Agent actions should occur at a fixed rate per second
        self.time_since_last_simulation += Timer.prev_frame_duration
        num_agent_actions: int = self.time_since_last_simulation * agent_acts_per_sec
        self.move_and_look(int(num_agent_actions))

        # Occasionally a frame will pass quicker than 1/60 seconds
        if self.time_since_last_simulation >= 1.0 / self.fps:
            if self.simulating or self.simulate_single_step:
                self.sim.step_world(1.0 / self.fps)
                self.simulate_single_step = False
                if simulation_call is not None:
                    simulation_call()
            if global_call is not None:
                global_call()

            # reset time_since_last_simulation, accounting for potential overflow
            self.time_since_last_simulation = math.fmod(
                self.time_since_last_simulation, 1.0 / self.fps
            )

        if self.enable_batch_renderer:
            self.render_batch()
        else:
            agent_sensors = self.sim._Simulator__sensors[self.agent_id]
            color_sensor = None
            if isinstance(agent_sensors, dict):
                color_sensor = agent_sensors.get("color_sensor")
            else:
                try:
                    color_sensor = agent_sensors["color_sensor"]  # type: ignore[index]
                except Exception:
                    if isinstance(agent_sensors, (list, tuple)):
                        for sensor in agent_sensors:
                            spec_fn = getattr(sensor, "specification", None)
                            spec = spec_fn() if callable(spec_fn) else None
                            if (
                                spec is not None
                                and getattr(spec, "sensor_type", None)
                                == habitat_sim.SensorType.COLOR
                            ):
                                color_sensor = sensor
                                break
            if color_sensor is None:
                logger.error("Color sensor not available; skipping draw.")
                return
            color_sensor.draw_observation()
            agent = self.sim.get_agent(self.agent_id)
            self.render_camera = agent.scene_node.node_sensor_suite.get("color_sensor")
            self.debug_draw()
            self.render_camera.render_target.blit_rgba_to_default()

        # draw CPU/GPU usage data and other info to the app window
        mn.gl.default_framebuffer.bind()
        self.draw_text(self.render_camera.specification())
        self._draw_bbox_label_overlay()

        self.swap_buffers()
        Timer.next_frame()
        self.redraw()

    def move_and_look(self, repetitions: int) -> None:
        """
        This method is called continuously with `self.draw_event` to monitor
        any changes in the movement keys map `Dict[KeyEvent.key, Bool]`.
        When a key in the map is set to `True` the corresponding action is taken.
        """
        super().move_and_look(repetitions)

        self._process_queued_actions()  # process any queued actions from the other thread

    def print_agent_state(self) -> None:
        """
        Logs the current agent's position and rotation in the simulator.
        """
        agent = self.sim.get_agent(self.agent_id)
        agent_state = agent.get_state()
        print(f"Agent State: pos: {agent_state.position}, rot: {agent_state.rotation}")

    def get_object_position(
        self, object_name: Optional[str], room_name: Optional[str]
    ) -> Optional[mn.Vector3]:
        """
        IMPORTANT: This function uses the data FROM THE SIMULATION! It only takes the name and position for each region inside the json file.
        Get the centroid position of an object or region from its category name.
        """
        scene = self.sim.semantic_scene
        if scene is None:
            print("Error: Semantic scene is not loaded.")
            return None

        for region in scene.regions:
            region_id = region.id.strip("_").lower() if region and region.id else ""
            region_meta = self.map_room_id_to_name.get(region_id, {})
            region_name = region_meta.get("name", region.id)
            region_pos = mn.Vector3(region_meta.get("position", (-1, -1, -1)))

            region_name_l = region_name.lower()

            # Case 1: user wants the region (room) position
            if object_name is None and room_name and region_name_l == room_name.lower():
                return region_pos

            # Case 2: user wants a specific object in a specific room
            if object_name and room_name and region_name_l != room_name.lower():
                continue  # Skip regions not matching the target room

            # Case 3: search for the object (in matching room or in all rooms)
            if object_name:
                for obj in region.objects:
                    if (
                        obj
                        and obj.category
                        and obj.category.name().lower() == object_name.lower()
                    ):
                        return mn.Vector3(obj.obb.center)

        return None

    def check_object_in_room(
        self, object_name: Optional[str], room_name: Optional[str]
    ) -> bool:
        """
        Verifies whether the given object exists in the given room.
        If room_name is None, always returns False.
        If object_name is None, returns True if room_name exists.
        """

        # * JSON already loaded (under self.room_objects_occurences)
        if room_name is None:
            return False

        room_name = room_name.strip().lower()
        object_name = object_name.strip().lower() if object_name else None

        # Check if room exists
        if room_name not in self.room_objects_occurences:
            return False

        # Case: only checking if room exists
        if object_name is None:
            return True

        # Check if object exists in that room
        room_objects = self.room_objects_occurences[room_name]
        return object_name in (obj.lower() for obj in room_objects.keys())

    def key_press_event(self, event: Application.KeyEvent) -> None:
        """
        Handles `Application.KeyEvent` on a key press by performing the corresponding functions.
        If the key pressed is part of the movement keys map `Dict[KeyEvent.key, Bool]`, then the
        key will be set to False for the next `self.move_and_look()` to update the current actions.
        """
        key = event.key
        pressed = Application.Key
        mod = Application.Modifier

        # warning: ctrl doesn't always pass through with other key-presses
        if key == pressed.B:
            self.show_object_bboxes = not self.show_object_bboxes
            state = "enabled" if self.show_object_bboxes else "disabled"
            logger.info(f"Object bounding boxes {state}.")

        super().key_press_event(event)

    def get_response_LLM(self, user_input):
        # This function calls an LLM to retrieve the landmark / room information from the Human Request
        if client is None:
            print("OpenAI client not initialized. Cannot get landmark room.")
            return None

        # OpenAI is loaded
        prompt = """
        You are an assistant for a home navigation system.
        You are given a dictionary describing the house structure: each key is a room name, and each value is a dictionary of landmarks (objects) with their number of occurrences in that room.

        Your task is to interpret natural language queries from the user who might:
        - ask to go to a room, or
        - ask where to find an object.

        Users may use synonyms or similar terms (for example: "clock" = "wall clock", "toy" = "plush toy", etc.). You must identify such equivalences before deciding your answer.

        When responding, follow these rules strictly:

        1. If the user mentions an object inside a room and there is one occurrence of that object in that room, respond with the name of the room and the object.
        Example: "1. bathroom_4, door"

        2. If the user requests a room and it exists in the dictionary, respond only with the exact name of the room.
        Example: "2. kitchen"

        3. If the object appears in multiple rooms, ask in which room it is located, listing all rooms that contain it.
        Example: "3. The object appears in multiple rooms, do you mean the one in kitchen or in dining_room?"

        4. If the user mentions a room, but there are multiple rooms of that type (e.g. "bedroom", "bathroom"), ask in which room to go, listing all possible candidates.
        Example: "4. There are multiple bathrooms, do you mean bathroom_1, bathroom_2, bathroom_3 or bathroom_4?"

        5. If an object appears multiple times in the same room but not in other rooms, respond with the name of the room.
        Example: "5. bedroom_1"

        6. If the user mentions a room and an object that appears multiple times in the room, respond with the name of the room.
        Example: "6. laundry_room"

        7. If no match or synonym is found, say you couldn't find the object and ask for more details.
        Example: "7. I couldn't find the object. Can you describe it or specify where it might be located?"

        8. If the user conversates without asking for a room or object, respond in a friendly manner.

        Output format rule: Always respond in the format
        <rule number>. <response text>
        and nothing else.
        """

        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": user_input + "\n" + str(self.room_objects_occurences),
            },
        ]

        response = client.chat.completions.create(model="gpt-4o", messages=messages)
        print("Response from GPT:")
        print(response.choices[0].message.content)
        return response.choices[0].message.content


def get_goal_from_response(response: str) -> object:
    response_list = response.split(".", 1)
    if len(response_list) < 2:
        raise ValueError(f"Invalid LLM response format: {response}")
    rule_number = int(response_list[0].strip())
    content = response_list[1].strip()

    if rule_number == 1:
        try:
            room, obj = map(str.strip, content.split(",", 1))
            return {"type": "object_in_room", "room": room, "object": obj}
        except ValueError:
            raise ValueError(f"Rule 1: Expected format 'room, object'. Got: {content}")
    elif rule_number == 2:
        # Format: "room_name"
        return {"type": "room_only", "room": content}
    elif rule_number == 3:
        # Object is ambiguous across rooms
        return {"type": "ambiguous_object_rooms", "message": content}
    elif rule_number == 4:
        # Room type is ambiguous
        return {"type": "ambiguous_room", "message": content}
    elif rule_number == 5:
        # Object repeated in one room only
        return {"type": "object_in_single_room", "room": content}
    elif rule_number == 6:
        # Object repeated in specified room
        return {"type": "object_repeated_in_room", "room": content}
    elif rule_number == 7:
        # No match found
        return {"type": "not_found", "message": content}
    elif rule_number == 8:
        # Friendly conversation
        return {"type": "friendly_conversation", "message": content}
    else:
        raise ValueError(f"Unexpected rule number: {rule_number}")


def user_input_logic_loop(
    viewer: NewViewer, input_q: queue.Queue, output_q: queue.Queue
):
    while True:
        try:
            user_input = input_q.get()
            print("Received user input:", user_input)
            if not user_input:
                continue

            # output_q.put("Processing your request...")

            response = viewer.get_response_LLM(user_input)  # * API Call to ChatGPT
            print("Response from ChatGPT: ", response)
            goal_info = get_goal_from_response(
                response
            )  # * Handle response and distinguish cases
            print("Handled Response: ", goal_info)
            response = response.split(".", 1)[
                1
            ].strip()  # Remove numbering from response for user display
            res_type = goal_info["type"]

            if res_type == "object_in_room":
                target_name = goal_info["object"]
                room_name = goal_info["room"]
            elif (
                res_type == "room_only"
                or res_type == "object_in_single_room"
                or res_type == "object_repeated_in_room"
            ):
                target_name = None
                room_name = goal_info["room"]
            elif (
                res_type == "ambiguous_room"
                or res_type == "ambiguous_object_rooms"
                or res_type == "not_found"
                or res_type == "friendly_conversation"
            ):
                print(goal_info["message"])
                output_q.put(response)
                continue
            else:
                print(f"Unhandled goal type: {res_type}")

            # * === SANITY CHECK ===
            if not viewer.check_object_in_room(target_name, room_name):
                print(f"Sanity check failed: '{target_name}' not in '{room_name}'")
                continue
            else:
                print(f"Sanity check passed: '{target_name}' in '{room_name}'")

            # * Query scene (and retrieve a point in the 3D space)
            goal_pos = viewer.get_object_position(
                object_name=target_name, room_name=room_name
            )
            print(f"Navigating to: '{room_name}/{target_name}' at position {goal_pos}")

            if goal_pos is None:
                print(f"Warning: '{room_name}/{target_name}' not found in the scene.")
                continue

            if goal_pos.y < 2.0:
                goal_pos.y = 0.163378  # Adjust height

            viewer.enqueue_shortest_path(goal_pos)
            # output_q.put(f"Generating navigation instructions...")
            time.sleep(0.3)

            ############ Generate Instruction ###############
            # print("Current working dir:", os.getcwd())
            input_dir = Path(os.getcwd()) / "output"
            instructions = generate_path_description(
                input_dir, user_input=user_input, model="gpt-4o", dry_run=False
            )
            print("\n--- GENERATED DESCRIPTION ---\n")
            print(instructions)
            output_q.put(instructions)

        except EOFError:
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # optional arguments
    parser.add_argument(
        "--scene",
        default="./data/scene_datasets/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb",
        type=str,
        help='scene/stage file to load (default: "./data/test_assets/scenes/simple_room.glb")',
    )
    parser.add_argument(
        "--dataset",
        default="./data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
        type=str,
        metavar="DATASET",
        help='dataset configuration file to use (default: "default")',
    )
    parser.add_argument(
        "--disable-physics",
        action="store_true",
        help="disable physics simulation (default: False)",
    )
    parser.add_argument(
        "--use-default-lighting",
        action="store_true",
        help="Override configured lighting to use default lighting for the stage.",
    )
    parser.add_argument(
        "--hbao",
        action="store_true",
        help="Enable horizon-based ambient occlusion, which provides soft shadows in corners and crevices.",
    )
    parser.add_argument(
        "--enable-batch-renderer",
        action="store_true",
        help="Enable batch rendering mode. The number of concurrent environments is specified with the num-environments parameter.",
    )
    parser.add_argument(
        "--num-environments",
        default=1,
        type=int,
        help="Number of concurrent environments to batch render. Note that only the first environment simulates physics and can be controlled.",
    )
    parser.add_argument(
        "--composite-files",
        type=str,
        nargs="*",
        help="Composite files that the batch renderer will use in-place of simulation assets to improve memory usage and performance. If none is specified, the original scene files will be loaded from disk.",
    )
    parser.add_argument(
        "--width",
        default=800,
        type=int,
        help="Horizontal resolution of the window.",
    )
    parser.add_argument(
        "--height",
        default=600,
        type=int,
        help="Vertical resolution of the window.",
    )

    args = parser.parse_args()

    if args.num_environments < 1:
        parser.error("num-environments must be a positive non-zero integer.")
    if args.width < 1:
        parser.error("width must be a positive non-zero integer.")
    if args.height < 1:
        parser.error("height must be a positive non-zero integer.")

    # Setting up sim_settings
    sim_settings: Dict[str, Any] = default_sim_settings
    sim_settings["scene"] = args.scene
    sim_settings["scene_dataset_config_file"] = args.dataset
    sim_settings["enable_physics"] = not args.disable_physics
    sim_settings["use_default_lighting"] = args.use_default_lighting
    sim_settings["enable_batch_renderer"] = args.enable_batch_renderer
    sim_settings["num_environments"] = args.num_environments
    sim_settings["composite_files"] = args.composite_files
    sim_settings["window_width"] = args.width
    sim_settings["window_height"] = args.height
    sim_settings["default_agent_navmesh"] = False
    sim_settings["enable_hbao"] = args.hbao
    sim_settings["semantic_sensor"] = True
    sim_settings["depth_sensor"] = True
    # start the application
    # HabitatSimInteractiveViewer(sim_settings).exec()

    input_from_gui_q = queue.Queue()
    output_to_gui_q = queue.Queue()

    # 1. Crea la GUI di Tkinter nel thread principale
    #    (ma non avviarla ancora con mainloop)
    q_app = QApplication(sys.argv or [])
    gui_window = create_gui(
        input_from_gui_q,
        output_to_gui_q,
        window_width=args.width * 3 // 5,
        window_height=args.height,
    )
    gui_window.show()

    viewer = NewViewer(sim_settings, q_app=q_app)

    logic_thread = threading.Thread(
        target=user_input_logic_loop,
        args=(viewer, input_from_gui_q, output_to_gui_q),
        daemon=True,
    )
    logic_thread.start()

    viewer.exec()

    sys.exit(q_app.exec_())
