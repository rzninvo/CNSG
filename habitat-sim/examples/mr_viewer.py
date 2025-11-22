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

from pathlib import Path
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
from viewer import (HabitatSimInteractiveViewer as BaseViewer, MouseMode, MouseGrabber, Timer)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.generate_description import generate_path_description
from utils.conversation_gui import *

from habitat.utils.visualizations import maps
from habitat_sim.utils.common import d3_40_colors_rgb

from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


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
    MOVE, LOOK = 0.07, 1.5  # New definition for these two attributes

    def __init__(self, sim_settings: Dict[str, Any], q_app: QApplication) -> None:
        scene_path = sim_settings["scene"]
        super().__init__(sim_settings)
        self.q_app = q_app
        self.objects = {}
        self.clusters_to_draw = None # List of 'Numbers' e.g. ['1', '645', ...]
        self.prev_objs_to_draw = None
        self.action_queue = queue.Queue()
        self.scene = self.sim.semantic_scene        

        # * Draw object bounding boxes when enabled
        self.show_object_bboxes = False
        self.show_room_bboxes = False
        self._object_bbox_colors: Dict[int, mn.Color4] = {}
        self._bbox_label_screen_positions: List[Tuple[str, mn.Vector2]] = []

        self.map_room_id_to_name = {}
        self.room_objects_occurences = {}

        # * Load semantic info files
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

        ignore_categories = ["ceiling", "floor", "wall", "handle", "window frame", "door frame", "frame", "unknown", "stairs", "staircase", "stair", "stairway"]
        semantic_info = self.get_semantic_info(
            semantic_path,
            map_room_id_to_name=self.map_room_id_to_name,
            ignore_categories=ignore_categories,
        )

        self.room_objects_occurences = semantic_info

        self.objects = self.get_objs_from_sim()
        self.cluster_cnt = 0
        self.clusters = self.cluster_objs(distance_thresh=0.5)
        self.rooms = self.get_rooms_from_sim()


        # self.print_scene_semantic_info()


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

    def print_scene_semantic_info(self) -> None:
        scene = self.sim.semantic_scene
        if scene is not None:
            print(f"Scene has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects")

            for region in scene.regions:
                print(f"\nRegion id:{region.id}" f" center:{region.aabb.center}")
                for obj in region.objects:
                    print(f"\tObject id:{obj.id}, category:{obj.category.name()}," f" center:{self.compute_xyz_center(obj.obb)}")

    def compute_xyz_center(self, obj_obb):
        # ottieni i vertici min e max dell'OBB
        vmin = (obj_obb.min() if callable(getattr(obj_obb, "min", None)) else obj_obb.min)
        vmax = (obj_obb.max() if callable(getattr(obj_obb, "max", None)) else obj_obb.max)

        # accedi per indice (funziona sia per Vector3 di Magnum che per array-like)
        cx = 0.5 * (float(vmin[0]) + float(vmax[0]))
        cy = 0.5 * (float(vmin[1]) + float(vmax[1]))
        cz = 0.5 * (float(vmin[2]) + float(vmax[2]))
        return cx, cy, cz
    

    def get_objs_from_sim(self):
        objects = {}
        objs_rooms = {}
        rooms_floors = {} # 
        rooms_heights = []
        for region in self.scene.regions:
            region_id = region.id.strip("_").lower() if region and region.id else ""
            room_name = self.map_room_id_to_name.get(region_id, {}).get("name", "unknown_room")
            if "unknown" in room_name.lower():
                continue 
            room_height = self.map_room_id_to_name.get(region_id, {})["position"][1]
            rooms_heights.append(room_height)
            for obj in region.objects:
                objs_rooms[obj.id] = room_name
        
        # Remove duplicates from heights
        rooms_heights = list(set(rooms_heights))

        # Sort heights ascending
        rooms_heights.sort()

        # Assign floor level to each room based on height (and add at floors_rooms)
        for region in self.scene.regions:
            region_id = region.id.strip("_").lower() if region and region.id else ""
            room_name = self.map_room_id_to_name.get(region_id, {}).get("name", "unknown_room")
            if "unknown" in room_name.lower():
                continue

            room_height = self.map_room_id_to_name.get(region_id, {})["position"][1]
            # Get the floor number based on the index of the room_height in the room_heights list
            floor_number = rooms_heights.index(room_height)  # Floors start at 0
            # print(f"Room: {room_name}, Height: {room_height}, Floor number: {floor_number}")
            rooms_floors[room_name] = floor_number

        # ^ Debug print per rooms_floors
        # print("Rooms and their floor numbers:")
        # for room, floor in rooms_floors.items():
        #     print(f"Room: {room}, Floor: {floor}")

        if self.scene is not None:
            
            for sim_obj in self.scene.objects:
                if "unknown" in sim_obj.id.lower():
                    continue 
                room_name = objs_rooms.get(sim_obj.id, "unknown_room")
                if "unknown" in room_name.lower():
                    continue
                
                obb = getattr(sim_obj, "obb", None)

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
                    corners_world = [rotation.transform_vector(offset) + center for offset in corner_offsets]

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

                    objects[sim_obj.id] = {
                        "obj_str_id": sim_obj.id,
                        "obj_num_id": int(sim_obj.id.split("_")[-1]),
                        "sim_obj" : sim_obj,
                        "label": sim_obj.category.name() if sim_obj.category is not None else f"sim_object_{sim_obj.id}",
                        "centroid_world": centroid_world,
                        "bbox_world": bbox_world,
                        "linear_size": self.compute_object_size({"bbox_world": bbox_world}),
                        "room": room_name,
                        "floor_number": rooms_floors[room_name],
                    }
                
        return objects
    
    def get_rooms_from_sim(self):
        rooms = {}
        ignore_categories = ["ceiling", "floor", "wall", "handle", "window", "frame", "unknown", "stair"]

        rooms_heights = []
        for region in self.scene.regions:
        
            region_id = region.id.strip("_").lower() if region and region.id else ""
            room_name = self.map_room_id_to_name.get(region_id, {}).get("name", "unknown_room")
            if "unknown" in room_name.lower():
                continue 

            room_height = self.map_room_id_to_name.get(region_id, {})["position"][1]
            rooms_heights.append(room_height)
      

            # build the bbox of the room from its objects
            for obj in region.objects:
                obj_str_id = obj.id
                for cat in ignore_categories:
                    if cat in obj_str_id.lower():
                        obj_str_id = None
                        break
                if obj_str_id is None:
                    continue
                if obj_str_id not in self.objects:
                    continue

                obj_data = self.objects[obj_str_id]
                if "bbox_world" not in obj_data:
                    continue
                obj_bbox = obj_data["bbox_world"]

                diff_x = obj_bbox[1][0] - obj_bbox[0][0]
                diff_y = obj_bbox[1][1] - obj_bbox[0][1]
                diff_z = obj_bbox[1][2] - obj_bbox[0][2]
                volume = diff_x * diff_y * diff_z
                if volume == 0:
                    continue
                if "bbox_world" not in rooms.get(region_id, {}):
                    rooms[region_id] = {
                        "bbox_world": obj_bbox
                    }
                else:
                    
                    room_bbox = rooms[region_id]["bbox_world"]
                    # update room bbox to include obj bbox
                    room_bbox[0][0] = min(room_bbox[0][0], obj_bbox[0][0])
                    room_bbox[0][1] = min(room_bbox[0][1], obj_bbox[0][1])
                    room_bbox[0][2] = min(room_bbox[0][2], obj_bbox[0][2])
                    room_bbox[1][0] = max(room_bbox[1][0], obj_bbox[1][0])
                    room_bbox[1][1] = max(room_bbox[1][1], obj_bbox[1][1])
                    room_bbox[1][2] = max(room_bbox[1][2], obj_bbox[1][2])
                    rooms[region_id]["bbox_world"] = room_bbox


            rooms[region_id]["region_id"] = region_id
            rooms[region_id]["name"] = room_name

        rooms_heights = list(set(rooms_heights))
        # Sort heights ascending
        rooms_heights.sort()

        for region_id in rooms.keys():
            room_height = self.map_room_id_to_name.get(region_id, {})["position"][1]
            # Get the floor number based on the index of the room_height in the room_heights list
            floor_number = rooms_heights.index(room_height)  # Floors start at 0
            rooms[region_id]["floor_number"] = floor_number

        return rooms
    
    def cluster_objs(self, distance_thresh=1.0):
        buckets = {}  # {label_norm : list[obj_str_ids]
        for obj in self.objects.values():
            obj_str_id = obj.get("obj_str_id")
            label_norm = self._normalize_label(obj.get("label"))
            if not self._is_informative(label_norm, mode="blacklist"):
                continue

            buckets.setdefault(label_norm, []).append(obj_str_id)


        clusters_list = []
        for label_norm, obj_str_ids in buckets.items():
            current_clusters = self._cluster_same_label(obj_str_ids, label_norm, distance_thresh=distance_thresh)
            clusters_list.extend(current_clusters)

        clusters = {}
        for cluster in clusters_list:
            label = cluster["label"]
            cluster_str_id = f"{label}_{self.cluster_cnt}"
            cluster["cluster_str_id"] = cluster_str_id
            clusters[cluster_str_id] = cluster
            self.cluster_cnt += 1

        return clusters
    
    def _cluster_same_label(self, obj_str_ids, label_norm, distance_thresh=1.0):
        clusters = []
        
        def bbox_distance(bbox_a, bbox_b):
            """
            Calcola la distanza minima tra due bounding boxes 3D.
            Ogni bbox è nel formato [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            """
            # Estrai i punti min e max per entrambi i bbox
            min_a, max_a = bbox_a[0], bbox_a[1]
            min_b, max_b = bbox_b[0], bbox_b[1]
            
            # Calcola la distanza lungo ogni asse
            dx = max(0, max(min_a[0] - max_b[0], min_b[0] - max_a[0]))
            dy = max(0, max(min_a[1] - max_b[1], min_b[1] - max_a[1]))
            dz = max(0, max(min_a[2] - max_b[2], min_b[2] - max_a[2]))
            
            # Distanza euclidea 3D
            return math.sqrt(dx*dx + dy*dy + dz*dz)
        
        def merge_obbs(obb_a, obb_b):
            return [
                [
                    min(obb_a[0][0], obb_b[0][0]),
                    min(obb_a[0][1], obb_b[0][1]),
                    min(obb_a[0][2], obb_b[0][2]),
                ],
                [
                    max(obb_a[1][0], obb_b[1][0]),
                    max(obb_a[1][1], obb_b[1][1]),
                    max(obb_a[1][2], obb_b[1][2]),
                ]
            ]
        
        for obj_str_id in obj_str_ids:
            assigned = False
            obj = self.objects[obj_str_id]
            for cluster in clusters:
                # Usa la distanza tra bbox invece che tra centroidi
                if (bbox_distance(obj["bbox_world"], cluster["bbox_world"]) <= distance_thresh):
                    count_cluster = len(cluster["obj_str_ids"])
                    # centroid (world)
                    new_cx = (cluster["centroid_world"][0] * count_cluster + obj["centroid_world"][0]) / (count_cluster + 1)
                    new_cy = (cluster["centroid_world"][1] * count_cluster + obj["centroid_world"][1]) / (count_cluster + 1)
                    new_cz = (cluster["centroid_world"][2] * count_cluster + obj["centroid_world"][2]) / (count_cluster + 1)
                    cluster["centroid_world"] = [new_cx, new_cy, new_cz]

                    # union bbox
                    cluster["bbox_world"] = merge_obbs(cluster["bbox_world"], obj["bbox_world"])

                    cluster["linear_size"] += obj.get("linear_size") 
                    # track raw semantic ids merged
                    cluster["obj_str_ids"].append(obj_str_id) 
                    assigned = True
                    break

            if not assigned:
                clusters.append({"obj_str_ids": [obj_str_id],
                                 "label": label_norm,
                                "centroid_world": obj["centroid_world"],
                                "bbox_world": obj["bbox_world"], 
                                "linear_size": obj.get("linear_size", 0.0),
                                "room": obj["room"],
                                "floor_number": obj["floor_number"],
                                })
                                 
        return clusters
        

        


    def set_clusters_to_draw(self, clusters_to_draw: List[str]):
        self.clusters_to_draw = clusters_to_draw
        # print(f"Updated clusters_to_draw: {self.clusters_to_draw}")  


    def start_navigation(self, sim, target_name: str, room_name: str = "", user_input=None, output_q=None):
        goal_pos = viewer.get_object_position(object_name=target_name, room_name=room_name)
        print(f"Navigating to: '{room_name}/{target_name}' at position {goal_pos}")

        if goal_pos is None:
            print(f"Object '{target_name}' not found in the scene.")
            return

        if goal_pos.y < 2.0:
            goal_pos.y = 0.163378  # Adjust height
        frames = self.shortest_path(sim, goal_pos, target_name)

        if len(frames) == 0:
            goal_pos = viewer.get_object_position(object_name=None, room_name=room_name)
            print(f"Retrying navigation to room center at position {goal_pos}")
            if goal_pos is None:
                print(f"Room '{room_name}' not found in the scene.")
                return
            if goal_pos.y < 2.0:
                goal_pos.y = 0.163378  # Adjust height
            frames = self.shortest_path(sim, goal_pos, target_name)

        if len(frames) == 0:
            print("No path frames generated, aborting navigation.")
            return

        instructions, clusters_to_draw = generate_path_description(frames, user_input=user_input, model=_LOCAL_MODEL, tokenizer=_LOCAL_TOKENIZER, dry_run=False, target_name=target_name, room_name=room_name) # dry run = not llm_enabled # to allow instructions but not user input menagement
        self.set_clusters_to_draw(clusters_to_draw)

        print("\n--- GENERATED DESCRIPTION ---\n")
        print(instructions)
        output_q.put(instructions)
        




    def shortest_path(self, sim, goal: mn.Vector3, target_object: str = ""): 
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


            save_images = False

            output_dir = "output"
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            frames = []

            if found_path:
                path_points = self.densify_path(path_points, step_size=3.0)
                if save_images:
                    meters_per_pixel = 0.025
                    height = sim.scene_aabb.y().min

                    top_down_map = maps.get_topdown_map(sim.pathfinder, height, meters_per_pixel=meters_per_pixel)
                    recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
                    top_down_map = recolor_map[top_down_map]
                    grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
                    # convert world trajectory points to maps module grid points
                    trajectory = [maps.to_grid(path_point[2], path_point[0], grid_dimensions, pathfinder=sim.pathfinder) for path_point in path_points]
                    grid_tangent = mn.Vector2(trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0],)
                    path_initial_tangent = grid_tangent / grid_tangent.length()
                    initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
                    # draw the agent and trajectory on the map
                    maps.draw_path(top_down_map, trajectory)
                    maps.draw_agent(top_down_map, trajectory[0], initial_angle, agent_radius_px=8)
                    # print("\nDisplay the map with agent and path overlay:")
                    self.display_map(top_down_map)

                # @markdown 4. (optional) Place agent and render images at trajectory points (if found).
                display_path_agent_renders = True  # @param{type:"boolean"}
                if display_path_agent_renders:
                    for i, point in enumerate(path_points):
                        if i < len(path_points) - 1:
                            tangent = path_points[i + 1] - point
                            agent_state.position = point
                            tangent_orientation_matrix = mn.Matrix4.look_at(point, point + tangent, np.array([0.0, 1.0, 0]))
                            tangent_orientation_q = mn.Quaternion.from_matrix(tangent_orientation_matrix.rotation())
                            agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)

                            
                            # Compute initial angle for eventual "Turn to your right / left" instructions
                            # Need to get the angle between the initial_rotation and agent_state.rotation around Y axis
                            initial_rot_matrix = utils.quat_to_magnum(initial_agent_state_rotation).to_matrix()
                            tangent_rot_matrix = utils.quat_to_magnum(agent_state.rotation).to_matrix()
                            # Get forward vectors (row 2 is the forward direction in Magnum's convention)
                            initial_forward = mn.Vector3(initial_rot_matrix[2][0], initial_rot_matrix[2][1], initial_rot_matrix[2][2])
                            tangent_forward = mn.Vector3(tangent_rot_matrix[2][0], tangent_rot_matrix[2][1], tangent_rot_matrix[2][2])
                            # Project on XZ plane (ignore Y component)
                            initial_forward_proj = mn.Vector3(initial_forward[0], 0.0, initial_forward[2]).normalized()
                            tangent_forward_proj = mn.Vector3(tangent_forward[0], 0.0, tangent_forward[2]).normalized()
                            
                            # Calculate angle using dot product
                            dot_product = mn.math.dot(initial_forward_proj, tangent_forward_proj)
                            angle = math.acos(np.clip(dot_product, -1.0, 1.0))
                            
                            # Calculate cross product to determine turn direction (left or right)
                            cross_product = mn.math.cross(initial_forward_proj, tangent_forward_proj)
                            # If Y component of cross product is positive, turn is to the left; if negative, to the right
                            angle_deg = math.degrees(angle)
                            if 45.0 < angle_deg < 110.0:
                                if cross_product[1] > 0:
                                    turn_direction = "left"
                                else:
                                    turn_direction = "right"
                            elif angle_deg <= 45.0:
                                turn_direction = "forward"
                            else:
                                turn_direction = "behind"
                            
                            # print(f"\n\n[PAPERINO] Initial angle to first path segment: {math.degrees(angle):.2f} degrees ({turn_direction})\n\n")

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
                                        self.display_sample(rgb_obs=rgb, semantic_obs=semantic)
                                    else:
                                        self.display_sample(rgb_obs=rgb)

                                # Extract visible objects + relations
                                visible_objs = self.extract_visible_objs(sim, observations)
                                if visible_objs is not None:
                                    visible_clusters = self.cluster_visible_objs(visible_objs, pixel_percent_min=0.02, visible_objs_only=True)
                                    
                                    processed_visible_clusters = self.process_visible_clusters(visible_clusters, target_object)

                                    sensor_state = (
                                        sim.get_agent(self.agent_id)
                                        .get_state()
                                        .sensor_states["color_sensor"]
                                    )
                                    rot_mn = utils.quat_to_magnum(sensor_state.rotation)
                                    T_world_sensor = mn.Matrix4.from_(rot_mn.to_matrix(), sensor_state.position)

                                    # add a field in frame data with the room name where the agent is located in that frame
                                    current_room = self.get_room_from_position(agent_state.position) # TODO doesn't work -> try to fix it

                                    # print(f"[PLUTO] Oggetti: {processed_objs}")
                                    frame_data = {
                                        "scene_index": sim.curr_scene_name,
                                        "image_index": f"frame-{i:06d}",
                                        "scene_pose": np.array(T_world_sensor).tolist(),
                                        "current_room": current_room,
                                        "objects": processed_visible_clusters,  # NOTE we use "objects" even if they are clusters
                                        "spatial_relations": self.compute_spatial_relations(processed_visible_clusters, sim.get_agent(self.agent_id).get_state()),
                                        "timestamp": datetime.datetime.now().isoformat(),
                                        "turn_direction": turn_direction,
                                    }

                                    # save frame data in a list of frames
                                    frames.append(frame_data)

                                    # with open(f"output/frame_{i:06d}.json", "w") as f:
                                    #     json.dump(frame_data, f, indent=2)
                                    #     print(f"✅ Saved metadata: output/frame_{i:06d}.json")
                            else:
                                print("No color sensor found in observations.")
            else: 
                print("No path found to the goal.")
            agent_state.position = initial_agent_state_position
            agent_state.rotation = initial_agent_state_rotation
            agent = sim.get_agent(self.agent_id)
            agent.set_state(agent_state)

        return frames

    def get_room_from_position(self, position: mn.Vector3) -> str:
        """
        Trova tutte le stanze che contengono la posizione e seleziona quella più probabile
        basandosi sulla distanza dall'oggetto più vicino in quelle stanze.
        """
        candidate_rooms = []
        
        # Trova tutte le stanze candidate che contengono la posizione
        for room in self.rooms.values():
            bbox = room.get("bbox_world", None)
            if bbox is None:
                continue
            
            # Verifica se la posizione è dentro il bbox
            if (bbox[0][0] <= position[0] <= bbox[1][0] and
                bbox[0][1] <= position[1] <= bbox[1][1] and
                bbox[0][2] <= position[2] <= bbox[1][2]):
                
                candidate_rooms.append(room)
        
        if not candidate_rooms:
            return None 
        
        # Se c'è solo una stanza candidata, ritornala direttamente
        if len(candidate_rooms) == 1:
            return candidate_rooms[0]
        
        # Se ci sono più stanze candidate, trova quella con l'oggetto più vicino
        min_distance = float('inf')
        selected_room = None
        
        for room in candidate_rooms:
            room_name = room.get("name", "unknown_room")
            
            # Trova tutti gli oggetti in questa stanza
            for obj in self.objects.values():
                if obj.get("room") == room_name:
                    obj_centroid = obj.get("centroid_world")
                    if obj_centroid is None:
                        continue
                    
                    # Calcola la distanza dall'oggetto alla posizione
                    distance = np.linalg.norm(
                        np.array([position[0], position[1], position[2]]) - 
                        np.array(obj_centroid)
                    )
                    
                    # Aggiorna la stanza selezionata se questo oggetto è più vicino
                    if distance < min_distance:
                        min_distance = distance
                        selected_room = room
        
        return selected_room
        
        
    def densify_path(self, path_points, step_size=1.0, min_step_size=0.7):
        points = np.array(path_points)

        new_points = [points[0]]
        for i in range(1, len(points)):
            p0, p1 = points[i - 1], points[i]
            segment = p1 - p0
            dist = np.linalg.norm(segment)
            if dist == 0:
                continue
            if (dist <= step_size and np.linalg.norm(new_points[-1] - p1) > min_step_size):
                new_points.append(p1)
                continue
            direction = segment / dist
            n_steps = int(dist / step_size)
            for s in range(1, n_steps + 1):
                new_point = p0 + direction * step_size * s
                new_points.append(new_point)
        if np.linalg.norm(new_points[-1] - points[-1]) > 0.3: 
            new_points.append(points[-1])
        if len(new_points) < 2:
            new_points.append(points[-1])
        return np.array(new_points)

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
        plt.savefig(f"output/topdown_map{self.topdown_map_counter}.png", bbox_inches="tight")
        self.topdown_map_counter += 1
        plt.close()
        # logger.info(f"Saved: output/topdown_map.png")

    def display_sample(self, rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
        rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

        arr = [rgb_img]
        titles = ["rgb"]
        if semantic_obs.size != 0:
            semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
            semantic_img.putpalette(d3_40_colors_rgb.flatten())
            semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
            semantic_img = semantic_img.convert("RGBA")
            arr.append(semantic_img)
            titles.append("semantic")

        if depth_obs.size != 0:
            depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
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
    
    def extract_visible_objs(self, sim, observations) -> Optional[Dict[str, Any]]:
        """
        Given current sim and observations (must include semantic sensor),
        returns a dict of visible objects and spatial relations between them.
        """
        if "semantic_sensor" not in observations:
            logger.warning("No semantic sensor found; skipping visible object extraction.")
            return None

        semantic = observations["semantic_sensor"]
        total_pixels = semantic.size
        obj_num_ids, counts = np.unique(semantic, return_counts=True)

        visible_objects = {}


        for obj_num_id, pixel_count in zip(obj_num_ids, counts): # obj_num_id is an int
            if obj_num_id == 0 or pixel_count < 50:
                continue  # skip background/noise/small fragments

            try:
                sim_obj = sim.semantic_scene.objects[obj_num_id]
            except IndexError:
                print(f"[WARNING] Object with num_id {obj_num_id} not found in semantic scene.")
                continue
            
            obj = self.objects.get(sim_obj.id)
            if obj is None:
                continue
            centroid_world = obj.get("centroid_world")

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

            # Project to NDC (Normalized Device Coordinates) for accurate screen position
            # Get camera sensor to access projection matrix
            cam_sensor = sim.get_agent(self.agent_id).scene_node.node_sensor_suite.get("color_sensor")
            if cam_sensor is not None:
                # Get projection matrix (FOV-based perspective projection)
                proj_mat = cam_sensor.render_camera.projection_matrix
                
                # Transform from camera space to clip space (homogeneous coords)
                centroid_cam_vec = mn.Vector3(centroid_cam[0], centroid_cam[1], centroid_cam[2])
                clip_pos = proj_mat.transform_point(centroid_cam_vec)
                
                # NDC coordinates are already normalized (-1 to +1 range)
                # x: -1 (left) to +1 (right)
                # y: -1 (bottom) to +1 (top)
                ndc_x, ndc_y = float(clip_pos.x), float(clip_pos.y)
            else:
                # Fallback if projection matrix not available
                ndc_x, ndc_y = None, None

            obj_str_id = str(sim_obj.id)  # -> Eg. 'wall_clock_231'
            # print(f"[MINNIE] Visible obj: id={obj_str_id}, label={label}, pixels={pixel_count}, centroid_world={centroid_world}, dist={dist:.2f}m")
            visible_objects[obj_str_id] = {
                **obj, # "sim_obj", "obj_str_id", "obj_num_id", "label", "centroid_world", "bbox_world", "linear_size", "room", "floor_number"
                "pixel_count": int(pixel_count),
                "pixel_percent": float(100 * pixel_count / total_pixels),
                "centroid_cam": centroid_cam.tolist(),
                "distance_from_camera": dist,
                "ndc_x": ndc_x,  # Screen position: -1 (left) to +1 (right)
                "ndc_y": ndc_y,  # Screen position: -1 (bottom) to +1 (top)
            }

        return dict(sorted(visible_objects.items(), key=lambda item: (item[1].get("pixel_count", 0.0),),reverse=True,))
        
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

    def compute_spatial_relations(self, visible_clusters, agent_state, max_distance=1.5, vertical_thresh=0.25, horizontal_bias=1.2, size_ratio_thresh=3.0):
        """
        Compute spatial relations between nearby objects, filtering out irrelevant ones.
        - Ignora relazioni tra oggetti con scala troppo diversa
        - Favorisce relazioni tra oggetti visibili di scala comparabile
        - Usa la direzione più dominante per definire la relazione
        - Applica regole semantiche per oggetti specifici (es. tavoli, porte, ecc.)
        """
        # EXAMPLE
        SEMANTIC_RULES = {
            "table": {"allowed": ["on_top_of", "beneath_of", "left_of", "right_of"],},
            "desk": {"allowed": ["on_top_of", "beneath_of", "left_of", "right_of"],},
            "door": {"allowed": ["on_top_of", "left_of", "right_of", "in_front_of"],},
            "window": {"allowed": [ "on_top_of", "left_of", "right_of"],},
            "ceiling": {"allowed": ["beneath_of"],},
            "floor": {"allowed": ["on_top_of"],},
        }

        agent_position = agent_state.position
        agent_rotation = utils.quat_to_magnum(agent_state.rotation)

        def is_relation_valid(obj_label, relation):
            """
            Verifica se una relazione è semanticamente valida per un dato oggetto.
            """
            if obj_label not in SEMANTIC_RULES:
                return True
            rules = SEMANTIC_RULES[obj_label]
            if "allowed" in rules and relation not in rules["allowed"]:
                return False
            return True

        relations = []
        keys = list(visible_clusters.keys())

        if len(keys) <= 1:
            return relations

        centroids = {
            k: np.array(visible_clusters[k]["centroid_cam"], dtype=float) for k in keys
        }

        sizes = {k: float(visible_clusters[k].get("linear_size", 0.0)) for k in keys}

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                obj_a, obj_b = visible_clusters[keys[i]], visible_clusters[keys[j]]
                size_a, size_b = sizes[keys[i]], sizes[keys[j]]

                if size_a <= 0 or size_b <= 0:
                    continue

                ratio = max(size_a, size_b) / min(size_a, size_b) # TODO why this? check
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
                if (abs_dy > vertical_thresh and abs_dy > (abs_dx + abs_dz) / horizontal_bias):
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
                        "subject": obj_a,
                        "subject_str_id": keys[i],
                        "object": obj_b,
                        "object_str_id": keys[j],
                        "relation": rel_ab,
                        "distance_m": round(float(dist), 3),
                    }
                )

        return relations
    
  

    def cluster_visible_objs(self, visible_objs: dict, *, pixel_percent_min=0.02, visible_objs_only=True) -> dict:
        """
        Cluster visible objects based on label and proximity.
        Returns a dict of clustered objects with relevant info.
        """
        # Initialize dictionary to aggregate visible data per cluster
        visible_clusters = {}
        total_visible_pixels = sum(v["pixel_count"] for v in visible_objs.values())

        # Find the cluster for each visible object and aggregate data
        for visible_obj in visible_objs.values():
            cluster_str_id = None
            for cluster in self.clusters.values():
                if visible_obj["obj_str_id"] in cluster["obj_str_ids"]:
                    cluster_str_id = cluster["cluster_str_id"]
                    break

            if cluster_str_id is None:
                continue  # Skip if no cluster found

            # Initialize cluster data if first time seeing this cluster
            if cluster_str_id not in visible_clusters:
                cluster = self.clusters[cluster_str_id]
                visible_clusters[cluster_str_id] = {
                    "cluster_str_id": cluster_str_id,
                    "label_norm": cluster["label"],  # Assumed key based on final structure
                    "pixel_count": 0,
                    "pixel_percent": 0.0,
                    "centroid_world_sum": np.zeros(3),
                    "centroid_cam_sum": np.zeros(3),
                    "distance_from_camera_sum": 0.0,
                    "ndc_x_sum": 0.0,
                    "ndc_y_sum": 0.0,
                    "ndc_count": 0,  # Track how many objects have valid NDC
                    "obj_count": 0,
                    "bbox_worlds": [],
                    "linear_size": 0.0,  # Sum of linear_size of visible objects
                    "obj_str_ids": [],   # Only visible object IDs
                    "room": cluster["room"],
                    "floor_number": cluster["floor_number"],
                }
            
            visible_cluster = visible_clusters[cluster_str_id]
            
            # Sum scalar values
            visible_cluster["pixel_count"] += visible_obj["pixel_count"]
            visible_cluster["distance_from_camera_sum"] += visible_obj["distance_from_camera"]
            visible_cluster["linear_size"] += visible_obj["linear_size"]
            visible_cluster["obj_count"] += 1
            
            # Sum vector/list values
            visible_cluster["centroid_world_sum"] += np.array(visible_obj["centroid_world"])
            visible_cluster["centroid_cam_sum"] += np.array(visible_obj["centroid_cam"])
            
            # Accumulate NDC coordinates (if available)
            ndc_x = visible_obj["ndc_x"]
            ndc_y = visible_obj["ndc_y"]
            if ndc_x is not None and ndc_y is not None:
                visible_cluster["ndc_x_sum"] += ndc_x
                visible_cluster["ndc_y_sum"] += ndc_y
                visible_cluster["ndc_count"] += 1
            
            # Collect for merging
            visible_cluster["bbox_worlds"].append(visible_obj["bbox_world"])
            
            # Collect visible IDs
            visible_cluster["obj_str_ids"].append(visible_obj["obj_str_id"]), 
        

        if not visible_objs_only:
            for cluster_str_id, visible_cluster in visible_clusters.items():
                
                full_cluster = self.clusters[cluster_str_id]

                visible_cluster["obj_str_ids"] = full_cluster["obj_str_ids"]
                visible_cluster["obj_count"] = len(visible_cluster["obj_str_ids"])
                visible_cluster["centroid_world_sum"] = np.array(full_cluster["centroid_world"]) * visible_cluster["obj_count"]
                visible_cluster["linear_size"] = full_cluster["linear_size"]
                visible_cluster["bbox_worlds"] = [full_cluster["bbox_world"]]



        
        # 2. Finalize calculations and apply filter
        updated_visible_clusters = {}

        for cluster_str_id, visible_cluster in visible_clusters.items():
            # Calculate averages and percentages
            if visible_cluster["obj_count"] == 0:
                continue
                
            visible_cluster["pixel_percent"] = 100 * visible_cluster["pixel_count"] / total_visible_pixels

            # Apply minimum pixel percentage filter
            if visible_cluster["pixel_percent"] < pixel_percent_min:
                continue
                
            # Calculate centroids and distance averages
            avg_centroid_world = (visible_cluster["centroid_world_sum"] / visible_cluster["obj_count"]).tolist()
            avg_centroid_cam = (visible_cluster["centroid_cam_sum"] / visible_cluster["obj_count"]).tolist()
            avg_distance_from_camera = visible_cluster["distance_from_camera_sum"] / visible_cluster["obj_count"]
            
            # Calculate average NDC coordinates (if available)
            if visible_cluster["ndc_count"] > 0:
                avg_ndc_x = visible_cluster["ndc_x_sum"] / visible_cluster["ndc_count"]
                avg_ndc_y = visible_cluster["ndc_y_sum"] / visible_cluster["ndc_count"]
            else:
                avg_ndc_x = None
                avg_ndc_y = None
            
            # Merge bounding boxes
              
            def _merge_bboxes(bboxes):
                """
                Merges a list of bounding boxes (min_x, min_y, min_z, max_x, max_y, max_z)
                into a single encompassing bounding box.
                """
                if not bboxes:
                    return None
                
                # Convert to NumPy array for easier min/max calculation
                np_bboxes = np.array(bboxes)
                
                # Calculate min corners (0:3) and max corners (3:6) across all bboxes
                min_corners = np.min(np_bboxes[:, :3], axis=0)
                max_corners = np.max(np_bboxes[:, 3:], axis=0)
                
                return np.concatenate((min_corners, max_corners)).tolist()
            
            merged_bbox = _merge_bboxes(visible_cluster["bbox_worlds"])
            
            # Build final structure
            updated_visible_clusters[cluster_str_id] = {
                "cluster_str_id": cluster_str_id,
                "label": visible_cluster["label_norm"],
                "pixel_count": int(visible_cluster["pixel_count"]),
                "pixel_percent": float(visible_cluster["pixel_percent"]),
                "centroid_world": [float(v) for v in avg_centroid_world],
                "bbox_world": merged_bbox,
                "centroid_cam": [float(v) for v in avg_centroid_cam],
                "distance_from_camera": float(avg_distance_from_camera),
                "ndc_x": float(avg_ndc_x) if avg_ndc_x is not None else None,
                "ndc_y": float(avg_ndc_y) if avg_ndc_y is not None else None,
                # Sum of linear_size of visible objects
                "linear_size": visible_cluster["linear_size"], 
                # List of obj_str_ids that are currently visible
                "obj_str_ids": sorted(visible_cluster["obj_str_ids"]), 
                "room": visible_cluster["room"],
                "floor_number": visible_cluster["floor_number"],
            }


        # 3. Sort the resulting dictionary by pixel_count (descending)
        visible_clusters = dict(sorted(updated_visible_clusters.items(), key=lambda item: item[1].get("pixel_count", 0.0),reverse=True,))

        return visible_clusters

    def process_visible_clusters(self, visible_clusters: dict, target_object: str = "") -> dict:
        """
        Process clustered objects into a list of dicts with relevant info.
        Also filters out objects and do more interesting computations if needed.
        """

        # The input is already sorted by pixel_count descending
        # * 1. We can simply use the top-10 most visible clusters
        visible_clusters_list = list(visible_clusters.items())[:10]

        for cluster_str_id, visible_cluster in visible_clusters.items():
            # Further processing per cluster can be done here if needed
            if target_object and visible_cluster["label"] == target_object and target_object != "":
                visible_clusters_list.insert(0, (cluster_str_id, visible_cluster))
                visible_clusters_list.pop()  # remove last to keep size
                
        visible_clusters = {key: value for key, value in visible_clusters_list}

            
        # * 2. Further processing can be done here if needed
        return visible_clusters

    def _normalize_label(self, label: str) -> str:
        """Normalize object labels to handle synonyms and groupings."""
        l = label.strip().lower()
        if l in {"wall", "ceiling", "floor", "window frame", "frame", "unknown"}:
            return l

        if l in {"door", "doorway", "door frame", "attic door"}:
            return "door"

        if l in {"stairs", "stair", "step", "stairway"}:
            return "stairs"
        
        # TODO add more classes synonyms/groupings as needed

        return l

    def _is_informative(self, label_norm: str, *, mode: str = "blacklist", blacklist=None, whitelist=None) -> bool:
        """
        mode="blacklist": keep everything except the blacklist
        mode="whitelist": keep only the whitelist
        """
        if mode not in {"blacklist", "whitelist"}:
            mode = "blacklist"
        if mode == "blacklist":
            blacklist = set(blacklist or ["wall", "floor", "ceiling", "frame", "window frame", "unknown", "ceiling_light", "light", "lamp"])
            return label_norm not in blacklist
        else:
            whitelist = set(whitelist or ["doorway", "staircase", "elevator", "escalator", "corridor", "intersection", "railing", "exit_sign", "sign", "sofa", "table", "wardrobe", "balcony", "bridge"])
            return label_norm in whitelist
        
    def _cluster_visible_same_label(self, obj_str_ids, visible_objs, cluster_cnt, label_norm, distance_thresh=1.0):
        """
        Greedy clustering per label on ground plane.
        - obj_str_ids: list of dicts with keys:
            label_norm, centroid_world, bbox_world, pixel_count, pixel_percent, distance_from_camera, raw_ids (set)
        - visible_objs: dict of all visible objects (to access data by obj_str_id)
        - distance_thresh: meters (tune per scene scale; 0.8~1.5 works well)

        Returns list of merged clusters for that label.
        """
        clusters = []  # each: dict like instances, aggregated

        # Sort big to small so large areas seed clusters
        obj_str_ids_sorted = sorted(obj_str_ids, key=lambda x: visible_objs[x]["pixel_count"], reverse=True)
        def xz_dist(a, b):
            return math.sqrt((a[0]-b[0])**2 + (a[2]-b[2])**2)  # ignore Y
        
        def merge_obbs(obb_a, obb_b):
            return [
                [
                    min(obb_a[0][0], obb_b[0][0]),
                    min(obb_a[0][1], obb_b[0][1]),
                    min(obb_a[0][2], obb_b[0][2]),
                ],
                [
                    max(obb_a[1][0], obb_b[1][0]),
                    max(obb_a[1][1], obb_b[1][1]),
                    max(obb_a[1][2], obb_b[1][2]),
                ]
            ]
        
        for obj_str_id in obj_str_ids_sorted:
            assigned = False
            visible_obj = visible_objs[obj_str_id]
            for cluster in clusters:
                if (xz_dist(visible_obj["centroid_world"], cluster["centroid_world"]) <= distance_thresh):
                    # merge into cluster (weighted by pixel_count)
                    w_old = cluster["pixel_count"]
                    w_new = visible_obj["pixel_count"]
                    w_sum = w_old + w_new

                    # weighted centroid (world)
                    cx = (cluster["centroid_world"][0] * w_old + visible_obj["centroid_world"][0] * w_new) / w_sum
                    cy = (cluster["centroid_world"][1] * w_old + visible_obj["centroid_world"][1] * w_new) / w_sum
                    cz = (cluster["centroid_world"][2] * w_old + visible_obj["centroid_world"][2] * w_new) / w_sum
                    cluster["centroid_world"] = [cx, cy, cz]

                    # choose min distance to camera (useful for narration)
                    cluster["distance_from_camera"] = min(cluster["distance_from_camera"], visible_obj["distance_from_camera"])

                    # union bbox
                    cluster["bbox_world"] = merge_obbs(cluster["bbox_world"], visible_obj["bbox_world"])

                    # accumulate pixels
                    cluster["pixel_count"] = w_sum
                    cluster["pixel_percent"] += visible_obj["pixel_percent"]

                    cluster["linear_size"] += visible_obj.get("linear_size", 0.0) 
                    # track raw semantic ids merged
                    cluster["obj_str_ids"].append(obj_str_id) # TODO check di questa porcata
                    assigned = True
                    break

            if not assigned:
                clusters.append({**visible_obj, 
                                 "obj_str_ids": [obj_str_id], 
                                 "cluster_id": cluster_cnt, 
                                 "linear_size": visible_obj.get("linear_size", 0.0), 
                                 "label_norm": label_norm})
                cluster_cnt += 1
        return clusters, cluster_cnt

    def draw_event(self, simulation_call: Optional[Callable] = None, global_call: Optional[Callable] = None, active_agent_id_and_sensor_name: Tuple[int, str] = (0, "color_sensor")) -> None:
        """
        Calls continuously to re-render frames and swap the two frame buffers
        at a fixed rate.
        """
        if self.q_app:
            self.q_app.processEvents()

        agent_acts_per_sec = self.fps

        mn.gl.default_framebuffer.clear(mn.gl.FramebufferClear.COLOR | mn.gl.FramebufferClear.DEPTH)

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
            self.time_since_last_simulation = math.fmod(self.time_since_last_simulation, 1.0 / self.fps)

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
                    print("Falling back to searching for color sensor...")
                    if isinstance(agent_sensors, (list, tuple)):
                        for sensor in agent_sensors:
                            spec_fn = getattr(sensor, "specification", None)
                            spec = spec_fn() if callable(spec_fn) else None
                            if (spec is not None and getattr(spec, "sensor_type", None) == habitat_sim.SensorType.COLOR):
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
        # self._draw_bbox_label_overlay()

        self.swap_buffers()
        Timer.next_frame()
        self.redraw()

    def debug_draw(self):
        """
        Additional draw commands to be called during draw_event.
        """
        super().debug_draw()
        if self.show_object_bboxes:
            self._draw_object_bboxes(self.debug_line_render, self.clusters_to_draw)
        else:
            self._bbox_label_screen_positions.clear()

        if self.show_room_bboxes:
            self._draw_room_bboxes(self.debug_line_render)


    def _draw_room_bboxes(self, debug_line_render: Any) -> None:
        """
        Draw bounding boxes for all rooms in self.rooms.
        """
        if not self.rooms:
            return
        
        debug_line_render.set_line_width(3.0)
        
        # Define a color palette for rooms
        ROOM_COLORS = [
            mn.Color4(1.0, 0.0, 0.0, 0.8),    # Red
            mn.Color4(0.0, 1.0, 0.0, 0.8),    # Green
            mn.Color4(0.0, 0.0, 1.0, 0.8),    # Blue
            mn.Color4(1.0, 1.0, 0.0, 0.8),    # Yellow
            mn.Color4(1.0, 0.0, 1.0, 0.8),    # Magenta
            mn.Color4(0.0, 1.0, 1.0, 0.8),    # Cyan
            mn.Color4(1.0, 0.5, 0.0, 0.8),    # Orange
            mn.Color4(0.5, 0.0, 1.0, 0.8),    # Purple
            mn.Color4(0.0, 0.5, 0.5, 0.8),    # Teal
            mn.Color4(0.5, 0.5, 0.0, 0.8),    # Olive
        ]
        
        for idx, (region_id, room) in enumerate(self.rooms.items()):
            bbox = room.get("bbox_world")
            if bbox is None or len(bbox) != 2:
                continue
            
            # Extract min and max corners
            min_corner = mn.Vector3(bbox[0][0], bbox[0][1], bbox[0][2])
            max_corner = mn.Vector3(bbox[1][0], bbox[1][1], bbox[1][2])
            
            # Define the 8 corners of the bounding box
            corners = [
                mn.Vector3(min_corner[0], min_corner[1], min_corner[2]),  # 0
                mn.Vector3(max_corner[0], min_corner[1], min_corner[2]),  # 1
                mn.Vector3(min_corner[0], max_corner[1], min_corner[2]),  # 2
                mn.Vector3(max_corner[0], max_corner[1], min_corner[2]),  # 3
                mn.Vector3(min_corner[0], min_corner[1], max_corner[2]),  # 4
                mn.Vector3(max_corner[0], min_corner[1], max_corner[2]),  # 5
                mn.Vector3(min_corner[0], max_corner[1], max_corner[2]),  # 6
                mn.Vector3(max_corner[0], max_corner[1], max_corner[2]),  # 7
            ]
            
            # Define the 12 edges of the bounding box
            edges = [
                (0, 1), (0, 2), (0, 4),
                (1, 3), (1, 5),
                (2, 3), (2, 6),
                (3, 7),
                (4, 5), (4, 6),
                (5, 7),
                (6, 7),
            ]
            
            # Select color for this room
            color = ROOM_COLORS[idx % len(ROOM_COLORS)]
            
            # Draw all edges
            for edge in edges:
                start = corners[edge[0]]
                end = corners[edge[1]]
                debug_line_render.draw_transformed_line(start, end, color)
            
            # Optionally, add room name label at the center top of the bbox
            center = mn.Vector3(
                (min_corner[0] + max_corner[0]) / 2.0,
                max_corner[1] + 0.1,  # Slightly above the top
                (min_corner[2] + max_corner[2]) / 2.0
            )
            screen_pos = self._project_to_screen(center)
            room_name = room.get("name", region_id)
            if screen_pos is not None:
                self._bbox_label_screen_positions.append((room_name, screen_pos))

    def _draw_object_bboxes(self, debug_line_render: Any, clusters_to_draw: list = None) -> None:
    
        """
        Draw axis-aligned bounding boxes for every semantic object.
        """
        # print(f" _draw_object_bboxes called with objs_str_ids_to_draw: {objs_str_ids_to_draw}")
        # clusters_to_draw: {"cluster_str_id": ["obj_str_id1", "obj_str_id2", ...], ...} 
        scene = self.sim.semantic_scene
        if scene is None:
            return
    

        objs_to_draw = []

        if clusters_to_draw is not None:
            for cluster_str_id, obj_str_ids in clusters_to_draw.items():
                for obj_str_id in obj_str_ids: 
                    sim_obj = self.objects[obj_str_id]["sim_obj"]
                    objs_to_draw.append((sim_obj, cluster_str_id))
            # if objs_to_draw != self.prev_objs_to_draw:
                # for sim_obj, cluster_str_id in objs_to_draw:
                #     print(f"Cluster {cluster_str_id} includes object ID {sim_obj.id}")
        else:
            for sim_obj in scene.objects:
                objs_to_draw.append((sim_obj, sim_obj.id))     

        self._bbox_label_screen_positions.clear()
        debug_line_render.set_line_width(2.5)
        target_labels = []
        max_boxes = 1000
        candidates = []



        
        self.prev_objs_to_draw = objs_to_draw

        for (sim_obj, str_id) in objs_to_draw:

            label = ""
            if sim_obj.category is not None and hasattr(sim_obj.category, "name"):
                label = sim_obj.category.name()
            label_norm = label.strip().lower()
            if clusters_to_draw is None and label_norm not in target_labels:
                continue

            if not label:
                label = f"object_{sim_obj.id}"

            obb = getattr(sim_obj, "obb", None)
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
            corners = [rotation.transform_vector(offset) + center for offset in corner_offsets]

            volume = max(8.0 * half_extents[0] * half_extents[1] * half_extents[2], 0.0)
            # print(f"Object ID {obj.id} ('{label}') volume: {volume:.4f} m^3")
            candidates.append((volume, str_id, label, corners, center, rotation, half_extents))

        candidates.sort(key=lambda item: item[0], reverse=True)
        # print("Len candidates", len(candidates))

        edges = [(0, 1),(0, 2),(0, 4),(1, 3),(1, 5),(2, 3),(2, 6),(3, 7),(4, 5),(4, 6),(5, 7),(6, 7),]

        for (volume, str_id, label, corners, center, rotation, half_extents) in candidates[:max_boxes]:
            color = self._get_bbox_color(str_id)
            # if objs_to_draw != self.prev_objs_to_draw:
            #     print("Color", color)

            for edge in edges:
                start = corners[edge[0]]
                end = corners[edge[1]]
                debug_line_render.draw_transformed_line(start, end, color)
                debug_line_render.draw_transformed_line(start, end, color)

            top_center = (center + rotation.transform_vector(mn.Vector3(0.0, half_extents[1], 0.0)) + mn.Vector3(0.0, 0.05, 0.0))
            screen_pos = self._project_to_screen(top_center)
            if screen_pos is not None:
                self._bbox_label_screen_positions.append((label, screen_pos))

    def _get_bbox_color(self, obj_id: int) -> mn.Color4:
        """
        Returns a consistent color for the given semantic object id.
        """
        if obj_id in self._object_bbox_colors:
            return self._object_bbox_colors[obj_id]
        
        VIVID_PALETTE_RGB = [
            [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], # Red, Green, Yellow, Blue
            [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230], # Orange, Purple, Cyan, Magenta
            [170, 110, 40], [255, 250, 200], [128, 0, 0], [154, 99, 36], # Brown, Beige, Maroon, Olive
            [128, 128, 0], [0, 0, 128], [0, 0, 0], [170, 255, 195], # Teal, Navy, Black, Mint
            [255, 215, 180], [255, 255, 255] # Pink, White
        ]

        # palette = 
        palette = np.array(VIVID_PALETTE_RGB)
        palette_len = len(palette)
        if palette_len == 0:
            color = mn.Color4(1.0, 0.0, 0.0, 1.0)
            self._object_bbox_colors[obj_id] = color
            return color
        
        if isinstance(obj_id, str):
            num_id = int(obj_id.split("_")[-1])
            idx = hash(num_id) % palette_len
        elif isinstance(obj_id, int):
            idx = obj_id % palette_len
        else:
            raise ValueError(f"Unsupported obj_id type: {type(obj_id)}")
       
    
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

            label_renderer = text.Renderer2D(self.display_font, self.glyph_cache, BaseViewer.DISPLAY_FONT_SIZE, text.Alignment.TOP_CENTER)
            label_renderer.reserve(len(label))
            label_renderer.render(label)

            transform = mn.Matrix3.projection(framebuffer) @ mn.Matrix3.translation(screen_pos)
            self.shader.transformation_projection_matrix = transform
            self.shader.color = mn.Color4(1.0, 1.0, 1.0, 1.0)
            self.shader.draw(label_renderer.mesh)

        mn.gl.Renderer.disable(mn.gl.Renderer.Feature.BLENDING)

    def move_and_look(self, repetitions: int) -> None:
        """
        This method is called continuously with `self.draw_event` to monitor
        any changes in the movement keys map `Dict[KeyEvent.key, Bool]`.
        When a key in the map is set to `True` the corresponding action is taken.
        """
        super().move_and_look(repetitions)

        self._process_queued_actions()  # process any queued actions from the other thread
        
    def _process_queued_actions(self):
        """Execute actions enqueued from other threads."""
        try:
            while True:
                action, args, kwargs = self.action_queue.get_nowait()
                # try:
                action(*args, **kwargs)
                # except Exception as e:
                #     print(f"Error executing queued action {action}: {e}")

                self.action_queue.task_done()

        except queue.Empty:
            pass

    def print_agent_state(self) -> None:
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
                    if (obj and obj.category and obj.category.name().lower() == object_name.lower()):
                        return mn.Vector3(obj.obb.center)

        return None

    def check_object_in_room(self, object_name: Optional[str], room_name: Optional[str]) -> bool:
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
            {"role": "user", "content": user_input + "\n" + str(self.room_objects_occurences)},
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

def user_input_logic_loop(viewer: NewViewer, input_q: queue.Queue, output_q: queue.Queue):
    while True:
        try:
            user_input = input_q.get()
            print("Received user input:", user_input)
            if not user_input:
                continue

            llm_enabled = True
            # output_q.put("Processing your request...")
            if not llm_enabled:
                try:
                    target_name, room_name = user_input.split("/")[0].strip(), user_input.split("/")[1].strip()
                    output_q.put(f"Navigating to {room_name}/{target_name}...")
                except Exception as e:
                    print("Error parsing input without LLM. Please use 'object/room' format.")
                    continue
            else: 
                try:           
                    response = viewer.get_response_LLM(user_input)  # * API Call to ChatGPT
                except Exception as e:
                    print("Error getting response from LLM:", e)
                    continue
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


            
            viewer.action_queue.put((viewer.start_navigation, (viewer.sim, target_name, room_name, user_input, output_q), {}))
            # output_q.put(f"Generating navigation instructions...")



        except EOFError:
            break

_LOCAL_MODEL = None
_LOCAL_TOKENIZER = None

def load_local_model(repo_id="microsoft/Phi-3-mini-4k-instruct"):
    global _LOCAL_MODEL, _LOCAL_TOKENIZER
    if _LOCAL_MODEL is not None:
        print("[LOCAL-LLM] Model already loaded, reusing the cached instance.")
        return _LOCAL_MODEL, _LOCAL_TOKENIZER

    print("[INFO] Loading HF model (cached if present)...")
    print(f"[LOCAL-LLM] Loading model from HuggingFace repo: {repo_id}")
    print("[LOCAL-LLM] Step 1/3: Loading tokenizer...")
    _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(repo_id)

    print("[LOCAL-LLM] Step 2/3: Loading model weights (this may take a while)...")
    _LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(
        repo_id,
        dtype=torch.float16,
        device_map="auto",
    )
    print(f"[LOCAL-LLM] Step 3/3: Model loaded successfully on device: {_LOCAL_MODEL.device}")
    print("[LOCAL-LLM] Ready for inference.")
    return _LOCAL_MODEL, _LOCAL_TOKENIZER

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
    parser.add_argument(
        "--backend",
        default="openai",
        type=str,
        help="LLM backend to use: openai (default), local.",
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

    # * Depending on the backend flag, load the local model
    if args.backend.lower() == "local":
        try:
            model, tokenizer = load_local_model()
            print("[mr_viewer.py main] Local model loaded and ready.")
        except Exception as e:
            print(f"[mr_viewer.py main] Error loading local model: {e}")
            sys.exit(1)

    viewer.exec()

    sys.exit(q_app.exec_())
