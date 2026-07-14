#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import math
import os
import string
import sys
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import magnum as mn
import numpy as np
from magnum import shaders, text
from magnum.platform.glfw import Application

import habitat_sim
from habitat_sim import ReplayRenderer, ReplayRendererConfiguration, physics
from habitat_sim.logging import LoggingContext, logger
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_sim.utils.settings import default_sim_settings, make_cfg


class HabitatSimInteractiveViewer(Application):
    # the maximum number of chars displayable in the app window
    # using the magnum text module. These chars are used to
    # display the CPU/GPU usage data
    MAX_DISPLAY_TEXT_CHARS = 256

    # how much to displace window text relative to the center of the
    # app window (e.g if you want the display text in the top left of
    # the app window, you will displace the text
    # window width * -TEXT_DELTA_FROM_CENTER in the x axis and
    # window height * TEXT_DELTA_FROM_CENTER in the y axis, as the text
    # position defaults to the middle of the app window)
    TEXT_DELTA_FROM_CENTER = 0.49

    # font size of the magnum in-window display text that displays
    # CPU and GPU usage info
    DISPLAY_FONT_SIZE = 16.0

    MOVE, LOOK = 0.07, 1.5

    def __init__(self, sim_settings: Dict[str, Any]) -> None:
        self.sim_settings: Dict[str:Any] = sim_settings

        self.enable_batch_renderer: bool = self.sim_settings["enable_batch_renderer"]
        self.num_env: int = (
            self.sim_settings["num_environments"] if self.enable_batch_renderer else 1
        )

        # Compute environment camera resolution based on the number of environments to render in the window.
        window_size: mn.Vector2 = (
            self.sim_settings["window_width"],
            self.sim_settings["window_height"],
        )

        configuration = self.Configuration()
        configuration.title = "Habitat Sim Interactive Viewer"
        configuration.size = window_size
        # Optionally create the window hidden. Used for GPU-headless rendering:
        # a real display (e.g. WSLg :0) provides GPU acceleration while the
        # window stays invisible.
        if self.sim_settings.get("hidden_window", False):
            cfg_cls = self.Configuration
            hidden_flag = None
            for _enum_name in ("WindowFlags", "WindowFlag"):
                _enum = getattr(cfg_cls, _enum_name, None)
                if _enum is not None and hasattr(_enum, "HIDDEN"):
                    hidden_flag = _enum.HIDDEN
                    break
            if hidden_flag is not None:
                try:
                    configuration.window_flags = hidden_flag
                except Exception as _e:
                    print(f"[viewer] could not set hidden window flag: {_e}")
        Application.__init__(self, configuration)
        self.fps: float = 60.0

        # Compute environment camera resolution based on the number of environments to render in the window.
        grid_size: mn.Vector2i = ReplayRenderer.environment_grid_size(self.num_env)
        camera_resolution: mn.Vector2 = mn.Vector2(self.framebuffer_size) / mn.Vector2(
            grid_size
        )
        self.sim_settings["width"] = camera_resolution[0]
        self.sim_settings["height"] = camera_resolution[1]

        # draw Bullet debug line visualizations (e.g. collision meshes)
        self.debug_bullet_draw = False
        # draw active contact point debug line visualizations
        self.contact_debug_draw = False
        # draw semantic region debug visualizations if present
        self.semantic_region_debug_draw = False

        # cache most recently loaded URDF file for quick-reload
        self.cached_urdf = ""

        # set up our movement map

        key = Application.Key
        self.pressed = {
            key.UP: False,
            key.DOWN: False,
            key.LEFT: False,
            key.RIGHT: False,
            key.A: False,
            key.D: False,
            key.S: False,
            key.W: False,
            key.X: False,
            key.Z: False,
        }

        # set up our movement key bindings map
        self.key_to_action = {
            key.UP: "look_up",
            key.DOWN: "look_down",
            key.LEFT: "turn_left",
            key.RIGHT: "turn_right",
            key.A: "move_left",
            key.D: "move_right",
            key.S: "move_backward",
            key.W: "move_forward",
            key.X: "move_down",
            key.Z: "move_up",
        }

        # Load a TrueTypeFont plugin and open the font file
        self.display_font = text.FontManager().load_and_instantiate("TrueTypeFont")
        relative_path_to_font = "../data/fonts/ProggyClean.ttf"
        self.display_font.open_file(
            os.path.join(os.path.dirname(__file__), relative_path_to_font),
            13,
        )

        # Glyphs we need to render everything
        self.glyph_cache = text.GlyphCacheGL(
            mn.PixelFormat.R8_UNORM, mn.Vector2i(256), mn.Vector2i(1)
        )
        self.display_font.fill_glyph_cache(
            self.glyph_cache,
            string.ascii_lowercase
            + string.ascii_uppercase
            + string.digits
            + ":-_+,.! %µ",
        )

        # magnum text object that displays CPU/GPU usage data in the app window
        self.window_text = text.Renderer2D(
            self.display_font,
            self.glyph_cache,
            HabitatSimInteractiveViewer.DISPLAY_FONT_SIZE,
            text.Alignment.TOP_LEFT,
        )
        self.window_text.reserve(HabitatSimInteractiveViewer.MAX_DISPLAY_TEXT_CHARS)

        # text object transform in window space is Projection matrix times Translation Matrix
        # put text in top left of window
        self.window_text_transform = mn.Matrix3.projection(
            self.framebuffer_size
        ) @ mn.Matrix3.translation(
            mn.Vector2(self.framebuffer_size)
            * mn.Vector2(
                -HabitatSimInteractiveViewer.TEXT_DELTA_FROM_CENTER,
                HabitatSimInteractiveViewer.TEXT_DELTA_FROM_CENTER,
            )
        )
        self.shader = shaders.VectorGL2D()

        # Set blend function
        mn.gl.Renderer.set_blend_equation(
            mn.gl.Renderer.BlendEquation.ADD, mn.gl.Renderer.BlendEquation.ADD
        )

        # variables that track app data and CPU/GPU usage
        self.num_frames_to_track = 60

        # Cycle mouse utilities
        self.mouse_interaction = MouseMode.LOOK
        self.mouse_grabber: Optional[MouseGrabber] = None
        self.previous_mouse_point = None

        # toggle physics simulation on/off
        self.simulating = True

        # toggle a single simulation step at the next opportunity if not
        # simulating continuously.
        self.simulate_single_step = False

        # configure our simulator
        self.cfg: Optional[habitat_sim.simulator.Configuration] = None
        self.sim: Optional[habitat_sim.simulator.Simulator] = None
        self.tiled_sims: list[habitat_sim.simulator.Simulator] = None
        self.replay_renderer_cfg: Optional[ReplayRendererConfiguration] = None
        self.replay_renderer: Optional[ReplayRenderer] = None
        self.reconfigure_sim()
        self.debug_semantic_colors = {}

        # compute NavMesh if not already loaded by the scene.
        if (
            not self.sim.pathfinder.is_loaded
            and self.cfg.sim_cfg.scene_id.lower() != "none"
        ):
            self.navmesh_config_and_recompute()

        self.time_since_last_simulation = 0.0
        LoggingContext.reinitialize_from_env()
        logger.setLevel("INFO")
        self.print_help_text()

    def save_semantic_image(self, side_by_side: bool = True) -> None:
        """
        Captures the current semantic sensor view and saves it as a colored JPG file.
        
        Args:
            side_by_side (bool): If True, saves a single image with RGB on the left
                and semantic on the right. If False, saves them separately. Default: False.
        """
        try:
            # Get observations from the simulator
            observations = self.sim.get_sensor_observations()

            # Check if semantic sensor exists
            if "semantic_sensor" not in observations:
                logger.warning("No semantic sensor found in observations. Cannot save semantic image.")
                return

            rgb_obs = observations["color_sensor"]
            semantic_obs = observations["semantic_sensor"]
            
            # Process RGB observation to handle RGBA or other formats
            rgb_img = rgb_obs
            
            # Handle different array formats from Habitat-sim
            if rgb_img.ndim == 3:
                # If RGBA (4 channels), convert to RGB
                if rgb_img.shape[2] == 4:
                    rgb_img = rgb_img[:, :, :3]
                # If single channel, replicate to 3 channels
                elif rgb_img.shape[2] == 1:
                    rgb_img = np.repeat(rgb_img, 3, axis=2)
            
            # Ensure uint8 type
            if rgb_img.dtype != np.uint8:
                if np.issubdtype(rgb_img.dtype, np.floating):
                    # If float in [0, 1], scale to [0, 255]
                    if rgb_img.max() <= 1.0:
                        rgb_img = (rgb_img * 255).astype(np.uint8)
                    else:
                        rgb_img = rgb_img.astype(np.uint8)
                else:
                    rgb_img = rgb_img.astype(np.uint8)

            # Create output directory if it doesn't exist
            output_dir = "semantic_captures"
            os.makedirs(output_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_semantic = os.path.join(output_dir, f"semantic_{timestamp}.jpg")
            filename_rgb = os.path.join(output_dir, f"rgb_{timestamp}.jpg")

            # Convert semantic IDs to colored image
            # Get unique semantic IDs
            unique_ids = np.unique(semantic_obs)

            # Create a colormap for semantic IDs
            # Use a deterministic color mapping based on ID
            semantic_img_rgb = np.zeros((semantic_obs.shape[0], semantic_obs.shape[1], 3), dtype=np.uint8)

            for semantic_id in unique_ids:
                # Generate a unique color for each semantic ID using a hash-like function
                # This ensures consistent colors across saves
                np.random.seed(int(semantic_id))
                color = np.random.randint(0, 256, 3, dtype=np.uint8)

                # Apply color to all pixels with this semantic ID
                mask = semantic_obs == semantic_id
                semantic_img_rgb[mask] = color

            # Reset random seed to avoid affecting other random operations
            np.random.seed(None)
            from PIL import Image
            # Save the image(s)
            if side_by_side:
                # Ensure both images have the same height for concatenation
                if rgb_img.shape[0] != semantic_img_rgb.shape[0] or rgb_img.shape[1] != semantic_img_rgb.shape[1]:
                    # Resize semantic to match RGB size
                    from PIL import Image as PILImage
                    sem_pil = PILImage.fromarray(semantic_img_rgb)
                    sem_pil = sem_pil.resize((rgb_img.shape[1], rgb_img.shape[0]), resample=PILImage.NEAREST)
                    semantic_img_rgb = np.array(sem_pil)
                
                # Concatenate horizontally (RGB on left, semantic on right)
                combined = np.concatenate([rgb_img, semantic_img_rgb], axis=1)
                filename_combined = os.path.join(output_dir, f"rgb_semantic_{timestamp}.jpg")
                Image.fromarray(combined, mode='RGB').save(filename_combined)
                logger.info(f"✅ Saved RGB + semantic side-by-side image to: {filename_combined}")
                print(f"✅ Saved RGB + semantic side-by-side image to: {filename_combined}")
            else:
                # Save separately
                Image.fromarray(semantic_img_rgb, mode='RGB').save(filename_semantic)
                Image.fromarray(rgb_img, mode='RGB').save(filename_rgb)
                logger.info(f"✅ Saved colored semantic image to: {filename_semantic}")
                logger.info(f"✅ Saved RGB image to: {filename_rgb}")
                print(f"✅ Saved colored semantic image to: {filename_semantic}")
                print(f"✅ Saved RGB image to: {filename_rgb}")

        except Exception as e:
            logger.error(f"Failed to save semantic image: {e}")
            print(f"❌ Failed to save semantic image: {e}")

    def draw_contact_debug(self, debug_line_render: Any):
        """
        This method is called to render a debug line overlay displaying active contact points and normals.
        Yellow lines show the contact distance along the normal and red lines show the contact normal at a fixed length.
        """
        yellow = mn.Color4.yellow()
        red = mn.Color4.red()
        cps = self.sim.get_physics_contact_points()
        debug_line_render.set_line_width(1.5)
        camera_position = self.render_camera.render_camera.node.absolute_translation
        # only showing active contacts
        active_contacts = (x for x in cps if x.is_active)
        for cp in active_contacts:
            # red shows the contact distance
            debug_line_render.draw_transformed_line(
                cp.position_on_b_in_ws,
                cp.position_on_b_in_ws
                + cp.contact_normal_on_b_in_ws * -cp.contact_distance,
                red,
            )
            # yellow shows the contact normal at a fixed length for visualization
            debug_line_render.draw_transformed_line(
                cp.position_on_b_in_ws,
                # + cp.contact_normal_on_b_in_ws * cp.contact_distance,
                cp.position_on_b_in_ws + cp.contact_normal_on_b_in_ws * 0.1,
                yellow,
            )
            debug_line_render.draw_circle(
                translation=cp.position_on_b_in_ws,
                radius=0.005,
                color=yellow,
                normal=camera_position - cp.position_on_b_in_ws,
            )

    def draw_region_debug(self, debug_line_render: Any) -> None:
        """
        Draw the semantic region wireframes.
        """

        for region in self.sim.semantic_scene.regions:
            color = self.debug_semantic_colors.get(region.id, mn.Color4.magenta())
            for edge in region.volume_edges:
                debug_line_render.draw_transformed_line(
                    edge[0],
                    edge[1],
                    color,
                )

    def debug_draw(self):
        """
        Additional draw commands to be called during draw_event.
        """
        if self.debug_bullet_draw:
            render_cam = self.render_camera.render_camera
            proj_mat = render_cam.projection_matrix.__matmul__(render_cam.camera_matrix)
            self.sim.physics_debug_draw(proj_mat)

        self.debug_line_render = self.sim.get_debug_line_render()
        if self.contact_debug_draw:
            self.draw_contact_debug(self.debug_line_render)

        if self.semantic_region_debug_draw:
            if len(self.debug_semantic_colors) != len(self.sim.semantic_scene.regions):
                for region in self.sim.semantic_scene.regions:
                    self.debug_semantic_colors[region.id] = mn.Color4(
                        mn.Vector3(np.random.random(3))
                    )
            self.draw_region_debug(self.debug_line_render)

    def compute_object_center(self, obj_obb):
        """
        Compute the center of an object's oriented bounding box.
        """
        try:
            # OBB in habitat-sim has a 'center' property
            center = mn.Vector3(obj_obb.center)
            return center
        except Exception as e:
            logger.warning(f"Could not compute object center: {e}")
            return mn.Vector3(0, 0, 0)

    def compute_object_bbox(self, obj_obb):
        """
        Compute axis-aligned bounding box from oriented bounding box.
        Returns [[min_x, min_y, min_z], [max_x, max_y, max_z]]

        The OBB has center, half_extents, and rotation. We compute the 8 corners
        of the oriented box, then find the min/max to get an axis-aligned box.
        """
        try:
            # Get OBB properties
            center = mn.Vector3(obj_obb.center)
            half_extents = mn.Vector3(obj_obb.half_extents)
            rot_vec = mn.Vector4(obj_obb.rotation)
            rotation = mn.Quaternion(
                mn.Vector3(rot_vec[0], rot_vec[1], rot_vec[2]), rot_vec[3]
            )

            # Compute the 8 corners of the oriented bounding box
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

            # Transform corners to world space
            corners_world = [rotation.transform_vector(offset) + center for offset in corner_offsets]

            # Compute axis-aligned bounding box from corners
            bbox = [
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
            return bbox
        except Exception as e:
            logger.warning(f"Could not compute object bbox: {e}")
            return [[0, 0, 0], [0, 0, 0]]
        
    def print_agent_state(self) -> None:
        agent = self.sim.get_agent(self.agent_id)
        agent_state = agent.get_state()
        print(f"Agent State: pos: {agent_state.position}, rot: {agent_state.rotation}")

    def print_semantic_objects_info(self) -> None:
        """
        Prints detailed information about all semantic objects in the scene.
        This includes regions and their objects with bounding boxes.
        Useful for validating that semantics have been loaded correctly.
        """
        print("\n" + "="*80)
        print("SEMANTIC OBJECTS VALIDATION")
        print("="*80)

        scene = self.sim.semantic_scene

        if scene is None:
            print("❌ No semantic scene loaded!")
            print("="*80 + "\n")
            return

        print(f"✅ Semantic Scene Loaded")
        print(f"   Levels: {len(scene.levels)}")
        print(f"   Regions: {len(scene.regions)}")
        print(f"   Objects: {len(scene.objects)}")
        print()

        # Print information for each region
        for region in scene.regions:
            region_center = region.aabb.center if hasattr(region.aabb, 'center') else "N/A"
            print(f"\n📍 Region ID: {region.id}")
            print(f"   Category: {region.category.name() if region.category else 'Unknown'}")
            print(f"   Center: {region_center}")
            print(f"   Objects in region: {len(region.objects)}")

            # Print information for each object in the region
            for obj in region.objects:
                obj_center = self.compute_object_center(obj.obb)
                obj_bbox = self.compute_object_bbox(obj.obb)
                category_name = obj.category.name() if obj.category else "Unknown"

                print(f"      └─ Object ID: {obj.id}")
                print(f"         Category: {category_name}")
                print(f"         Center: [{obj_center.x:.3f}, {obj_center.y:.3f}, {obj_center.z:.3f}]")
                print(f"         BBox Min: {obj_bbox[0]}")
                print(f"         BBox Max: {obj_bbox[1]}")

                # Calculate and print bbox dimensions
                bbox_size = [
                    obj_bbox[1][0] - obj_bbox[0][0],
                    obj_bbox[1][1] - obj_bbox[0][1],
                    obj_bbox[1][2] - obj_bbox[0][2]
                ]
                print(f"         Size (WxHxD): [{bbox_size[0]:.3f} x {bbox_size[1]:.3f} x {bbox_size[2]:.3f}] m")

        print("\n" + "="*80)
        print(f"Total: {len(scene.regions)} regions, {len(scene.objects)} objects")
        print("="*80 + "\n")

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

        keys = active_agent_id_and_sensor_name

        if self.enable_batch_renderer:
            self.render_batch()
        else:
            self.sim._Simulator__sensors[keys[0]][keys[1]].draw_observation()
            agent = self.sim.get_agent(keys[0])
            self.render_camera = agent.scene_node.node_sensor_suite.get(keys[1])
            self.debug_draw()
            self.render_camera.render_target.blit_rgba_to_default()

        # draw CPU/GPU usage data and other info to the app window
        mn.gl.default_framebuffer.bind()
        self.draw_text(self.render_camera.specification())

        self.swap_buffers()
        Timer.next_frame()
        self.redraw()

    def default_agent_config(self) -> habitat_sim.agent.AgentConfiguration:
        """
        Set up our own agent and agent controls
        """
        make_action_spec = habitat_sim.agent.ActionSpec
        make_actuation_spec = habitat_sim.agent.ActuationSpec
        # MOVE, LOOK = 0.07, 1.5

        # all of our possible actions' names
        action_list = [
            "move_left",
            "turn_left",
            "move_right",
            "turn_right",
            "move_backward",
            "look_up",
            "move_forward",
            "look_down",
            "move_down",
            "move_up",
        ]

        action_space: Dict[str, habitat_sim.agent.ActionSpec] = {}

        # build our action space map
        for action in action_list:
            actuation_spec_amt = self.MOVE if "move" in action else self.LOOK
            action_spec = make_action_spec(
                action, make_actuation_spec(actuation_spec_amt)
            )
            action_space[action] = action_spec

        sensor_spec: List[habitat_sim.sensor.SensorSpec] = self.cfg.agents[
            self.agent_id
        ].sensor_specifications

        agent_config = habitat_sim.agent.AgentConfiguration(
            height=1.5,
            radius=0.1,
            sensor_specifications=sensor_spec,
            action_space=action_space,
            body_type="cylinder",
        )
        return agent_config

    def reconfigure_sim(self) -> None:
        """
        Utilizes the current `self.sim_settings` to configure and set up a new
        `habitat_sim.Simulator`, and then either starts a simulation instance, or replaces
        the current simulator instance, reloading the most recently loaded scene
        """
        # configure our sim_settings but then set the agent to our default
        self.cfg = make_cfg(self.sim_settings)
        self.agent_id: int = self.sim_settings["default_agent"]
        self.cfg.agents[self.agent_id] = self.default_agent_config()

        if self.enable_batch_renderer:
            self.cfg.enable_batch_renderer = True
            self.cfg.sim_cfg.create_renderer = False
            self.cfg.sim_cfg.enable_gfx_replay_save = True

        if self.sim_settings["use_default_lighting"]:
            logger.info("Setting default lighting override for scene.")
            self.cfg.sim_cfg.override_scene_light_defaults = True
            self.cfg.sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

        if self.sim is None:
            self.tiled_sims = []
            for _i in range(self.num_env):
                self.tiled_sims.append(habitat_sim.Simulator(self.cfg))
            self.sim = self.tiled_sims[0]
        else:  # edge case
            for i in range(self.num_env):
                if (
                    self.tiled_sims[i].config.sim_cfg.scene_id
                    == self.cfg.sim_cfg.scene_id
                ):
                    # we need to force a reset, so change the internal config scene name
                    self.tiled_sims[i].config.sim_cfg.scene_id = "NONE"
                self.tiled_sims[i].reconfigure(self.cfg)

        # post reconfigure
        self.default_agent = self.sim.get_agent(self.agent_id)
        self.render_camera = self.default_agent.scene_node.node_sensor_suite.get(
            "color_sensor"
        )

        # set sim_settings scene name as actual loaded scene
        self.sim_settings["scene"] = self.sim.curr_scene_name

        # Initialize replay renderer
        if self.enable_batch_renderer and self.replay_renderer is None:
            self.replay_renderer_cfg = ReplayRendererConfiguration()
            self.replay_renderer_cfg.num_environments = self.num_env
            self.replay_renderer_cfg.standalone = (
                False  # Context is owned by the GLFW window
            )
            self.replay_renderer_cfg.sensor_specifications = self.cfg.agents[
                self.agent_id
            ].sensor_specifications
            self.replay_renderer_cfg.gpu_device_id = self.cfg.sim_cfg.gpu_device_id
            self.replay_renderer_cfg.force_separate_semantic_scene_graph = False
            self.replay_renderer_cfg.leave_context_with_background_renderer = False
            self.replay_renderer = ReplayRenderer.create_batch_replay_renderer(
                self.replay_renderer_cfg
            )
            # Pre-load composite files
            if sim_settings["composite_files"] is not None:
                for composite_file in sim_settings["composite_files"]:
                    self.replay_renderer.preload_file(composite_file)

        Timer.start()
        self.step = -1

        # --- Validation Print ---
        if self.sim.semantic_scene:
            print(f"✅ Semantic Scene Loaded Successfully")
            print(f"   - Regions: {len(self.sim.semantic_scene.regions)}")
            print(f"   - Objects: {len(self.sim.semantic_scene.objects)}")

            # DEBUG: Check if any objects have non-zero bounding boxes
            non_zero_bbox_count = 0
            for obj in self.sim.semantic_scene.objects:
                half_ext = obj.obb.half_extents
                if half_ext.x > 0 or half_ext.y > 0 or half_ext.z > 0:
                    non_zero_bbox_count += 1

            print(f"   - Objects with non-zero BBoxes: {non_zero_bbox_count}/{len(self.sim.semantic_scene.objects)}")

            if non_zero_bbox_count == 0:
                print(f"\n   ⚠️  WARNING: ALL objects have zero-sized bounding boxes!")
                print(f"   This means semantic mesh color matching FAILED.")
                print(f"\n   Possible causes:")
                print(f"      1. Colors in .semantic.glb don't exactly match .semantic.txt")
                print(f"      2. Semantic mesh file not found/loaded")
                print(f"      3. Habitat-sim needs recompiling after source changes")
                print(f"      4. Semantic mesh has no vertex colors")
                print(f"\n   Try pressing 'B' to see individual object details.\n")
        else:
            print(f"❌ WARNING: Semantic Scene is NULL. Check your dataset config.")
        # ----------------------------------

    def render_batch(self):
        """
        This method updates the replay manager with the current state of environments and renders them.
        """
        for i in range(self.num_env):
            # Apply keyframe
            keyframe = self.tiled_sims[i].gfx_replay_manager.extract_keyframe()
            self.replay_renderer.set_environment_keyframe(i, keyframe)
            # Copy sensor transforms
            sensor_suite = self.tiled_sims[i]._sensors
            for sensor_uuid, sensor in sensor_suite.items():
                transform = sensor._sensor_object.node.absolute_transformation()
                self.replay_renderer.set_sensor_transform(i, sensor_uuid, transform)
        # Render
        self.replay_renderer.render(mn.gl.default_framebuffer)

    def move_and_look(self, repetitions: int) -> None:
        """
        This method is called continuously with `self.draw_event` to monitor
        any changes in the movement keys map `Dict[KeyEvent.key, Bool]`.
        When a key in the map is set to `True` the corresponding action is taken.
        """
        # avoids unnecessary updates to grabber's object position
        if repetitions == 0:
            return

        agent = self.sim.agents[self.agent_id]
        press: Dict[Application.Key.key, bool] = self.pressed
        act: Dict[Application.Key.key, str] = self.key_to_action

        action_queue: List[str] = [act[k] for k, v in press.items() if v]

        for _ in range(int(repetitions)):
            [agent.act(x) for x in action_queue]

        # update the grabber transform when our agent is moved
        if self.mouse_grabber is not None:
            # update location of grabbed object
            self.update_grab_position(self.previous_mouse_point)

    def invert_gravity(self) -> None:
        """
        Sets the gravity vector to the negative of it's previous value. This is
        a good method for testing simulation functionality.
        """
        gravity: mn.Vector3 = self.sim.get_gravity() * -1
        self.sim.set_gravity(gravity)

    def key_press_event(self, event: Application.KeyEvent) -> None:
        """
        Handles `Application.KeyEvent` on a key press by performing the corresponding functions.
        If the key pressed is part of the movement keys map `Dict[KeyEvent.key, Bool]`, then the
        key will be set to False for the next `self.move_and_look()` to update the current actions.
        """
        key = event.key
        pressed = Application.Key
        mod = Application.Modifier

        shift_pressed = bool(event.modifiers & mod.SHIFT)
        alt_pressed = bool(event.modifiers & mod.ALT)
        # warning: ctrl doesn't always pass through with other key-presses

        if key == pressed.ESC:
            event.accepted = True
            self.exit_event(Application.ExitEvent)
            return

        elif key == pressed.H:
            self.print_help_text()
        elif key == pressed.J:
            logger.info(
                f"Toggle Region Draw from {self.semantic_region_debug_draw } to {not self.semantic_region_debug_draw}"
            )
            # Toggle visualize semantic bboxes. Currently only regions supported
            self.semantic_region_debug_draw = not self.semantic_region_debug_draw

        elif key == pressed.K:
            # Save semantic sensor view as JPG
            logger.info("Command: saving semantic sensor view")
            self.save_semantic_image(side_by_side=True)

        elif key == pressed.B:
            # Print bounding boxes and object info for validation
            logger.info("Command: printing semantic objects bounding boxes")
            self.print_semantic_objects_info()

        elif key == pressed.A:
            logger.info("Command: print agent state")
            self.print_agent_state()

        elif key == pressed.TAB:
            # NOTE: (+ALT) - reconfigure without cycling scenes
            if not alt_pressed:
                # cycle the active scene from the set available in MetadataMediator
                inc = -1 if shift_pressed else 1
                scene_ids = self.sim.metadata_mediator.get_scene_handles()
                cur_scene_index = 0
                if self.sim_settings["scene"] not in scene_ids:
                    matching_scenes = [
                        (ix, x)
                        for ix, x in enumerate(scene_ids)
                        if self.sim_settings["scene"] in x
                    ]
                    if not matching_scenes:
                        logger.warning(
                            f"The current scene, '{self.sim_settings['scene']}', is not in the list, starting cycle at index 0."
                        )
                    else:
                        cur_scene_index = matching_scenes[0][0]
                else:
                    cur_scene_index = scene_ids.index(self.sim_settings["scene"])

                next_scene_index = min(
                    max(cur_scene_index + inc, 0), len(scene_ids) - 1
                )
                self.sim_settings["scene"] = scene_ids[next_scene_index]
            self.reconfigure_sim()
            logger.info(
                f"Reconfigured simulator for scene: {self.sim_settings['scene']}"
            )

        elif key == pressed.SPACE:
            if not self.sim.config.sim_cfg.enable_physics:
                logger.warn("Warning: physics was not enabled during setup")
            else:
                self.simulating = not self.simulating
                logger.info(f"Command: physics simulating set to {self.simulating}")

        elif key == pressed.PERIOD:
            if self.simulating:
                logger.warn("Warning: physics simulation already running")
            else:
                self.simulate_single_step = True
                logger.info("Command: physics step taken")

        elif key == pressed.COMMA:
            self.debug_bullet_draw = not self.debug_bullet_draw
            logger.info(f"Command: toggle Bullet debug draw: {self.debug_bullet_draw}")

        elif key == pressed.C:
            if shift_pressed:
                self.contact_debug_draw = not self.contact_debug_draw
                logger.info(
                    f"Command: toggle contact debug draw: {self.contact_debug_draw}"
                )
            else:
                # perform a discrete collision detection pass and enable contact debug drawing to visualize the results
                logger.info(
                    "Command: perform discrete collision detection and visualize active contacts."
                )
                self.sim.perform_discrete_collision_detection()
                self.contact_debug_draw = True
                # TODO: add a nice log message with concise contact pair naming.

        elif key == pressed.T:
            # load URDF
            fixed_base = alt_pressed
            urdf_file_path = ""
            if shift_pressed and self.cached_urdf:
                urdf_file_path = self.cached_urdf
            else:
                urdf_file_path = input("Load URDF: provide a URDF filepath:").strip()

            if not urdf_file_path:
                logger.warn("Load URDF: no input provided. Aborting.")
            elif not urdf_file_path.endswith((".URDF", ".urdf")):
                logger.warn("Load URDF: input is not a URDF. Aborting.")
            elif os.path.exists(urdf_file_path):
                self.cached_urdf = urdf_file_path
                aom = self.sim.get_articulated_object_manager()
                ao = aom.add_articulated_object_from_urdf(
                    urdf_file_path,
                    fixed_base,
                    1.0,
                    1.0,
                    True,
                    maintain_link_order=False,
                    intertia_from_urdf=False,
                )
                ao.translation = (
                    self.default_agent.scene_node.transformation.transform_point(
                        [0.0, 1.0, -1.5]
                    )
                )
                # check removal and auto-creation
                joint_motor_settings = habitat_sim.physics.JointMotorSettings(
                    position_target=0.0,
                    position_gain=1.0,
                    velocity_target=0.0,
                    velocity_gain=1.0,
                    max_impulse=1000.0,
                )
                existing_motor_ids = ao.existing_joint_motor_ids
                for motor_id in existing_motor_ids:
                    ao.remove_joint_motor(motor_id)
                ao.create_all_motors(joint_motor_settings)
            else:
                logger.warn("Load URDF: input file not found. Aborting.")

        elif key == pressed.M:
            self.cycle_mouse_mode()
            logger.info(f"Command: mouse mode set to {self.mouse_interaction}")

        elif key == pressed.V:
            self.invert_gravity()
            logger.info("Command: gravity inverted")
        elif key == pressed.N:
            # (default) - toggle navmesh visualization
            # NOTE: (+ALT) - re-sample the agent position on the NavMesh
            # NOTE: (+SHIFT) - re-compute the NavMesh
            if alt_pressed:
                logger.info("Command: resample agent state from navmesh")
                if self.sim.pathfinder.is_loaded:
                    new_agent_state = habitat_sim.AgentState()
                    new_agent_state.position = (
                        self.sim.pathfinder.get_random_navigable_point()
                    )
                    new_agent_state.rotation = quat_from_angle_axis(
                        self.sim.random.uniform_float(0, 2.0 * np.pi),
                        np.array([0, 1, 0]),
                    )
                    self.default_agent.set_state(new_agent_state)
                else:
                    logger.warning(
                        "NavMesh is not initialized. Cannot sample new agent state."
                    )
            elif shift_pressed:
                logger.info("Command: recompute navmesh")
                self.navmesh_config_and_recompute()
            else:
                if self.sim.pathfinder.is_loaded:
                    self.sim.navmesh_visualization = not self.sim.navmesh_visualization
                    logger.info("Command: toggle navmesh")
                else:
                    logger.warn("Warning: recompute navmesh first")

        # update map of moving/looking keys which are currently pressed
        if key in self.pressed:
            self.pressed[key] = True
        event.accepted = True
        self.redraw()

    def key_release_event(self, event: Application.KeyEvent) -> None:
        """
        Handles `Application.KeyEvent` on a key release. When a key is released, if it
        is part of the movement keys map `Dict[KeyEvent.key, Bool]`, then the key will
        be set to False for the next `self.move_and_look()` to update the current actions.
        """
        key = event.key

        # update map of moving/looking keys which are currently pressed
        if key in self.pressed:
            self.pressed[key] = False
        event.accepted = True
        self.redraw()

    def pointer_move_event(self, event: Application.PointerMoveEvent) -> None:
        """
        Handles `Application.PointerMoveEvent`. When in LOOK mode, enables the left
        mouse button to steer the agent's facing direction. When in GRAB mode,
        continues to update the grabber's object position with our agents position.
        """
        # if interactive mode -> LOOK MODE
        if (
            event.pointers & Application.Pointer.MOUSE_LEFT
            and self.mouse_interaction == MouseMode.LOOK
        ):
            agent = self.sim.agents[self.agent_id]
            delta = self.get_mouse_position(event.relative_position) / 2
            action = habitat_sim.agent.ObjectControls()
            act_spec = habitat_sim.agent.ActuationSpec

            # left/right on agent scene node
            action(agent.scene_node, "turn_right", act_spec(delta.x))

            # up/down on cameras' scene nodes
            action = habitat_sim.agent.ObjectControls()
            sensors = list(self.default_agent.scene_node.subtree_sensors.values())
            [action(s.object, "look_down", act_spec(delta.y), False) for s in sensors]

        # if interactive mode is TRUE -> GRAB MODE
        elif self.mouse_interaction == MouseMode.GRAB and self.mouse_grabber:
            # update location of grabbed object
            self.update_grab_position(self.get_mouse_position(event.position))

        self.previous_mouse_point = self.get_mouse_position(event.position)
        self.redraw()
        event.accepted = True

    def pointer_press_event(self, event: Application.PointerEvent) -> None:
        """
        Handles `Application.PointerEvent`. When in GRAB mode, click on
        objects to drag their position. (right-click for fixed constraints)
        """
        physics_enabled = self.sim.get_physics_simulation_library()

        # if interactive mode is True -> GRAB MODE
        if self.mouse_interaction == MouseMode.GRAB and physics_enabled:
            render_camera = self.render_camera.render_camera
            ray = render_camera.unproject(self.get_mouse_position(event.position))
            raycast_results = self.sim.cast_ray(ray=ray)

            if raycast_results.has_hits():
                hit_object, ao_link = -1, -1
                hit_info = raycast_results.hits[0]

                if hit_info.object_id > habitat_sim.stage_id:
                    # we hit an non-staged collision object
                    ro_mngr = self.sim.get_rigid_object_manager()
                    ao_mngr = self.sim.get_articulated_object_manager()
                    ao = ao_mngr.get_object_by_id(hit_info.object_id)
                    ro = ro_mngr.get_object_by_id(hit_info.object_id)

                    if ro:
                        # if grabbed an object
                        hit_object = hit_info.object_id
                        object_pivot = ro.transformation.inverted().transform_point(
                            hit_info.point
                        )
                        object_frame = ro.rotation.inverted()
                    elif ao:
                        # if grabbed the base link
                        hit_object = hit_info.object_id
                        object_pivot = ao.transformation.inverted().transform_point(
                            hit_info.point
                        )
                        object_frame = ao.rotation.inverted()
                    else:
                        for ao_handle in ao_mngr.get_objects_by_handle_substring():
                            ao = ao_mngr.get_object_by_handle(ao_handle)
                            link_to_obj_ids = ao.link_object_ids

                            if hit_info.object_id in link_to_obj_ids:
                                # if we got a link
                                ao_link = link_to_obj_ids[hit_info.object_id]
                                object_pivot = (
                                    ao.get_link_scene_node(ao_link)
                                    .transformation.inverted()
                                    .transform_point(hit_info.point)
                                )
                                object_frame = ao.get_link_scene_node(
                                    ao_link
                                ).rotation.inverted()
                                hit_object = ao.object_id
                                break
                    # done checking for AO

                    if hit_object >= 0:
                        node = self.default_agent.scene_node
                        constraint_settings = physics.RigidConstraintSettings()

                        constraint_settings.object_id_a = hit_object
                        constraint_settings.link_id_a = ao_link
                        constraint_settings.pivot_a = object_pivot
                        constraint_settings.frame_a = (
                            object_frame.to_matrix() @ node.rotation.to_matrix()
                        )
                        constraint_settings.frame_b = node.rotation.to_matrix()
                        constraint_settings.pivot_b = hit_info.point

                        # by default use a point 2 point constraint
                        if event.pointer == Application.Pointer.MOUSE_RIGHT:
                            constraint_settings.constraint_type = (
                                physics.RigidConstraintType.Fixed
                            )

                        grip_depth = (
                            hit_info.point - render_camera.node.absolute_translation
                        ).length()

                        self.mouse_grabber = MouseGrabber(
                            constraint_settings,
                            grip_depth,
                            self.sim,
                        )
                    else:
                        logger.warning(
                            "Oops, couldn't find the hit object. That's odd."
                        )
                # end if didn't hit the scene
            # end has raycast hit
        # end has physics enabled

        self.previous_mouse_point = self.get_mouse_position(event.position)
        self.redraw()
        event.accepted = True

    def scroll_event(self, event: Application.ScrollEvent) -> None:
        """
        Handles `Application.ScrollEvent`. When in LOOK mode, enables camera
        zooming (fine-grained zoom using shift) When in GRAB mode, adjusts the depth
        of the grabber's object. (larger depth change rate using shift)
        """
        scroll_mod_val = (
            event.offset.y
            if abs(event.offset.y) > abs(event.offset.x)
            else event.offset.x
        )
        if not scroll_mod_val:
            return

        # use shift to scale action response
        shift_pressed = bool(event.modifiers & Application.Modifier.SHIFT)
        alt_pressed = bool(event.modifiers & Application.Modifier.ALT)
        ctrl_pressed = bool(event.modifiers & Application.Modifier.CTRL)

        # if interactive mode is False -> LOOK MODE
        if self.mouse_interaction == MouseMode.LOOK:
            # use shift for fine-grained zooming
            mod_val = 1.01 if shift_pressed else 1.1
            mod = mod_val if scroll_mod_val > 0 else 1.0 / mod_val
            cam = self.render_camera
            cam.zoom(mod)
            self.redraw()

        elif self.mouse_interaction == MouseMode.GRAB and self.mouse_grabber:
            # adjust the depth
            mod_val = 0.1 if shift_pressed else 0.01
            scroll_delta = scroll_mod_val * mod_val
            if alt_pressed or ctrl_pressed:
                # rotate the object's local constraint frame
                agent_t = self.default_agent.scene_node.transformation_matrix()
                # ALT - yaw
                rotation_axis = agent_t.transform_vector(mn.Vector3(0, 1, 0))
                if alt_pressed and ctrl_pressed:
                    # ALT+CTRL - roll
                    rotation_axis = agent_t.transform_vector(mn.Vector3(0, 0, -1))
                elif ctrl_pressed:
                    # CTRL - pitch
                    rotation_axis = agent_t.transform_vector(mn.Vector3(1, 0, 0))
                self.mouse_grabber.rotate_local_frame_by_global_angle_axis(
                    rotation_axis, mn.Rad(scroll_delta)
                )
            else:
                # update location of grabbed object
                self.mouse_grabber.grip_depth += scroll_delta
                self.update_grab_position(self.get_mouse_position(event.position))
        self.redraw()
        event.accepted = True

    def pointer_release_event(self, event: Application.PointerEvent) -> None:
        """
        Release any existing constraints.
        """
        del self.mouse_grabber
        self.mouse_grabber = None
        event.accepted = True

    def update_grab_position(self, point: mn.Vector2i) -> None:
        """
        Accepts a point derived from a mouse click event and updates the
        transform of the mouse grabber.
        """
        # check mouse grabber
        if not self.mouse_grabber:
            return

        render_camera = self.render_camera.render_camera
        ray = render_camera.unproject(point)

        rotation: mn.Matrix3x3 = self.default_agent.scene_node.rotation.to_matrix()
        translation: mn.Vector3 = (
            render_camera.node.absolute_translation
            + ray.direction * self.mouse_grabber.grip_depth
        )
        self.mouse_grabber.update_transform(mn.Matrix4.from_(rotation, translation))

    def get_mouse_position(self, mouse_event_position: mn.Vector2i) -> mn.Vector2i:
        """
        This function will get a screen-space mouse position appropriately
        scaled based on framebuffer size and window size.  Generally these would be
        the same value, but on certain HiDPI displays (Retina displays) they may be
        different.
        """
        scaling = mn.Vector2i(self.framebuffer_size) / mn.Vector2i(self.window_size)
        return mouse_event_position * scaling

    def cycle_mouse_mode(self) -> None:
        """
        This method defines how to cycle through the mouse mode.
        """
        if self.mouse_interaction == MouseMode.LOOK:
            self.mouse_interaction = MouseMode.GRAB
        elif self.mouse_interaction == MouseMode.GRAB:
            self.mouse_interaction = MouseMode.LOOK

    def navmesh_config_and_recompute(self) -> None:
        """
        This method is setup to be overridden in for setting config accessibility
        in inherited classes.

        Navmesh recompute on a 4 M-vert HGE stage is ~12 s (Recast is single-
        threaded). Cache the result to a `.navmesh` sidecar keyed by
        `(scene_id, height, radius)` so subsequent launches skip the recompute.
        """
        self.navmesh_settings = habitat_sim.NavMeshSettings()
        self.navmesh_settings.set_defaults()
        self.navmesh_settings.agent_height = self.cfg.agents[self.agent_id].height
        self.navmesh_settings.agent_radius = self.cfg.agents[self.agent_id].radius
        self.navmesh_settings.include_static_objects = True

        # Cache path lives next to the scene file so it invalidates naturally
        # when the scene moves or gets rebuilt. Hash the settings that affect
        # the navmesh so different agent sizes don't share a cache file.
        scene_id = str(self.cfg.sim_cfg.scene_id)
        scene_path = Path(scene_id)
        import hashlib
        settings_key = hashlib.sha1(
            f"{self.navmesh_settings.agent_height:.4f}|"
            f"{self.navmesh_settings.agent_radius:.4f}|"
            f"{self.navmesh_settings.include_static_objects}".encode("utf-8")
        ).hexdigest()[:12]
        cache_path = scene_path.with_suffix(f".{settings_key}.navmesh")

        if cache_path.exists():
            try:
                t0 = time.time()
                loaded = self.sim.pathfinder.load_nav_mesh(str(cache_path))
                if loaded:
                    logger.info(
                        f"navmesh cache HIT: {cache_path.name} "
                        f"({time.time()-t0:.2f}s)"
                    )
                    return
                logger.warning(
                    f"[WARN] navmesh load: expected=readable {cache_path}, "
                    f"got=load_nav_mesh returned False, fallback=recompute"
                )
            except Exception as exc:
                logger.warning(
                    f"[WARN] navmesh load: expected=readable {cache_path}, "
                    f"got={type(exc).__name__}: {exc}, fallback=recompute"
                )

        t0 = time.time()
        self.sim.recompute_navmesh(
            self.sim.pathfinder,
            self.navmesh_settings,
        )
        logger.info(f"navmesh recomputed in {time.time()-t0:.2f}s")
        try:
            self.sim.pathfinder.save_nav_mesh(str(cache_path))
            logger.info(f"navmesh cached to {cache_path.name}")
        except Exception as exc:
            logger.warning(
                f"[WARN] navmesh save: expected=writable {cache_path}, "
                f"got={type(exc).__name__}: {exc}, fallback=will recompute next launch"
            )

    def exit_event(self, event: Application.ExitEvent):
        """
        Overrides exit_event to properly close the Simulator before exiting the
        application.
        """
        for i in range(self.num_env):
            self.tiled_sims[i].close(destroy=True)
            event.accepted = True
        exit(0)

    def draw_text(self, sensor_spec):
        # make magnum text background transparent for text
        mn.gl.Renderer.enable(mn.gl.Renderer.Feature.BLENDING)
        mn.gl.Renderer.set_blend_function(
            mn.gl.Renderer.BlendFunction.ONE,
            mn.gl.Renderer.BlendFunction.ONE_MINUS_SOURCE_ALPHA,
        )

        self.shader.bind_vector_texture(self.glyph_cache.texture)
        self.shader.transformation_projection_matrix = self.window_text_transform
        self.shader.color = [1.0, 1.0, 1.0]

        sensor_type_string = str(sensor_spec.sensor_type.name)
        sensor_subtype_string = str(sensor_spec.sensor_subtype.name)
        if self.mouse_interaction == MouseMode.LOOK:
            mouse_mode_string = "LOOK"
        elif self.mouse_interaction == MouseMode.GRAB:
            mouse_mode_string = "GRAB"
        self.window_text.render(
            f"""
{self.fps} FPS
Sensor Type: {sensor_type_string}
Sensor Subtype: {sensor_subtype_string}
Mouse Interaction Mode: {mouse_mode_string}
            """
        )
        self.shader.draw(self.window_text.mesh)

        # Disable blending for text
        mn.gl.Renderer.disable(mn.gl.Renderer.Feature.BLENDING)

    def print_help_text(self) -> None:
        """
        Print the Key Command help text.
        """
        logger.info(
            """
=====================================================
Welcome to the Habitat-sim Python Viewer application!
=====================================================
Mouse Functions ('m' to toggle mode):
----------------
In LOOK mode (default):
    LEFT:
        Click and drag to rotate the agent and look up/down.
    WHEEL:
        Modify orthographic camera zoom/perspective camera FOV (+SHIFT for fine grained control)

In GRAB mode (with 'enable-physics'):
    LEFT:
        Click and drag to pickup and move an object with a point-to-point constraint (e.g. ball joint).
    RIGHT:
        Click and drag to pickup and move an object with a fixed frame constraint.
    WHEEL (with picked object):
        default - Pull gripped object closer or push it away.
        (+ALT) rotate object fixed constraint frame (yaw)
        (+CTRL) rotate object fixed constraint frame (pitch)
        (+ALT+CTRL) rotate object fixed constraint frame (roll)
        (+SHIFT) amplify scroll magnitude


Key Commands:
-------------
    esc:        Exit the application.
    'h':        Display this help message.
    'm':        Cycle mouse interaction modes.

    Agent Controls:
    'wasd':     Move the agent's body forward/backward and left/right.
    'zx':       Move the agent's body up/down.
    arrow keys: Turn the agent's body left/right and camera look up/down.

    Utilities:
    'r':        Reset the simulator with the most recently loaded scene.
    'n':        Show/hide NavMesh wireframe.
                (+SHIFT) Recompute NavMesh with default settings.
                (+ALT) Re-sample the agent(camera)'s position and orientation from the NavMesh.
    ',':        Render a Bullet collision shape debug wireframe overlay (white=active, green=sleeping, blue=wants sleeping, red=can't sleep).
    'c':        Run a discrete collision detection pass and render a debug wireframe overlay showing active contact points and normals (yellow=fixed length normals, red=collision distances).
                (+SHIFT) Toggle the contact point debug render overlay on/off.
    'j':        Toggle Semantic visualization bounds (currently only Semantic Region annotations)
    'k':        Save current semantic sensor view as JPG image to 'semantic_captures/' directory
    'b':        Print detailed semantic objects info with bounding boxes (for validation)

    Object Interactions:
    SPACE:      Toggle physics simulation on/off.
    '.':        Take a single simulation step if not simulating continuously.
    'v':        (physics) Invert gravity.
    't':        Load URDF from filepath
                (+SHIFT) quick re-load the previously specified URDF
                (+ALT) load the URDF with fixed base
=====================================================
"""
        )


class MouseMode(Enum):
    LOOK = 0
    GRAB = 1
    MOTION = 2


class MouseGrabber:
    """
    Create a MouseGrabber from RigidConstraintSettings to manipulate objects.
    """

    def __init__(
        self,
        settings: physics.RigidConstraintSettings,
        grip_depth: float,
        sim: habitat_sim.simulator.Simulator,
    ) -> None:
        self.settings = settings
        self.simulator = sim

        # defines distance of the grip point from the camera for pivot updates
        self.grip_depth = grip_depth
        self.constraint_id = sim.create_rigid_constraint(settings)

    def __del__(self):
        self.remove_constraint()

    def remove_constraint(self) -> None:
        """
        Remove a rigid constraint by id.
        """
        self.simulator.remove_rigid_constraint(self.constraint_id)

    def updatePivot(self, pos: mn.Vector3) -> None:
        self.settings.pivot_b = pos
        self.simulator.update_rigid_constraint(self.constraint_id, self.settings)

    def update_frame(self, frame: mn.Matrix3x3) -> None:
        self.settings.frame_b = frame
        self.simulator.update_rigid_constraint(self.constraint_id, self.settings)

    def update_transform(self, transform: mn.Matrix4) -> None:
        self.settings.frame_b = transform.rotation()
        self.settings.pivot_b = transform.translation
        self.simulator.update_rigid_constraint(self.constraint_id, self.settings)

    def rotate_local_frame_by_global_angle_axis(
        self, axis: mn.Vector3, angle: mn.Rad
    ) -> None:
        """rotate the object's local constraint frame with a global angle axis input."""
        object_transform = mn.Matrix4()
        rom = self.simulator.get_rigid_object_manager()
        aom = self.simulator.get_articulated_object_manager()
        if rom.get_library_has_id(self.settings.object_id_a):
            object_transform = rom.get_object_by_id(
                self.settings.object_id_a
            ).transformation
        else:
            # must be an ao
            object_transform = (
                aom.get_object_by_id(self.settings.object_id_a)
                .get_link_scene_node(self.settings.link_id_a)
                .transformation
            )
        local_axis = object_transform.inverted().transform_vector(axis)
        R = mn.Matrix4.rotation(angle, local_axis.normalized())
        self.settings.frame_a = R.rotation().__matmul__(self.settings.frame_a)
        self.simulator.update_rigid_constraint(self.constraint_id, self.settings)


class Timer:
    """
    Timer class used to keep track of time between buffer swaps
    and guide the display frame rate.
    """

    start_time = 0.0
    prev_frame_time = 0.0
    prev_frame_duration = 0.0
    running = False

    @staticmethod
    def start() -> None:
        """
        Starts timer and resets previous frame time to the start time.
        """
        Timer.running = True
        Timer.start_time = time.time()
        Timer.prev_frame_time = Timer.start_time
        Timer.prev_frame_duration = 0.0

    @staticmethod
    def stop() -> None:
        """
        Stops timer and erases any previous time data, resetting the timer.
        """
        Timer.running = False
        Timer.start_time = 0.0
        Timer.prev_frame_time = 0.0
        Timer.prev_frame_duration = 0.0

    @staticmethod
    def next_frame() -> None:
        """
        Records previous frame duration and updates the previous frame timestamp
        to the current time. If the timer is not currently running, perform nothing.
        """
        if not Timer.running:
            return
        Timer.prev_frame_duration = time.time() - Timer.prev_frame_time
        Timer.prev_frame_time = time.time()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # optional arguments
    parser.add_argument(
        "--scene",
        # default="./data/test_assets/scenes/simple_room.glb",
        default="./data/scene_datasets/HGE/HGE.basis.glb",
        type=str,
        help='scene/stage file to load (default: "./data/test_assets/scenes/simple_room.glb")',
    )
    parser.add_argument(
        "--dataset",
        # default="default",
        default="data/scene_datasets/HGE.scene_dataset_config.json",
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
    HabitatSimInteractiveViewer(sim_settings).exec()
