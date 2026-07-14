#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Conversational Navigation Assistant - web backend.

Streams the live Habitat-Sim render to the browser (MJPEG), accepts the same
keyboard controls used by the native ``mr_viewer.py`` window (WASD / ZX +
arrow keys) and exposes the LLM conversation / navigation pipeline over HTTP.

It reuses *all* of the existing logic from ``mr_viewer.py`` (the ``NewViewer``
class, the ``user_input_logic_loop`` conversation pipeline and the local model
loader) so the web experience is identical to the desktop viewer, only the I/O
surface changes from a native GLFW window + Qt GUI to a browser.

Endpoints
---------
GET  /health   -> liveness probe
GET  /status   -> llm status, scene, current room, control map
GET  /video    -> multipart MJPEG stream of the live render
GET  /frame    -> single JPEG snapshot of the live render
POST /key      -> {"key": "w", "down": true}  set a movement key state
POST /action   -> {"action": "move_forward"}  single-step action (touch UI)
POST /chat     -> {"message": "..."}          converse / navigate, returns text
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from io import BytesIO
from typing import Any, Dict

import numpy as np
from PIL import Image

# Import the desktop viewer module. Importing it pulls in magnum / habitat_sim /
# torch exactly like running mr_viewer.py, but does not execute its __main__
# block, so we can reuse every class and helper without duplicating logic.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import mr_viewer  # noqa: E402
from mr_viewer import (  # noqa: E402
    NewViewer,
    load_local_model,
    user_input_logic_loop,
)

from flask import Flask, Response, jsonify, request, send_from_directory  # noqa: E402
from flask_cors import CORS  # noqa: E402

try:
    from flask_sock import Sock  # noqa: E402

    _HAS_SOCK = True
except Exception:  # pragma: no cover - optional dependency
    Sock = None  # type: ignore[assignment]
    _HAS_SOCK = False
from habitat_sim.utils.settings import default_sim_settings  # noqa: E402
from magnum.platform.glfw import Application  # noqa: E402


# Ordered list of the actions the agent understands (same as the desktop viewer).
ACTION_NAMES = (
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
)

# Browser key name -> agent action, mirroring mr_viewer.NewViewer.key_to_action.
WEB_KEY_TO_ACTION = {
    "w": "move_forward",
    "a": "move_left",
    "s": "move_backward",
    "d": "move_right",
    "z": "move_up",
    "x": "move_down",
    "arrowup": "look_up",
    "arrowdown": "look_down",
    "arrowleft": "turn_left",
    "arrowright": "turn_right",
    # convenience aliases
    "up": "look_up",
    "down": "look_down",
    "left": "turn_left",
    "right": "turn_right",
}

# Keys that must NOT be forwarded from the web (would freeze or kill the server):
#   escape -> exits the app, tab -> heavy scene reconfigure, t -> blocking input()
DENY_KEYS = {"escape", "esc", "tab", "enter", "return", "t"}

# Categories ignored when rebuilding per-scene semantic metadata (same as mr_viewer).
IGNORE_CATEGORIES = [
    "ceiling", "floor", "wall", "handle", "window frame", "door frame",
    "frame", "unknown", "stairs", "staircase", "stair", "stairway",
]


class _FakeKeyEvent:
    """Minimal stand-in for a Magnum KeyEvent to drive the viewer's handlers."""

    __slots__ = ("key", "modifiers", "accepted")

    def __init__(self, key, modifiers) -> None:
        self.key = key
        self.modifiers = modifiers
        self.accepted = False


class WebViewer(NewViewer):
    """``NewViewer`` that also captures frames and accepts remote key input.

    The heavy lifting (rendering, movement, navigation, LLM) is unchanged - we
    only add a throttled frame grab inside ``draw_event`` and a couple of
    thread-safe hooks the Flask server can call from its own threads.
    """

    def __init__(
        self,
        sim_settings: Dict[str, Any],
        stream_fps: int = 30,
        move_amount: float = 0.07,
        look_amount: float = 1.5,
    ) -> None:
        # Same navigation feel as the desktop viewer (0.07 / 1.5).
        # Set before super().__init__ so the agent action space is built with
        # these amounts (default_agent_config reads self.MOVE / self.LOOK).
        self.MOVE = move_amount
        self.LOOK = look_amount
        super().__init__(sim_settings, q_app=None)

        # Throttled JPEG frame shared with the Flask streaming thread.
        self._latest_frame: bytes | None = None
        self._frame_id = 0
        self._frame_lock = threading.Lock()
        self._stream_interval = 1.0 / max(1, stream_fps)
        self._last_capture = 0.0
        self._jpeg_quality = 70
        self._color_sensor = None

        # Raw-frame hand-off so JPEG encoding runs OFF the render thread
        # (the render thread only does a fast GPU read-back -> much less lag).
        self._raw_frame = None
        self._raw_lock = threading.Lock()
        self._raw_event = threading.Event()
        self._encoder_thread = threading.Thread(
            target=self._encoder_loop, daemon=True
        )
        self._encoder_thread.start()

        # Map browser key names to the Magnum keys already tracked in self.pressed.
        k = Application.Key
        self._web_key_map = {
            "w": k.W,
            "a": k.A,
            "s": k.S,
            "d": k.D,
            "z": k.Z,
            "x": k.X,
            "arrowup": k.UP,
            "arrowdown": k.DOWN,
            "arrowleft": k.LEFT,
            "arrowright": k.RIGHT,
            "up": k.UP,
            "down": k.DOWN,
            "left": k.LEFT,
            "right": k.RIGHT,
        }

        # A zero-valued Modifier (SHIFT & ALT are disjoint flags -> empty set),
        # used when synthesizing key events for non-movement keys.
        try:
            self._mod_zero = Application.Modifier.SHIFT & Application.Modifier.ALT
        except Exception:
            self._mod_zero = 0

    # ------------------------------------------------------------------ render
    def draw_event(self, *args, **kwargs) -> None:  # type: ignore[override]
        super().draw_event(*args, **kwargs)
        self._maybe_capture_frame()

    def _get_color_sensor(self):
        if self._color_sensor is not None:
            return self._color_sensor
        try:
            sensors = self.sim._Simulator__sensors[self.agent_id]
            if isinstance(sensors, dict):
                self._color_sensor = sensors.get("color_sensor")
            else:
                self._color_sensor = sensors["color_sensor"]
        except Exception:
            self._color_sensor = None
        return self._color_sensor

    def _maybe_capture_frame(self) -> None:
        now = time.time()
        if now - self._last_capture < self._stream_interval:
            return
        self._last_capture = now
        try:
            sensor = self._get_color_sensor()
            if sensor is None:
                return
            # The color sensor was already drawn by draw_event this frame, so we
            # only read it back here (no semantic/depth re-render). We just copy
            # the pixels and hand them to the encoder thread, keeping the render
            # loop responsive.
            rgb = np.asarray(sensor.get_observation())
            if rgb is None or rgb.size == 0:
                return
            if rgb.ndim == 3 and rgb.shape[2] >= 3:
                rgb = rgb[..., :3]
            # Single contiguous uint8 copy (conversion + copy in one step).
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            with self._raw_lock:
                self._raw_frame = rgb
            self._raw_event.set()
        except Exception as err:  # pragma: no cover - defensive, keep loop alive
            print(f"[CNA] frame capture failed: {err}")

    def _encoder_loop(self) -> None:
        """Encode the latest raw frame to JPEG off the render thread."""
        while True:
            self._raw_event.wait()
            self._raw_event.clear()
            with self._raw_lock:
                rgb = self._raw_frame
            if rgb is None:
                continue
            try:
                buffer = BytesIO()
                Image.fromarray(rgb, mode="RGB").save(
                    buffer, format="JPEG", quality=self._jpeg_quality
                )
                data = buffer.getvalue()
            except Exception as err:  # pragma: no cover
                print(f"[CNA] frame encode failed: {err}")
                continue
            with self._frame_lock:
                self._latest_frame = data
                self._frame_id += 1

    def get_latest_frame(self) -> bytes | None:
        with self._frame_lock:
            return self._latest_frame

    def get_latest_frame_versioned(self) -> tuple[int, bytes | None]:
        with self._frame_lock:
            return self._frame_id, self._latest_frame

    # ---------------------------------------------------------------- controls
    def web_key_event(
        self,
        key_str: str,
        down: bool,
        shift: bool = False,
        alt: bool = False,
        ctrl: bool = False,
    ) -> bool:
        """Handle a key from the web client.

        Movement/look keys are held (continuous) via the ``pressed`` map, exactly
        like the desktop viewer. Every other key (e.g. ``B`` for bounding boxes)
        is replayed through the viewer's real ``key_press_event`` on the render
        thread, so all viewer shortcuts work identically.
        """
        s = (key_str or "").lower()
        if s in DENY_KEYS:
            return False

        # Movement / look keys -> held state (clean, no per-press side effects).
        magnum_key = self._web_key_map.get(s)
        if magnum_key is not None:
            self.pressed[magnum_key] = bool(down)
            return True

        # Any other key -> replicate the native key press/release on the GL thread.
        resolved = self._resolve_extra_key(s)
        if resolved is None:
            return False
        if down:
            self.action_queue.put(
                (self._dispatch_key_press, (resolved, shift, alt, ctrl), {})
            )
        else:
            self.action_queue.put((self._dispatch_key_release, (resolved,), {}))
        return True

    def _resolve_extra_key(self, s: str):
        """Map a browser key name to a Magnum Application.Key (letters + a few)."""
        if len(s) == 1 and "a" <= s <= "z":
            return getattr(Application.Key, s.upper(), None)
        extras = {" ": "SPACE", "space": "SPACE", ",": "COMMA", ".": "PERIOD"}
        name = extras.get(s)
        if name is not None:
            return getattr(Application.Key, name, None)
        return None

    def _make_modifiers(self, shift: bool, alt: bool, ctrl: bool):
        mods = self._mod_zero
        try:
            if shift:
                mods = mods | Application.Modifier.SHIFT
            if alt:
                mods = mods | Application.Modifier.ALT
            if ctrl:
                mods = mods | Application.Modifier.CTRL
        except Exception:
            pass
        return mods

    def _dispatch_key_press(self, magnum_key, shift, alt, ctrl) -> None:
        event = _FakeKeyEvent(magnum_key, self._make_modifiers(shift, alt, ctrl))
        try:
            self.key_press_event(event)
        except Exception as err:
            print(f"[CNA] key_press_event failed for {magnum_key}: {err}")

    def _dispatch_key_release(self, magnum_key) -> None:
        event = _FakeKeyEvent(magnum_key, self._mod_zero)
        try:
            self.key_release_event(event)
        except Exception as err:
            print(f"[CNA] key_release_event failed for {magnum_key}: {err}")

    def web_set_key(self, key_str: str, down: bool) -> bool:
        """Backward-compatible movement-only helper."""
        return self.web_key_event(key_str, down)

    def web_release_all(self) -> None:
        for magnum_key in self._web_key_map.values():
            self.pressed[magnum_key] = False

    def web_tap_action(self, action_name: str) -> bool:
        """Queue a single-step action (used by the on-screen touch pad)."""
        if action_name not in ACTION_NAMES:
            return False
        # Run on the render thread via the existing action queue.
        self.action_queue.put((self._web_do_action, (action_name,), {}))
        return True

    def _web_do_action(self, action_name: str) -> None:
        agent = self.sim.agents[self.agent_id]
        agent.act(action_name)

    # ------------------------------------------------------------ scene switch
    def reconfigure_scene(self, scene_path: str, done: "queue.Queue | None" = None) -> None:
        """Reload a different scene at runtime (keeps the LLM loaded).

        Must run on the render thread (queued via action_queue).
        """
        try:
            self.sim_settings["scene"] = scene_path
            # Recreate the simulator from scratch. An in-place reconfigure leaves
            # the semantic mesh unmatched -> all object OBBs become zero-sized ->
            # room bounding boxes degenerate and rooms/objects are not detected.
            # A fresh Simulator reloads the semantics correctly, like at startup.
            try:
                if self.sim is not None:
                    self.sim.close(destroy=True)
            except Exception as close_err:
                print(f"[CNA] sim close warning: {close_err}")
            self.sim = None
            self.reconfigure_sim()
            self.scene = self.sim.semantic_scene
            self._color_sensor = None
            self.clusters_to_draw = None
            self.show_object_bboxes = False
            self.show_all_object_bboxes = False
            self._object_bbox_colors = {}
            self.prev_objs_to_draw = None

            base_path = os.path.dirname(scene_path)
            scene_name = os.path.splitext(os.path.basename(scene_path))[0]
            semantic_path = os.path.join(
                base_path, f"{scene_name.split('.')[0]}.semantic.txt"
            )
            map_file_path = os.path.join(base_path, "room_id_to_name_map.json")
            with open(map_file_path, "r", encoding="utf-8") as f:
                self.map_room_id_to_name = json.load(f)

            self.room_objects_occurences = self.get_semantic_info(
                semantic_path,
                map_room_id_to_name=self.map_room_id_to_name,
                ignore_categories=IGNORE_CATEGORIES,
            )
            self.objects = self.get_objs_from_sim()
            self.cluster_cnt = 0
            self.clusters = self.cluster_objs(distance_thresh=0.5)
            self.rooms = self.get_rooms_from_sim()

            # --- Diagnostics: verify semantic metadata actually reloaded ---
            try:
                regions = list(self.scene.regions) if self.scene else []
                sample_ids = [getattr(r, "id", "?") for r in regions[:6]]
                print(
                    f"[CNA][scene] {scene_path}\n"
                    f"[CNA][scene]   regions={len(regions)} "
                    f"sample_region_ids={sample_ids}\n"
                    f"[CNA][scene]   map_keys={list(self.map_room_id_to_name.keys())}\n"
                    f"[CNA][scene]   objects={len(self.objects)} "
                    f"clusters={len(self.clusters)} "
                    f"rooms={len(self.rooms)} "
                    f"room_names={[r.get('name') for r in self.rooms.values()]}",
                    flush=True,
                )
            except Exception as diag_err:
                print(f"[CNA][scene] diagnostics failed: {diag_err}", flush=True)

            # Deterministic "default" spawn for the house: seeding the pathfinder
            # makes get_random_navigable_point return the same spot every time you
            # load this scene (instead of a different random point each switch).
            if self.sim.pathfinder.is_loaded:
                self.sim.pathfinder.seed(1)
                agent = self.sim.get_agent(self.agent_id)
                state = agent.get_state()
                state.position = self.sim.pathfinder.get_random_navigable_point()
                agent.set_state(state)

            print(f"[CNA] Scene switched to {scene_path}")
            if done is not None:
                done.put(None)
        except Exception as err:  # pragma: no cover
            import traceback

            traceback.print_exc()
            if done is not None:
                done.put(str(err))

    # ------------------------------------------------------------------- state
    def current_room_name(self) -> str | None:
        try:
            agent_state = self.sim.get_agent(self.agent_id).get_state()
            room = self.get_room_from_position(agent_state.position)
            if room is not None:
                return room.get("name", "unknown_room")
        except Exception:
            pass
        return None


class ConversationalNavigationServer:
    """Flask app exposing the streaming + control + conversation endpoints."""

    def __init__(
        self,
        viewer: WebViewer,
        input_q: queue.Queue,
        output_q: queue.Queue,
        model,
        backend: str,
        stream_fps: int = 30,
        chat_timeout: float = 120.0,
        webapp_dist: str | None = None,
        scenes_dir: str | None = None,
        current_scene: str | None = None,
        finetuned: bool = False,
    ) -> None:
        self.viewer = viewer
        self.input_q = input_q
        self.output_q = output_q
        self.model = model
        self.backend = backend
        self.finetuned = bool(finetuned)
        self.stream_fps = max(1, stream_fps)
        self.chat_timeout = chat_timeout
        self.webapp_dist = webapp_dist
        self.scenes_dir = scenes_dir
        self.current_scene = os.path.normpath(current_scene) if current_scene else ""
        self._chat_lock = threading.Lock()
        self._scene_lock = threading.Lock()
        self._llm_lock = threading.Lock()

        self.app = Flask(__name__, static_folder=None)
        CORS(self.app, resources={r"/*": {"origins": "*"}})
        # Quiet the dev server's per-request access log (e.g. the /status polls).
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

        self.app.route("/health", methods=["GET"])(self.health)
        self.app.route("/status", methods=["GET"])(self.status)
        self.app.route("/video", methods=["GET"])(self.video)
        self.app.route("/frame", methods=["GET"])(self.frame)
        self.app.route("/key", methods=["POST"])(self.key)
        self.app.route("/action", methods=["POST"])(self.action)
        self.app.route("/chat", methods=["POST"])(self.chat)
        self.app.route("/scenes", methods=["GET"])(self.scenes)
        self.app.route("/scene", methods=["POST"])(self.set_scene)
        self.app.route("/llm", methods=["POST"])(self.set_llm)

        # Low-latency WebSocket channel: instant key input + backpressured video
        # (only the freshest frame is sent, and only once the client is ready,
        # so the input->display loop stays tight).
        if _HAS_SOCK:
            self.sock = Sock(self.app)
            self.sock.route("/ws")(self.ws_handler)
            print("[CNA] WebSocket channel enabled at /ws")
        else:
            print("[CNA] flask-sock not installed; WebSocket disabled (MJPEG only)")

        # Serve the built single-page webapp from the same origin (so the whole
        # demo lives on one host:port and needs no separate frontend server).
        if self.webapp_dist and os.path.isdir(self.webapp_dist):
            self.app.route("/", methods=["GET"])(self.serve_index)
            self.app.route("/<path:path>", methods=["GET"])(self.serve_static)
            print(f"[CNA] Serving webapp from {self.webapp_dist}")
        elif self.webapp_dist:
            print(
                f"[CNA] Warning: --webapp-dist '{self.webapp_dist}' not found; "
                "serving API only."
            )

    # --------------------------------------------------------------- endpoints
    def health(self):
        return jsonify({"status": "ok", "service": "conversational-navigation-assistant"})

    def serve_index(self):
        return send_from_directory(self.webapp_dist, "index.html")

    def serve_static(self, path: str):
        full = os.path.join(self.webapp_dist, path)
        if os.path.isfile(full):
            return send_from_directory(self.webapp_dist, path)
        # SPA fallback: unknown client-side routes return index.html.
        return send_from_directory(self.webapp_dist, "index.html")

    def status(self):
        return jsonify(
            {
                "llm_loaded": self.viewer.model is not None or self.backend == "openai",
                "backend": self.backend,
                "finetuned": self.finetuned,
                "scene": os.path.basename(self.current_scene),
                "scene_path": self.current_scene,
                "current_room": self.viewer.current_room_name(),
                "controls": WEB_KEY_TO_ACTION,
            }
        )

    def apply_llm_config(self, backend: str, finetuned: bool) -> str | None:
        """Switch backend (local/openai) and finetuned/base at runtime.

        Frees the current model BEFORE loading the new one so the quantized
        weights fit in VRAM (otherwise two copies coexist and OOM). On any
        failure the previous configuration is reloaded.
        """
        with self._llm_lock:
            prev_backend = self.backend
            prev_finetuned = self.finetuned

            def _free_local():
                self.viewer.model = None
                self.viewer.tokenizer = None
                self.viewer.model_intent = None
                self.model = None
                mr_viewer._LOCAL_MODEL = None
                mr_viewer._LOCAL_TOKENIZER = None
                mr_viewer._LOCAL_MODEL_INTENT = None
                import gc

                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass

            def _load_local(ft: bool):
                m, tok, mi = mr_viewer.load_local_model(fine_tuned_model=ft)
                self.viewer.model = m
                self.viewer.tokenizer = tok
                self.viewer.model_intent = mi
                self.model = m

            try:
                if backend == "openai":
                    if getattr(mr_viewer, "client", None) is None:
                        raise RuntimeError(
                            "OpenAI backend requested but OPENAI_API_KEY is not "
                            "configured (set it in the project .env)."
                        )
                    _free_local()  # OpenAI path needs no local model; free VRAM
                else:
                    # Free the current model FIRST, then load fresh -> no OOM.
                    _free_local()
                    _load_local(finetuned)
                self.backend = backend
                self.finetuned = bool(finetuned)
                print(f"[CNA] LLM config -> backend={backend} finetuned={finetuned}")
                return None
            except Exception as err:
                import traceback

                traceback.print_exc()
                # Best-effort revert: reload the previous configuration.
                try:
                    _free_local()
                    if prev_backend == "local":
                        _load_local(prev_finetuned)
                except Exception:
                    traceback.print_exc()
                finally:
                    self.backend = prev_backend
                    self.finetuned = prev_finetuned
                return str(err)

    def set_llm(self):
        data = request.get_json(force=True, silent=True) or {}
        backend = (data.get("backend") or self.backend).lower()
        finetuned = bool(data.get("finetuned", self.finetuned))
        if backend not in ("local", "openai"):
            return jsonify({"error": f"invalid backend: {backend}"}), 400
        err = self.apply_llm_config(backend, finetuned)
        if err:
            return (
                jsonify(
                    {
                        "error": err,
                        "backend": self.backend,
                        "finetuned": self.finetuned,
                    }
                ),
                500,
            )
        return jsonify(
            {"status": "ok", "backend": self.backend, "finetuned": self.finetuned}
        )

    def scenes(self):
        """List scenes available for switching (folders with a glb + room map)."""
        result = []
        base = self.scenes_dir
        if base and os.path.isdir(base):
            for name in sorted(os.listdir(base)):
                folder = os.path.join(base, name)
                if not os.path.isdir(folder):
                    continue
                if not os.path.isfile(os.path.join(folder, "room_id_to_name_map.json")):
                    continue
                glbs = sorted(g for g in os.listdir(folder) if g.endswith(".basis.glb"))
                if not glbs:
                    continue
                scene_abs = os.path.join(folder, glbs[0])
                scene_rel = os.path.normpath(os.path.relpath(scene_abs, os.getcwd()))
                result.append({"label": name, "scene": scene_rel})
        return jsonify({"scenes": result, "current": self.current_scene})

    def set_scene(self):
        """Switch the active scene at runtime (no process restart)."""
        data = request.get_json(force=True, silent=True) or {}
        scene = os.path.normpath((data.get("scene") or "").strip())
        if not scene or scene == ".":
            return jsonify({"error": "no scene provided"}), 400
        if not os.path.isfile(scene):
            return jsonify({"error": f"scene not found: {scene}"}), 404
        # Only one switch at a time.
        with self._scene_lock:
            done: queue.Queue = queue.Queue()
            self.viewer.action_queue.put(
                (self.viewer.reconfigure_scene, (scene, done), {})
            )
            try:
                err = done.get(timeout=180)
            except queue.Empty:
                return jsonify({"error": "scene switch timeout"}), 504
            if err:
                return jsonify({"error": err}), 500
            self.current_scene = scene
        return jsonify(
            {"status": "ok", "scene": os.path.basename(scene), "scene_path": scene}
        )

    def video(self):
        def generate():
            last_id = -1
            idle = 0.003  # 3 ms poll; only yields when a NEW frame is ready
            while True:
                frame_id, frame = self.viewer.get_latest_frame_versioned()
                if frame is not None and frame_id != last_id:
                    last_id = frame_id
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                        + frame + b"\r\n"
                    )
                else:
                    time.sleep(idle)

        return Response(
            generate(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    def frame(self):
        frame = self.viewer.get_latest_frame()
        if not frame:
            return jsonify({"error": "no frame available yet"}), 503
        return Response(frame, mimetype="image/jpeg")

    def ws_handler(self, ws):
        """Bidirectional low-latency channel.

        Client -> server (JSON text):
            {"t":"key","key":"w","down":true,"shift":false,"alt":false,"ctrl":false}
            {"t":"ready"}        # client finished drawing the previous frame
            {"t":"releaseAll"}
        Server -> client: binary JPEG frames (only the newest, one in flight).
        """
        import json as _json

        last_id = -1
        ready = True  # send the first frame immediately
        try:
            while True:
                # Drain any pending client messages without blocking.
                while True:
                    msg = ws.receive(timeout=0)
                    if msg is None:
                        break
                    try:
                        data = _json.loads(msg)
                    except Exception:
                        continue
                    kind = data.get("t")
                    if kind == "ready":
                        ready = True
                    elif kind == "key":
                        self.viewer.web_key_event(
                            data.get("key", ""),
                            bool(data.get("down", False)),
                            shift=bool(data.get("shift", False)),
                            alt=bool(data.get("alt", False)),
                            ctrl=bool(data.get("ctrl", False)),
                        )
                    elif kind == "releaseAll":
                        self.viewer.web_release_all()

                # Send the freshest frame, but only when the client is ready.
                frame_id, frame = self.viewer.get_latest_frame_versioned()
                if ready and frame is not None and frame_id != last_id:
                    last_id = frame_id
                    ready = False
                    ws.send(frame)
                else:
                    time.sleep(0.002)
        except Exception:
            # Client disconnected or socket error -> just end the handler.
            return

    def key(self):
        data = request.get_json(force=True, silent=True) or {}
        if data.get("release_all"):
            self.viewer.web_release_all()
            return jsonify({"ok": True})
        ok = self.viewer.web_key_event(
            data.get("key", ""),
            bool(data.get("down", False)),
            shift=bool(data.get("shift", False)),
            alt=bool(data.get("alt", False)),
            ctrl=bool(data.get("ctrl", False)),
        )
        return jsonify({"ok": ok})

    def action(self):
        data = request.get_json(force=True, silent=True) or {}
        action_name = data.get("action", "")
        if not self.viewer.web_tap_action(action_name):
            return jsonify({"error": f"unknown action: {action_name}"}), 400
        return jsonify({"ok": True})

    def chat(self):
        data = request.get_json(force=True, silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"error": "empty message"}), 400

        # Only one conversation turn at a time (single simulated agent).
        with self._chat_lock:
            # Drop any stale responses left over from a previous timeout.
            while not self.output_q.empty():
                try:
                    self.output_q.get_nowait()
                except queue.Empty:
                    break

            self.input_q.put(message)
            try:
                response = self.output_q.get(timeout=self.chat_timeout)
            except queue.Empty:
                return jsonify({"error": "assistant timeout"}), 504

        return jsonify({"response": response, "message": message})

    def run(self, host: str = "0.0.0.0", port: int = 5001) -> None:
        print(f"[CNA] Conversational Navigation Assistant server on {host}:{port}")
        self.app.run(host=host, port=port, threaded=True)


def build_sim_settings(args: argparse.Namespace) -> Dict[str, Any]:
    sim_settings: Dict[str, Any] = dict(default_sim_settings)
    sim_settings["scene"] = args.scene
    sim_settings["scene_dataset_config_file"] = args.dataset
    sim_settings["enable_physics"] = not args.disable_physics
    sim_settings["use_default_lighting"] = args.use_default_lighting
    sim_settings["enable_batch_renderer"] = False
    sim_settings["num_environments"] = 1
    sim_settings["composite_files"] = None
    sim_settings["window_width"] = args.width
    sim_settings["window_height"] = args.height
    sim_settings["default_agent_navmesh"] = False
    sim_settings["enable_hbao"] = args.hbao
    sim_settings["semantic_sensor"] = True
    sim_settings["depth_sensor"] = True
    sim_settings["hidden_window"] = getattr(args, "hidden_window", False)
    return sim_settings


def _wait_until_serving(host: str, port: int, timeout: float = 30.0) -> None:
    """Block until the HTTP server accepts connections (or timeout)."""
    import socket

    connect_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((connect_host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conversational Navigation Assistant web backend"
    )
    parser.add_argument(
        "--scene",
        default="./data/scene_datasets/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb",
        type=str,
        help="scene/stage file to load",
    )
    parser.add_argument(
        "--dataset",
        default="./data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
        type=str,
        help="dataset configuration file to use",
    )
    parser.add_argument("--disable-physics", action="store_true")
    parser.add_argument("--use-default-lighting", action="store_true")
    parser.add_argument("--hbao", action="store_true")
    parser.add_argument(
        "--hidden-window",
        action="store_true",
        help="create the render window hidden (GPU headless on a real display)",
    )
    parser.add_argument("--width", default=800, type=int, help="render width")
    parser.add_argument("--height", default=600, type=int, help="render height")
    parser.add_argument(
        "--backend",
        default="local",
        type=str,
        help="LLM backend to use: openai / local (default: local)",
    )
    parser.add_argument(
        "--finetuned-model",
        action="store_true",
        help="use the finetuned local LoRA model (if backend=local)",
    )
    parser.add_argument("--host", default="0.0.0.0", type=str, help="server host")
    parser.add_argument("--port", default=5001, type=int, help="server port")
    parser.add_argument(
        "--stream-fps", default=60, type=int, help="MJPEG stream frame-rate cap"
    )
    parser.add_argument(
        "--move",
        default=0.07,
        type=float,
        help="per-step movement amount (desktop viewer default 0.07)",
    )
    parser.add_argument(
        "--look",
        default=1.5,
        type=float,
        help="per-step turn/look degrees (desktop viewer default 1.5)",
    )
    parser.add_argument(
        "--webapp-dist",
        default=None,
        type=str,
        help="path to the built webapp (dist/) to serve on the same host:port",
    )
    args = parser.parse_args()

    if args.width < 1 or args.height < 1:
        parser.error("width and height must be positive non-zero integers.")

    sim_settings = build_sim_settings(args)

    # Conversation queues shared with the (unchanged) mr_viewer pipeline.
    input_q: queue.Queue = queue.Queue()
    output_q: queue.Queue = queue.Queue()

    # Load the local model if requested (populates mr_viewer module globals too).
    model = tokenizer = model_intent = None
    if args.backend.lower() == "local":
        try:
            model, tokenizer, model_intent = load_local_model(
                fine_tuned_model=args.finetuned_model
            )
            print("[CNA] Local model loaded and ready.")
        except Exception as err:
            print(f"[CNA] Error loading local model: {err}")
            sys.exit(1)
    else:
        print(f"[CNA] Using backend: {args.backend} (no local model loaded)")

    # Create the web-enabled viewer and register it as the module-level `viewer`
    # so the reused navigation methods (which reference the global) resolve to it.
    viewer = WebViewer(
        sim_settings,
        stream_fps=args.stream_fps,
        move_amount=args.move,
        look_amount=args.look,
    )
    mr_viewer.viewer = viewer
    # Runtime-swappable model refs read by user_input_logic_loop.
    viewer.model = model
    viewer.tokenizer = tokenizer
    viewer.model_intent = model_intent

    # Conversation loop: identical to the desktop GUI, driven by the queues.
    logic_thread = threading.Thread(
        target=user_input_logic_loop,
        args=(viewer, input_q, output_q, model, tokenizer, model_intent),
        daemon=True,
    )
    logic_thread.start()

    # HTTP server in a background thread; the render loop owns the main thread.
    server = ConversationalNavigationServer(
        viewer=viewer,
        input_q=input_q,
        output_q=output_q,
        model=model,
        backend=args.backend.lower(),
        stream_fps=args.stream_fps,
        webapp_dist=args.webapp_dist,
        scenes_dir=os.path.dirname(os.path.dirname(os.path.abspath(args.scene))),
        current_scene=args.scene,
        finetuned=args.finetuned_model,
    )
    server_thread = threading.Thread(
        target=server.run, args=(args.host, args.port), daemon=True
    )
    server_thread.start()

    # Only announce the URL once everything is loaded AND the server is serving.
    _wait_until_serving(args.host, args.port)
    view_host = "localhost" if args.host in ("0.0.0.0", "::", "") else args.host
    local_url = f"http://{view_host}:{args.port}/assistant"

    # If a public tunnel (run_demo.sh --public) is starting in parallel, wait
    # briefly for its URL so we can show it in this banner too.
    public_url = None
    url_file = os.environ.get("CNA_PUBLIC_URL_FILE")
    if url_file:
        for _ in range(40):
            try:
                with open(url_file) as fh:
                    raw = fh.read().strip()
                if raw:
                    public_url = raw.rstrip("/") + "/assistant"
                    break
            except Exception:
                pass
            time.sleep(0.25)

    lines = [
        "\n" + "=" * 66,
        "  Conversational Navigation Assistant - READY",
        "  Everything is loaded. Open one of these URLs:",
        "",
        f"      Local:   {local_url}",
    ]
    if public_url:
        lines.append(f"      Public:  {public_url}")
    else:
        lines.append(
            f"      Public:  (start with --public, or run:  ngrok http {args.port})"
        )
    lines.append("=" * 66 + "\n")
    print("\n".join(lines), flush=True)

    # Blocking Magnum render loop (owns the GL context / frame capture).
    viewer.exec()


if __name__ == "__main__":
    main()
