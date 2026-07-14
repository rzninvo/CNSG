#!/bin/bash
# Conversational Navigation Assistant - one-command demo launcher.
#
# Starts EVERYTHING with a single command:
#   - builds the webapp (only the first time)
#   - runs the Habitat-Sim backend HEADLESS (no local window, via xvfb)
#   - serves the webapp + live stream + LLM chat on one host:port
#
# Usage:
#   ./run_demo.sh                          # http://0.0.0.0:5001/assistant
#   ./run_demo.sh --host 0.0.0.0 --port 8080
#   ./run_demo.sh --scene <glb> --dataset <json>
#   ./run_demo.sh --base-model             # use base (non-finetuned) LLM
#   ./run_demo.sh --gpu                    # GPU-accelerated AND no window (xvfb + Mesa d3d12, WSL)
#   ./run_demo.sh --no-xvfb                # GPU with a visible local window (WSLg)
#   ./run_demo.sh --public                 # also expose it to remote users via a tunnel
#
# Remote access (a user outside your network / VPN):
#   - Same LAN only: they open  http://<your-LAN-ip>:<port>/assistant
#   - Anywhere:      add --public (needs cloudflared or ngrok) -> shareable https URL
#     or manually:   cloudflared tunnel --url http://localhost:<port>
#                    ngrok http <port>
set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
WEBAPP="$REPO/webapp"
HSIM="$REPO/habitat-sim"

# Make user-installed tools (e.g. cloudflared in ~/.local/bin) discoverable.
export PATH="$HOME/.local/bin:$PATH"

HOST="0.0.0.0"
PORT="5001"
BACKEND="local"
FINETUNED="--finetuned-model"
SCENE="./data/scene_datasets/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
DATASET="./data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
STREAM_FPS="60"
WIDTH="720"
HEIGHT="540"
DO_BUILD="auto"      # auto | force | skip
USE_XVFB="1"
HIDDEN="0"
GPU="0"
SCREEN="1280x960x24"
PUBLIC="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)        HOST="$2"; shift 2;;
    --port)        PORT="$2"; shift 2;;
    --scene)       SCENE="$2"; shift 2;;
    --dataset)     DATASET="$2"; shift 2;;
    --backend)     BACKEND="$2"; shift 2;;
    --base-model)  FINETUNED=""; shift;;
    --stream-fps)  STREAM_FPS="$2"; shift 2;;
    --width)       WIDTH="$2"; shift 2;;
    --height)      HEIGHT="$2"; shift 2;;
    --gpu)         GPU="1"; USE_XVFB="1"; shift;;
    --hidden-window) HIDDEN="1"; shift;;
    --public)      PUBLIC="1"; shift;;
    --build)       DO_BUILD="force"; shift;;
    --no-build)    DO_BUILD="skip"; shift;;
    --no-xvfb)     USE_XVFB="0"; shift;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0;;
    *) echo "[demo] Unknown option: $1"; exit 1;;
  esac
done

# 1. Build the webapp if needed -------------------------------------------------
# Auto mode rebuilds when the build is missing OR any source file changed, so a
# plain `./run_demo.sh` always serves the latest UI.
NEEDS_BUILD=0
if [[ "$DO_BUILD" == "force" ]]; then
  NEEDS_BUILD=1
elif [[ "$DO_BUILD" == "auto" ]]; then
  if [[ ! -f "$WEBAPP/dist/index.html" ]]; then
    NEEDS_BUILD=1
  elif [[ -n "$(find "$WEBAPP/src" "$WEBAPP/index.html" "$WEBAPP/package.json" \
                 "$WEBAPP/tailwind.config.ts" "$WEBAPP/vite.config.ts" \
                 -newer "$WEBAPP/dist/index.html" 2>/dev/null | head -1)" ]]; then
    NEEDS_BUILD=1
    echo "[demo] Detected webapp source changes -> rebuilding."
  fi
fi

if [[ "$NEEDS_BUILD" == "1" ]]; then
  echo "[demo] Building webapp..."
  cd "$WEBAPP"
  [[ -d node_modules ]] || npm install
  npm run build
else
  echo "[demo] Using existing webapp build ($WEBAPP/dist)."
fi

# 2. Launch the headless backend serving everything ----------------------------
cd "$HSIM"

CMD=(python3 examples/cna_server.py
  --host "$HOST" --port "$PORT"
  --scene "$SCENE" --dataset "$DATASET"
  --backend "$BACKEND"
  --stream-fps "$STREAM_FPS"
  --width "$WIDTH" --height "$HEIGHT"
  --webapp-dist "$WEBAPP/dist")
[[ -n "$FINETUNED" ]] && CMD+=("$FINETUNED")
[[ "$HIDDEN" == "1" ]] && CMD+=(--hidden-window)

echo "[demo] Loading scene and model… the connect URL will appear when ready."

# Optional public tunnel so a remote user (outside your network/VPN) can connect.
TUNNEL_PID=""
cleanup() { [[ -n "$TUNNEL_PID" ]] && kill "$TUNNEL_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

if [[ "$PUBLIC" == "1" ]]; then
  # File used to hand the public URL to the backend so it can show it in the
  # final READY banner too.
  PUB_URL_FILE="$(mktemp -t cna_public_url.XXXXXX)"
  export CNA_PUBLIC_URL_FILE="$PUB_URL_FILE"
  if command -v cloudflared >/dev/null 2>&1; then
    echo "[demo] Starting public tunnel with cloudflared (no signup needed)…"
    CF_LOG="$(mktemp -t cna_cloudflared.XXXXXX.log)"
    cloudflared tunnel --url "http://localhost:$PORT" >"$CF_LOG" 2>&1 &
    TUNNEL_PID=$!
    # Watch the log and print the public URL prominently once it appears.
    (
      for _ in $(seq 1 40); do
        url=$(grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' "$CF_LOG" | head -1)
        if [ -n "$url" ]; then
          printf '%s' "$url" > "$PUB_URL_FILE"
          echo ""
          echo "================================================================"
          echo "  PUBLIC URL — open THIS on the other device (NOT the 172.x IP):"
          echo ""
          echo "      $url/assistant"
          echo ""
          echo "================================================================"
          echo ""
          break
        fi
        sleep 1
      done
    ) &
  elif command -v ngrok >/dev/null 2>&1; then
    echo "[demo] Starting public tunnel with ngrok…"
    ngrok http "$PORT" >/dev/null 2>&1 &
    TUNNEL_PID=$!
    (
      for _ in $(seq 1 40); do
        url=$(curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null \
              | grep -oE 'https://[a-zA-Z0-9.-]*ngrok[a-zA-Z0-9.-]*' | head -1)
        if [ -n "$url" ]; then
          printf '%s' "$url" > "$PUB_URL_FILE"
          echo "[demo] Public URL: $url/assistant"
          break
        fi
        sleep 1
      done
    ) &
  else
    echo "[demo] --public requested but no tunnel tool found."
    echo "[demo] Install one:  (recommended) cloudflared  or  ngrok"
    echo "[demo]   cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
    echo "[demo]   ngrok:       https://ngrok.com/download   (then: ngrok config add-authtoken <token>)"
  fi
fi

if [[ "$USE_XVFB" == "1" ]]; then
  if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "[demo] ERROR: 'xvfb-run' not found (needed for headless rendering)."
    echo "[demo]        Install it:  sudo apt-get install -y xvfb"
    echo "[demo]        Or run with a local window:  ./run_demo.sh --no-xvfb"
    exit 1
  fi
  # GPU rendering via Mesa's d3d12 driver (uses the RTX through WSL) while xvfb
  # keeps the window invisible -> GPU-accelerated AND no local window.
  if [[ "$GPU" == "1" ]]; then
    if [[ -d /usr/lib/wsl/lib ]]; then
      export GALLIUM_DRIVER=d3d12
      export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
      echo "[demo] GPU rendering via Mesa d3d12 (RTX) — headless, no visible window."
    else
      echo "[demo] --gpu: /usr/lib/wsl/lib not found; falling back to software (xvfb)."
    fi
  else
    echo "[demo] Running headless (xvfb, no local window)."
    if [[ -e /dev/dxg ]]; then
      echo "[demo] TIP: WSL GPU detected — for GPU rendering (still no window) use:  ./run_demo.sh --gpu"
    fi
  fi
  exec xvfb-run -a -s "-screen 0 $SCREEN" "${CMD[@]}"
else
  if [[ "$HIDDEN" == "1" ]]; then
    if [[ -z "${DISPLAY:-}" ]]; then
      echo "[demo] ERROR: --hidden-window needs a real display (DISPLAY is empty)."
      exit 1
    fi
    echo "[demo] Running on DISPLAY=$DISPLAY with a hidden window (if supported by GLFW)."
  else
    echo "[demo] Running with a local window (--no-xvfb)."
  fi
  exec "${CMD[@]}"
fi
