// Client for the Conversational Navigation Assistant backend (cna_server.py).
import { config } from "@/lib/config";

// When VITE_ASSISTANT_URL is unset, talk to the same origin that served the app
// (the backend serves the built webapp too, so the whole demo is one host:port).
const base =
  config.assistantUrl ||
  (typeof window !== "undefined" ? window.location.origin : "");

export type AssistantStatus = {
  llm_loaded: boolean;
  backend: string;
  finetuned?: boolean;
  scene: string;
  scene_path?: string;
  current_room: string | null;
  overlays?: {
    bboxes?: boolean;
    all_bboxes?: boolean;
    rooms?: boolean;
    save_frames?: boolean;
  };
  controls: Record<string, string>;
};

export type SceneEntry = { label: string; scene: string; dataset?: string };

/** List scenes available for switching from the webapp. */
export async function getScenes(): Promise<{
  scenes: SceneEntry[];
  current: string;
} | null> {
  if (!base) return null;
  try {
    const res = await fetch(`${base}/scenes`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

/** Switch the active scene at runtime (backend reloads it, LLM stays loaded). */
export async function setScene(
  scene: string,
  dataset?: string,
  timeoutMs = 180000,
): Promise<void> {
  if (!base) throw new Error("Assistant backend URL is not configured.");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${base}/scene`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scene, dataset }),
      signal: controller.signal,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data?.error || `Server error (${res.status})`);
  } finally {
    clearTimeout(timer);
  }
}

/** Switch LLM backend (local/openai) and finetuned/base at runtime. */
export async function setLlm(
  cfg: { backend: string; finetuned: boolean },
  timeoutMs = 300000,
): Promise<void> {
  if (!base) throw new Error("Assistant backend URL is not configured.");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${base}/llm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
      signal: controller.signal,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data?.error || `Server error (${res.status})`);
  } finally {
    clearTimeout(timer);
  }
}

/** URL of the live MJPEG stream (bind directly to an <img src>). */
export function videoStreamUrl(): string {
  return base ? `${base}/video` : "";
}

/** WebSocket URL for the low-latency input + video channel. */
export function wsUrl(): string {
  if (!base) return "";
  // http -> ws, https -> wss
  return `${base.replace(/^http/, "ws")}/ws`;
}

/** Send a key state (held for movement, or a single press for other viewer keys). */
export async function sendKey(
  key: string,
  down: boolean,
  mods?: { shift?: boolean; alt?: boolean; ctrl?: boolean },
): Promise<void> {
  if (!base) return;
  try {
    await fetch(`${base}/key`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        key,
        down,
        shift: mods?.shift ?? false,
        alt: mods?.alt ?? false,
        ctrl: mods?.ctrl ?? false,
      }),
      keepalive: true,
    });
  } catch {
    /* movement is best-effort; ignore transient failures */
  }
}

/** Release all held keys (called on blur / unmount). */
export async function releaseAllKeys(): Promise<void> {
  if (!base) return;
  try {
    await fetch(`${base}/key`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ release_all: true }),
      keepalive: true,
    });
  } catch {
    /* ignore */
  }
}

/** Single-step action (used by the on-screen control pad on tap). */
export async function sendAction(action: string): Promise<void> {
  if (!base) return;
  try {
    await fetch(`${base}/action`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
  } catch {
    /* ignore */
  }
}

/** Send a conversation / navigation message and get the assistant's reply. */
export async function sendChat(
  message: string,
  timeoutMs = 120000,
): Promise<string> {
  if (!base) throw new Error("Assistant backend URL is not configured.");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${base}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
      signal: controller.signal,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data?.error || `Server error (${res.status})`);
    }
    return data.response ?? "";
  } finally {
    clearTimeout(timer);
  }
}

/** Poll backend status (LLM readiness, current room, scene). */
export async function getStatus(): Promise<AssistantStatus | null> {
  if (!base) return null;
  try {
    const res = await fetch(`${base}/status`);
    if (!res.ok) return null;
    return (await res.json()) as AssistantStatus;
  } catch {
    return null;
  }
}

export const assistantConfigured = Boolean(base);
