import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowLeft,
  Compass,
  Mic,
  MicOff,
  Send,
  Volume2,
  VolumeX,
  Keyboard,
  Loader2,
  RotateCw,
  SlidersHorizontal,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import {
  assistantConfigured,
  getScenes,
  getStatus,
  releaseAllKeys,
  sendChat,
  sendKey,
  setLlm,
  setNatural,
  setDirections,
  setScene,
  recalculateFromHere,
  savePreference,
  videoStreamUrl,
  wsUrl,
  type AssistantStatus,
  type SceneEntry,
} from "@/lib/assistant";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  // When set, the assistant reply contains both a landmark description and
  // geometric directions, shown side by side.
  dual?: {
    llm: string;
    geometric: string;
    meta?: {
      scene?: string;
      destination?: string;
      question?: string;
      user_position?: number[] | null;
      destination_position?: number[] | null;
    };
  };
};

// User-study questions comparing the landmark (LLM) vs geometric answers.
const STUDY_QUESTIONS: { key: "easier" | "natural" | "reach"; label: string }[] = [
  { key: "easier", label: "Easier to follow" },
  { key: "natural", label: "More natural" },
  { key: "reach", label: "Better to reach the destination" },
];
type StudyKey = (typeof STUDY_QUESTIONS)[number]["key"];

// Keys we forward to the simulator when the viewer has keyboard focus:
// all letters/digits, the arrow keys, space and comma/period (viewer shortcuts).
const isControlKey = (k: string) =>
  /^[a-z0-9]$/.test(k) ||
  k === " " ||
  k === "," ||
  k === "." ||
  ["arrowup", "arrowdown", "arrowleft", "arrowright"].includes(k);

const uid = () => Math.random().toString(36).slice(2);

const INTRO_MESSAGE: ChatMessage = {
  id: "intro",
  role: "assistant",
  text: "Hi! I'm your navigation assistant. Move with WASD, look by dragging inside the view (or with the arrow keys), and ask me how to reach any room or object.",
};

const ConversationalNavigationAssistant = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "intro",
      role: "assistant",
      text: "Hi! I'm your navigation assistant. Move with WASD, look by dragging inside the view (or with the arrow keys), and ask me how to reach any room or object.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const [status, setStatus] = useState<AssistantStatus | null>(null);
  const [videoOk, setVideoOk] = useState(true);
  const [wsConnected, setWsConnected] = useState(false);
  const [viewerHovered, setViewerHovered] = useState(false);
  const [micOn, setMicOn] = useState(false);
  const [ttsOn, setTtsOn] = useState(false);
  const [scenes, setScenes] = useState<SceneEntry[]>([]);
  const [currentScene, setCurrentScene] = useState<string>("");
  const [switching, setSwitching] = useState(false);
  const [llmBusy, setLlmBusy] = useState(false);
  // Which user-study questions to ask under a "both" answer (all on by default).
  const [studyEnabled, setStudyEnabled] = useState<Record<StudyKey, boolean>>({
    easier: true,
    natural: true,
    reach: true,
  });

  const pressedRef = useRef<Set<string>>(new Set());
  const scrollRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);
  const viewerHoveredRef = useRef(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // ----------------------------------------------------------- status polling
  useEffect(() => {
    if (!assistantConfigured) return;
    let alive = true;
    const poll = async () => {
      const s = await getStatus();
      if (alive) setStatus(s);
    };
    poll();
    const t = setInterval(poll, 4000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  // ------------------------------------------------------------ scene list
  useEffect(() => {
    if (!assistantConfigured) return;
    let alive = true;
    getScenes().then((data) => {
      if (!alive || !data) return;
      // Expose the 00800/00808 HM3D houses plus any MP3D example scenes.
      const allowed = data.scenes.filter(
        (s) =>
          s.label.includes("00800") ||
          s.label.includes("00808") ||
          s.label.startsWith("MP3D"),
      );
      setScenes(allowed);
    });
    return () => {
      alive = false;
    };
  }, []);

  // Keep the dropdown in sync with the scene actually loaded by the server
  // (status.scene_path), so it can never show the wrong house.
  useEffect(() => {
    const sp = status?.scene_path;
    if (sp && !switching) {
      setCurrentScene(sp);
    }
  }, [status, switching]);

  const handleSceneChange = useCallback(
    async (scene: string) => {
      if (!scene || scene === currentScene || switching) return;
      setSwitching(true);
      try {
        const dataset = scenes.find((s) => s.scene === scene)?.dataset;
        await setScene(scene, dataset);
        setCurrentScene(scene);
        toast.success("Scene switched");
      } catch (err) {
        toast.error(
          err instanceof Error ? err.message : "Failed to switch scene",
        );
      } finally {
        setSwitching(false);
      }
    },
    [currentScene, switching, scenes],
  );

  // The backend applies the change and reverts on failure; the UI reflects the
  // server's status (re-fetched after each attempt), so a failed switch snaps
  // back to the previous model automatically.
  const applyLlm = useCallback(
    async (next: { backend: string; finetuned: boolean }) => {
      if (llmBusy) return;
      setLlmBusy(true);
      try {
        await setLlm(next);
        toast.success("Model updated");
      } catch (err) {
        toast.error(
          err instanceof Error ? err.message : "Failed to switch model",
        );
      } finally {
        const s = await getStatus();
        setStatus(s);
        setLlmBusy(false);
      }
    },
    [llmBusy],
  );

  // ---------------------------------- low-latency WebSocket (video + input)
  useEffect(() => {
    if (!assistantConfigured) return;
    const url = wsUrl();
    if (!url || typeof WebSocket === "undefined") return;

    let ws: WebSocket | null = null;
    let closed = false;
    let reconnectTimer: number | null = null;

    const connect = () => {
      ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        if (!closed) setWsConnected(true);
      };
      ws.onmessage = async (ev: MessageEvent) => {
        if (typeof ev.data === "string") return;
        try {
          const bmp = await createImageBitmap(
            new Blob([ev.data as ArrayBuffer], { type: "image/jpeg" }),
          );
          const canvas = canvasRef.current;
          if (canvas) {
            if (canvas.width !== bmp.width || canvas.height !== bmp.height) {
              canvas.width = bmp.width;
              canvas.height = bmp.height;
            }
            const ctx = canvas.getContext("2d");
            if (ctx) ctx.drawImage(bmp, 0, 0);
          }
          bmp.close?.();
        } catch {
          /* ignore decode errors */
        }
        // Ask for the next (freshest) frame -> backpressure, no lag build-up.
        if (ws && ws.readyState === WebSocket.OPEN)
          ws.send(JSON.stringify({ t: "ready" }));
      };
      ws.onclose = () => {
        wsRef.current = null;
        if (!closed) {
          setWsConnected(false);
          reconnectTimer = window.setTimeout(connect, 1500);
        }
      };
      ws.onerror = () => {
        try {
          ws?.close();
        } catch {
          /* ignore */
        }
      };
    };

    connect();
    return () => {
      closed = true;
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      try {
        ws?.close();
      } catch {
        /* ignore */
      }
      wsRef.current = null;
      setWsConnected(false);
    };
  }, []);

  // ------------------------------------------------------------ keyboard input
  const forwardKey = useCallback(
    (
      key: string,
      down: boolean,
      mods?: { shift?: boolean; alt?: boolean; ctrl?: boolean },
    ) => {
      const k = key.toLowerCase();
      if (down) {
        if (pressedRef.current.has(k)) return; // ignore auto-repeat
        pressedRef.current.add(k);
      } else {
        if (!pressedRef.current.has(k)) return;
        pressedRef.current.delete(k);
      }
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(
          JSON.stringify({
            t: "key",
            key: k,
            down,
            shift: mods?.shift ?? false,
            alt: mods?.alt ?? false,
            ctrl: mods?.ctrl ?? false,
          }),
        );
      } else {
        void sendKey(k, down, mods);
      }
    },
    [],
  );

  const releaseAll = useCallback(() => {
    pressedRef.current.clear();
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ t: "releaseAll" }));
    } else {
      void releaseAllKeys();
    }
  }, []);

  // Tap a viewer shortcut key (toggles like B/G/R/P) from the Tools menu.
  const tapKey = useCallback(
    (k: string) => {
      forwardKey(k, true);
      window.setTimeout(() => forwardKey(k, false), 60);
    },
    [forwardKey],
  );
  const clearChat = useCallback(() => {
    setMessages([INTRO_MESSAGE]);
  }, []);

  // Toggle a viewer overlay/capture, then re-read status so the menu checkbox
  // reflects the backend's real state (also keeps it in sync with keyboard keys).
  const toggleTool = useCallback(
    (k: string) => {
      tapKey(k);
      window.setTimeout(async () => {
        const s = await getStatus();
        setStatus(s);
      }, 300);
    },
    [tapKey],
  );

  // Force the natural-language prompt on/off, then refresh status.
  const toggleNatural = useCallback(async (next: boolean) => {
    try {
      await setNatural(next);
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Failed to update prompt mode",
      );
    }
    const s = await getStatus();
    setStatus(s);
  }, []);

  // Toggle geometric (landmark-free) directions, then refresh status.
  const toggleGeometric = useCallback(async (mode: "llm" | "both" | "geometric") => {
    try {
      await setDirections(mode);
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Failed to update directions mode",
      );
    }
    const s = await getStatus();
    setStatus(s);
  }, []);

  useEffect(() => {
    if (!assistantConfigured) return;

    const isTyping = () => {
      const el = document.activeElement;
      return (
        el instanceof HTMLInputElement ||
        el instanceof HTMLTextAreaElement ||
        (el as HTMLElement | null)?.isContentEditable === true
      );
    };

    const onKeyDown = (e: KeyboardEvent) => {
      // Control the sim only when the cursor is over the viewer and we're not
      // typing in the chat box.
      if (!viewerHoveredRef.current || isTyping()) return;
      const k = e.key.toLowerCase();
      if (!isControlKey(k)) return;
      if (e.repeat) return; // one action per physical press (e.g. B toggle)
      e.preventDefault();
      forwardKey(k, true, {
        shift: e.shiftKey,
        alt: e.altKey,
        ctrl: e.ctrlKey,
      });
    };
    const onKeyUp = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (!isControlKey(k)) return;
      forwardKey(k, false);
    };
    const onBlur = () => releaseAll();
    const onVisibility = () => {
      if (document.hidden) releaseAll();
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
      document.removeEventListener("visibilitychange", onVisibility);
      releaseAll();
    };
  }, [forwardKey, releaseAll]);

  // ---------------------------------------------------- mouse drag "look" control
  // Left-drag inside the viewer turns/looks just like holding the arrow keys.
  const dragRef = useRef(false);
  const lastPosRef = useRef<{ x: number; y: number } | null>(null);
  const mouseLookKeysRef = useRef<Set<string>>(new Set());
  const lookStopTimerRef = useRef<number | null>(null);

  const pressLook = useCallback(
    (key: string) => {
      if (!mouseLookKeysRef.current.has(key)) {
        mouseLookKeysRef.current.add(key);
        forwardKey(key, true);
      }
    },
    [forwardKey],
  );
  const releaseLook = useCallback(
    (key: string) => {
      if (mouseLookKeysRef.current.has(key)) {
        mouseLookKeysRef.current.delete(key);
        forwardKey(key, false);
      }
    },
    [forwardKey],
  );
  const releaseAllLook = useCallback(() => {
    ["arrowup", "arrowdown", "arrowleft", "arrowright"].forEach(releaseLook);
  }, [releaseLook]);

  const onViewerPointerDown = useCallback((e: React.PointerEvent) => {
    if (e.button !== 0 && e.button !== 2) return; // left or right button
    // Take keyboard control away from the chat box so WASD steer the agent.
    (document.activeElement as HTMLElement | null)?.blur?.();
    dragRef.current = true;
    lastPosRef.current = { x: e.clientX, y: e.clientY };
    (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
    e.preventDefault();
  }, []);

  const onViewerPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragRef.current || !lastPosRef.current) return;
      const dx = e.clientX - lastPosRef.current.x;
      const dy = e.clientY - lastPosRef.current.y;
      lastPosRef.current = { x: e.clientX, y: e.clientY };
      const TH = 1.2; // px dead-zone
      if (dx > TH) {
        pressLook("arrowright");
        releaseLook("arrowleft");
      } else if (dx < -TH) {
        pressLook("arrowleft");
        releaseLook("arrowright");
      }
      if (dy < -TH) {
        pressLook("arrowup");
        releaseLook("arrowdown");
      } else if (dy > TH) {
        pressLook("arrowdown");
        releaseLook("arrowup");
      }
      // Stop turning shortly after the mouse stops moving (mouse-look feel).
      if (lookStopTimerRef.current)
        window.clearTimeout(lookStopTimerRef.current);
      lookStopTimerRef.current = window.setTimeout(releaseAllLook, 110);
    },
    [pressLook, releaseLook, releaseAllLook],
  );

  const onViewerPointerEnd = useCallback(() => {
    dragRef.current = false;
    lastPosRef.current = null;
    if (lookStopTimerRef.current) window.clearTimeout(lookStopTimerRef.current);
    releaseAllLook();
  }, [releaseAllLook]);

  // ----------------------------------------------------------------- chat / TTS
  const speak = useCallback(
    (text: string) => {
      if (!ttsOn || typeof window === "undefined" || !window.speechSynthesis)
        return;
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
    },
    [ttsOn],
  );

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, isThinking]);

  const submit = useCallback(
    async (text: string) => {
      const message = text.trim();
      if (!message || isThinking) return;
      if (!assistantConfigured) {
        toast.error("Assistant backend URL is not configured (VITE_ASSISTANT_URL).");
        return;
      }
      setMessages((m) => [...m, { id: uid(), role: "user", text: message }]);
      setInput("");
      setIsThinking(true);
      try {
        const response = await sendChat(message);
        if (typeof response === "string") {
          const reply = response || "(no response)";
          setMessages((m) => [
            ...m,
            { id: uid(), role: "assistant", text: reply },
          ]);
          speak(reply);
        } else {
          const llm = response.llm || "(no response)";
          const geometric = response.geometric || "(no directions)";
          setMessages((m) => [
            ...m,
            {
              id: uid(),
              role: "assistant",
              text: llm,
              dual: { llm, geometric, meta: response.meta },
            },
          ]);
          speak(llm);
        }
      } catch (err) {
        const msg =
          err instanceof Error ? err.message : "Failed to reach the assistant.";
        setMessages((m) => [
          ...m,
          { id: uid(), role: "assistant", text: `⚠️ ${msg}` },
        ]);
        toast.error(msg);
      } finally {
        setIsThinking(false);
      }
    },
    [isThinking, speak],
  );

  // "From here" button: recompute the route to the last destination from the
  // current position (no NLP; the backend reuses the stored target).
  const handleRecalculate = useCallback(async () => {
    if (isThinking) return;
    setMessages((m) => [
      ...m,
      { id: uid(), role: "user", text: "From here, where should I go?" },
    ]);
    setIsThinking(true);
    try {
      const response = await recalculateFromHere();
      if (typeof response === "string") {
        const reply = response || "(no response)";
        setMessages((m) => [...m, { id: uid(), role: "assistant", text: reply }]);
        speak(reply);
      } else {
        const llm = response.llm || "(no response)";
        const geometric = response.geometric || "(no directions)";
        setMessages((m) => [
          ...m,
          { id: uid(), role: "assistant", text: llm, dual: { llm, geometric, meta: response.meta } },
        ]);
        speak(llm);
      }
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : "Failed to recalculate.";
      setMessages((m) => [
        ...m,
        { id: uid(), role: "assistant", text: `⚠️ ${msg}` },
      ]);
      toast.error(msg);
    } finally {
      setIsThinking(false);
    }
  }, [isThinking, speak]);

  // Persist a landmark-vs-geometric preference for a given "both" answer.
  const handleSavePreference = useCallback(
    async (
      dual: NonNullable<ChatMessage["dual"]>,
      ratings: Record<string, number>,
    ) => {
      try {
        await savePreference({
          scene: dual.meta?.scene || status?.scene || "",
          destination: dual.meta?.destination || "",
          question: dual.meta?.question || "",
          landmark: dual.llm,
          geometric: dual.geometric,
          ratings,
          user_position: dual.meta?.user_position ?? null,
          destination_position: dual.meta?.destination_position ?? null,
        });
        toast.success("Preference saved");
      } catch (err) {
        toast.error(
          err instanceof Error ? err.message : "Failed to save preference",
        );
        throw err;
      }
    },
    [status],
  );

  // ---------------------------------------------------------- speech recognition
  const toggleMic = useCallback(() => {
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      toast.error("Speech recognition is not supported in this browser.");
      return;
    }
    if (micOn) {
      recognitionRef.current?.stop();
      setMicOn(false);
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onresult = (event: any) => {
      const transcript = event.results?.[0]?.[0]?.transcript ?? "";
      if (transcript) void submit(transcript);
    };
    recognition.onerror = () => {
      setMicOn(false);
      toast.error("Voice input failed.");
    };
    recognition.onend = () => setMicOn(false);
    recognitionRef.current = recognition;
    recognition.start();
    setMicOn(true);
  }, [micOn, submit]);

  const llmReady = status?.llm_loaded ?? false;
  const connected = status !== null;
  const backend = status?.backend ?? "local";
  const finetuned = status?.finetuned ?? false;
  const bboxOn = status?.overlays?.bboxes ?? false;
  const allBboxOn = status?.overlays?.all_bboxes ?? false;
  const roomsOn = status?.overlays?.rooms ?? false;
  const saveOn = status?.overlays?.save_frames ?? false;
  const naturalOn = status?.overlays?.natural ?? false;
  const directionsMode = status?.overlays?.directions_mode ?? "llm";
  const hasLastGoal = status?.has_last_goal ?? false;

  return (
    <div className="h-screen bg-background text-foreground flex flex-col overflow-hidden">
      {/* Background glows */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-24 left-1/4 w-[32rem] h-[32rem] bg-primary/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-[28rem] h-[28rem] bg-accent/10 rounded-full blur-3xl" />
      </div>

      {/* Header */}
      <header className="relative z-10 glass border-b border-border/50 shrink-0">
        <div className="max-w-[1600px] mx-auto px-3 sm:px-5 py-2 flex items-center gap-3 flex-wrap">
          <Link
            to="/"
            className="text-muted-foreground hover:text-foreground transition-colors"
            aria-label="Back"
          >
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-accent flex items-center justify-center glow-primary">
              <Compass className="w-5 h-5 text-primary-foreground animate-spin-slow" />
            </div>
            <div>
              <h1 className="text-sm sm:text-base font-bold leading-tight gradient-text">
                Conversational Navigation Assistant
              </h1>
              <p className="text-[11px] text-muted-foreground leading-tight">
                Explore the scene &amp; ask for directions
              </p>
            </div>
          </div>

          <div className="ml-auto flex items-center gap-2 flex-wrap">
            {status && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="secondary"
                    size="sm"
                    className="h-8 gap-1.5 text-xs"
                  >
                    <SlidersHorizontal className="w-3.5 h-3.5" />
                    Tools
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-60">
                  <DropdownMenuLabel>Overlays</DropdownMenuLabel>
                  <DropdownMenuCheckboxItem
                    checked={bboxOn}
                    onCheckedChange={() => toggleTool("b")}
                  >
                    Target bounding boxes (B)
                  </DropdownMenuCheckboxItem>
                  <DropdownMenuCheckboxItem
                    checked={allBboxOn}
                    onCheckedChange={() => toggleTool("g")}
                  >
                    All object boxes (G)
                  </DropdownMenuCheckboxItem>
                  <DropdownMenuCheckboxItem
                    checked={roomsOn}
                    onCheckedChange={() => toggleTool("r")}
                  >
                    Room boxes (R)
                  </DropdownMenuCheckboxItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuLabel>Capture</DropdownMenuLabel>
                  <DropdownMenuCheckboxItem
                    checked={saveOn}
                    onCheckedChange={() => toggleTool("p")}
                  >
                    Save color + semantic frames (P)
                  </DropdownMenuCheckboxItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuLabel>Prompt</DropdownMenuLabel>
                  <DropdownMenuCheckboxItem
                    checked={naturalOn}
                    onCheckedChange={(v) => toggleNatural(!!v)}
                  >
                    Natural language
                  </DropdownMenuCheckboxItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuLabel>Directions</DropdownMenuLabel>
                  <DropdownMenuRadioGroup
                    value={directionsMode}
                    onValueChange={(v) =>
                      toggleGeometric(v as "llm" | "both" | "geometric")
                    }
                  >
                    <DropdownMenuRadioItem value="llm">
                      LLM only
                    </DropdownMenuRadioItem>
                    <DropdownMenuRadioItem value="both">
                      Both (LLM + geometric)
                    </DropdownMenuRadioItem>
                    <DropdownMenuRadioItem value="geometric">
                      Geometric only (no LLM)
                    </DropdownMenuRadioItem>
                  </DropdownMenuRadioGroup>
                  <DropdownMenuSeparator />
                  <DropdownMenuLabel>User study (Both mode)</DropdownMenuLabel>
                  {STUDY_QUESTIONS.map((q) => (
                    <DropdownMenuCheckboxItem
                      key={q.key}
                      checked={studyEnabled[q.key]}
                      onCheckedChange={(v) =>
                        setStudyEnabled((s) => ({ ...s, [q.key]: !!v }))
                      }
                    >
                      {q.label}
                    </DropdownMenuCheckboxItem>
                  ))}
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={clearChat}>
                    <Trash2 className="w-4 h-4 mr-2" />
                    Clear conversation
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}
            {scenes.length > 0 && (
              <Select
                value={currentScene}
                onValueChange={handleSceneChange}
                disabled={switching}
              >
                <SelectTrigger className="h-8 w-[160px] text-xs bg-background/60">
                  <SelectValue placeholder="Scene" />
                </SelectTrigger>
                <SelectContent>
                  {scenes.map((s) => (
                    <SelectItem key={s.scene} value={s.scene} className="text-xs">
                      {s.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            {status && (
              <Select
                value={backend}
                onValueChange={(v) => applyLlm({ backend: v, finetuned })}
                disabled={llmBusy || switching}
              >
                <SelectTrigger className="h-8 w-[104px] text-xs bg-background/60">
                  <SelectValue placeholder="Backend" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="local" className="text-xs">
                    Local
                  </SelectItem>
                  <SelectItem value="openai" className="text-xs">
                    OpenAI
                  </SelectItem>
                </SelectContent>
              </Select>
            )}
            {status && backend === "local" && (
              <Select
                value={finetuned ? "finetuned" : "base"}
                onValueChange={(v) =>
                  applyLlm({ backend, finetuned: v === "finetuned" })
                }
                disabled={llmBusy || switching}
              >
                <SelectTrigger className="h-8 w-[116px] text-xs bg-background/60">
                  <SelectValue placeholder="Model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="finetuned" className="text-xs">
                    Finetuned
                  </SelectItem>
                  <SelectItem value="base" className="text-xs">
                    Base
                  </SelectItem>
                </SelectContent>
              </Select>
            )}
            <StatusBadge
              ok={connected}
              label={connected ? "Connected" : "Offline"}
            />
            <StatusBadge
              ok={llmReady}
              label={llmReady ? "LLM ready" : "LLM loading"}
              pulse={connected && !llmReady}
            />
            <StatusBadge
              ok={wsConnected}
              label={wsConnected ? "Low-latency" : "Streaming"}
            />
            {status?.scene && (
              <Badge variant="secondary" className="font-normal">
                {status.scene}
              </Badge>
            )}
            {status?.current_room && (
              <Badge className="bg-primary/15 text-primary border-primary/30 font-normal">
                {status.current_room}
              </Badge>
            )}
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="relative z-10 flex-1 min-h-0 w-full max-w-[1600px] mx-auto p-3 sm:p-4 grid gap-4 lg:grid-cols-[1fr_400px] lg:grid-rows-1">
        {/* Viewer */}
        <section className="flex flex-col gap-3 min-w-0 min-h-0">
          <div
            className="relative glass-strong rounded-2xl overflow-hidden flex-1 min-h-0 flex items-center justify-center cursor-grab active:cursor-grabbing touch-none select-none"
            onPointerDown={onViewerPointerDown}
            onPointerMove={onViewerPointerMove}
            onPointerUp={onViewerPointerEnd}
            onPointerLeave={onViewerPointerEnd}
            onPointerCancel={onViewerPointerEnd}
            onContextMenu={(e) => e.preventDefault()}
            onMouseEnter={() => {
              viewerHoveredRef.current = true;
              setViewerHovered(true);
            }}
            onMouseLeave={() => {
              viewerHoveredRef.current = false;
              setViewerHovered(false);
              releaseAll();
            }}
          >
            {/* WebSocket video (low latency). The canvas stays mounted so its
                ref is ready even before the socket connects. */}
            <canvas
              ref={canvasRef}
              className={cn(
                "w-full h-full object-contain bg-black pointer-events-none",
                !wsConnected && "hidden",
              )}
            />

            {/* Fallback to MJPEG (or a placeholder) while the socket is down. */}
            {!wsConnected &&
              (assistantConfigured && videoOk ? (
                <img
                  src={videoStreamUrl()}
                  alt="Live scene view"
                  className="w-full h-full object-contain bg-black pointer-events-none"
                  onLoad={() => setVideoOk(true)}
                  onError={() => setVideoOk(false)}
                  draggable={false}
                />
              ) : (
                <div className="text-center px-6">
                  <Compass className="w-12 h-12 text-muted-foreground mx-auto mb-3 animate-pulse" />
                  <p className="text-sm text-muted-foreground">
                    {assistantConfigured
                      ? "Connecting to the live stream…"
                      : "Set VITE_ASSISTANT_URL to connect to the backend."}
                  </p>
                  {assistantConfigured && !videoOk && (
                    <Button
                      variant="secondary"
                      size="sm"
                      className="mt-3"
                      onClick={() => setVideoOk(true)}
                    >
                      Retry stream
                    </Button>
                  )}
                </div>
              ))}

            {/* Keyboard focus indicator */}
            <div
              className={cn(
                "absolute top-3 left-3 flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium backdrop-blur-md transition-colors pointer-events-none",
                viewerHovered
                  ? "bg-primary/20 text-primary border border-primary/40"
                  : "bg-card/70 text-muted-foreground border border-border/50",
              )}
              title="Keyboard controls the viewer while the cursor is over it"
            >
              <Keyboard className="w-3.5 h-3.5" />
              {viewerHovered ? "Keyboard → viewer" : "Hover to control"}
            </div>

            {/* Hint */}
            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full bg-card/70 backdrop-blur-md border border-border/50 text-[11px] text-muted-foreground pointer-events-none whitespace-nowrap">
              Hover the view · drag to look · WASD move · G all boxes · R rooms · P save frames
            </div>

            {/* Scene switch overlay */}
            {(switching || llmBusy) && (
              <div className="absolute inset-0 z-10 bg-background/70 backdrop-blur-sm flex flex-col items-center justify-center gap-3">
                <Loader2 className="w-8 h-8 animate-spin text-primary" />
                <p className="text-sm text-muted-foreground">
                  {switching ? "Switching scene…" : "Switching model…"}
                </p>
              </div>
            )}
          </div>
        </section>

        {/* Chat */}
        <section className="glass-strong rounded-2xl flex flex-col min-h-0 h-full min-w-0">
          <div className="px-4 py-3 border-b border-border/50 flex items-center justify-between">
            <h2 className="font-semibold text-sm">Conversation</h2>
            <div className="flex items-center gap-1">
              <IconToggle
                active={ttsOn}
                onClick={() => setTtsOn((v) => !v)}
                onIcon={<Volume2 className="w-4 h-4" />}
                offIcon={<VolumeX className="w-4 h-4" />}
                title="Speak responses"
              />
              <IconToggle
                active={micOn}
                onClick={toggleMic}
                onIcon={<Mic className="w-4 h-4" />}
                offIcon={<MicOff className="w-4 h-4" />}
                title="Voice input"
              />
            </div>
          </div>

          <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.map((m) => (
              <div
                key={m.id}
                className={cn(
                  "flex",
                  m.role === "user" ? "justify-end" : "justify-start",
                )}
              >
                {m.dual ? (
                  <div className="max-w-[95%] w-full space-y-2">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      <div className="rounded-2xl rounded-bl-sm bg-secondary text-secondary-foreground px-4 py-2.5 text-sm whitespace-pre-wrap break-words">
                        <div className="text-[10px] uppercase tracking-wide text-primary/80 font-semibold mb-1">
                          Landmark (LLM)
                        </div>
                        {m.dual.llm}
                      </div>
                      <div className="rounded-2xl rounded-bl-sm bg-secondary/60 text-secondary-foreground px-4 py-2.5 text-sm whitespace-pre-wrap break-words">
                        <div className="text-[10px] uppercase tracking-wide text-accent/80 font-semibold mb-1">
                          Geometric
                        </div>
                        {m.dual.geometric}
                      </div>
                    </div>
                    <PreferencePanel
                      dual={m.dual}
                      enabled={studyEnabled}
                      onSave={(ratings) => handleSavePreference(m.dual!, ratings)}
                    />
                  </div>
                ) : (
                  <div
                    className={cn(
                      "max-w-[85%] rounded-2xl px-4 py-2.5 text-sm whitespace-pre-wrap break-words",
                      m.role === "user"
                        ? "bg-primary text-primary-foreground rounded-br-sm"
                        : "bg-secondary text-secondary-foreground rounded-bl-sm",
                    )}
                  >
                    {m.text}
                  </div>
                )}
              </div>
            ))}
            {isThinking && (
              <div className="flex justify-start">
                <div className="bg-secondary text-muted-foreground rounded-2xl rounded-bl-sm px-4 py-2.5 text-sm flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Thinking…
                </div>
              </div>
            )}
          </div>

          {hasLastGoal && (
            <div className="px-3 pt-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={isThinking}
                onClick={() => void handleRecalculate()}
                className="w-full h-8 gap-1.5 text-xs"
              >
                <RotateCw className="w-3.5 h-3.5" />
                From here, where should I go?
              </Button>
            </div>
          )}

          <form
            className="p-3 border-t border-border/50 flex items-end gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              void submit(input);
            }}
          >
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void submit(input);
                }
              }}
              placeholder="Ask how to reach a room or object…"
              className="min-h-[44px] max-h-32 resize-none bg-background/60"
              rows={1}
            />
            <Button
              type="submit"
              size="icon"
              disabled={isThinking || !input.trim()}
              className="shrink-0 h-11 w-11"
              aria-label="Send"
            >
              <Send className="w-4 h-4" />
            </Button>
          </form>
        </section>
      </main>
    </div>
  );
};

// ------------------------------------------------------------------ subcomponents

const PreferencePanel = ({
  dual,
  enabled,
  onSave,
}: {
  dual: NonNullable<ChatMessage["dual"]>;
  enabled: Record<StudyKey, boolean>;
  onSave: (ratings: Record<string, number>) => Promise<void>;
}) => {
  const questions = STUDY_QUESTIONS.filter((q) => enabled[q.key]);
  const [ratings, setRatings] = useState<Record<string, number>>({});
  const [saved, setSaved] = useState(false);
  const [saving, setSaving] = useState(false);

  if (questions.length === 0) return null;
  if (saved) {
    return (
      <div className="text-[11px] text-muted-foreground px-1">
        Preference saved ✓
      </div>
    );
  }

  const allAnswered = questions.every((q) => ratings[q.key]);

  return (
    <div className="rounded-xl border border-border/50 bg-background/40 p-2.5 space-y-2.5">
      <div className="text-[11px] font-semibold text-muted-foreground">
        Rate each answer (1 = Landmark · 5 = Geometric)
      </div>
      {questions.map((q) => (
        <div key={q.key} className="space-y-1">
          <div className="text-[11px]">{q.label}</div>
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] text-primary/80 w-14 shrink-0">
              Landmark
            </span>
            {[1, 2, 3, 4, 5].map((v) => (
              <button
                key={v}
                type="button"
                onClick={() => setRatings((r) => ({ ...r, [q.key]: v }))}
                className={cn(
                  "w-6 h-6 rounded-full border text-[10px] transition-colors",
                  ratings[q.key] === v
                    ? "bg-primary text-primary-foreground border-primary"
                    : "border-border/60 hover:border-primary/60",
                )}
                aria-label={`${q.label}: ${v}`}
              >
                {v}
              </button>
            ))}
            <span className="text-[10px] text-accent/80 w-14 shrink-0 text-right">
              Geometric
            </span>
          </div>
        </div>
      ))}
      <Button
        size="sm"
        className="h-7 text-xs"
        disabled={!allAnswered || saving}
        onClick={async () => {
          setSaving(true);
          try {
            await onSave(ratings);
            setSaved(true);
          } catch {
            /* toast handled by caller */
          } finally {
            setSaving(false);
          }
        }}
      >
        Save rating
      </Button>
    </div>
  );
};

const StatusBadge = ({
  ok,
  label,
  pulse,
}: {
  ok: boolean;
  label: string;
  pulse?: boolean;
}) => (
  <Badge
    variant="outline"
    className={cn(
      "font-normal gap-1.5",
      ok
        ? "border-success/40 text-success"
        : "border-muted-foreground/30 text-muted-foreground",
    )}
  >
    <span
      className={cn(
        "w-2 h-2 rounded-full",
        ok ? "bg-success" : "bg-muted-foreground",
        pulse && "animate-pulse",
      )}
    />
    {label}
  </Badge>
);

const IconToggle = ({
  active,
  onClick,
  onIcon,
  offIcon,
  title,
}: {
  active: boolean;
  onClick: () => void;
  onIcon: React.ReactNode;
  offIcon: React.ReactNode;
  title: string;
}) => (
  <button
    type="button"
    onClick={onClick}
    title={title}
    aria-label={title}
    aria-pressed={active}
    className={cn(
      "w-9 h-9 rounded-lg flex items-center justify-center transition-colors",
      active
        ? "bg-primary/20 text-primary"
        : "text-muted-foreground hover:text-foreground hover:bg-secondary",
    )}
  >
    {active ? onIcon : offIcon}
  </button>
);

export default ConversationalNavigationAssistant;
