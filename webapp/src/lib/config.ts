// Configuration for external services.
// Set these at build/dev time in `webapp/.env` (see `.env.example`).

export const config = {
  googleSheetsUrl: import.meta.env.VITE_GOOGLE_SHEETS_URL ?? "",
  serverUrl: import.meta.env.VITE_SERVER_URL ?? "",
  // Base URL of the Conversational Navigation Assistant backend (cna_server.py).
  // Example: http://localhost:5001  or  https://<tunnel>.ngrok-free.dev
  assistantUrl:
    (import.meta.env.VITE_ASSISTANT_URL as string | undefined)?.replace(/\/$/, "") ??
    "",
} as const;

if (!config.serverUrl) {
  console.warn(
    "[WARN] config.serverUrl: expected=VITE_SERVER_URL env var, got=empty, " +
    "fallback=none (requests to /process will fail). " +
    "Create webapp/.env with VITE_SERVER_URL=<backend URL>."
  );
}
