import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "0.0.0.0", // Accetta connessioni da qualsiasi IP
    port: 8080,
    strictPort: true,

    // ⚠️ SOLUZIONE: Permetti host ngrok
    allowedHosts: [
      "eustatic-unenviable-sunshine.ngrok-free.dev", // Il tuo URL specifico
      "monasterial-daine-swirlier.ngrok-free.dev",
      ".ngrok-free.dev", // Oppure tutti i domini ngrok
      ".ngrok-free.app", // E anche questi
      ".ngrok.io", // E questi (vecchi domini ngrok)
    ],

    hmr: {
      protocol: "wss",
      clientPort: 443,
    },
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(
    Boolean
  ),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
