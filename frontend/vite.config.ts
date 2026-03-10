import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    // Force a single resolved copy of these packages.
    // Without this, Vite can pre-bundle @react-three/fiber independently
    // inside the Drei chunk, producing two separate Context instances —
    // Drei's useThree() then fails with "Hooks can only be used within Canvas".
    dedupe: ["@react-three/fiber", "@react-three/drei", "three", "react", "react-dom"],
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  // @niivue/dcm2niix uses `new Worker(new URL('./worker.js', import.meta.url))`
  // which Vite handles natively. Pre-bundling it breaks the worker URL resolution.
  optimizeDeps: {
    exclude: ["@niivue/dcm2niix"],
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/upload": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
