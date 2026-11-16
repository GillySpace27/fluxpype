import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies the API to a running `flux-plumb` (default port 8765).
// Open the tokenized URL it prints once on the dev server's origin to get the cookie.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 700,
  },
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:8765", ws: true, changeOrigin: false },
    },
  },
});
