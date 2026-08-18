import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env.VITE_PORT ?? 5173),
    host: "0.0.0.0",
    proxy: {
      "/api": {
        target: process.env.VITE_API_PROXY_TARGET ?? "http://localhost",
        changeOrigin: true,
        ...(process.env.VITE_API_PROXY_STRIP_PREFIX === "1"
          ? { rewrite: (path: string) => path.replace(/^\/api/, "") }
          : {}),
      },
    },
  },
});
