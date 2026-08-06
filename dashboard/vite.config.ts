import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const bcgApi = env.VITE_BCG_API_URL || "http://127.0.0.1:8848";
  return {
    server: {
      proxy: {
        // Dev proxy for the BCG construct server (contracts/http.schema.json).
        "/graph": {
          target: bcgApi,
          changeOrigin: true,
        },
        "/health": {
          target: bcgApi,
          changeOrigin: true,
        },
      },
    },
  };
});
