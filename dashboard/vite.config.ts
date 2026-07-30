import { defineConfig } from "vite";

export default defineConfig({
  server: {
    proxy: {
      "/api": {
        target: "http://172.25.10.2:23456",
        changeOrigin: true,
      },
    },
  },
});
