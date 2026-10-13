import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vitest/config";

// The components are plain Vue (explicit imports, no Nuxt auto-imports), so they are
// tested with Vue's own plugin, the setup Nuxt's testing guide gives for components
// that do not rely on Nuxt.
export default defineConfig({
  plugins: [vue()],
  test: {
    environment: "jsdom",
    setupFiles: ["app/test-setup.ts"],
    include: ["app/**/*.test.ts"],
    coverage: {
      include: ["app/**/*.{ts,vue}"],
      exclude: [
        "app/app.vue",
        "app/lib/api-types.ts",
        "app/lib/fixtures.ts",
        "app/test-setup.ts",
        "app/**/*.test.ts",
      ],
      thresholds: { lines: 100, branches: 100, functions: 100, statements: 100 },
    },
  },
});
