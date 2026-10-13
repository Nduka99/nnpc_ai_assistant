import { defineConfig } from "@playwright/test";

// The thread in a real browser, served by the API itself (`petroknowledge serve`), so
// its security headers apply as in use. Each test answers the API (`page.route`), so
// runs are fast and need no model servers or database. It uses the installed
// Microsoft Edge (Playwright's `msedge` channel): no browser download.
export default defineConfig({
  testDir: "e2e",
  // Reduced motion: accessibility checks then measure settled colours, not a frame of
  // a fade, and the reduced-motion styles are exercised.
  use: { baseURL: "http://127.0.0.1:4173", channel: "msedge", reducedMotion: "reduce" },
  webServer: {
    command:
      "npm --prefix web run build && uv run --no-sync petroknowledge serve --port 4173",
    cwd: "..",
    url: "http://127.0.0.1:4173/healthz",
    reuseExistingServer: false,
    timeout: 300_000,
  },
});
