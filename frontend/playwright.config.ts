import { defineConfig, devices } from "@playwright/test";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const frontendRoot = resolve(fileURLToPath(new URL(".", import.meta.url)));
const repoRoot = resolve(frontendRoot, "..");
const reportPath = resolve(repoRoot, "data/eval/results/e2e/playwright-results.json");
const useExternal = Boolean(process.env.PLAYWRIGHT_BASE_URL);

export default defineConfig({
  testDir: "./e2e",
  testIgnore: [
    ...(process.env.E2E_LIVE === "1" ? [] : ["**/live/**"]),
    ...(process.env.E2E_VISUAL === "1" ? [] : ["**/visual.spec.ts"]),
  ],
  fullyParallel: false,
  timeout: 90_000,
  expect: { timeout: 15_000 },
  reporter: [
    ["list"],
    ["json", { outputFile: reportPath }],
    ["html", { outputFolder: resolve(repoRoot, "playwright-report"), open: "never" }],
  ],
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL ?? "http://127.0.0.1:5173",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: useExternal ? undefined : [
    {
      command: "node e2e/start-backend.mjs",
      cwd: frontendRoot,
      url: "http://127.0.0.1:8000/health",
      timeout: 180_000,
      reuseExistingServer: true,
    },
    {
      command: "cross-env VITE_API_PROXY_TARGET=http://127.0.0.1:8000 VITE_API_PROXY_STRIP_PREFIX=1 npm run dev -- --host 127.0.0.1",
      cwd: frontendRoot,
      url: "http://127.0.0.1:5173",
      timeout: 120_000,
      reuseExistingServer: true,
    },
  ],
});
