import { defineConfig, devices } from "@playwright/test";
import { join } from "node:path";

const PORT = 3210;

export default defineConfig({
  testDir: "./e2e",
  // Marking a pick runs a 3000-simulation Monte Carlo in the browser, and the
  // build step in `webServer` is not fast either.
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  // Serial: every spec shares one file-backed store, and the point of these
  // tests is state that survives, so isolating them from each other by luck of
  // scheduling would defeat it.
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",

  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },

  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],

  webServer: {
    command: `npx next build && npx next start -p ${PORT}`,
    url: `http://localhost:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 240_000,
    env: {
      // Its own store, so a test run never touches a real draft.
      FANTASY_DATA_FILE: join(process.cwd(), ".data", "e2e-leagues.json"),
    },
  },
});
