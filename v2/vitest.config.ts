import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    environment: "node",
    // The golden fixtures are a megabyte of JSON and the survival tests run
    // real Monte Carlo; the default 5s timeout is too tight.
    testTimeout: 30_000,
  },
});
