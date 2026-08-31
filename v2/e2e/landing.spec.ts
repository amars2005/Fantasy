/**
 * Creating a league, and getting back to one.
 *
 * There are no accounts, so the "your leagues" list and the league URL are the
 * only routes back in. Both are covered here.
 */

import { expect, test } from "@playwright/test";

import { createLeague } from "./helpers";

test.describe("creating a league", () => {
  test("the form builds a board and lands on it", async ({ page }) => {
    await page.goto("/");

    await page.getByLabel("Name").fill("Playwright League");
    await page.getByLabel("Teams", { exact: true }).fill("10");
    await page.getByLabel("Per reception", { exact: true }).fill("0.5");

    // The anchor is worked out live from the scoring.
    await expect(page.getByText(/anchored to the half-ppr board/i)).toBeVisible();

    await page.getByRole("button", { name: "Create league" }).click();

    await page.waitForURL(/\/league\/.+/, { timeout: 60_000 });
    await page.waitForSelector("main.board");
    await expect(page.locator("header.app h1")).toHaveText("Playwright League");
    await expect(page.locator("#slot option")).toHaveCount(10);
  });

  test("a lineup that cannot be drafted is refused with a reason", async ({ page }) => {
    await page.goto("/");

    await page.getByLabel("Rounds", { exact: true }).fill("3");
    await expect(page.getByText(/cannot be drafted in this many rounds/i)).toBeVisible();

    await page.getByRole("button", { name: "Create league" }).click();
    await expect(page.locator(".err")).toContainText(/rounds|starting slots/i);
    // Still on the form, with the entered values intact.
    await expect(page).toHaveURL(/\/$/);
  });

  test("scoring that is identically zero is refused", async ({ page, request }) => {
    const res = await request.post("/api/leagues", {
      data: {
        name: "Zero scoring",
        config: {
          teams: 12, rounds: 15, bench: 6, irSlots: 1,
          starters: { QB: 1, RB: 2, WR: 2, TE: 1, FLEX: 1, K: 1, DST: 1 },
          flexEligible: ["RB", "WR", "TE"],
          positionMax: {},
          schedule: { regularSeasonWeeks: 14, playoffWeeks: [15, 16, 17], playoffTeams: 6 },
          scoring: { receptions: 0 },
          kickerScoring: {},
          dst: { events: {}, pointsAllowedBands: [], yardsAllowedBands: [] },
        },
      },
    });

    expect(res.status()).toBe(422);
    expect(await res.text()).toMatch(/identically zero/i);
  });

  test("every scoring rule is reachable, not just the headline ones", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByLabel("fumbles lost")).toBeHidden();
    await page.getByRole("button", { name: /Show every scoring rule/i }).click();
    await expect(page.getByLabel("fumbles lost")).toBeVisible();
  });
});

test.describe("getting back to a league", () => {
  test("a created league is remembered on this device", async ({ page }) => {
    await page.goto("/");
    await page.getByLabel("Name").fill("Remembered League");
    await page.getByRole("button", { name: "Create league" }).click();
    await page.waitForURL(/\/league\/.+/, { timeout: 60_000 });

    await page.goto("/");
    await expect(page.getByRole("link", { name: "Remembered League" })).toBeVisible();
  });

  test("forgetting a league removes it from the list but not the server", async ({ page }) => {
    await page.goto("/");
    await page.getByLabel("Name").fill("Forgettable League");
    await page.getByRole("button", { name: "Create league" }).click();
    await page.waitForURL(/\/league\/.+/, { timeout: 60_000 });
    const url = page.url();

    await page.goto("/");
    await page
      .locator(".leaguelist li")
      .filter({ hasText: "Forgettable League" })
      .getByRole("button", { name: "Forget" })
      .click();
    await expect(page.getByRole("link", { name: "Forgettable League" })).toBeHidden();

    // The link still works: forgetting is a local convenience, not a delete.
    await page.goto(url);
    await page.waitForSelector("main.board");
    await expect(page.locator("header.app h1")).toHaveText("Forgettable League");
  });

  test("an unknown league id says so rather than hanging", async ({ page }) => {
    await page.goto("/league/does-not-exist");
    await expect(page.getByRole("heading", { name: "Not found" })).toBeVisible();
  });

  test("a league URL works in a browser that has never seen it", async ({ browser, request }) => {
    const league = await createLeague(request, "Shared By Link");
    // A completely fresh context: no localStorage, no cache.
    const fresh = await browser.newContext();
    const page = await fresh.newPage();

    await page.goto(`/league/${league.id}`);
    await page.waitForSelector("main.board");
    await expect(page.locator("header.app h1")).toHaveText("Shared By Link");

    await fresh.close();
  });
});
