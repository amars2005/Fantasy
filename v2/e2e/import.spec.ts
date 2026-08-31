/**
 * Platform imports, from the form's point of view.
 *
 * What matters here is not "does the mapping work" -- `tests/import.test.ts`
 * pins that against the mapper directly -- but whether the *review flow* stays
 * honest: values land in the form, unmapped keys are named, nothing is saved
 * until a person says so.
 *
 * The upstream call happens on the server, so the browser cannot intercept
 * `api.sleeper.app`. These stub our own `/api/import` instead, and build the
 * stubbed body with the real mapper, so a mapping regression still shows up
 * here rather than being papered over by a hand-written fixture.
 */

import { expect, test } from "@playwright/test";

import { mapEspnLeague } from "../lib/import/espn";
import { mapSleeperLeague } from "../lib/import/sleeper";

const SLEEPER_RAW = {
  name: "Sleeper Import Test",
  total_rosters: 10,
  roster_positions: ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF", "BN", "BN"],
  settings: { playoff_week_start: 15, playoff_teams: 6 },
  scoring_settings: {
    pass_yd: 0.04,
    pass_td: 4,
    pass_int: -2,
    rush_yd: 0.1,
    rush_td: 6,
    rec: 0.5,
    rec_yd: 0.1,
    rec_td: 6,
    fum_lost: -2,
    sack: 1,
    int: 2,
    a_completely_new_key: 7,
  },
};

const ESPN_RAW = {
  settings: {
    name: "ESPN Import Test",
    size: 12,
    scoringSettings: {
      scoringItems: [
        { statId: 3, points: 0.04 },
        { statId: 4, points: 6 },
        { statId: 53, points: 1 },
        { statId: 4242, points: 9 },
      ],
    },
    rosterSettings: {
      lineupSlotCounts: { "0": 1, "2": 2, "4": 2, "6": 1, "23": 1, "16": 1, "17": 1, "20": 6 },
    },
    scheduleSettings: { matchupPeriodCount: 14, playoffTeamCount: 6 },
  },
};

test.describe("Sleeper", () => {
  test("prefills the form and lands on a working board", async ({ page }) => {
    await page.route("**/api/import", (route) =>
      route.fulfill({ json: mapSleeperLeague(SLEEPER_RAW) }),
    );

    await page.goto("/");
    await page.getByRole("button", { name: /Import from Sleeper/i }).click();
    await page.getByLabel("League id").fill("123456789012345678");
    await page.getByRole("button", { name: "Import settings" }).click();

    await expect(page.getByLabel("Name")).toHaveValue("Sleeper Import Test");
    await expect(page.getByLabel("Teams", { exact: true })).toHaveValue("10");
    await expect(page.getByLabel("Per reception", { exact: true })).toHaveValue("0.5");
    // Half-PPR, so the board is anchored to the half-PPR ADP.
    await expect(page.getByText(/anchored to the half-ppr board/i)).toBeVisible();

    await page.getByRole("button", { name: "Create league" }).click();
    await page.waitForURL(/\/league\/.+/, { timeout: 90_000 });
    await page.waitForSelector("main.board");
    await expect(page.locator("#slot option")).toHaveCount(10);
  });

  test("names the keys it could not place rather than dropping them", async ({ page }) => {
    await page.route("**/api/import", (route) =>
      route.fulfill({ json: mapSleeperLeague(SLEEPER_RAW) }),
    );

    await page.goto("/");
    await page.getByRole("button", { name: /Import from Sleeper/i }).click();
    await page.getByLabel("League id").fill("123456789012345678");
    await page.getByRole("button", { name: "Import settings" }).click();

    await expect(page.getByText(/a_completely_new_key/)).toBeVisible();
    await expect(page.getByText(/kept their defaults/i)).toBeVisible();
  });

  test("reports a bad league id in language a person can act on", async ({ page }) => {
    await page.route("**/api/import", (route) =>
      route.fulfill({
        status: 502,
        json: {
          error: "No Sleeper league with that id. It is the long number in the league URL.",
        },
      }),
    );

    await page.goto("/");
    await page.getByRole("button", { name: /Import from Sleeper/i }).click();
    await page.getByLabel("League id").fill("nope");
    await page.getByRole("button", { name: "Import settings" }).click();

    await expect(page.locator(".err")).toContainText(/long number in the league URL/i);
  });
});

test.describe("ESPN", () => {
  test("always shows a caution, because the id tables drift", async ({ page }) => {
    await page.route("**/api/import", (route) => route.fulfill({ json: mapEspnLeague(ESPN_RAW) }));

    await page.goto("/");
    await page.getByRole("button", { name: /Import from ESPN/i }).click();
    await page.getByLabel("League id").fill("987654");
    await page.getByRole("button", { name: "Import settings" }).click();

    await expect(page.getByText(/undocumented/i)).toBeVisible();
    await expect(page.getByText(/statId 4242/)).toBeVisible();
    // Six-point passing touchdowns diverge from the anchored board, and that
    // has to be said before the league is created, not after.
    await expect(page.getByText(/Passing touchdowns are worth 6/i)).toBeVisible();
  });

  test("explains what a private league needs", async ({ page }) => {
    await page.route("**/api/import", (route) =>
      route.fulfill({
        status: 502,
        json: {
          error: "ESPN refused the request. Private leagues need your espn_s2 and SWID cookies.",
        },
      }),
    );

    await page.goto("/");
    await page.getByRole("button", { name: /Import from ESPN/i }).click();
    await page.getByLabel("League id").fill("987654");
    await page.getByRole("button", { name: "Import settings" }).click();

    await expect(page.locator(".err")).toContainText(/espn_s2 and SWID/i);
  });

  test("nothing is saved until the form is submitted", async ({ page }) => {
    await page.route("**/api/import", (route) =>
      route.fulfill({
        json: mapEspnLeague({
          settings: {
            name: "Never Saved",
            size: 8,
            scoringSettings: { scoringItems: [{ statId: 53, points: 1 }] },
            rosterSettings: { lineupSlotCounts: { "0": 1, "2": 2, "4": 2, "20": 5 } },
          },
        }),
      }),
    );

    await page.goto("/");
    await page.getByRole("button", { name: /Import from ESPN/i }).click();
    await page.getByLabel("League id").fill("987654");
    await page.getByRole("button", { name: "Import settings" }).click();
    await expect(page.getByLabel("Name")).toHaveValue("Never Saved");

    // Navigate away without submitting; it must not exist anywhere.
    await page.goto("/");
    await expect(page.getByRole("link", { name: "Never Saved" })).toBeHidden();
  });

  test("the real importer is still reachable and fails cleanly", async ({ request }) => {
    // No stub: this one genuinely calls out, to prove the route is wired up and
    // that a refusal surfaces as a message rather than a stack trace.
    const res = await request.post("/api/import", {
      data: { source: "espn", leagueId: "0", season: 2026 },
    });
    expect([400, 401, 404, 502]).toContain(res.status());
    expect(await res.text()).toMatch(/espn|refused|returned/i);
  });
});
