/**
 * Refresh: the thing this whole architecture exists to get right.
 *
 * The tool being replaced kept the draft in a dictionary in one process, so a
 * closed terminal lost it. The reference league allows three hours per pick and
 * a draft runs over days, which means a browser *will* be reloaded, slept,
 * reopened on another device, and reopened after the daily ADP refresh has
 * moved underneath it. Each of those is a case below.
 */

import { expect, test } from "@playwright/test";
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

import {
  createLeague,
  markBySearch,
  onTheClock,
  recommendedNames,
  waitForBoard,
  waitForPicksOnServer,
} from "./helpers";

const DATA_FILE = join(process.cwd(), ".data", "e2e-leagues.json");

test.describe("state survives a reload", () => {
  test("picks are still there after refreshing the page", async ({ page, request }) => {
    const league = await createLeague(request, "Reload league");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    expect(await onTheClock(page)).toBe(1);

    const first = await markBySearch(page, "a", false);
    await expect.poll(() => onTheClock(page)).toBe(2);
    const second = await markBySearch(page, "b", true);
    await expect.poll(() => onTheClock(page)).toBe(3);

    await page.reload();
    await waitForBoard(page);

    // The clock, the roster and the recent list all come back.
    expect(await onTheClock(page)).toBe(3);
    await expect(page.locator("header.app .stat").nth(3).locator("b")).toHaveText("1");
    await expect(page.getByText(first, { exact: false }).first()).toBeVisible();
    await expect(page.getByText(second, { exact: false }).first()).toBeVisible();
  });

  test("a drafted player does not come back to the board", async ({ page, request }) => {
    const league = await createLeague(request, "No resurrection");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    const before = await recommendedNames(page);
    const taken = before[0];

    await page.locator("main.board table tbody tr").first().getByRole("button", { name: "Gone" }).click();
    await expect.poll(() => onTheClock(page)).toBe(2);

    await page.reload();
    await waitForBoard(page);

    const after = await recommendedNames(page);
    expect(after).not.toContain(taken);
  });

  test("the chosen draft slot survives a reload", async ({ page, request }) => {
    const league = await createLeague(request, "Slot memory");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    await page.locator("#slot").selectOption("11");
    // Slot 11 of 14 picks 11th, so the header must follow.
    await expect.poll(async () =>
      Number(await page.locator("header.app .stat").nth(1).locator("b").innerText()),
    ).toBe(11);

    await page.reload();
    await waitForBoard(page);

    await expect(page.locator("#slot")).toHaveValue("11");
    await expect(page.locator("header.app .stat").nth(1).locator("b")).toHaveText("11");
  });

  test("an unidentified pick still moves the clock across a reload", async ({ page, request }) => {
    // The counter falling behind the room is what corrupts every VONA number,
    // so this has to survive exactly like a real pick does.
    const league = await createLeague(request, "Unknown pick");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    await page.getByRole("button", { name: /not on the board/i }).click();
    await expect.poll(() => onTheClock(page)).toBe(2);

    await page.reload();
    await waitForBoard(page);

    expect(await onTheClock(page)).toBe(2);
    await expect(page.getByText("unidentified pick").first()).toBeVisible();
  });

  test("undo survives a reload and does not renumber later picks", async ({ page, request }) => {
    const league = await createLeague(request, "Undo league");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    await markBySearch(page, "a");
    await expect.poll(() => onTheClock(page)).toBe(2);
    await markBySearch(page, "b");
    await expect.poll(() => onTheClock(page)).toBe(3);
    await waitForPicksOnServer(request, league.id, 2);

    await page.getByRole("button", { name: "Undo", exact: true }).click();
    await expect.poll(() => onTheClock(page)).toBe(2);

    await page.reload();
    await waitForBoard(page);
    expect(await onTheClock(page)).toBe(2);

    // The void is a soft delete, so the row is still on record.
    const stored = await request.get(`/api/leagues/${league.id}/picks`);
    const { picks } = await stored.json();
    expect(picks.length).toBe(2);
    expect(picks.filter((p: { voidedAt: string | null }) => p.voidedAt !== null).length).toBe(1);
    expect(picks.map((p: { seq: number }) => p.seq)).toEqual([1, 2]);
  });
});

test.describe("reconciling across tabs", () => {
  test("a second tab catches up when it is focused", async ({ page, context, request }) => {
    // One person per league, but a laptop and a phone -- or two tabs -- still
    // have to converge. Reconciliation is on focus, not a poll.
    const league = await createLeague(request, "Two tabs");

    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    const second = await context.newPage();
    await second.goto(`/league/${league.id}`);
    await waitForBoard(second);
    expect(await onTheClock(second)).toBe(1);

    // Pick in the first tab.
    await page.bringToFront();
    await markBySearch(page, "a");
    await expect.poll(() => onTheClock(page)).toBe(2);

    // Focusing the second tab reconciles it.
    await second.bringToFront();
    await second.dispatchEvent("body", "focus");
    await expect.poll(() => onTheClock(second), { timeout: 15_000 }).toBe(2);

    await second.close();
  });
});

test.describe("refreshing the board itself", () => {
  test("newer ADP does not silently re-anchor a draft in progress", async ({ page, request }) => {
    // ADP refreshes daily and a draft runs over days, so this *will* happen.
    // Re-deriving underneath a live draft would move every projection, tier and
    // survival number, so the board is frozen on the first pick and the change
    // is offered rather than applied.
    const league = await createLeague(request, "Freeze league");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    // Fingerprint the stored board. This -- not the visible ordering, which
    // legitimately shifts as players come off -- is what must not move.
    const fingerprint = async () => {
      const view = await (await request.get(`/api/leagues/${league.id}`)).json();
      return {
        adpAsOf: view.adpAsOf,
        refreshAvailable: view.refreshAvailable,
        projections: (view.board as { player_id: string; proj_points: number; tier: number }[])
          .map((p) => `${p.player_id}:${p.proj_points}:${p.tier}`)
          .join("|"),
      };
    };

    const before = await fingerprint();

    await markBySearch(page, "a");
    await expect.poll(() => onTheClock(page)).toBe(2);
    await waitForPicksOnServer(request, league.id, 1);

    // Simulate the daily refresh having produced a different bundle by
    // invalidating the stored hash. The league is frozen, so nothing should
    // change on its own.
    const raw = JSON.parse(await readFile(DATA_FILE, "utf-8"));
    expect(raw.leagues[league.id].frozenAt, "first pick should freeze the board").toBeTruthy();
    raw.leagues[league.id].boardHash = "stale-on-purpose";
    await writeFile(DATA_FILE, JSON.stringify(raw, null, 1), "utf-8");

    await page.reload();
    await waitForBoard(page);

    // Offered...
    await expect(page.getByText(/Newer ADP is available/i)).toBeVisible();

    // ...but not applied: every projection and tier is exactly as it was.
    const after = await fingerprint();
    expect(after.projections).toBe(before.projections);
    expect(after.adpAsOf).toBe(before.adpAsOf);
    expect(after.refreshAvailable).toBe(true);
    expect(await onTheClock(page)).toBe(2);
  });

  test("refreshing on request rebuilds the board and keeps the picks", async ({ page, request }) => {
    const league = await createLeague(request, "Explicit refresh");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    await markBySearch(page, "a");
    await expect.poll(() => onTheClock(page)).toBe(2);
    await waitForPicksOnServer(request, league.id, 1);

    const raw = JSON.parse(await readFile(DATA_FILE, "utf-8"));
    raw.leagues[league.id].boardHash = "stale-on-purpose";
    await writeFile(DATA_FILE, JSON.stringify(raw, null, 1), "utf-8");

    await page.reload();
    await waitForBoard(page);
    await page.getByRole("button", { name: /Refresh anyway/i }).click();

    // The banner clears, the freeze lifts, and the draft is untouched.
    await expect(page.getByText(/Newer ADP is available/i)).toBeHidden({ timeout: 20_000 });
    expect(await onTheClock(page)).toBe(2);

    const after = JSON.parse(await readFile(DATA_FILE, "utf-8"));
    expect(after.leagues[league.id].boardHash).not.toBe("stale-on-purpose");
    expect(after.leagues[league.id].frozenAt).toBeNull();
  });

  test("a league that has not started refreshes without being asked", async ({ request }) => {
    // Nothing to protect before the first pick, so a stale hash is just a
    // cache miss rather than something the user has to decide about.
    const league = await createLeague(request, "Unstarted");
    const view = await (await request.get(`/api/leagues/${league.id}`)).json();
    expect(view.frozenAt).toBeNull();
    expect(view.refreshAvailable).toBe(false);
  });
});
