/**
 * Shared setup for the end-to-end tests.
 *
 * Leagues are created through the API rather than by driving the form, because
 * almost every test needs one and only `landing.spec.ts` is actually about the
 * form. Board derivation costs a couple of hundred milliseconds, so tests that
 * can share a league do.
 */

import { expect, type APIRequestContext, type Page } from "@playwright/test";

import { REFERENCE_LEAGUE } from "../lib/config";
import type { LeagueConfig } from "../lib/types";

export const reference = (): LeagueConfig => structuredClone(REFERENCE_LEAGUE);

/** A 12-team superflex league with six-point passing touchdowns. */
export function superflex(): LeagueConfig {
  const config = reference();
  config.teams = 12;
  config.rounds = 16;
  config.scoring = { ...config.scoring, passing_tds: 6 };
  config.starters = { QB: 1, RB: 2, WR: 2, TE: 1, FLEX: 1, SUPERFLEX: 1, K: 1, DST: 1 };
  return config;
}

export interface CreatedLeague {
  id: string;
  name: string;
  format: string;
}

export async function createLeague(
  request: APIRequestContext,
  name: string,
  config: LeagueConfig = reference(),
): Promise<CreatedLeague> {
  const res = await request.post("/api/leagues", { data: { name, config } });
  if (!res.ok()) throw new Error(`create failed: ${res.status()} ${await res.text()}`);
  return (await res.json()) as CreatedLeague;
}

/** Wait for the board to have finished its first recommendation pass. */
export async function waitForBoard(page: Page): Promise<void> {
  await page.waitForSelector("main.board", { state: "visible" });
  await page.waitForSelector("table tbody tr", { state: "visible" });
}

/** The names currently listed in the recommendations table, in order. */
export async function recommendedNames(page: Page): Promise<string[]> {
  return page.locator("main.board table tbody tr td.name").allInnerTexts();
}

/** The "on the clock" number from the header. */
export async function onTheClock(page: Page): Promise<number> {
  const text = await page.locator("header.app .stat").first().locator("b").innerText();
  return Number(text);
}

/** Mark the first search result, by keyboard, the way it is used at a table. */
export async function markBySearch(
  page: Page,
  query: string,
  mine = false,
): Promise<string> {
  const search = page.locator('input[type="search"]');
  await search.fill(query);
  // Wait for the result list to catch up with the query.
  const firstResult = page.locator("main.board .panel .slot .name").first();
  await firstResult.waitFor({ state: "visible" });
  const name = await firstResult.innerText();

  await search.press(mine ? "Shift+Enter" : "Enter");
  return name;
}

/**
 * Wait until the server has actually recorded `count` picks.
 *
 * The header clock updates optimistically the moment a pick is marked -- that
 * is the whole point of computing in the browser -- so polling the clock says
 * nothing about whether the write has landed. Any assertion about server state
 * has to wait for this instead.
 */
export async function waitForPicksOnServer(
  request: APIRequestContext,
  leagueId: string,
  count: number,
): Promise<void> {
  await expect
    .poll(
      async () => {
        const res = await request.get(`/api/leagues/${leagueId}/picks`);
        return res.ok() ? ((await res.json()).picks as unknown[]).length : -1;
      },
      { timeout: 20_000 },
    )
    .toBe(count);
}
