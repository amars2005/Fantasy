/**
 * The board under a clock.
 *
 * These exercise the interactions a person actually performs during a draft:
 * type a name, hit Enter, watch the ranking move. The keyboard path matters
 * most -- there are sixty seconds on the clock and nobody is reaching for a
 * mouse.
 */

import { expect, test } from "@playwright/test";

import {
  createLeague,
  markBySearch,
  onTheClock,
  recommendedNames,
  superflex,
  waitForBoard,
} from "./helpers";

test.describe("marking picks", () => {
  test("Enter marks a player as taken by someone else", async ({ page, request }) => {
    const league = await createLeague(request, "Keyboard league");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    const name = await markBySearch(page, "a", false);

    await expect.poll(() => onTheClock(page)).toBe(2);
    // Taken by the room, so it does not join my roster.
    await expect(page.locator("header.app .stat").nth(3).locator("b")).toHaveText("0");
    await expect(page.getByText(name, { exact: false }).first()).toBeVisible();
  });

  test("Shift+Enter marks a player as mine", async ({ page, request }) => {
    const league = await createLeague(request, "Mine league");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    await markBySearch(page, "a", true);

    await expect.poll(() => onTheClock(page)).toBe(2);
    await expect(page.locator("header.app .stat").nth(3).locator("b")).toHaveText("1");
  });

  test("the search box clears and refocuses, ready for the next pick", async ({ page, request }) => {
    const league = await createLeague(request, "Focus league");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    const search = page.locator('input[type="search"]');
    await markBySearch(page, "a");

    await expect(search).toHaveValue("");
    await expect(search).toBeFocused();
  });

  test("the board re-ranks after a pick", async ({ page, request }) => {
    const league = await createLeague(request, "Rerank league");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    const before = await recommendedNames(page);
    await page.locator("main.board table tbody tr").first()
      .getByRole("button", { name: "Gone" }).click();
    await expect.poll(() => onTheClock(page)).toBe(2);

    const after = await recommendedNames(page);
    expect(after[0]).not.toBe(before[0]);
    expect(after).not.toContain(before[0]);
  });

  test("taking players changes which position is most urgent", async ({ page, request }) => {
    // Position urgency is the pick signal; it has to respond to the roster.
    const league = await createLeague(request, "Urgency league");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    const urgencyOrder = async () =>
      page.locator("main.board .rail-right .urgrow .pos").allInnerTexts();

    const before = await urgencyOrder();

    // Take three running backs for myself.
    for (let i = 0; i < 3; i++) {
      const rbRow = page.locator("main.board table tbody tr").filter({ has: page.locator("td .pos.RB") }).first();
      await rbRow.getByRole("button", { name: "Mine" }).click();
      await expect.poll(() => onTheClock(page)).toBe(i + 2);
    }

    const after = await urgencyOrder();
    expect(after).not.toEqual(before);
  });
});

test.describe("search", () => {
  test("finds a defence by nickname, not just abbreviation", async ({ page, request }) => {
    // The room says "Ravens D/ST"; nobody types BAL.
    const league = await createLeague(request, "Nickname league");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    await page.locator('input[type="search"]').fill("ravens");
    await expect(page.locator("main.board .panel .slot .pos.DST").first()).toBeVisible();
  });

  test("a stray bracket matches nothing instead of throwing", async ({ page, request }) => {
    // This killed the request mid-draft in the Python version until the search
    // was made literal rather than a regex.
    const league = await createLeague(request, "Bracket league");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));

    await page.locator('input[type="search"]').fill("(");
    await page.waitForTimeout(300);

    expect(errors).toEqual([]);
    await expect(page.locator("main.board table tbody tr").first()).toBeVisible();
  });

  test("does not offer a player who is already gone", async ({ page, request }) => {
    const league = await createLeague(request, "Gone league");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    const name = await markBySearch(page, "a");
    await expect.poll(() => onTheClock(page)).toBe(2);

    await page.locator('input[type="search"]').fill(name);
    await page.waitForTimeout(300);
    const offered = await page.locator("main.board .panel .slot .name").allInnerTexts();
    expect(offered).not.toContain(name);
  });
});

test.describe("league shape drives the board", () => {
  test("superflex with six-point passing TDs values quarterbacks far higher", async ({
    page,
    request,
  }) => {
    // The clearest proof configurability is doing real work: in the reference
    // 14-team league the top quarterback is nowhere near the top of the board.
    const flexLeague = await createLeague(request, "Standard shape");
    const sfLeague = await createLeague(request, "Superflex shape", superflex());

    const topQbRank = async (id: string) => {
      const view = await (await request.get(`/api/leagues/${id}`)).json();
      const byVor = [...view.board].sort((a, b) => b.vor - a.vor);
      return byVor.findIndex((p) => p.pos === "QB") + 1;
    };

    const standardRank = await topQbRank(flexLeague.id);
    const superflexRank = await topQbRank(sfLeague.id);

    expect(superflexRank).toBeLessThan(standardRank);
    expect(superflexRank).toBe(1);

    await page.goto(`/league/${sfLeague.id}`);
    await waitForBoard(page);
    // And the anchoring caveat is shown rather than buried.
    await expect(page.getByText(/Passing touchdowns are worth 6/i)).toBeVisible();
  });

  test("the slot selector offers exactly one option per team", async ({ page, request }) => {
    const league = await createLeague(request, "Slots", superflex());
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    await expect(page.locator("#slot option")).toHaveCount(12);
  });
});
