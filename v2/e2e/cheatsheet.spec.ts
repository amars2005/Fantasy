/**
 * The offline escape hatch.
 *
 * Hosting this trades away the one property the local tool had: it worked with
 * no network and nothing installed. The cheatsheet buys part of that back, so
 * it has to be genuinely self-contained -- no scripts, no fetches, no fonts.
 */

import { expect, test } from "@playwright/test";

import { createLeague, markBySearch, onTheClock, superflex, waitForBoard } from "./helpers";

test.describe("cheatsheet", () => {
  test("downloads as a file from the board", async ({ page, request }) => {
    const league = await createLeague(request, "Cheatsheet league");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: "Cheatsheet" }).click(),
    ]);

    expect(download.suggestedFilename()).toMatch(/cheatsheet\.html$/);
  });

  test("is a complete standalone document with no network dependencies", async ({ request }) => {
    const league = await createLeague(request, "Standalone");
    const res = await request.get(`/api/leagues/${league.id}/cheatsheet?slot=8`);

    expect(res.status()).toBe(200);
    expect(res.headers()["content-type"]).toContain("text/html");
    expect(res.headers()["content-disposition"]).toContain("attachment");

    const html = await res.text();
    expect(html.startsWith("<!doctype html>")).toBe(true);
    expect(html).toContain("</html>");
    // The whole point: nothing to fetch, nothing to execute.
    expect(html).not.toMatch(/<script/i);
    expect(html).not.toMatch(/https?:\/\//);
    expect(html).not.toMatch(/<link[^>]+href/i);
  });

  test("renders in a browser with the network cut off", async ({ browser, request }) => {
    const league = await createLeague(request, "Airgapped");
    const html = await (
      await request.get(`/api/leagues/${league.id}/cheatsheet?slot=3`)
    ).text();

    const context = await browser.newContext();
    const page = await context.newPage();
    // Fail every network request, then load the file from memory.
    await context.route("**/*", (route) => route.abort());
    await page.setContent(html, { waitUntil: "domcontentloaded" });

    await expect(page.locator("h1")).toContainText("Airgapped");
    await expect(page.locator("table tbody tr").first()).toBeVisible();
    await expect(page.getByText("Replacement level")).toBeVisible();
    await expect(page.getByText("Your picks")).toBeVisible();

    await context.close();
  });

  test("shows the picks for the slot that was asked for", async ({ browser, request }) => {
    const league = await createLeague(request, "Slot five");
    const html = await (
      await request.get(`/api/leagues/${league.id}/cheatsheet?slot=5`)
    ).text();

    const context = await browser.newContext();
    const page = await context.newPage();
    await page.setContent(html);

    // Slot 5 of 14 picks 5th, then 24th on the way back.
    await expect(page.getByText("pick 5", { exact: false }).first()).toBeVisible();
    await expect(page.getByText("pick 24", { exact: false }).first()).toBeVisible();

    await context.close();
  });

  test("carries the anchoring caveat into the printed copy", async ({ request }) => {
    // A printed sheet outlives the tab that warned you, so the warning goes
    // with it.
    const league = await createLeague(request, "Caveat league", superflex());
    const html = await (
      await request.get(`/api/leagues/${league.id}/cheatsheet?slot=1`)
    ).text();

    expect(html).toMatch(/Read first/i);
    expect(html).toMatch(/Passing touchdowns are worth 6/i);
  });

  test("reflects the draft as it stands, not as it started", async ({ page, request }) => {
    const league = await createLeague(request, "Live sheet");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    const taken = await markBySearch(page, "a");
    await expect.poll(() => onTheClock(page)).toBe(2);

    const html = await (
      await request.get(`/api/leagues/${league.id}/cheatsheet?slot=1`)
    ).text();

    // The board is the league's full board; the sheet is a planning artefact
    // rather than a live mirror, so a drafted player is still listed. Pin the
    // behaviour so a change to it is deliberate.
    expect(html).toContain(taken);
  });
});
