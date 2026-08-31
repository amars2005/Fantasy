/**
 * Pricing in news at the table.
 *
 * The headline lands twenty minutes before the draft and the board is still
 * anchored to yesterday's ADP. These cover the whole path a person actually
 * walks: find him, say what you think, watch the ranking move, and still have
 * the tag when the tab is reloaded three rounds later.
 */

import { expect, test } from "@playwright/test";

import { createLeague, recommendedNames, waitForBoard } from "./helpers";

/** Open the news editor from the top row of the recommendation table. */
async function editTopRecommendation(page: import("@playwright/test").Page) {
  const top = (await recommendedNames(page))[0];
  await page
    .locator("main.board table tbody tr")
    .first()
    .getByRole("button", { name: "News" })
    .click();
  await expect(page.locator(".newsedit")).toBeVisible();
  return top;
}

test.describe("news and risk", () => {
  test("a do-not-draft tag drops a player off the recommendations", async ({
    page,
    request,
  }) => {
    const league = await createLeague(request, "News league");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    const top = await editTopRecommendation(page);
    await page.locator(".presets button", { hasText: "Do not draft" }).click();

    await expect.poll(async () => (await recommendedNames(page))[0]).not.toContain(top);
  });

  test("a discounted player is re-priced, not hidden", async ({ page, request }) => {
    const league = await createLeague(request, "Discount league");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    const top = await editTopRecommendation(page);
    await page.locator(".presets button", { hasText: "Suspension risk" }).click();

    // The badge says what came off, and the readout says where it went.
    await expect(page.locator(".newsedit .newsbadge")).toBeVisible();
    await expect(page.locator(".newsread")).toContainText("off his projection");

    // He leaves the recommendations but not the board: still findable, still
    // draftable, in case the room overreacts and he falls to you cheap.
    await page.locator('input[type="search"]').fill(top.split(" ").pop() ?? top);
    await expect(page.getByText(top, { exact: false }).first()).toBeVisible();
  });

  test("a bigger games-missed figure takes more off the projection", async ({
    page,
    request,
  }) => {
    const league = await createLeague(request, "Custom league");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    await editTopRecommendation(page);
    const badge = page.locator(".newsedit .newsbadge");
    const percent = async () => Number((await badge.innerText()).replace(/[-%]/g, ""));

    await page.locator("#gamesMissed").fill("2");
    await expect(badge).toBeVisible();
    const small = await percent();

    await page.locator("#gamesMissed").fill("8");
    await expect.poll(percent).toBeGreaterThan(small);
    expect(small).toBeGreaterThan(0);
  });

  test("tags survive a reload, because the draft outlasts the tab", async ({
    page,
    request,
  }) => {
    const league = await createLeague(request, "Persist league");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    await editTopRecommendation(page);
    await page.locator(".presets button", { hasText: "Do not draft" }).click();
    await expect(page.locator(".newsbadge.avoid").first()).toBeVisible();

    await page.reload();
    await waitForBoard(page);

    await expect(page.locator(".newsbadge.avoid").first()).toBeVisible();
  });

  test("clearing a tag puts the player back where the market had him", async ({
    page,
    request,
  }) => {
    const league = await createLeague(request, "Clear league");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);

    const top = await editTopRecommendation(page);
    await page.locator(".presets button", { hasText: "Do not draft" }).click();
    await expect.poll(async () => (await recommendedNames(page))[0]).not.toContain(top);

    await page.getByRole("button", { name: "Clear this player" }).click();

    await expect.poll(async () => (await recommendedNames(page))[0]).toContain(top);
  });
});
