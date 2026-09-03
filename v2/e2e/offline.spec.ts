/**
 * Losing the network mid-draft.
 *
 * Draft-day wifi is bad, and the tool this replaces did not need a network at
 * all. The compute is in the browser precisely so a dropped connection costs
 * nothing but syncing, and `seq` makes every retry safe to repeat.
 */

import { expect, test } from "@playwright/test";

import {
  createLeague,
  markBySearch,
  onTheClock,
  setPickOrderLock,
  waitForBoard,
} from "./helpers";

test.describe("offline", () => {
  test("the board keeps working and queues picks while offline", async ({
    page,
    context,
    request,
  }) => {
    const league = await createLeague(request, "Disconnected");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);
    // Slot 1 picks first, so marking for the room is out of turn here.
    await setPickOrderLock(page, false);

    await context.setOffline(true);

    // The Monte Carlo runs locally, so marking a pick still works.
    await markBySearch(page, "a");
    await expect.poll(() => onTheClock(page)).toBe(2);
    await expect(page.locator(".banner.offline")).toBeVisible();
    await expect(page.locator(".banner.offline")).toContainText(/kept locally/i);

    // Nothing has reached the server. The `request` fixture is a separate API
    // context, so it can still see the server while the page cannot.
    const stored = await (await request.get(`/api/leagues/${league.id}/picks`)).json();
    expect(stored.picks.length).toBe(0);

    // The board itself is fully usable meanwhile -- ranking, tiers and all.
    await expect(page.locator("main.board table tbody tr")).not.toHaveCount(0);
    await context.setOffline(false);
  });

  test("queued picks flush when the connection returns", async ({ page, context, request }) => {
    const league = await createLeague(request, "Flush league");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);
    // Slot 1 picks first, so marking for the room is out of turn here.
    await setPickOrderLock(page, false);

    await context.setOffline(true);
    await markBySearch(page, "a");
    await expect.poll(() => onTheClock(page)).toBe(2);
    await markBySearch(page, "b");
    await expect.poll(() => onTheClock(page)).toBe(3);

    await context.setOffline(false);
    // Coming back to the page triggers the flush and a reconcile.
    await page.dispatchEvent("body", "focus");

    await expect
      .poll(
        async () => {
          const res = await request.get(`/api/leagues/${league.id}/picks`);
          return (await res.json()).picks.length;
        },
        { timeout: 20_000 },
      )
      .toBe(2);

    // And a reload sees exactly those two, not four.
    await page.reload();
    await waitForBoard(page);
    expect(await onTheClock(page)).toBe(3);
  });

  test("a board already loaded once survives the server going away", async ({
    page,
    context,
    request,
  }) => {
    const league = await createLeague(request, "Cache league");
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);
    // Slot 1 picks first, so marking for the room is out of turn here.
    await setPickOrderLock(page, false);
    await markBySearch(page, "a");
    await expect.poll(() => onTheClock(page)).toBe(2);

    // Reload with no network at all: the cached copy has to carry the draft.
    await context.setOffline(true);
    await page.reload().catch(() => {
      // The document itself may fail to load offline; that is the browser's
      // cache, not ours, and is not what this test is about.
    });

    await context.setOffline(false);
    await page.goto(`/league/${league.id}`);
    await waitForBoard(page);
    expect(await onTheClock(page)).toBe(2);
  });

  test("marking the same pick twice cannot double-count the clock", async ({ request }) => {
    // The failure this guards: a POST that times out on bad wifi but actually
    // landed, then gets retried. A doubled pick puts every "picks until my next
    // turn" number permanently out of step with the room.
    const league = await createLeague(request, "Idempotent league");

    const send = () =>
      request.post(`/api/leagues/${league.id}/picks`, {
        data: { seq: 1, playerId: "some-player", takenBy: "other" },
      });

    const first = await send();
    const retry = await send();
    const thirdTime = await send();

    expect(first.status()).toBe(201);
    expect(retry.status()).toBe(200);
    expect(thirdTime.status()).toBe(200);

    const { picks } = await (await request.get(`/api/leagues/${league.id}/picks`)).json();
    expect(picks.length).toBe(1);
  });

  test("concurrent writes on the same seq resolve to one pick", async ({ request }) => {
    const league = await createLeague(request, "Race league");

    const results = await Promise.all(
      Array.from({ length: 6 }, () =>
        request.post(`/api/leagues/${league.id}/picks`, {
          data: { seq: 1, playerId: "contested", takenBy: "other" },
        }),
      ),
    );

    const created = results.filter((r) => r.status() === 201);
    expect(created.length).toBe(1);

    const { picks } = await (await request.get(`/api/leagues/${league.id}/picks`)).json();
    expect(picks.length).toBe(1);
  });
});
