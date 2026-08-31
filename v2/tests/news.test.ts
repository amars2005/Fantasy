/**
 * The news layer: pricing what ADP has not caught up to yet.
 *
 * ADP is a snapshot. When a headline lands the morning of the draft it breaks
 * the board in two independent places -- the player's value, and the market's
 * appetite for him -- and fixing only one is worse than fixing neither.
 *
 * The property guarded hardest here is that a tagged player *stays on the
 * board*. Filtering him out would take his picks away from the room in the
 * survival simulation, spend them on the players behind him, and quietly make
 * every one of them look scarcer than he is.
 */

import { describe, expect, it } from "vitest";

import { deriveBoard, DraftBoard } from "../lib/board";
import { applyNews, PRESETS, type NewsMap } from "../lib/news";
import type { Player } from "../lib/types";
import { loadBundle, loadFixture, loadManifest } from "./golden";

const fixture = loadFixture("reference_ppr_14");
const players = deriveBoard(loadBundle(fixture.format), fixture.league);
const nicknames = loadManifest().team_nicknames;

/** The board's top running back -- the shape of the Jacobs problem. */
const topRb = players.filter((p) => p.pos === "RB").sort((a, b) => a.adp - b.adp)[0];

const byId = (list: Player[], id: string) => list.find((p) => p.player_id === id)!;

describe("applying news to a board", () => {
  it("leaves an untagged board alone", () => {
    const out = applyNews(players, {});
    expect(out).toHaveLength(players.length);
    expect(out.map((p) => p.player_id)).toEqual(players.map((p) => p.player_id));
    expect(out.map((p) => p.proj_points)).toEqual(players.map((p) => p.proj_points));
  });

  it("discounts value in proportion to the player's own expected season", () => {
    // Half his expected games missed is half his expected points, because the
    // curve he is priced off is a full-season curve.
    const half = topRb.games / 2;
    const news: NewsMap = { [topRb.player_id]: { gamesMissed: half, adpShift: 0 } };

    const tagged = byId(applyNews(players, news), topRb.player_id);

    // Loose on the last decimal: rounding to one place is cosmetic here, and
    // halving an already-rounded number lands on a .05 boundary often enough
    // to make a tighter assertion flap for no reason.
    expect(tagged.proj_points).toBeCloseTo(topRb.proj_points / 2, 0);
    expect(tagged.games).toBeCloseTo(half, 0);
  });

  it("keeps the pre-news projection so the discount is visible", () => {
    const news: NewsMap = { [topRb.player_id]: { gamesMissed: 3, adpShift: 0 } };
    const tagged = byId(applyNews(players, news), topRb.player_id);

    expect(tagged.proj_before_news).toBe(topRb.proj_points);
    expect(tagged.proj_points).toBeLessThan(topRb.proj_points);
  });

  it("zeroes a do-not-draft player outright", () => {
    const news: NewsMap = { [topRb.player_id]: { gamesMissed: 0, adpShift: 0, avoid: true } };
    const tagged = byId(applyNews(players, news), topRb.player_id);

    expect(tagged.proj_points).toBe(0);
    expect(tagged.games).toBe(0);
  });

  it("floors at zero rather than paying a player negative points", () => {
    const news: NewsMap = { [topRb.player_id]: { gamesMissed: 99, adpShift: 0 } };
    const tagged = byId(applyNews(players, news), topRb.player_id);

    expect(tagged.proj_points).toBe(0);
    expect(tagged.games).toBe(0);
  });

  it("slides the latent draft mean later, because the room read the same headline", () => {
    const news: NewsMap = { [topRb.player_id]: { gamesMissed: 0, adpShift: 18 } };
    const tagged = byId(applyNews(players, news), topRb.player_id);

    expect(tagged.adp_mu).toBeCloseTo(topRb.adp_mu + 18, 6);
  });

  it("reports the market's own stale ADP unchanged, since the slide is our estimate", () => {
    const news: NewsMap = { [topRb.player_id]: { gamesMissed: 0, adpShift: 18 } };
    const tagged = byId(applyNews(players, news), topRb.player_id);

    expect(tagged.adp).toBe(topRb.adp);
  });

  it("falls back to a full season when the player has no fitted games", () => {
    const noGames: Player[] = [{ ...topRb, games: 0, games_raw: 0 }];
    const news: NewsMap = { [topRb.player_id]: { gamesMissed: 17, adpShift: 0 } };

    // Missing all 17 is a total write-off; without the fallback this divides
    // by zero and hands back NaN, which sorts unpredictably.
    expect(byId(applyNews(noGames, news), topRb.player_id).proj_points).toBe(0);
  });

  it("keeps the tagged player on the board", () => {
    const news: NewsMap = { [topRb.player_id]: { gamesMissed: 6, adpShift: 18, avoid: true } };
    const out = applyNews(players, news);

    expect(out).toHaveLength(players.length);
    expect(byId(out, topRb.player_id)).toBeDefined();
  });

  it("does not mutate the board it was given", () => {
    const before = topRb.proj_points;
    applyNews(players, { [topRb.player_id]: { gamesMissed: 8, adpShift: 20 } });

    expect(topRb.proj_points).toBe(before);
  });

  it("touches nothing but the tagged player", () => {
    // The whole design rests on this. Discounting one running back must not
    // move a single number on anyone else -- their survival odds shift later,
    // in the simulation, which is a different thing and deliberate.
    const news: NewsMap = { [topRb.player_id]: { gamesMissed: 6, adpShift: 18 } };
    const out = applyNews(players, news);

    const others = (list: Player[]) =>
      list
        .filter((p) => p.player_id !== topRb.player_id)
        .map((p) => `${p.player_id}:${p.proj_points}:${p.adp_mu}:${p.games}`);

    expect(others(out)).toEqual(others(players));
  });

  it("ignores a tag for a player who is not on the board", () => {
    const out = applyNews(players, { __nobody__: { gamesMissed: 6, adpShift: 12 } });
    expect(out.map((p) => p.proj_points)).toEqual(players.map((p) => p.proj_points));
  });
});

describe("presets", () => {
  it("orders severity: day-to-day costs less than a suspension, which costs less than out", () => {
    expect(PRESETS.dtd.gamesMissed).toBeLessThan(PRESETS.risk.gamesMissed);
    expect(PRESETS.risk.gamesMissed).toBeLessThan(PRESETS.out.gamesMissed);
  });

  it("slides further the worse the news is", () => {
    expect(PRESETS.dtd.adpShift).toBeLessThan(PRESETS.risk.adpShift);
    expect(PRESETS.risk.adpShift).toBeLessThan(PRESETS.out.adpShift);
  });

  it("marks only do-not-draft as avoid", () => {
    expect(PRESETS.avoid.avoid).toBe(true);
    expect(PRESETS.dtd.avoid).toBeFalsy();
    expect(PRESETS.risk.avoid).toBeFalsy();
    expect(PRESETS.out.avoid).toBeFalsy();
  });
});

describe("the effect on a live recommendation", () => {
  const newBoard = (list: Player[], slot = 3) =>
    new DraftBoard(list, fixture.league, slot, nicknames);

  it("drops a do-not-draft player out of the top recommendations", () => {
    const clean = newBoard(players).recommend(14, 400);
    expect(clean.recommendations.some((p) => p.player_id === topRb.player_id)).toBe(true);

    const news: NewsMap = { [topRb.player_id]: PRESETS.avoid };
    const tagged = newBoard(applyNews(players, news)).recommend(14, 400);

    expect(tagged.recommendations.some((p) => p.player_id === topRb.player_id)).toBe(false);
  });

  it("still lets you find and draft a tagged player, in case the room is wrong", () => {
    const news: NewsMap = { [topRb.player_id]: PRESETS.avoid };
    const board = newBoard(applyNews(players, news));

    const hits = board.find(topRb.name.split(" ")[1] ?? topRb.name, 20);
    expect(hits.some((p) => p.player_id === topRb.player_id)).toBe(true);
  });

  it("promotes a real alternative rather than leaving a hole", () => {
    const news: NewsMap = { [topRb.player_id]: PRESETS.avoid };
    const clean = newBoard(players).recommend(14, 400);
    const tagged = newBoard(applyNews(players, news)).recommend(14, 400);

    expect(tagged.recommendations).toHaveLength(clean.recommendations.length);
    expect(tagged.recommendations[0].player_id).not.toBe(topRb.player_id);
  });
});
