/**
 * Tiering: the thing that tells you when to reach.
 *
 * The parity of `addTiers` against Python is pinned by `pipeline.test.ts`
 * against the golden fixtures. What is checked here is the behaviour around it
 * that no fixture covers: what a tier means once a player has been re-priced by
 * hand, and that the numbering everyone reads off the board stays still while
 * that happens.
 */

import { describe, expect, it } from "vitest";

import { addTiers, playersLeftInTier, tierSummary, untieredPositions } from "../lib/tiers";
import { applyNews } from "../lib/news";
import { renderCheatsheet } from "../lib/cheatsheet";
import { REFERENCE_LEAGUE } from "../lib/config";
import { addVor } from "../lib/replacement";
import type { Player, Position } from "../lib/types";

const player = (pos: Position, name: string, points: number): Player =>
  ({
    player_id: `${pos}-${name}`,
    name,
    pos,
    tm: "SEA",
    adp: points > 0 ? 400 - points : 400,
    adp_mu: 400 - points,
    stdev: 10,
    bye: 8,
    pos_rank: 1,
    proj_points: points,
    proj_raw: points,
    sd: 10,
    games: 16,
    games_raw: 16,
    source: "test",
  }) as Player;

/**
 * Three positions: two with genuine plateaus, and a kicker position where the
 * fit separated everyone -- which is what really happens to kickers and
 * defences, and is the case the board must not print a tier for.
 */
const board = (): Player[] =>
  addTiers([
    player("WR", "wr-a", 300),
    player("WR", "wr-b", 300),
    player("WR", "wr-c", 280),
    player("WR", "wr-d", 280),
    player("WR", "wr-e", 280),
    player("WR", "wr-f", 150),
    player("RB", "rb-a", 290),
    player("RB", "rb-b", 210),
    player("RB", "rb-c", 210),
    player("K", "k-a", 130),
    player("K", "k-b", 120),
    player("K", "k-c", 110),
  ]);

const find = (players: Player[], name: string) => players.find((p) => p.name === name)!;

describe("addTiers", () => {
  it("pools players the fitted curve could not separate", () => {
    const b = board();
    expect(find(b, "wr-a").tier).toBe(1);
    expect(find(b, "wr-b").tier).toBe(1);
    expect(find(b, "wr-c").tier).toBe(2);
    expect(find(b, "wr-e").tier).toBe(2);
    expect(find(b, "wr-f").tier).toBe(3);
  });

  it("numbers tiers within a position, not across the board", () => {
    // A WR tier 1 and an RB tier 1 are not the same thing, and never compare.
    const b = board();
    expect(find(b, "rb-a").tier).toBe(1);
    expect(find(b, "rb-b").tier).toBe(2);
    expect(find(b, "rb-c").tier).toBe(2);
  });

  it("reports the size of a tier and the points it is worth", () => {
    const b = board();
    expect(find(b, "wr-c").tier_size).toBe(3);
    expect(find(b, "wr-c").tier_points).toBe(280);
    expect(find(b, "wr-a").tier_size).toBe(2);
  });

  it("counts who is left in a tier", () => {
    expect(playersLeftInTier(board(), "WR", 2)).toBe(3);
    expect(playersLeftInTier(board(), "WR", 3)).toBe(1);
  });

  it("does not mutate the board it was given", () => {
    const raw = [player("WR", "wr-a", 300)];
    addTiers(raw);
    expect(raw[0].tier).toBeUndefined();
  });

  it("summarises a tier by where it runs out", () => {
    const summary = tierSummary(board());
    const wr2 = summary.find((t) => t.pos === "WR" && t.tier === 2)!;
    expect(wr2.n).toBe(3);
    expect(wr2.points).toBe(280);
    expect(wr2.firstAdp).toBeLessThanOrEqual(wr2.lastAdp);
  });
});

describe("a tier after the player has been re-priced by hand", () => {
  const tagged = () =>
    applyNews(board(), { "WR-wr-a": { gamesMissed: 8, adpShift: 0 } });

  it("moves him down to the tier his new projection earns", () => {
    // Half a season off a 300-point receiver leaves him worth 150, which is
    // the bottom rung of this board -- not the top one he arrived on.
    const p = find(tagged(), "wr-a");
    expect(p.proj_points).toBe(150);
    expect(p.tier).toBe(3);
    expect(p.tier_points).toBe(150);
  });

  it("stops counting him as the last of the tier he has left", () => {
    // The costly version of the bug: "one left in tier 1, reach now" about a
    // player you had just written down.
    const after = tagged();
    expect(playersLeftInTier(after, "WR", 1)).toBe(1);
    expect(find(after, "wr-b").tier).toBe(1);
    expect(playersLeftInTier(after, "WR", 3)).toBe(2);
  });

  it("leaves every other player's tier number exactly where it was", () => {
    // Re-running the tiering on adjusted points would insert a new value into
    // the position and renumber everything below it, so tagging one receiver
    // would shift the tier printed against all the others.
    const before = board();
    const after = tagged();
    for (const p of after) {
      if (p.name === "wr-a") continue;
      expect(p.tier, `${p.name} tier`).toBe(find(before, p.name).tier);
    }
  });

  it("puts a do-not-draft player on the bottom rung", () => {
    const after = applyNews(board(), {
      "WR-wr-a": { gamesMissed: 0, adpShift: 0, avoid: true },
    });
    const p = find(after, "wr-a");
    expect(p.proj_points).toBe(0);
    expect(p.tier).toBe(3);
  });

  it("keeps tier sizes consistent with where players have ended up", () => {
    const after = tagged();
    expect(find(after, "wr-a").tier_size).toBe(2);
    expect(find(after, "wr-b").tier_size).toBe(1);
  });

  it("passes an untagged board straight through", () => {
    const b = board();
    expect(applyNews(b, {})).toBe(b);
  });
});


describe("the cheatsheet's tier rule", () => {
  // The sheet is ordered by value across every position, so "the next row is a
  // different tier" is true of almost every row -- it was marking 123 of the
  // 150 printed rows, which is a heavy rule that tells you nothing. The mark
  // has to say "last of his tier".
  const sheet = () =>
    renderCheatsheet({
      leagueName: "Tier sheet",
      league: REFERENCE_LEAGUE,
      players: addVor(board(), REFERENCE_LEAGUE),
      slot: 1,
      adpAsOf: "2026-08-30",
    });

  const marks = (html: string) =>
    [...html.matchAll(/<tr class="([^"]*)">[\s\S]*?<td>([^<]*)<\/td>/g)].map((m) => ({
      name: m[2],
      end: m[1].includes("tier-end"),
    }));

  it("marks the last player of a tier, and only him", () => {
    const rows = marks(sheet());
    const ended = rows.filter((r) => r.end).map((r) => r.name).sort();
    // One mark per tier: wr-b ends WR 1, wr-e ends WR 2, wr-f ends WR 3,
    // rb-a ends RB 1, rb-b ends RB 2.
    expect(ended).toEqual(["rb-a", "rb-c", "wr-b", "wr-e", "wr-f"]);
  });

  it("does not mark a player with tier-mates still below him", () => {
    const rows = marks(sheet());
    for (const name of ["wr-a", "wr-c", "wr-d", "rb-b"]) {
      expect(rows.find((r) => r.name === name)!.end, `${name} marked`).toBe(false);
    }
  });

  it("marks nothing at a position where every player is his own tier", () => {
    // Every kicker would otherwise carry the rule, which is not a tier break;
    // it is just the list continuing.
    const rows = marks(sheet());
    for (const name of ["k-a", "k-b", "k-c"]) {
      expect(rows.find((r) => r.name === name)!.end, `${name} marked`).toBe(false);
    }
  });

  it("prints no tier number for a position that has no tiers", () => {
    const html = sheet();
    expect(html).toContain(">WR1<");
    expect(html).not.toContain(">K1<");
    expect(html).toContain("&mdash;");
  });

  it("leaves a tier that runs past the printed cut unmarked", () => {
    // Three receivers share tier 2; print only as far as the first of them and
    // the tier has not been exhausted by anything on the page.
    const html = renderCheatsheet({
      leagueName: "Short sheet",
      league: REFERENCE_LEAGUE,
      players: addVor(board(), REFERENCE_LEAGUE),
      slot: 1,
      adpAsOf: "2026-08-30",
      limit: 4,
    });
    const rows = marks(html);
    expect(rows.length).toBe(4);
    expect(rows.find((r) => r.name === "wr-c")?.end).toBe(false);
  });
});


describe("untieredPositions", () => {
  it("names the positions where the fit pooled nobody", () => {
    // Kickers and defences are the real case: their curve is fit on outcome
    // rank, which separates all of them, so every one is a tier of one.
    expect([...untieredPositions(board())].sort()).toEqual(["K"]);
  });

  it("does not name a position with even one real plateau", () => {
    const untiered = untieredPositions(board());
    expect(untiered.has("WR")).toBe(false);
    expect(untiered.has("RB")).toBe(false);
  });
});
