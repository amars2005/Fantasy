/**
 * Live draft state: the clock, the roster, search, and the recommendation.
 *
 * The clock is the part worth guarding hardest. Every VONA number is built on
 * "how many picks until my next turn", so a counter that drifts from the room
 * quietly corrupts the entire board while still looking perfectly reasonable.
 */

import { describe, expect, it } from "vitest";

import { deriveBoard, DraftBoard, UNKNOWN_PREFIX } from "../lib/board";
import { renderCheatsheet } from "../lib/cheatsheet";
import { loadBundle, loadFixture, loadManifest } from "./golden";

const fixture = loadFixture("reference_ppr_14");
const players = deriveBoard(loadBundle(fixture.format), fixture.league);
const nicknames = loadManifest().team_nicknames;

const newBoard = (slot = 8) => new DraftBoard(players, fixture.league, slot, nicknames);

describe("the clock", () => {
  it("starts on pick one with nothing drafted", () => {
    const b = newBoard();
    expect(b.pickNumber).toBe(1);
    expect(b.nextPick()).toBe(8);
  });

  it("advances by one per pick, whoever made it", () => {
    const b = newBoard();
    b.draft(players[0].player_id, "other");
    b.draft(players[1].player_id, "me");
    expect(b.pickNumber).toBe(3);
  });

  it("counts an unidentified pick, so it cannot fall behind the room", () => {
    const b = newBoard();
    b.skip(3);
    expect(b.pickNumber).toBe(4);
    expect([...b.drafted.keys()].every((k) => k.startsWith(UNKNOWN_PREFIX))).toBe(true);
  });

  it("undoes a skip", () => {
    const b = newBoard();
    b.skip(2);
    b.undoSkip();
    expect(b.pickNumber).toBe(2);
  });

  it("reports the gap between my consecutive picks", () => {
    // Slot 8 of 14 picks at 8 and 21, so 13 players come off in between.
    const b = newBoard(8);
    expect(b.picksUntilNext()).toBe(13);
  });

  it("keeps my roster separate from the rest of the room", () => {
    const b = newBoard();
    b.draft(players[0].player_id, "other");
    b.draft(players[1].player_id, "me");
    b.draft(players[2].player_id, "me");
    expect(b.myRoster.length).toBe(2);
  });

  it("removes drafted players from what is available", () => {
    const b = newBoard();
    const before = b.available.length;
    b.draft(players[0].player_id, "other");
    expect(b.available.length).toBe(before - 1);
    expect(b.available.some((p) => p.player_id === players[0].player_id)).toBe(false);
  });
});

describe("search", () => {
  it("finds a player by name", () => {
    const b = newBoard();
    const target = players.find((p) => p.pos === "RB")!;
    const hits = b.find(target.name.split(" ")[0]);
    expect(hits.length).toBeGreaterThan(0);
  });

  it("finds a defence by its nickname", () => {
    // The room says "Ravens D/ST"; nobody types a team abbreviation.
    const b = newBoard();
    const anyDst = players.find((p) => p.pos === "DST");
    if (!anyDst) return;
    const nick = (nicknames[anyDst.tm] ?? "").split(" ")[0];
    if (!nick) return;
    expect(b.find(nick).some((p) => p.pos === "DST")).toBe(true);
  });

  it("treats a stray bracket as text, not a pattern", () => {
    // This threw mid-draft in the Python version until `literal=True` was set.
    const b = newBoard();
    expect(() => b.find("(")).not.toThrow();
    expect(b.find("(")).toEqual([]);
  });

  it("never returns a player who is already gone", () => {
    const b = newBoard();
    const target = players[0];
    b.draft(target.player_id, "other");
    expect(b.find(target.name).some((p) => p.player_id === target.player_id)).toBe(false);
  });
});

describe("recommendation", () => {
  it("ranks by VONA and reports the pick context", () => {
    const b = newBoard();
    const rec = b.recommend(10, 800);

    expect(rec.recommendations.length).toBe(10);
    expect(rec.on_the_clock).toBe(1);
    expect(rec.picks_until_next).toBeGreaterThan(0);

    const vonas = rec.recommendations.map((p) => p.vona ?? 0);
    expect([...vonas].sort((a, b2) => b2 - a)).toEqual(vonas);
  });

  it("declines a third quarterback without a hand-written rule", () => {
    const b = newBoard();
    // Take the two best quarterbacks.
    const qbs = players.filter((p) => p.pos === "QB").slice(0, 2);
    for (const qb of qbs) b.draft(qb.player_id, "me");

    const rec = b.recommend(20, 800);
    const qbDropoff = rec.dropoff.find((d) => d.pos === "QB")!;
    const bestOther = Math.max(
      ...rec.dropoff.filter((d) => d.pos !== "QB" && d.pos !== "K" && d.pos !== "DST")
        .map((d) => d.dropoff),
    );
    expect(qbDropoff.dropoff).toBeLessThan(bestOther);
  });

  it("counts how many players remain in each candidate's tier", () => {
    const b = newBoard();
    const rec = b.recommend(5, 400);
    for (const p of rec.recommendations) {
      const left = (p as { tier_left?: number }).tier_left ?? 0;
      expect(left).toBeGreaterThan(0);
    }
  });

  it("flags a bye collision with players already rostered", () => {
    const b = newBoard();
    const withBye = players.find((p) => p.bye != null)!;
    b.draft(withBye.player_id, "me");

    const rec = b.recommend(60, 400);
    const sameBye = rec.recommendations.find(
      (p) => p.bye === withBye.bye && p.player_id !== withBye.player_id,
    );
    if (sameBye) {
      expect((sameBye as { bye_conflicts?: number }).bye_conflicts).toBeGreaterThan(0);
    }
  });
});

describe("cheatsheet", () => {
  const html = renderCheatsheet({
    leagueName: "Test League",
    league: fixture.league,
    players,
    slot: 8,
    adpAsOf: "2026-08-30",
    warnings: ["Anchored to the PPR board."],
  });

  it("is a complete standalone document", () => {
    expect(html.startsWith("<!doctype html>")).toBe(true);
    expect(html).toContain("</html>");
  });

  it("carries no scripts or network references, so it works offline", () => {
    expect(html).not.toMatch(/<script/i);
    expect(html).not.toMatch(/https?:\/\//);
  });

  it("lists players in value order with their tiers", () => {
    const best = [...players].sort((a, b) => (b.vor ?? 0) - (a.vor ?? 0))[0];
    expect(html).toContain(best.name);
    expect(html).toContain("Replacement level");
    expect(html).toContain("Your picks");
  });

  it("escapes names rather than injecting them raw", () => {
    const nasty = renderCheatsheet({
      leagueName: '<img src=x onerror="alert(1)">',
      league: fixture.league,
      players: players.slice(0, 3),
      slot: 1,
      adpAsOf: null,
    });
    expect(nasty).not.toContain("<img src=x");
    expect(nasty).toContain("&lt;img");
  });
});
