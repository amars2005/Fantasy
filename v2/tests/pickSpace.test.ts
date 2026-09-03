/**
 * ADP is a 12-team pick number; a league's snake usually is not.
 *
 * These need no bundle: the transform is pure arithmetic on a board, and the
 * property that matters -- that kickers and defences move while skill players
 * do not -- is checkable on a handful of rows.
 */

import { describe, expect, it } from "vitest";

import { ADP_TEAMS, toLeaguePickSpace } from "../lib/board";
import type { Player, Position } from "../lib/types";

const player = (pos: Position, adp: number): Player =>
  ({
    player_id: `${pos}-${adp}`,
    name: `${pos} ${adp}`,
    pos,
    tm: "SEA",
    adp,
    adp_mu: adp + 1,
    stdev: 10,
    bye: 8,
    pos_rank: 1,
    proj_points: 100,
    sd: 10,
    games: 16,
    source: "test",
  }) as Player;

// The shape of the real board: skill players throughout, a defence in the
// middle rounds and a kicker late. ADPs are the 2026-08-29 FFC snapshot.
const board = (): Player[] => [
  player("RB", 1.5),
  player("WR", 100),
  player("DST", 82.1),
  player("K", 127.5),
  player("TE", 183.8),
];

const byId = (players: Player[], id: string) => players.find((p) => p.player_id === id)!;

describe("toLeaguePickSpace", () => {
  it("leaves a 12-team league exactly alone", () => {
    const before = board();
    expect(toLeaguePickSpace(before, ADP_TEAMS)).toBe(before);
  });

  it("does not move skill players, whose rank does not depend on league size", () => {
    const after = toLeaguePickSpace(board(), 14);
    for (const id of ["RB-1.5", "WR-100", "TE-183.8"]) {
      expect(byId(after, id).adp).toBe(byId(board(), id).adp);
      expect(byId(after, id).adp_mu).toBe(byId(board(), id).adp_mu);
      expect(byId(after, id).stdev).toBe(byId(board(), id).stdev);
    }
  });

  it("scales kickers and defences by the team count", () => {
    const after = toLeaguePickSpace(board(), 14);
    // Round 7 of a 12-team draft is round 7 of a 14-team draft, two picks later
    // per round that has passed.
    expect(byId(after, "DST-82.1").adp).toBeCloseTo(82.1 * (14 / 12), 6);
    expect(byId(after, "K-127.5").adp).toBeCloseTo(127.5 * (14 / 12), 6);
    // The latent mean and the dispersion are in the same units.
    expect(byId(after, "DST-82.1").adp_mu).toBeCloseTo(83.1 * (14 / 12), 6);
    expect(byId(after, "DST-82.1").stdev).toBeCloseTo(10 * (14 / 12), 6);
  });

  it("moves them the other way in a small league", () => {
    const after = toLeaguePickSpace(board(), 8);
    expect(byId(after, "DST-82.1").adp).toBeCloseTo(82.1 * (8 / 12), 6);
    expect(byId(after, "DST-82.1").adp).toBeLessThan(82.1);
  });

  it("hands the board back in ADP order, which moving the kickers disturbs", () => {
    const after = toLeaguePickSpace(board(), 14);
    const adps = after.map((p) => p.adp);
    expect(adps).toEqual([...adps].sort((a, b) => a - b));
  });

  it("does not mutate the board it was given", () => {
    const before = board();
    toLeaguePickSpace(before, 14);
    expect(byId(before, "DST-82.1").adp).toBe(82.1);
  });

  it("is a no-op rather than a crash on a nonsense team count", () => {
    const before = board();
    expect(toLeaguePickSpace(before, 0)).toBe(before);
    expect(toLeaguePickSpace(before, Number.NaN)).toBe(before);
  });
});
