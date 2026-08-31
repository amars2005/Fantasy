/**
 * The whole derivation, against Python's: projections, replacement, VOR, tiers.
 *
 * This is the test that would catch a port that looks right and is not.
 *
 * One nuance runs through it. Projections are rounded to one decimal, and a
 * value like 84.85 is an exact tie, so a 1e-14 difference in the order two
 * implementations sum floats flips it by 0.1 -- and that flip propagates into
 * VOR and tier boundaries. Neither answer is more correct, and matching
 * sklearn's Cython accumulation bit-for-bit is not realistically achievable.
 *
 * So the *unrounded* values are asserted to 1e-9 for every player with no
 * exemptions -- that is the real invariant -- and the rounded values are
 * asserted exactly for everyone not sitting on a tie. Ties are tracked per
 * quantity rather than in one blended set, because they behave differently:
 * projected points tie rarely and matter (VOR, tiers), games played ties
 * constantly and matters to nobody (it is fitted on small integers, so pooled
 * means land on a quarter), and spread never ties at all. Each gets the cap it
 * deserves, so a systematic rounding bug still fails loudly.
 */

import { describe, expect, it } from "vitest";

import { deriveBoard } from "../lib/board";
import {
  byPlayer,
  FIXTURE_NAMES,
  loadBundle,
  loadFixture,
  tieBoundaryPlayers,
} from "./golden";

const TOL = 1e-9;

describe.each(FIXTURE_NAMES)("derived board matches Python: %s", (name) => {
  const fixture = loadFixture(name);
  const players = deriveBoard(loadBundle(fixture.format), fixture.league);
  const got = byPlayer(players);
  const ties = tieBoundaryPlayers(fixture);

  it("produces the same players", () => {
    expect(players.length).toBe(fixture.projections.length);
    for (const row of fixture.projections) {
      expect(got.has(row.player_id), `missing ${row.player_id}`).toBe(true);
    }
  });

  it("matches unrounded projections for every player", () => {
    // The real invariant. No exemptions here.
    for (const row of fixture.projections_raw) {
      const p = got.get(row.player_id)!;
      expect(Math.abs(p.proj_raw! - row.proj_raw), `${p.name} points`).toBeLessThan(TOL);
      expect(Math.abs(p.sd_raw! - row.sd_raw), `${p.name} sd`).toBeLessThan(TOL);
      expect(Math.abs(p.games_raw! - row.games_raw), `${p.name} games`).toBeLessThan(TOL);
    }
  });

  it("ties on projected points only rarely", () => {
    // Projection ties are the consequential ones: they move VOR and tier
    // boundaries. If this grows, the exemptions below have stopped being narrow
    // and the comparison has stopped meaning much.
    expect(ties.points.size).toBeLessThan(Math.max(2, players.length * 0.05));
  });

  it("never ties on spread", () => {
    // Measured, not assumed: the spread curve is a pooled standard deviation
    // and does not land on exact tenths.
    expect(ties.sd.size).toBe(0);
  });

  it("matches rounded points, spread and games", () => {
    for (const row of fixture.projections) {
      const p = got.get(row.player_id)!;
      if (!ties.points.has(row.player_id)) {
        expect(Math.abs(p.proj_points - row.proj_points), `${p.name} points`).toBeLessThan(TOL);
      }
      expect(Math.abs(p.sd - row.sd), `${p.name} sd`).toBeLessThan(TOL);
      if (!ties.games.has(row.player_id)) {
        expect(Math.abs(p.games - row.games), `${p.name} games`).toBeLessThan(TOL);
      }
    }
  });

  it("keeps every tie within one rounding step", () => {
    for (const row of fixture.projections) {
      const p = got.get(row.player_id)!;
      expect(
        Math.abs(p.proj_points - row.proj_points), `${p.name} points`,
      ).toBeLessThanOrEqual(0.1 + TOL);
      expect(Math.abs(p.games - row.games), `${p.name} games`).toBeLessThanOrEqual(0.1 + TOL);
    }
  });

  it("matches replacement levels", () => {
    const derived = new Map<string, number>(players.map((p) => [p.pos, p.replacement!]));
    for (const [pos, level] of Object.entries(fixture.replacement)) {
      expect(Math.abs(derived.get(pos)! - level), pos).toBeLessThan(TOL);
    }
  });

  it("matches value over replacement", () => {
    for (const row of fixture.vor) {
      if (ties.points.has(row.player_id)) continue;
      const p = got.get(row.player_id)!;
      expect(Math.abs(p.vor! - row.vor), `${p.name} vor`).toBeLessThan(TOL);
    }
  });

  it("orders the board the same way by VOR", () => {
    // Players tied on VOR have no meaningful order -- Python breaks those ties
    // by frame order and we break them by player id. So assert the two things
    // that do matter: the sequence of VOR values, and that each tie group holds
    // the same players.
    const mine = [...players]
      .filter((p) => !ties.points.has(p.player_id))
      .sort((a, b) => a.vor_rank! - b.vor_rank!);
    const theirs = [...fixture.vor]
      .filter((r) => !ties.points.has(r.player_id))
      .sort((a, b) => a.vor_rank - b.vor_rank);

    mine.forEach((p, i) => {
      expect(Math.abs(p.vor! - theirs[i].vor), `position ${i} (${p.name})`).toBeLessThan(TOL);
    });

    const groupBy = (rows: { vor: number; player_id: string }[]) => {
      const groups = new Map<string, Set<string>>();
      for (const r of rows) {
        const key = r.vor.toFixed(9);
        if (!groups.has(key)) groups.set(key, new Set());
        groups.get(key)!.add(r.player_id);
      }
      return groups;
    };

    const mineGroups = groupBy(mine.map((p) => ({ vor: p.vor!, player_id: p.player_id })));
    const theirGroups = groupBy(theirs);
    expect([...mineGroups.keys()].sort()).toEqual([...theirGroups.keys()].sort());
    for (const [key, ids] of mineGroups) {
      expect([...ids].sort(), `tie group ${key}`).toEqual([...theirGroups.get(key)!].sort());
    }
  });

  it("assigns the same tiers", () => {
    for (const row of fixture.tiers) {
      if (ties.points.has(row.player_id)) continue;
      const p = got.get(row.player_id)!;
      expect(p.tier, `${p.name} tier`).toBe(row.tier);
      expect(Math.abs(p.tier_points! - row.tier_points), `${p.name} tier points`).toBeLessThan(
        TOL,
      );
    }
  });

  it("produces the same number of tiers per position", () => {
    // The check that fails loudly if plateaus shatter: a broken fit gives one
    // tier per player while every value still looks perfectly plausible.
    const counts: Record<string, Set<number>> = {};
    for (const p of players) {
      (counts[p.pos] ??= new Set()).add(p.tier!);
    }
    for (const [pos, expected] of Object.entries(fixture.tier_counts)) {
      // A tie flip can merge or split exactly one tier, so allow one.
      expect(Math.abs(counts[pos].size - expected), `${pos} tier count`).toBeLessThanOrEqual(1);
    }
  });

  it("does not degenerate into one tier per skill player", () => {
    // A standing guard independent of the fixture.
    //
    // Skill positions only, deliberately. K and DST are fitted on finish rank
    // over ten seasons of ~32 teams, which is enough data to separate nearly
    // every rank -- so one tier per defence is the real answer there, not a
    // symptom. Python produces the same and the fixture agrees.
    for (const pos of ["QB", "RB", "WR", "TE"]) {
      const atPos = players.filter((p) => p.pos === pos);
      if (atPos.length < 10) continue;
      const tiers = new Set(atPos.map((p) => p.tier));
      expect(tiers.size, `${pos} has a tier per player`).toBeLessThan(atPos.length);
    }
  });
});
