/**
 * Lineup optimisation and replacement level.
 *
 * The lineup maths is what makes the tool decline a third quarterback without a
 * hand-written rule, so it is worth testing on its own as well as against the
 * fixtures.
 */

import { describe, expect, it } from "vitest";

import { REFERENCE_LEAGUE } from "../lib/config";
import { optimalLineupPoints, BENCH_WEIGHT } from "../lib/lineup";
import { allocateFlex } from "../lib/replacement";
import type { LeagueConfig, Position, RosterSlot } from "../lib/types";
import { FIXTURE_NAMES, loadFixture } from "./golden";

const TOL = 1e-9;

const roster = (specs: [Position, number][]): RosterSlot[] =>
  specs.map(([pos, proj_points]) => ({ pos, proj_points }));

describe.each(FIXTURE_NAMES)("lineup points match Python: %s", (name) => {
  const fixture = loadFixture(name);

  it("scores every fixture roster identically", () => {
    for (const [i, sample] of fixture.lineups.entries()) {
      const got = optimalLineupPoints(sample.roster as RosterSlot[], fixture.league);
      expect(Math.abs(got - sample.points), `roster ${i}`).toBeLessThan(TOL);
    }
  });
});

describe("lineup structure", () => {
  it("values an empty roster at zero", () => {
    expect(optimalLineupPoints([], REFERENCE_LEAGUE)).toBe(0);
  });

  it("prices a second quarterback below a startable third receiver", () => {
    // The property `src/draft/vona.py` is built to produce: with QB1 already
    // rostered, another quarterback only reaches the bench.
    const base = roster([["QB", 300], ["RB", 200], ["RB", 190], ["WR", 180], ["WR", 170]]);
    const withQb2 = optimalLineupPoints([...base, { pos: "QB", proj_points: 290 }], REFERENCE_LEAGUE);
    const withWr3 = optimalLineupPoints([...base, { pos: "WR", proj_points: 150 }], REFERENCE_LEAGUE);
    expect(withWr3).toBeGreaterThan(withQb2);
  });

  it("counts a surplus quarterback as bench, not zero", () => {
    const one = optimalLineupPoints(roster([["QB", 300]]), REFERENCE_LEAGUE);
    const two = optimalLineupPoints(roster([["QB", 300], ["QB", 200]]), REFERENCE_LEAGUE);
    expect(two - one).toBeCloseTo(BENCH_WEIGHT * 200, 9);
  });

  it("sends the best flex-eligible leftover to the FLEX slot", () => {
    // RB3 at 195 should start over WR3 at 100.
    const r = roster([
      ["RB", 220], ["RB", 210], ["RB", 195],
      ["WR", 200], ["WR", 190], ["WR", 100],
    ]);
    const total = optimalLineupPoints(r, REFERENCE_LEAGUE);
    // Starters: RB 220+210, WR 200+190, FLEX 195. Bench: 100.
    expect(total).toBeCloseTo(220 + 210 + 200 + 190 + 195 + BENCH_WEIGHT * 100, 9);
  });

  it("lets a quarterback fill SUPERFLEX but never plain FLEX", () => {
    const superflex: LeagueConfig = {
      ...REFERENCE_LEAGUE,
      starters: { QB: 1, RB: 1, WR: 1, FLEX: 1, SUPERFLEX: 1 },
    };
    const r = roster([["QB", 300], ["QB", 280], ["RB", 200], ["WR", 190], ["WR", 120]]);
    const total = optimalLineupPoints(r, superflex);
    // QB 300, RB 200, WR 190, FLEX takes WR 120, SUPERFLEX takes QB 280.
    expect(total).toBeCloseTo(300 + 200 + 190 + 120 + 280, 9);

    const flexOnly: LeagueConfig = {
      ...REFERENCE_LEAGUE,
      starters: { QB: 1, RB: 1, WR: 1, FLEX: 1 },
    };
    // Same roster, no superflex: QB2 is bench, worth only its discount.
    expect(optimalLineupPoints(r, flexOnly)).toBeCloseTo(
      300 + 200 + 190 + 120 + BENCH_WEIGHT * 280,
      9,
    );
  });
});

describe("flex allocation", () => {
  const player = (pos: Position, pts: number, i: number) => ({
    player_id: `${pos}${i}`,
    name: `${pos}${i}`,
    pos,
    tm: "XX",
    adp: i + 1,
    stdev: 1,
    bye: null,
    pos_rank: i + 1,
    proj_points: pts,
    sd: 1,
    games: 17,
    source: "test",
    adp_mu: i + 1,
  });

  it("gives flex slots to whichever position actually earns them", () => {
    // Receivers strictly better than running backs past the base starters, so
    // every flex slot should go to WR.
    const players = [
      ...Array.from({ length: 40 }, (_, i) => player("WR", 200 - i, i)),
      ...Array.from({ length: 40 }, (_, i) => player("RB", 150 - i, i)),
    ];
    const league: LeagueConfig = {
      ...REFERENCE_LEAGUE,
      teams: 10,
      starters: { QB: 1, RB: 2, WR: 2, TE: 1, FLEX: 1 },
    };
    const counts = allocateFlex(players, league);
    expect(counts.RB).toBe(20); // 2 * 10, no flex
    expect(counts.WR).toBe(30); // 2 * 10 base, plus all 10 flex slots
  });
});
