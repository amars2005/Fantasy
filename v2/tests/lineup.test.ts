/**
 * Lineup optimisation and replacement level.
 *
 * The lineup maths is what makes the tool decline a third quarterback without a
 * hand-written rule, so it is worth testing on its own as well as against the
 * fixtures.
 */

import { describe, expect, it } from "vitest";

import { REFERENCE_LEAGUE } from "../lib/config";
import {
  BENCH_WEIGHT,
  benchModel,
  optimalLineupPoints,
  promotionWeights,
} from "../lib/lineup";
import { allocateFlex } from "../lib/replacement";
import type { LeagueConfig, Player, Position, RosterSlot } from "../lib/types";
import { FIXTURE_NAMES, loadFixture } from "./golden";

const TOL = 1e-9;

const roster = (specs: [Position, number][]): RosterSlot[] =>
  specs.map(([pos, proj_points]) => ({ pos, proj_points }));

describe.each(FIXTURE_NAMES)("lineup points match Python: %s", (name) => {
  const fixture = loadFixture(name);
  const board = fixture.projections as unknown as Player[];

  it("scores every fixture roster identically", () => {
    for (const [i, sample] of fixture.lineups.entries()) {
      const got = optimalLineupPoints(sample.roster as RosterSlot[], fixture.league);
      expect(Math.abs(got - sample.points), `roster ${i}`).toBeLessThan(TOL);
    }
  });

  it("derives the same bench model", () => {
    const got = benchModel(board, fixture.league);
    expect(Object.keys(got).sort()).toEqual(Object.keys(fixture.bench).sort());
    for (const [pos, value] of Object.entries(fixture.bench)) {
      expect(Math.abs(got[pos].vacancies - value.vacancies), pos).toBeLessThan(TOL);
      expect(Math.abs(got[pos].baseline - value.baseline), pos).toBeLessThan(TOL);
    }
  });

  it("scores every fixture roster identically with the bench model", () => {
    const bench = benchModel(board, fixture.league);
    for (const [i, sample] of fixture.lineups.entries()) {
      const got = optimalLineupPoints(sample.roster as RosterSlot[], fixture.league, bench);
      expect(Math.abs(got - sample.bench_points), `roster ${i}`).toBeLessThan(TOL);
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

describe("bench value", () => {
  // A pool with a realistic shape: quarterbacks score most and replace
  // cheaply, receivers score least and replace expensively. Pricing the bench
  // in raw points is what used to invert that.
  const pool: Player[] = [];
  const add = (pos: Position, n: number, top: number, step: number) => {
    for (let i = 0; i < n; i++) {
      pool.push({
        player_id: `${pos}${i}`,
        name: `${pos}${i}`,
        pos,
        tm: "XX",
        adp: i + 1,
        adp_mu: i + 1,
        stdev: 1,
        bye: null,
        pos_rank: i + 1,
        proj_points: top - i * step,
        sd: 1,
        games: 17,
        source: "test",
      });
    }
  };
  add("QB", 40, 320, 2);
  add("RB", 80, 260, 3);
  add("WR", 90, 250, 2.5);
  add("TE", 30, 220, 4);
  const league: LeagueConfig = { ...REFERENCE_LEAGUE, starters: { QB: 1, RB: 2, WR: 2, TE: 1, FLEX: 1 } };
  const bench = benchModel(pool, league);

  it("splits a position's cover across the backups who could fill it", () => {
    // The weights are P(at least k slots open), so they sum back to the
    // expected number of open slots -- the cover is divided, not duplicated.
    for (const vacancies of [0.2, 0.48, 0.6]) {
      const weights = promotionWeights(vacancies, 40);
      expect(weights.reduce((a, b) => a + b, 0)).toBeCloseTo(vacancies, 6);
      // Each further backup is worth strictly less than the one ahead of him,
      // over the depth anyone actually rosters; past that the tail is float
      // dust and only has to stay non-increasing.
      for (let i = 1; i < weights.length; i++) {
        if (i < 6) expect(weights[i]).toBeLessThan(weights[i - 1]);
        else expect(weights[i]).toBeLessThanOrEqual(weights[i - 1]);
      }
    }
  });

  // Every starting slot filled, so anything added lands on the bench.
  const started = roster([
    ["QB", 320], ["RB", 260], ["RB", 257], ["WR", 250], ["WR", 247], ["TE", 220], ["RB", 245],
  ]);

  it("prices a backup over what you could stream, not over zero", () => {
    const before = optimalLineupPoints(started, league, bench);
    const qb2 = 300;
    const after = optimalLineupPoints([...started, { pos: "QB", proj_points: qb2 }], league, bench);
    const weight = promotionWeights(bench.QB.vacancies, 1)[0];
    expect(after - before).toBeCloseTo(weight * (qb2 - bench.QB.baseline), 9);
    // ...which is a small fraction of what a flat share of raw points paid him.
    expect(after - before).toBeLessThan(BENCH_WEIGHT * qb2 * 0.5);
  });

  it("makes a bench receiver worth more than a replaceable quarterback", () => {
    // The complaint this exists to answer: a QB2 arriving in the middle rounds
    // beat every bench skill player, because quarterbacks simply score more.
    const value = (pos: Position, pts: number, model = bench) =>
      optimalLineupPoints([...started, { pos, proj_points: pts }], league, model) -
      optimalLineupPoints(started, league, model);

    // Both beat their own replacement level by the same margin, so all that
    // separates them is how many starting slots they stand behind: one for the
    // quarterback, two and the flex for the receiver.
    const edge = 20;
    const qb2 = bench.QB.baseline + edge;
    const wr = bench.WR.baseline + edge;
    expect(value("WR", wr)).toBeGreaterThan(value("QB", qb2));

    // Under the flat weight the ordering was the other way round, on nothing
    // but the quarterback's bigger raw total.
    const flat = (pos: Position, pts: number) =>
      optimalLineupPoints([...started, { pos, proj_points: pts }], league) -
      optimalLineupPoints(started, league);
    expect(flat("QB", qb2)).toBeGreaterThan(flat("WR", wr));
  });

  it("prices a third quarterback at almost nothing", () => {
    const two = [...started, { pos: "QB" as Position, proj_points: 300 }];
    const value = (pos: Position, pts: number) =>
      optimalLineupPoints([...two, { pos, proj_points: pts }], league, bench) -
      optimalLineupPoints(two, league, bench);
    expect(value("QB", 290)).toBeLessThan(1);
    expect(value("WR", 200)).toBeGreaterThan(10 * value("QB", 290));
  });

  it("is zero for a bench player who cannot beat the waiver wire", () => {
    const before = optimalLineupPoints(started, league, bench);
    const belowReplacement = bench.WR.baseline - 20;
    expect(
      optimalLineupPoints(
        [...started, { pos: "WR", proj_points: belowReplacement }],
        league,
        bench,
      ),
    ).toBeCloseTo(before, 9);
  });

  it("doubles a backup quarterback's cover in superflex", () => {
    const superflex: LeagueConfig = {
      ...REFERENCE_LEAGUE,
      starters: { QB: 1, RB: 2, WR: 2, TE: 1, FLEX: 1, SUPERFLEX: 1 },
    };
    const sf = benchModel(pool, superflex);
    expect(sf.QB.vacancies).toBeCloseTo(2 * bench.QB.vacancies, 9);
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
