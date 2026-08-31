/**
 * Draft simulation: snake order, quickselect, and survival probabilities.
 *
 * Survival is the stochastic half of the port. numpy's PCG64 stream is not
 * reproducible here and reproducing it would buy nothing, so agreement with
 * Python is asserted *distributionally*. The fixture records its simulation
 * count and this test sizes its own tolerance from it.
 */

import { describe, expect, it } from "vitest";

import { deriveBoard } from "../lib/board";
import { mulberry32 } from "../lib/rng";
import { kthSmallest, simulateSurvival, snakePicks, DEFAULT_SIMS } from "../lib/simDraft";
import { boardArrays } from "../lib/vona";
import { byPlayer, FIXTURE_NAMES, loadBundle, loadFixture } from "./golden";

describe("snake picks", () => {
  it("reverses every other round", () => {
    expect(snakePicks(1, 4, 4)).toEqual([1, 8, 9, 16]);
    expect(snakePicks(4, 4, 4)).toEqual([4, 5, 12, 13]);
    expect(snakePicks(2, 3, 3)).toEqual([2, 5, 8]);
  });

  it("covers every pick exactly once across all slots", () => {
    const teams = 12;
    const rounds = 15;
    const all = Array.from({ length: teams }, (_, i) => snakePicks(i + 1, teams, rounds)).flat();
    expect(new Set(all).size).toBe(teams * rounds);
    expect(Math.min(...all)).toBe(1);
    expect(Math.max(...all)).toBe(teams * rounds);
  });
});

describe("quickselect", () => {
  it("finds the k-th smallest for every k", () => {
    const values = [5, 1, 9, 3, 7, 2, 8];
    const sorted = [...values].sort((a, b) => a - b);
    for (let k = 1; k <= values.length; k++) {
      expect(kthSmallest(Float64Array.from(values), k)).toBe(sorted[k - 1]);
    }
  });

  it("handles duplicates and already-sorted input", () => {
    expect(kthSmallest(Float64Array.from([2, 2, 2, 2]), 3)).toBe(2);
    expect(kthSmallest(Float64Array.from([1, 2, 3, 4, 5]), 2)).toBe(2);
    expect(kthSmallest(Float64Array.from([5, 4, 3, 2, 1]), 4)).toBe(4);
  });
});

describe("survival", () => {
  const mu = Float64Array.from([1, 2, 3, 4, 5]);
  const sd = Float64Array.from([0.5, 0.5, 0.5, 0.5, 0.5]);
  const values = Float64Array.from([10, 8, 6, 4, 2]);
  const positions = ["RB", "RB", "WR", "WR", "TE"];

  it("keeps everyone alive at the very next pick", () => {
    const { pSurvives } = simulateSurvival(mu, sd, values, positions, 1, 500, mulberry32(1));
    for (const p of pSurvives) expect(p).toBe(1);
  });

  it("leaves nobody once the board is exhausted", () => {
    const { pSurvives } = simulateSurvival(mu, sd, values, positions, 99, 500, mulberry32(1));
    for (const p of pSurvives) expect(p).toBe(0);
  });

  it("is monotone: a later pick is never more likely to reach you", () => {
    const at3 = simulateSurvival(mu, sd, values, positions, 3, 4000, mulberry32(2)).pSurvives;
    const at5 = simulateSurvival(mu, sd, values, positions, 5, 4000, mulberry32(2)).pSurvives;
    for (let i = 0; i < at3.length; i++) {
      expect(at5[i]).toBeLessThanOrEqual(at3[i] + 1e-12);
    }
  });

  it("ranks earlier ADP as less likely to survive", () => {
    const { pSurvives } = simulateSurvival(mu, sd, values, positions, 3, 8000, mulberry32(3));
    for (let i = 1; i < pSurvives.length; i++) {
      expect(pSurvives[i]).toBeGreaterThanOrEqual(pSurvives[i - 1]);
    }
  });
});

describe.each(FIXTURE_NAMES)("survival agrees with Python: %s", (name) => {
  const fixture = loadFixture(name);
  const players = deriveBoard(loadBundle(fixture.format), fixture.league);
  const byId = byPlayer(players);

  it("matches within Monte Carlo error", () => {
    const { mu, sd } = boardArrays(players);
    const values = Float64Array.from(players.map((p) => p.vor ?? 0));
    const positions = players.map((p) => p.pos);

    const nSims = 40_000;
    const { pSurvives } = simulateSurvival(
      mu, sd, values, positions, fixture.survival.pick, nSims, mulberry32(11),
    );
    const mine = new Map(players.map((p, i) => [p.player_id, pSurvives[i]]));

    // Both sides are estimates. The standard error of a difference of two
    // proportions each at n simulations peaks at sqrt(2 * 0.25 / n); four of
    // those is a tolerance that is generous to noise and still tight enough to
    // catch a genuinely wrong model.
    const se = Math.sqrt((2 * 0.25) / Math.min(nSims, fixture.survival.n_sims));
    const tolerance = 4 * se;

    let worst = 0;
    let worstPlayer = "";
    for (const row of fixture.survival.values) {
      const got = mine.get(row.player_id);
      if (got === undefined) continue;
      const delta = Math.abs(got - row.p_available);
      if (delta > worst) {
        worst = delta;
        worstPlayer = byId.get(row.player_id)?.name ?? row.player_id;
      }
    }
    expect(worst, `worst disagreement on ${worstPlayer}`).toBeLessThan(tolerance);
  });

  it("orders players by survival the same way", () => {
    // Ordering is what a person acts on, and it is far less noisy than any
    // individual probability.
    const { mu, sd } = boardArrays(players);
    const values = Float64Array.from(players.map((p) => p.vor ?? 0));
    const { pSurvives } = simulateSurvival(
      mu, sd, values, players.map((p) => p.pos), fixture.survival.pick,
      DEFAULT_SIMS * 4, mulberry32(5),
    );

    const theirs = new Map(fixture.survival.values.map((r) => [r.player_id, r.p_available]));
    const ranked = players
      .map((p, i) => ({ id: p.player_id, mine: pSurvives[i], theirs: theirs.get(p.player_id) }))
      .filter((r) => r.theirs !== undefined)
      // Only players whose survival is genuinely uncertain; the 0s and 1s at
      // either end are ties and their relative order is meaningless.
      .filter((r) => r.theirs! > 0.02 && r.theirs! < 0.98);

    if (ranked.length < 5) return;
    const mineOrder = [...ranked].sort((a, b) => a.mine - b.mine).map((r) => r.id);
    const theirsOrder = [...ranked].sort((a, b) => a.theirs! - b.theirs!).map((r) => r.id);

    // Rank correlation rather than exact order: neighbouring probabilities
    // differ by less than simulation noise.
    const pos = new Map(theirsOrder.map((id, i) => [id, i]));
    const n = mineOrder.length;
    let d2 = 0;
    mineOrder.forEach((id, i) => {
      d2 += (i - pos.get(id)!) ** 2;
    });
    const rho = 1 - (6 * d2) / (n * (n * n - 1));
    expect(rho, "survival rank correlation with Python").toBeGreaterThan(0.98);
  });
});
