/**
 * Monte Carlo draft simulation.
 *
 * Port of `src/draft/sim_draft.py`, minus `calibrate()`.
 *
 * Everything that matters on draft day is a question about *what will still be
 * there later*, not about who is best now. Bots are modelled by perturbing each
 * player's calibrated draft mean by his own observed dispersion and drafting in
 * the resulting order -- deliberately simple, because ADP is itself the
 * aggregate of thousands of real drafts, so reproducing it with its real spread
 * reproduces the room.
 *
 * Two deliberate differences from the Python
 * ------------------------------------------
 * 1. `calibrate()` is *not* here. It solves for the latent means that make
 *    simulated draft position reproduce the input ADP, it costs 15 iterations
 *    of 4000 simulations, and it depends only on ADP and dispersion -- never on
 *    league scoring. So it runs once in Python at export time and arrives as
 *    the `adp_mu` column.
 *
 * 2. No argsort. The Python builds a full (n_sims x n_players) rank matrix, but
 *    every question asked of it is `positions >= pick`, and a player's rank is
 *    at or beyond `pick` exactly when his score exceeds the (pick-1)-th
 *    smallest score in that simulation. So we take one order statistic per
 *    simulation by quickselect -- O(n) instead of O(n log n) -- and compare.
 *    Scores are continuous, so ties are measure-zero.
 */

import type { Rng } from "./rng";
import { mulberry32 } from "./rng";

export const DEFAULT_SIMS = 4000;

/**
 * ADP dispersion widens for late picks; a floor keeps early studs from being
 * treated as perfectly predictable.
 */
export const MIN_STDEV = 0.5;

/** Overall pick numbers for a given draft slot in a snake draft. */
export function snakePicks(slot: number, teams: number, rounds: number): number[] {
  const picks: number[] = [];
  for (let rnd = 0; rnd < rounds; rnd++) {
    picks.push(rnd % 2 === 0 ? rnd * teams + slot : rnd * teams + (teams - slot + 1));
  }
  return picks;
}

/**
 * The k-th smallest value (1-indexed) of `arr`, found by quickselect.
 *
 * Destructive: `arr` is reordered. Callers pass a scratch buffer.
 */
export function kthSmallest(arr: Float64Array, k: number): number {
  let lo = 0;
  let hi = arr.length - 1;
  const target = k - 1;

  while (lo < hi) {
    // Median-of-three pivot, which keeps nearly-sorted input off the
    // quadratic path -- scores are correlated with ADP order across sims.
    const mid = (lo + hi) >> 1;
    const a = arr[lo];
    const b = arr[mid];
    const c = arr[hi];
    const pivot = Math.max(Math.min(a, b), Math.min(Math.max(a, b), c));

    let i = lo;
    let j = hi;
    while (i <= j) {
      while (arr[i] < pivot) i++;
      while (arr[j] > pivot) j--;
      if (i <= j) {
        const tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
        i++;
        j--;
      }
    }
    if (target <= j) hi = j;
    else if (target >= i) lo = i;
    else return arr[target];
  }
  return arr[target];
}

export interface SurvivalResult {
  /** P(player is still on the board at `pick`), one per player. */
  pSurvives: Float64Array;
  /** Expected value of the best survivor at each position. */
  expectedBest: Record<string, number>;
}

/**
 * Survival probabilities and expected best-available, in one pass.
 *
 * `pick` is counted in *remaining* picks from now: 1 is the very next selection
 * in the room. `values` is whatever quantity "best available" should be
 * measured in -- VOR for the board, lineup-marginal value for VONA.
 */
export function simulateSurvival(
  mu: Float64Array,
  sd: Float64Array,
  values: Float64Array,
  positions: string[],
  pick: number,
  nSims: number = DEFAULT_SIMS,
  rng: Rng = mulberry32(0),
): SurvivalResult {
  const n = mu.length;
  const pSurvives = new Float64Array(n);

  // Map positions to small integers so the hot loop compares numbers.
  const posNames: string[] = [];
  const posIndex = new Int32Array(n);
  const lookup = new Map<string, number>();
  for (let i = 0; i < n; i++) {
    let id = lookup.get(positions[i]);
    if (id === undefined) {
      id = posNames.length;
      lookup.set(positions[i], id);
      posNames.push(positions[i]);
    }
    posIndex[i] = id;
  }

  const nPos = posNames.length;
  const bestSum = new Float64Array(nPos);
  const best = new Float64Array(nPos);
  const scores = new Float64Array(n);
  const scratch = new Float64Array(n);

  const k = pick - 1;

  for (let s = 0; s < nSims; s++) {
    for (let i = 0; i < n; i++) scores[i] = mu[i] + rng.normal() * sd[i];

    // Below the first pick nobody is gone; past the end of the board nobody
    // is left. Between, one order statistic decides every player at once.
    let threshold = -Infinity;
    if (k >= n) threshold = Infinity;
    else if (k > 0) {
      scratch.set(scores);
      threshold = kthSmallest(scratch, k);
    }

    best.fill(-Infinity);
    for (let i = 0; i < n; i++) {
      if (scores[i] > threshold) {
        pSurvives[i] += 1;
        const p = posIndex[i];
        if (values[i] > best[p]) best[p] = values[i];
      }
    }
    // Simulations where nothing at a position survives contribute zero.
    for (let p = 0; p < nPos; p++) {
      if (best[p] > -Infinity) bestSum[p] += best[p];
    }
  }

  for (let i = 0; i < n; i++) pSurvives[i] /= nSims;

  const expectedBest: Record<string, number> = {};
  for (let p = 0; p < nPos; p++) expectedBest[posNames[p]] = bestSum[p] / nSims;

  return { pSurvives, expectedBest };
}
