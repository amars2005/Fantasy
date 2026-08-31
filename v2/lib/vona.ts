/**
 * VONA -- value over next available. The pick criterion.
 *
 * Port of the decision half of `src/draft/vona.py`.
 *
 * The alternative to drafting a receiver now is not a replacement-level
 * receiver; it is the best receiver still on the board at your next pick. VONA
 * subtracts the simulated expected best-available at that position when you
 * next choose, which is the number the board should actually be ranked by.
 */

import type { Rng } from "./rng";
import { mulberry32 } from "./rng";
import { marginalValue } from "./lineup";
import { DEFAULT_SIMS, MIN_STDEV, simulateSurvival } from "./simDraft";
import type { LeagueConfig, Player, Position, RosterSlot } from "./types";

/**
 * Latent draft means and dispersions for a board.
 *
 * Uses the precomputed `adp_mu` from the bundle. Falls back to raw ADP only if
 * it is absent, which biases late-round survival by around ten picks -- the
 * calibration exists precisely to remove that, so this is a loud fallback
 * rather than a silent one.
 */
export function boardArrays(players: Player[]): {
  mu: Float64Array;
  sd: Float64Array;
} {
  const mu = new Float64Array(players.length);
  const sd = new Float64Array(players.length);
  let missing = 0;

  players.forEach((p, i) => {
    if (p.adp_mu === undefined || p.adp_mu === null || Number.isNaN(p.adp_mu)) {
      missing++;
      mu[i] = p.adp;
    } else {
      mu[i] = p.adp_mu;
    }
    sd[i] = Math.max(p.stdev ?? MIN_STDEV, MIN_STDEV);
  });

  if (missing > 0) {
    console.warn(
      `boardArrays: ${missing} of ${players.length} players have no calibrated ` +
        "adp_mu; survival probabilities for them will be biased late.",
    );
  }
  return { mu, sd };
}

/**
 * Rank available players by value over what survives to your next pick.
 *
 * `picksUntilNext` is how many players come off the board before you choose
 * again -- for a snake draft, the gap between your consecutive picks.
 */
export function vonaTable(
  available: Player[],
  roster: RosterSlot[],
  picksUntilNext: number,
  league: LeagueConfig,
  nSims: number = DEFAULT_SIMS,
  rng: Rng = mulberry32(0),
): Player[] {
  const withMarginal = marginalValue(available, roster, league);

  const { mu, sd } = boardArrays(withMarginal);
  const values = new Float64Array(withMarginal.length);
  const positions: string[] = [];
  withMarginal.forEach((p, i) => {
    values[i] = p.marginal ?? 0;
    positions.push(p.pos);
  });

  const { pSurvives, expectedBest } = simulateSurvival(
    mu,
    sd,
    values,
    positions,
    picksUntilNext + 1,
    nSims,
    rng,
  );

  const out = withMarginal.map((p, i) => {
    const nextBest = expectedBest[p.pos] ?? 0;
    return {
      ...p,
      next_best: nextBest,
      p_survives: pSurvives[i],
      vona: (p.marginal ?? 0) - nextBest,
    };
  });

  return out.sort((a, b) => (b.vona ?? 0) - (a.vona ?? 0));
}

export interface Dropoff {
  pos: Position;
  best_now: number;
  next_best: number;
  best_player: string;
  dropoff: number;
}

/**
 * Per-position urgency: what you lose by waiting a round at each position.
 *
 * This is the pick signal. If quarterback urgency is zero, taking one is pure
 * waste.
 */
export function positionDropoff(vona: Player[]): Dropoff[] {
  const groups = new Map<Position, Player[]>();
  for (const p of vona) {
    if (!groups.has(p.pos)) groups.set(p.pos, []);
    groups.get(p.pos)!.push(p);
  }

  const out: Dropoff[] = [];
  for (const [pos, group] of groups) {
    const best = group.reduce((a, b) => ((b.marginal ?? 0) > (a.marginal ?? 0) ? b : a));
    const bestNow = best.marginal ?? 0;
    const nextBest = group[0].next_best ?? 0;
    out.push({
      pos,
      best_now: bestNow,
      next_best: nextBest,
      best_player: best.name,
      dropoff: bestNow - nextBest,
    });
  }
  return out.sort((a, b) => b.dropoff - a.dropoff);
}
