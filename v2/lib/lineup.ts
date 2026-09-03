/**
 * Lineup-marginal player value.
 *
 * Port of the lineup half of `src/draft/vona.py`.
 *
 * Value over replacement asks "how much better is this player than a waiver
 * pickup". That is the wrong question at the table. Player value here is
 * *lineup-marginal*: how much a player adds to your optimal starting lineup
 * given the roster you already own. That makes the tool decline a third
 * quarterback without any hand-written rule, and prices the FLEX slot
 * correctly.
 *
 * The same question has to be answered for a player who does *not* start, and
 * answering it in raw points is what used to put a QB2 in round nine: a flat
 * share of projected points pays a quarterback more for being a quarterback. A
 * bench player is insurance, so he is priced over what you could stream at his
 * position and by how many starting slots he stands behind -- see `benchModel`.
 */

import { allocateFlex, replacementLevels } from "./replacement";
import type { LeagueConfig, Player, Position, RosterSlot } from "./types";

/**
 * What one starting slot loses to byes and injuries over a season, and so how
 * much of a season a bench player behind that slot actually plays: one bye plus
 * a couple of missed weeks out of seventeen.
 */
export const BENCH_WEIGHT = 0.2;

/**
 * How to price a player at this position who does not crack the lineup.
 *
 * `vacancies` is the expected number of this position's starting slots open in
 * a given week -- `BENCH_WEIGHT` per slot he could be promoted into, flex share
 * included. `baseline` is what he is promoted *over*: the best player at his
 * position you could still have for nothing.
 */
export interface BenchValue {
  vacancies: number;
  baseline: number;
}

/**
 * Share of weeks the 1st, 2nd, ... `depth`-th backup actually starts.
 *
 * Slots go vacant independently, so the number open in a given week is Poisson
 * with mean `vacancies`, and your k-th backup starts in the weeks at least k of
 * them are open. The weights sum back to `vacancies`, so a position's cover is
 * split between however many backups you own instead of each being paid in full
 * for it. That is what a hand-written "no more than two quarterbacks" rule is
 * really trying to say: behind one starting slot the third quarterback comes out
 * at 0.001 of a season on his own.
 */
export function promotionWeights(vacancies: number, depth: number): number[] {
  const out: number[] = [];
  let pmf = Math.exp(-vacancies); // P(X = k - 1)
  let tail = 1.0; // P(X >= k - 1)
  for (let k = 1; k <= depth; k++) {
    tail = Math.max(tail - pmf, 0);
    out.push(tail);
    pmf *= vacancies / k;
  }
  return out;
}

/**
 * Per-position bench pricing, derived from the pool still on the board.
 *
 * Two corrections, both of which a flat weight gets wrong:
 *
 * - **Baseline.** A backup is insurance, and insurance is worth what it saves
 *   you over the claim you would otherwise make -- the best player at that
 *   position you could still get for free. A QB2 projecting 249 behind a
 *   replacement level of 235 is insuring fourteen points, not 249.
 * - **Weight.** How often that cover gets used scales with how many starting
 *   slots it stands behind. A bench running back backs up two starters and a
 *   share of the FLEX; a QB2 backs up one quarterback. `allocateFlex` already
 *   works out how the flex slots land across positions for this pool, so the
 *   slot count is derived rather than assumed. `promotionWeights` then splits
 *   that cover across however many backups you already own, which is what stops
 *   a third quarterback being worth as much as the second.
 *
 * Both are computed against the *available* board, so the baseline falls as the
 * draft empties: in the last rounds "what you could get for free" really is a
 * waiver-wire body, and late fliers separate again.
 */
export function benchModel(
  players: Player[],
  league: LeagueConfig,
): Record<string, BenchValue> {
  const counts = allocateFlex(players, league);
  const levels = replacementLevels(players, league);

  const out: Record<string, BenchValue> = {};
  for (const [pos, baseline] of Object.entries(levels)) {
    out[pos] = {
      vacancies: (BENCH_WEIGHT * (counts[pos] ?? 0)) / league.teams,
      baseline,
    };
  }
  return out;
}

function superflexEligible(league: LeagueConfig): Position[] {
  const set = new Set<Position>(league.flexEligible);
  set.add("QB");
  return [...set];
}

/**
 * Points from the best legal starting lineup, plus discounted bench.
 *
 * Greedy fill is exact for this roster structure: required slots are filled
 * best-first, then the flex slots take the best eligible players left.
 *
 * `bench` prices whoever is left over, per position; see `benchModel`. Omitting
 * it falls back to a flat share of raw projected points, which is only right
 * when every player being compared plays the same position.
 */
export function optimalLineupPoints(
  roster: RosterSlot[],
  league: LeagueConfig,
  bench?: Record<string, BenchValue>,
): number {
  const byPos: Record<string, number[]> = {};
  for (const p of roster) {
    (byPos[p.pos] ??= []).push(p.proj_points);
  }
  for (const pos in byPos) byPos[pos].sort((a, b) => b - a);

  let total = 0;
  const used: Record<string, number> = {};
  for (const pos in byPos) used[pos] = 0;

  for (const slot in league.starters) {
    if (slot === "FLEX" || slot === "SUPERFLEX") continue;
    const take = (byPos[slot] ?? []).slice(0, league.starters[slot as Position] ?? 0);
    for (const pts of take) total += pts;
    used[slot] = take.length;
  }

  const fillFlex = (slotName: "FLEX" | "SUPERFLEX", eligible: Position[]) => {
    const slots = league.starters[slotName] ?? 0;
    if (slots <= 0) return;
    const candidates: { pts: number; pos: string }[] = [];
    for (const pos of eligible) {
      for (const pts of (byPos[pos] ?? []).slice(used[pos] ?? 0)) {
        candidates.push({ pts, pos });
      }
    }
    // Descending by points; ties broken by position name descending, matching
    // the Python's `candidates.sort(reverse=True)` over (pts, pos) tuples.
    candidates.sort((a, b) => b.pts - a.pts || (a.pos < b.pos ? 1 : a.pos > b.pos ? -1 : 0));
    for (const c of candidates.slice(0, slots)) {
      total += c.pts;
      used[c.pos] = (used[c.pos] ?? 0) + 1;
    }
  };

  fillFlex("FLEX", league.flexEligible);
  fillFlex("SUPERFLEX", superflexEligible(league));

  // Everyone still unused is bench -- at *any* position. Counting only
  // flex-eligible leftovers here would price a backup quarterback at exactly
  // zero, which is wrong: he covers a bye and an injury. But he covers it only
  // as well as he beats the man you would stream instead, which is what the
  // baseline subtracts; below it there is nothing to insure and the term is zero
  // rather than negative.
  for (const pos in byPos) {
    const leftovers = byPos[pos].slice(used[pos] ?? 0);
    if (leftovers.length === 0) continue;

    const value = bench?.[pos];
    if (!value) {
      for (const pts of leftovers) total += BENCH_WEIGHT * pts;
      continue;
    }
    const weights = promotionWeights(value.vacancies, leftovers.length);
    leftovers.forEach((pts, i) => {
      total += weights[i] * Math.max(0, pts - value.baseline);
    });
  }

  return total;
}

/** How much each available player would add to your optimal lineup. */
export function marginalValue(
  available: Player[],
  roster: RosterSlot[],
  league: LeagueConfig,
  bench: Record<string, BenchValue> = benchModel(available, league),
): Player[] {
  const base = optimalLineupPoints(roster, league, bench);
  return available.map((p) => ({
    ...p,
    marginal:
      optimalLineupPoints(
        [...roster, { pos: p.pos, proj_points: p.proj_points }],
        league,
        bench,
      ) - base,
  }));
}
