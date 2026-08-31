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
 */

import type { LeagueConfig, Player, Position, RosterSlot } from "./types";

/**
 * A player who does not crack your starting lineup still has worth: bye cover,
 * injury insurance, and the chance he outperforms a starter. Valuing him at
 * zero would make the tool refuse to build any depth at all.
 */
export const BENCH_WEIGHT = 0.2;

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
 */
export function optimalLineupPoints(
  roster: RosterSlot[],
  league: LeagueConfig,
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
  // zero, which is wrong: he covers a bye and an injury.
  let bench = 0;
  for (const pos in byPos) {
    for (const pts of byPos[pos].slice(used[pos] ?? 0)) bench += pts;
  }

  return total + BENCH_WEIGHT * bench;
}

/** How much each available player would add to your optimal lineup. */
export function marginalValue(
  available: Player[],
  roster: RosterSlot[],
  league: LeagueConfig,
): Player[] {
  const base = optimalLineupPoints(roster, league);
  return available.map((p) => ({
    ...p,
    marginal:
      optimalLineupPoints([...roster, { pos: p.pos, proj_points: p.proj_points }], league) -
      base,
  }));
}
