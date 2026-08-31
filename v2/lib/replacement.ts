/**
 * Replacement level for a specific league shape.
 *
 * Port of `src/draft/replacement.py`. Replacement level is what converts
 * "projected points" (which says quarterbacks are the most valuable players in
 * football) into "value over replacement" (which says they are not, because in
 * a 14-team league the 14th quarterback is nearly as good as the 3rd).
 *
 * It is derived, not hardcoded. Base starters come from the league config; the
 * FLEX slots go to whichever flex-eligible players actually earn them given the
 * projections, so replacement responds to the shape of this year's pool rather
 * than a rule of thumb.
 */

import type { LeagueConfig, Player, Position } from "./types";

/** Positions eligible for a SUPERFLEX slot: the flex positions plus QB. */
function superflexEligible(league: LeagueConfig): Position[] {
  const set = new Set<Position>(league.flexEligible);
  set.add("QB");
  return [...set];
}

/**
 * Number of *starters* at each position across the league, flex included.
 *
 * Base starters are filled first. Remaining flex slots go to the best players
 * left across all flex-eligible positions, which is what managers collectively
 * do.
 */
export function allocateFlex(
  players: Player[],
  league: LeagueConfig,
): Record<string, number> {
  const teams = league.teams;
  const counts: Record<string, number> = {};
  for (const p of players) {
    if (!(p.pos in counts)) counts[p.pos] = (league.starters[p.pos] ?? 0) * teams;
  }

  // Rank within position by projection, descending, 1-based.
  const byPos = new Map<Position, Player[]>();
  for (const p of players) {
    if (!byPos.has(p.pos)) byPos.set(p.pos, []);
    byPos.get(p.pos)!.push(p);
  }
  for (const list of byPos.values()) list.sort((a, b) => b.proj_points - a.proj_points);

  const fill = (slotName: "FLEX" | "SUPERFLEX", eligible: Position[]) => {
    const slots = (league.starters[slotName] ?? 0) * teams;
    if (slots <= 0) return;

    // Everyone at an eligible position who is not already a starter.
    const pool: Player[] = [];
    for (const pos of eligible) {
      const list = byPos.get(pos) ?? [];
      pool.push(...list.slice(counts[pos] ?? 0));
    }
    pool.sort((a, b) => b.proj_points - a.proj_points);
    for (const p of pool.slice(0, slots)) counts[p.pos] = (counts[p.pos] ?? 0) + 1;
  };

  fill("FLEX", league.flexEligible);
  fill("SUPERFLEX", superflexEligible(league));

  return counts;
}

/** Projected points of the best non-starter at each position. */
export function replacementLevels(
  players: Player[],
  league: LeagueConfig,
): Record<string, number> {
  const counts = allocateFlex(players, league);
  const levels: Record<string, number> = {};

  const byPos = new Map<Position, Player[]>();
  for (const p of players) {
    if (!byPos.has(p.pos)) byPos.set(p.pos, []);
    byPos.get(p.pos)!.push(p);
  }

  for (const [pos, list] of byPos) {
    if (list.length === 0) continue;
    const ranked = [...list].sort((a, b) => b.proj_points - a.proj_points);
    const idx = counts[pos] ?? 0;
    levels[pos] =
      idx >= ranked.length
        // Pool shallower than the number of starting slots: replacement is the
        // worst rostered player, i.e. effectively nothing here is free.
        ? ranked[ranked.length - 1].proj_points
        : ranked[idx].proj_points;
  }
  return levels;
}

/** Attach value over replacement and rank the board by it. */
export function addVor(players: Player[], league: LeagueConfig): Player[] {
  const levels = replacementLevels(players, league);

  const scored = players.map((p) => {
    const replacement = levels[p.pos] ?? 0;
    return { ...p, replacement, vor: p.proj_points - replacement };
  });

  // Ties broken by player id. The Python breaks them by whatever order the
  // frame happened to be in, which is fine for a single-shot script but would
  // let the board reshuffle between renders here. Ties are arbitrary either
  // way; this at least makes them stable.
  scored.sort((a, b) => b.vor - a.vor || (a.player_id < b.player_id ? -1 : 1));
  scored.forEach((p, i) => {
    p.vor_rank = i + 1;
  });
  return scored;
}
