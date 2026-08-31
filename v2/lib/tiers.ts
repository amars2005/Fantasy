/**
 * Tiering, from plateaus in the projection curve.
 *
 * At the table you rarely need to know whether one receiver outranks another;
 * you need to know whether the player you want is the last of his kind. Tiers
 * answer that, and they are what tells you when to reach and when to wait.
 *
 * Port of `src/draft/tiers.py`. The boundaries come for free from the
 * projection method: isotonic regression pools adjacent ranks it cannot
 * statistically distinguish, so a plateau in the fitted curve *is* a set of
 * players the historical data says are interchangeable.
 *
 * This is the module that depends on `isotonic.ts` assigning a bit-identical
 * value to every point in a pooled block. The comparison below is an exact
 * float `!==`, exactly as the Python's `!=` is. If the fit ever returns values
 * that differ in the last bit within a plateau, every player becomes his own
 * tier and this silently stops doing anything useful -- which is why
 * `tiers.test.ts` asserts tier counts rather than only values.
 */

import type { Player, Position } from "./types";

/** Assign within-position tiers from plateaus in the projection curve. */
export function addTiers(
  players: Player[],
  valueKey: "proj_points" = "proj_points",
): Player[] {
  // Sort by position, then by value descending -- matching the Python's
  // sort(["pos", value_col], descending=[False, True]).
  const sorted = [...players].sort(
    (a, b) => (a.pos < b.pos ? -1 : a.pos > b.pos ? 1 : b[valueKey] - a[valueKey]),
  );

  const out: Player[] = [];
  let tier = 0;
  let prevPos: Position | null = null;
  let prevValue = Number.NaN;

  for (const p of sorted) {
    if (p.pos !== prevPos) {
      tier = 1;
      prevPos = p.pos;
    } else if (p[valueKey] !== prevValue) {
      // Exact float inequality, deliberately. See the header.
      tier += 1;
    }
    prevValue = p[valueKey];
    out.push({ ...p, tier });
  }

  // Second pass for tier size and the tier's headline value.
  const sizes = new Map<string, number>();
  const points = new Map<string, number>();
  for (const p of out) {
    const key = `${p.pos}|${p.tier}`;
    sizes.set(key, (sizes.get(key) ?? 0) + 1);
    points.set(key, Math.max(points.get(key) ?? -Infinity, p[valueKey]));
  }
  for (const p of out) {
    const key = `${p.pos}|${p.tier}`;
    p.tier_size = sizes.get(key);
    p.tier_points = points.get(key);
  }
  return out;
}

/** How many players remain in a given position's tier. */
export function playersLeftInTier(
  players: Player[],
  pos: Position,
  tier: number,
): number {
  return players.filter((p) => p.pos === pos && p.tier === tier).length;
}

/** One row per tier: who is in it and where it runs out. */
export interface TierSummary {
  pos: Position;
  tier: number;
  points: number;
  n: number;
  firstAdp: number;
  lastAdp: number;
  players: string[];
}

export function tierSummary(players: Player[]): TierSummary[] {
  const groups = new Map<string, Player[]>();
  for (const p of players) {
    const key = `${p.pos}|${p.tier}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(p);
  }

  const out: TierSummary[] = [];
  for (const [key, group] of groups) {
    const [pos, tier] = key.split("|");
    const byAdp = [...group].sort((a, b) => a.adp - b.adp);
    out.push({
      pos: pos as Position,
      tier: Number(tier),
      points: group[0].proj_points,
      n: group.length,
      firstAdp: byAdp[0].adp,
      lastAdp: byAdp[byAdp.length - 1].adp,
      players: byAdp.map((p) => p.name),
    });
  }
  return out.sort((a, b) => (a.pos < b.pos ? -1 : a.pos > b.pos ? 1 : a.tier - b.tier));
}
