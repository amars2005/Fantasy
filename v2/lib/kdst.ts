/**
 * Kicker and defence projections.
 *
 * Port of `src/project/kdst.py`.
 *
 * These are two of the roster spots and they must be on the board even though
 * they are barely worth projecting: a pick that cannot be marked makes the pick
 * counter lag reality, and everything keyed to "how many picks until my next
 * turn" -- which is the whole basis of VONA -- drifts with it.
 *
 * The method differs from the skill positions in one way worth noticing. Here
 * the curve is fitted on *outcome* rank, not ADP rank: it describes the spread
 * of season finishes at each position and is then applied to draft order. That
 * is weaker, and deliberately so. Kicker season points have a year-over-year
 * correlation near zero, so there is nothing better to be had.
 */

import { columnIndex, roundTo, type ColumnarTable } from "./bundle";
import { fitIsotonic, predictIsotonic } from "./isotonic";
import { scoreBand, scoreColumnar, type ScoringRules } from "./scoring";
import type { BoardEntry } from "./project";
import type { DstScoring, Player } from "./types";

/** Defences and kickers are drafted late; games played is not modelled. */
const ASSUMED_GAMES = 16.0;

interface FlatCurve {
  grid: number[];
  mean: number[];
  sd: number;
}

/**
 * Fit finish-rank -> points from a set of (season, points) observations.
 *
 * Mirrors `_curve_for`: rank within season by points descending, then a
 * monotone decreasing fit, with a single pooled residual spread.
 */
function curveFor(seasons: number[], points: number[]): FlatCurve {
  // Rank descending within each season, 1-based, ties by first occurrence --
  // matching polars' `rank("ordinal", descending=True).over("season")`.
  const bySeason = new Map<number, number[]>();
  seasons.forEach((s, i) => {
    if (!bySeason.has(s)) bySeason.set(s, []);
    bySeason.get(s)!.push(i);
  });

  const ranks = new Array<number>(points.length);
  for (const ids of bySeason.values()) {
    const ordered = [...ids].sort((a, b) => points[b] - points[a]);
    ordered.forEach((i, r) => {
      ranks[i] = r + 1;
    });
  }

  const fit = fitIsotonic(ranks, points, false);
  const maxRank = Math.max(...ranks);
  const grid: number[] = [];
  for (let r = 1; r <= maxRank; r++) grid.push(r);

  const fitted = predictIsotonic(fit, ranks);
  const resid = points.map((p, i) => p - fitted[i]);
  const mean = resid.reduce((a, b) => a + b, 0) / resid.length;
  const sd = Math.sqrt(resid.reduce((a, b) => a + (b - mean) ** 2, 0) / resid.length);

  return { grid, mean: predictIsotonic(fit, grid), sd };
}

/** Season points per kicker, under this league's banded kicker scoring. */
export function kickerSeasonPoints(
  components: ColumnarTable,
  kickerScoring: ScoringRules,
): { seasons: number[]; points: number[] } {
  const idx = columnIndex(components);
  const points = scoreColumnar(components.rows, components.columns, kickerScoring);
  return {
    seasons: components.rows.map((r) => Number(r[idx.season])),
    points,
  };
}

/**
 * Season points per defence, under this league's event and band scoring.
 *
 * Scored per game and then summed, because the points- and yards-allowed bands
 * are non-linear and cannot be applied to a season total.
 */
export function dstSeasonPoints(
  components: ColumnarTable,
  dst: DstScoring,
): { seasons: number[]; points: number[]; teams: string[] } {
  const idx = columnIndex(components);
  const eventPoints = scoreColumnar(components.rows, components.columns, dst.events);

  const totals = new Map<string, { season: number; team: string; points: number }>();
  components.rows.forEach((row, i) => {
    const season = Number(row[idx.season]);
    const team = String(row[idx.team]);
    const pa = Number(row[idx.points_allowed] ?? 0);
    const ya = Number(row[idx.yards_allowed] ?? 0);
    const game =
      eventPoints[i] +
      scoreBand(pa, dst.pointsAllowedBands) +
      scoreBand(ya, dst.yardsAllowedBands);

    const key = `${team}|${season}`;
    const acc = totals.get(key) ?? { season, team, points: 0 };
    acc.points += game;
    totals.set(key, acc);
  });

  const rows = [...totals.values()];
  return {
    seasons: rows.map((r) => r.season),
    points: rows.map((r) => r.points),
    teams: rows.map((r) => r.team),
  };
}

/**
 * K and DST rows for the draft board, in the projections contract.
 *
 * `board` should be the K/DST subset of the current-season board bundle.
 */
export function projectKdst(
  board: BoardEntry[],
  kickerComponents: ColumnarTable,
  dstComponents: ColumnarTable,
  kickerScoring: ScoringRules,
  dst: DstScoring,
): Player[] {
  const kickers = board.filter((p) => p.pos === "K");
  const defences = board.filter((p) => p.pos === "DST");
  const out: Player[] = [];

  const apply = (entries: BoardEntry[], curve: FlatCurve) => {
    for (const e of entries) {
      const i = Math.min(Math.max(Math.trunc(e.pos_rank) - 1, 0), curve.grid.length - 1);
      out.push({
        ...e,
        proj_points: roundTo(curve.mean[i], 1),
        sd: roundTo(curve.sd, 1),
        games: ASSUMED_GAMES,
        proj_raw: curve.mean[i],
        sd_raw: curve.sd,
        games_raw: ASSUMED_GAMES,
        source: "consensus",
      });
    }
  };

  if (kickers.length) {
    const k = kickerSeasonPoints(kickerComponents, kickerScoring);
    apply(kickers, curveFor(k.seasons, k.points));
  }
  if (defences.length) {
    const d = dstSeasonPoints(dstComponents, dst);
    apply(defences, curveFor(d.seasons, d.points));
  }
  return out;
}
