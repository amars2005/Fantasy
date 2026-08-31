/**
 * Market-implied projections: positional ADP rank -> expected fantasy points.
 *
 * Port of `src/project/consensus.py`.
 *
 * The market tells us the order players are drafted in. History tells us what a
 * player drafted at a given positional rank actually went on to score under
 * *this league's* rules. Composing the two gives a projection denominated in
 * points without inventing a single number of our own.
 *
 * Fitting against preseason rank rather than end-of-season finish is what keeps
 * this honest: the running back drafted first has averaged far less than the
 * eventual RB1 scores, and the difference is winner's curse. The same fit
 * yields empirically calibrated uncertainty, and it is large.
 *
 * This is the one place league scoring enters the projection. Everything the
 * curve is fitted on is re-scored from raw stat components first, which is what
 * makes arbitrary per-league scoring possible.
 */

import { columnIndex, roundTo, type ColumnarTable } from "./bundle";
import { fitIsotonic, predictIsotonic } from "./isotonic";
import { scoreColumnar, type ScoringRules } from "./scoring";
import type { Curve, Player, Position } from "./types";

/** Ranks either side of a point pooled when estimating outcome spread. */
export const SD_WINDOW = 8;

/** Below this many observations a position is not fitted at all. */
export const MIN_OBSERVATIONS = 30;

/** An NFL season is 18 weeks with one bye, so nobody plays more than 17. */
const MAX_GAMES = 17;

function populationStd(values: number[]): number {
  if (values.length === 0) return 0;
  let mean = 0;
  for (const v of values) mean += v;
  mean /= values.length;
  let acc = 0;
  for (const v of values) acc += (v - mean) ** 2;
  // Population standard deviation (ddof=0), matching numpy's default.
  return Math.sqrt(acc / values.length);
}

/**
 * Outcome spread at each rank, pooled over neighbouring ranks.
 *
 * Raw per-rank spread is estimated from about ten observations, which is far
 * too few. Pooling a window either side gives a usable number; a window with
 * fewer than five observations falls back to the overall spread.
 */
function rollingSd(ranks: number[], resid: number[], grid: number[]): number[] {
  const overall = populationStd(resid);
  return grid.map((r) => {
    const sample: number[] = [];
    for (let i = 0; i < ranks.length; i++) {
      if (Math.abs(ranks[i] - r) <= SD_WINDOW) sample.push(resid[i]);
    }
    return sample.length >= 5 ? populationStd(sample) : overall;
  });
}

/**
 * Fit per-position rank -> (expected points, spread, games) curves.
 *
 * `training` is a `training_<scoring>.json` table: one row per drafted
 * player-season, carrying its positional ADP rank and raw stat components.
 */
export function fitCurves(
  training: ColumnarTable,
  scoring: ScoringRules,
): Record<string, Curve> {
  const idx = columnIndex(training);
  const points = scoreColumnar(training.rows, training.columns, scoring);

  const byPos = new Map<string, number[]>();
  training.rows.forEach((row, i) => {
    const pos = String(row[idx.pos]);
    if (!byPos.has(pos)) byPos.set(pos, []);
    byPos.get(pos)!.push(i);
  });

  const curves: Record<string, Curve> = {};
  for (const [pos, rowIds] of byPos) {
    if (rowIds.length < MIN_OBSERVATIONS) continue;

    const ranks = rowIds.map((i) => Number(training.rows[i][idx.pos_rank]));
    const pts = rowIds.map((i) => points[i]);
    const games = rowIds.map((i) => Number(training.rows[i][idx.games] ?? 0));

    // Monotone decreasing: being drafted earlier should not predict fewer
    // points. Raw per-rank means are pure noise at the top, where there are
    // only about ten observations per rank.
    const fit = fitIsotonic(ranks, pts, false);
    const maxRank = Math.max(...ranks);
    const grid: number[] = [];
    for (let r = 1; r <= maxRank; r++) grid.push(r);

    const mean = predictIsotonic(fit, grid);
    const fittedAtRanks = predictIsotonic(fit, ranks);
    const resid = pts.map((p, i) => p - fittedAtRanks[i]);

    const gamesFit = fitIsotonic(ranks, games, false);
    const gamesGrid = predictIsotonic(gamesFit, grid).map((g) =>
      Math.min(Math.max(g, 0), MAX_GAMES),
    );

    curves[pos] = {
      grid,
      mean,
      sd: rollingSd(ranks, resid, grid),
      games: gamesGrid,
      n: rowIds.length,
    };
  }
  return curves;
}

function lookup(curve: Curve, rank: number, field: "mean" | "sd" | "games"): number {
  const i = Math.min(Math.max(Math.trunc(rank) - 1, 0), curve.grid.length - 1);
  return curve[field][i];
}

/** A draftable player as the board bundle supplies him, before projection. */
export interface BoardEntry {
  player_id: string;
  name: string;
  pos: Position;
  tm: string;
  adp: number;
  stdev: number;
  bye: number | null;
  pos_rank: number;
  adp_mu: number;
}

/**
 * Apply fitted curves to the current board.
 *
 * Players at positions with no fitted curve (K and DST, which are handled
 * separately) are skipped.
 */
export function projectPlayers(
  curves: Record<string, Curve>,
  board: BoardEntry[],
): Player[] {
  const out: Player[] = [];
  for (const entry of board) {
    const curve = curves[entry.pos];
    if (!curve) continue;
    const mean = lookup(curve, entry.pos_rank, "mean");
    const sd = lookup(curve, entry.pos_rank, "sd");
    const games = lookup(curve, entry.pos_rank, "games");
    out.push({
      ...entry,
      proj_points: roundTo(mean, 1),
      sd: roundTo(sd, 1),
      games: roundTo(games, 1),
      proj_raw: mean,
      sd_raw: sd,
      games_raw: games,
      source: "consensus",
    });
  }
  return out.sort((a, b) => a.adp - b.adp);
}
