/**
 * Fantasy scoring: a pure dot product of a stat line against a rules vector.
 *
 * Port of `src/scoring.py`, which is already I/O-free and exhaustively tested.
 * Every projection, replacement level and VOR number downstream is denominated
 * in the points this file produces.
 *
 * Missing stats count as zero, so a partial line (a WR with no passing stats)
 * scores correctly without the caller padding it. Note that the Python keeps
 * this tolerance deliberately, and `scripts/export_v2_bundle.py` carries the
 * guard that catches an upstream column rename -- the tolerance is safe only
 * because that guard exists at build time.
 */

export type ScoringRules = Record<string, number>;
export type StatLine = Record<string, number | null | undefined>;

/** Score a single stat line. */
export function scoreLine(stats: StatLine, rules: ScoringRules): number {
  let total = 0;
  for (const stat in rules) {
    const v = stats[stat];
    if (v) total += v * rules[stat];
  }
  return total;
}

/**
 * Score many rows at once, given a columnar table.
 *
 * `columns` maps a stat name to its column index; `rows` are raw arrays as they
 * arrive from the exported bundle.
 */
export function scoreColumnar(
  rows: (number | string | null)[][],
  columns: string[],
  rules: ScoringRules,
): number[] {
  const idx: [number, number][] = [];
  for (const stat in rules) {
    const i = columns.indexOf(stat);
    if (i >= 0) idx.push([i, rules[stat]]);
  }
  return rows.map((row) => {
    let total = 0;
    for (const [i, pts] of idx) {
      const v = row[i];
      if (typeof v === "number") total += v * pts;
    }
    return total;
  });
}

/**
 * Points for a value falling in a banded scale, e.g. points allowed by a
 * defence. Bands are inclusive on both ends; anything unmatched scores zero.
 *
 * Mirrors `_band_expr` in `src/dst.py`.
 */
export function scoreBand(
  value: number,
  bands: { low: number; high: number; points: number }[],
): number {
  for (const b of bands) {
    if (value >= b.low && value <= b.high) return b.points;
  }
  return 0;
}
