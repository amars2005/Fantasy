/**
 * Reading the static bundle emitted by `scripts/export_v2_bundle.py`.
 *
 * Everything arrives as compact columnar JSON: a `columns` header plus `rows`
 * of raw arrays. That keeps the files small enough to serve from the edge (the
 * whole bundle is under a megabyte) without pulling in a parquet reader.
 */

export interface ColumnarTable {
  name: string;
  generated_at: string;
  columns: string[];
  rows: (number | string | null)[][];
  [meta: string]: unknown;
}

export interface Manifest {
  generated_at: string;
  season: number;
  fit_seasons: number[];
  components: string[];
  kicker_stats: string[];
  dst_events: string[];
  scoring_formats: string[];
  boards: { scoring: string; file: string; players: number; adp_as_of: string | null }[];
  notes: Record<string, string>;
}

/** Index of column name to position, for hot loops that avoid indexOf. */
export function columnIndex(table: ColumnarTable): Record<string, number> {
  const idx: Record<string, number> = {};
  table.columns.forEach((c, i) => {
    idx[c] = i;
  });
  return idx;
}

/** Pull one column out as numbers, treating null as zero. */
export function numericColumn(table: ColumnarTable, name: string): Float64Array {
  const i = table.columns.indexOf(name);
  if (i < 0) throw new Error(`${table.name}: no column "${name}"`);
  const out = new Float64Array(table.rows.length);
  for (let r = 0; r < table.rows.length; r++) {
    const v = table.rows[r][i];
    out[r] = typeof v === "number" ? v : 0;
  }
  return out;
}

/** Pull one column out as strings. */
export function stringColumn(table: ColumnarTable, name: string): string[] {
  const i = table.columns.indexOf(name);
  if (i < 0) throw new Error(`${table.name}: no column "${name}"`);
  return table.rows.map((r) => String(r[i] ?? ""));
}

/** Materialise the whole table as objects. Convenient, not for hot loops. */
export function toObjects<T = Record<string, unknown>>(table: ColumnarTable): T[] {
  return table.rows.map((row) => {
    const obj: Record<string, unknown> = {};
    table.columns.forEach((c, i) => {
      obj[c] = row[i];
    });
    return obj as T;
  });
}

/**
 * Round the way polars' `.round()` does: scale, round half to *even*, unscale.
 *
 * Verified against polars 1.44 rather than assumed, because the obvious guesses
 * are both wrong. `Math.round` breaks ties toward positive infinity, and
 * round-half-away-from-zero gets 14.25 -> 14.3 where polars gives 14.2. The
 * rule that actually reproduces it is banker's rounding on the *scaled* value:
 *
 *   14.25 -> 142.5 -> 142 (even) -> 14.2
 *   14.35 -> 143.5 -> 144 (even) -> 14.4
 *
 * Note the second case only lands on an exact .5 because the multiplication
 * itself rounds, which is why comparing against Python's own `round()` would
 * also disagree here. The reference is polars, since that is what produced
 * every number in the bundle.
 *
 * This matters more than it looks: projections are rounded to one decimal, and
 * a single-step difference propagates into VOR, tier boundaries and the order
 * of the board.
 */
export function roundTo(value: number, places: number): number {
  const f = 10 ** places;
  const scaled = value * f;
  const lower = Math.floor(scaled);
  const diff = scaled - lower;

  let rounded: number;
  if (diff > 0.5) rounded = lower + 1;
  else if (diff < 0.5) rounded = lower;
  // Exact tie: go to the even neighbour. `%` on a negative gives a negative
  // remainder, which still tests evenness correctly.
  else rounded = lower % 2 === 0 ? lower : lower + 1;

  return rounded / f;
}
