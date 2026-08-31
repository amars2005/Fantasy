/**
 * Isotonic regression by pool-adjacent-violators.
 *
 * A faithful port of what `sklearn.isotonic.IsotonicRegression(increasing=False,
 * out_of_bounds="clip")` does in `src/project/consensus.py`, because the whole
 * projection method rests on it: the fitted curve is the projection, and its
 * plateaus *are* the tiers.
 *
 * The one non-negotiable property
 * ------------------------------
 * `src/draft/tiers.py` starts a new tier "wherever projected points actually
 * change" -- a literal `!=` on floats. That works only because isotonic
 * regression assigns a *bit-identical* value to every point in a pooled block.
 *
 * If this file computed each element's value separately (say, as a running
 * mean), values inside a plateau would differ in the last bit, the `!=` would
 * fire on every row, and every player would become his own tier. The board
 * would still render, the projections would still be right, and tiering -- the
 * one thing that tells you when to reach -- would be silently gone.
 *
 * So: a block holds one `value`, and every index in that block is assigned that
 * same number. `predict` short-circuits when the bracketing y values are equal.
 * `tiers.test.ts` asserts tier *counts*, not just values, because a tolerance
 * check on values would pass happily while tiers were shattered.
 */

export interface IsotonicFit {
  /** Ascending, de-duplicated x positions. */
  xThresholds: number[];
  /** Fitted y at each threshold. Equal values across a plateau are identical. */
  yThresholds: number[];
}

interface Block {
  /** Sum of weight * y over the block. */
  wy: number;
  /** Sum of weights. */
  w: number;
  /** Number of collapsed points, for expansion. */
  count: number;
  /** The single value shared by every point in this block. */
  value: number;
}

/**
 * Collapse duplicate x into one point carrying the weighted mean of y.
 *
 * sklearn does this in `_make_unique` before running PAVA, and it matters here:
 * the training x-axis is positional ADP rank, so rank 1 appears once per season
 * and would otherwise be ten separate points.
 */
function makeUnique(
  x: number[],
  y: number[],
  w: number[],
): { x: number[]; y: number[]; w: number[] } {
  const order = x
    .map((_, i) => i)
    // Sort by x, ties broken by y -- matching numpy's lexsort((y, X)).
    .sort((a, b) => (x[a] - x[b]) || (y[a] - y[b]));

  const ux: number[] = [];
  const uwy: number[] = [];
  const uw: number[] = [];

  // Accumulate weighted sums and divide once at the end rather than keeping a
  // running mean: fewer roundings, and it is what sklearn computes.
  for (const i of order) {
    const last = ux.length - 1;
    if (last >= 0 && ux[last] === x[i]) {
      uwy[last] += y[i] * w[i];
      uw[last] += w[i];
    } else {
      ux.push(x[i]);
      uwy.push(y[i] * w[i]);
      uw.push(w[i]);
    }
  }
  return { x: ux, y: uwy.map((wy, i) => wy / uw[i]), w: uw };
}

/**
 * Weighted pool-adjacent-violators, always fitting a non-decreasing sequence.
 *
 * Returns one value per input point, where points sharing a block share the
 * exact same number.
 */
function pava(y: number[], w: number[]): number[] {
  const blocks: Block[] = [];

  for (let i = 0; i < y.length; i++) {
    blocks.push({ wy: w[i] * y[i], w: w[i], count: 1, value: y[i] });

    // Merge backwards while the sequence decreases.
    while (blocks.length > 1) {
      const b = blocks[blocks.length - 1];
      const a = blocks[blocks.length - 2];
      if (a.value <= b.value) break;
      const wy = a.wy + b.wy;
      const wSum = a.w + b.w;
      blocks.splice(blocks.length - 2, 2, {
        wy,
        w: wSum,
        count: a.count + b.count,
        // Computed once. Every point in the block is assigned this exact
        // number below -- see the header comment.
        value: wy / wSum,
      });
    }
  }

  const out: number[] = [];
  for (const block of blocks) {
    for (let k = 0; k < block.count; k++) out.push(block.value);
  }
  return out;
}

/**
 * Fit a monotone curve of y on x.
 *
 * `increasing: false` (the case this project uses) imposes the one thing we are
 * confident about: being drafted earlier should not predict fewer points.
 */
export function fitIsotonic(
  x: number[],
  y: number[],
  increasing = true,
): IsotonicFit {
  if (x.length !== y.length) {
    throw new Error(`fitIsotonic: length mismatch ${x.length} vs ${y.length}`);
  }
  if (x.length === 0) {
    throw new Error("fitIsotonic: no observations");
  }

  const weights = new Array(x.length).fill(1);
  // A decreasing fit is an increasing fit of -y, negated back. Doing it this way
  // rather than writing a second loop keeps the block structure identical.
  const signed = increasing ? y : y.map((v) => -v);
  const unique = makeUnique(x, signed, weights);
  const fitted = pava(unique.y, unique.w);

  return {
    xThresholds: unique.x,
    yThresholds: increasing ? fitted : fitted.map((v) => -v),
  };
}

/**
 * Evaluate the fitted curve, clipping outside the observed range.
 *
 * Matches `out_of_bounds="clip"`: queries below the first threshold return the
 * first fitted value, queries above the last return the last.
 */
export function predictIsotonic(fit: IsotonicFit, xs: number[]): number[] {
  const { xThresholds: xt, yThresholds: yt } = fit;
  const n = xt.length;

  return xs.map((q) => {
    if (n === 1 || q <= xt[0]) return yt[0];
    if (q >= xt[n - 1]) return yt[n - 1];

    // Binary search for the bracketing interval [lo, lo + 1].
    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xt[mid] <= q) lo = mid;
      else hi = mid;
    }

    const y0 = yt[lo];
    const y1 = yt[lo + 1];
    // Inside a plateau, return the plateau's exact value. Interpolating would
    // give the same answer in IEEE arithmetic, but saying so explicitly is the
    // point: this is the invariant tiering depends on.
    if (y0 === y1) return y0;

    const x0 = xt[lo];
    const x1 = xt[lo + 1];
    return y0 + ((q - x0) / (x1 - x0)) * (y1 - y0);
  });
}

/** Convenience: fit and immediately evaluate on a grid. */
export function fitAndPredict(
  x: number[],
  y: number[],
  grid: number[],
  increasing = true,
): number[] {
  return predictIsotonic(fitIsotonic(x, y, increasing), grid);
}
