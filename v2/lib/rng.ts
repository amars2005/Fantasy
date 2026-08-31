/**
 * Seeded random numbers for the draft simulator.
 *
 * This deliberately does *not* reproduce numpy's PCG64 stream. Porting it would
 * be a lot of work for no benefit: the Monte Carlo is a statistical estimate, so
 * the right agreement test against Python is distributional, not bitwise. See
 * the tolerance classes in `v2/tests/README.md`.
 *
 * What does matter is that a given seed is reproducible *here*, so a board
 * re-renders identically for the same draft state.
 */

export interface Rng {
  /** Uniform in [0, 1). */
  next(): number;
  /** Standard normal. */
  normal(): number;
}

/**
 * mulberry32: small, fast, and good enough for 4000 draft simulations.
 * Period 2^32, which is far more than one board ever draws.
 */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;

  const next = (): number => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  // Box-Muller, caching the second variate rather than throwing it away.
  let spare: number | null = null;
  const normal = (): number => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    let v = 0;
    // Guard against log(0).
    while (u === 0) u = next();
    while (v === 0) v = next();
    const r = Math.sqrt(-2 * Math.log(u));
    const theta = 2 * Math.PI * v;
    spare = r * Math.sin(theta);
    return r * Math.cos(theta);
  };

  return { next, normal };
}

/** Fill a typed array with standard normals. */
export function fillNormals(out: Float64Array, rng: Rng): Float64Array {
  for (let i = 0; i < out.length; i++) out[i] = rng.normal();
  return out;
}
